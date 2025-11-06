#!/usr/bin/env python3
import os
import sys

# Critical: Set BEFORE any imports
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Import in this specific order to avoid mutex
import re
import json
import numpy as np

# Import torch WITHOUT calling any threading functions yet
import torch

# NOW set threading after import
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# Import transformers components one by one
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()

from datasets import Dataset, DatasetDict, logging as datasets_logging
datasets_logging.set_verbosity_error()

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

import evaluate

print("All imports successful!\n")

def normalize_abstract_text(abs_text):
    if isinstance(abs_text, str):
        return {"UNLABELED": abs_text.strip()}
    sections = {}
    if isinstance(abs_text, list):
        parts = []
        for chunk in abs_text:
            if isinstance(chunk, str):
                parts.append(("UNLABELED", chunk))
            elif isinstance(chunk, dict):
                label = (chunk.get("Label") or chunk.get("label") or "UNLABELED").upper()
                text  = chunk.get("#text") or chunk.get("text") or ""
                parts.append((label, text))
        for lab, txt in parts:
            sections[lab] = (sections.get(lab, "") + " " + (txt or "")).strip()
    return sections or {"UNLABELED": ""}

def pick(sentences, n):
    return " ".join(sentences[:n]).strip()

def split_sentences(text):
    text = re.sub(r'\s+', ' ', text).strip()
    return re.split(r'(?<=[\.!?])\s+', text) if text else []

def build_targets(example):
    mc   = example.get("MedlineCitation", {})
    art  = mc.get("Article", {}) or {}
    title = (art.get("ArticleTitle") or "").strip()
    abs_obj = art.get("Abstract", {}) or {}
    abs_sections = normalize_abstract_text(abs_obj.get("AbstractText"))

    bg   = abs_sections.get("BACKGROUND") or abs_sections.get("INTRODUCTION") or abs_sections.get("UNLABELED") or ""
    mtd  = abs_sections.get("METHODS") or abs_sections.get("MATERIALS AND METHODS") or ""
    res  = abs_sections.get("RESULTS") or ""
    conc = abs_sections.get("CONCLUSIONS") or abs_sections.get("INTERPRETATION") or ""

    all_text = " ".join(v for v in abs_sections.values() if v).strip()
    sents = split_sentences(all_text)

    plain_lang = (conc or bg or all_text)
    plain_lang = pick(split_sentences(plain_lang), 3)

    key_findings_source = (res or conc or all_text)
    key_sents = split_sentences(key_findings_source)[:4]

    clinical = conc or ""
    if not clinical and res:
        clinical = pick(split_sentences(res), 2)
    if not clinical and sents:
        clinical = pick(sents[-2:], 2)

    methodology = mtd or ""
    if not methodology and sents:
        meth_candidates = [x for x in sents if re.search(r'(random|double-blind|cohort|trial|prospective|retrospective|n=|participants|patients|subjects|dataset|sampling|assay|sequenc|R(CT)|placebo)', x, re.I)]
        methodology = " ".join(meth_candidates[:2]) if meth_candidates else pick(sents, 2)

    target = (
        f"# Plain-language summary\n"
        f"{plain_lang}\n\n"
        f"# Key findings\n" +
        "".join([f"- {kf}\n" for kf in key_sents if kf]) +
        (f"\n# Clinical relevance\n{clinical}\n\n" if clinical else "\n# Clinical relevance\n\n") +
        (f"# Methodology brief\n{methodology}\n" if methodology else "# Methodology brief\n")
    ).strip()

    input_text = f"{title}\n\n{all_text}".strip()

    return {
        "pmid": mc.get("PMID") or "",
        "title": title,
        "text": input_text,
        "summary": target
    }

if __name__ == '__main__':
    print("=== Medical Paper Summarizer ===\n")
    
    print("Loading data...")
    with open("pubmed_abstracts.json", "r") as f:
        data = json.load(f)
    
    articles = data.get("PubmedArticle", [])
    print(f"Loaded {len(articles)} articles")
    
    print("\nProcessing articles...")
    processed = []
    for i, article in enumerate(articles):
        try:
            result = build_targets(article)
            if 256 <= len(result["text"]) <= 8000 and len(result["summary"]) > 0:
                processed.append(result)
        except Exception as e:
            continue
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(articles)} articles...")
    
    print(f"\nValid articles: {len(processed)}")
    
    # Deduplicate
    seen = set()
    deduplicated = []
    for ex in processed:
        if ex["pmid"] not in seen:
            seen.add(ex["pmid"])
            deduplicated.append(ex)
    
    print(f"After deduplication: {len(deduplicated)} articles")
    
    # Create dataset
    print("\nCreating dataset...")
    dataset = Dataset.from_list(deduplicated)
    
    # Split
    print("Splitting dataset...")
    split = dataset.train_test_split(test_size=0.2, seed=42)
    dataset = DatasetDict({"train": split["train"], "validation": split["test"]})
    
    print(f"Train: {len(dataset['train'])} | Validation: {len(dataset['validation'])}")
    
    # Load tokenizer
    model_name = "google-t5/t5-small"
    print(f"\nLoading tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    prefix = "summarize: "
    
    def preprocess_function(examples):
        inputs = [prefix + x for x in examples["text"]]
        model_inputs = tokenizer(inputs, max_length=1024, truncation=True)
        labels = tokenizer(text_target=examples["summary"], max_length=384, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    print("Tokenizing dataset...")
    tokenized = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    print(f"\nLoading model: {model_name}...")
    device = "cpu"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    print("Model loaded!")
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    print("\nLoading ROUGE metric...")
    rouge = evaluate.load("rouge")
    
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        
        # Convert to proper format and handle overflow
        if isinstance(preds, tuple):
            preds = preds[0]
        
        # Ensure predictions are in valid range and convert to list of lists
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        preds = np.clip(preds, 0, tokenizer.vocab_size - 1).astype(np.int32)
        
        # Decode predictions
        decoded_preds = []
        for pred in preds:
            decoded_preds.append(tokenizer.decode(pred.tolist(), skip_special_tokens=True))
        
        # Process labels
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        labels = np.clip(labels, 0, tokenizer.vocab_size - 1).astype(np.int32)
        
        # Decode labels
        decoded_labels = []
        for label in labels:
            decoded_labels.append(tokenizer.decode(label.tolist(), skip_special_tokens=True))
        
        scores = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        return {k: round(v * 100, 2) for k, v in scores.items()}
    
    args = Seq2SeqTrainingArguments(
        output_dir="pubmed-sum",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        num_train_epochs=3,
        lr_scheduler_type="linear",
        warmup_ratio=0.03,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=1,
        predict_with_generate=True,
        generation_max_length=128,
        generation_num_beams=4,
        dataloader_num_workers=0,
        load_best_model_at_end=True,
        metric_for_best_model="rougeLsum",
        greater_is_better=True,
        fp16=False,
        bf16=False,
        report_to="none",
        seed=42,
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    print("\n=== Starting training ===\n")
    trainer.train()
    
    print("\n=== Training complete! ===")
    print("Saving model...")
    trainer.save_model("pubmed-summarizer-best")
    tokenizer.save_pretrained("pubmed-summarizer-best")
    print("Done!")

test_abstract = """
Background: Chronic obstructive pulmonary disease (COPD) is characterized by progressive airflow limitation. 
Current treatments focus on symptom management but have limited impact on disease progression. 
Novel anti-inflammatory approaches targeting the IL-17 pathway have shown promise in preclinical studies.
Methods: This multicenter, double-blind, randomized controlled trial enrolled 847 patients with moderate to severe COPD 
across 45 sites in 12 countries. Patients were randomized 1:1 to receive either monthly subcutaneous injections of 
IL-17 inhibitor (150mg) or placebo for 52 weeks. Primary endpoint was change in forced expiratory volume (FEV1) 
from baseline to week 52. Secondary endpoints included exacerbation rate, quality of life scores, and adverse events.
Results: The treatment group showed significant improvement in FEV1 compared to placebo (mean difference 125mL, 95% CI 89-161, p<0.001).
Annual exacerbation rate was reduced by 42% in the treatment group (rate ratio 0.58, 95% CI 0.46-0.73, p<0.001).
Quality of life scores improved significantly with treatment (SGRQ difference -4.2 points, p=0.002).
Serious adverse events were comparable between groups (18% treatment vs 21% placebo).
Conclusions: IL-17 inhibition represents a novel therapeutic approach for COPD with significant benefits in lung function,
exacerbation reduction, and quality of life, with an acceptable safety profile. Further long-term studies are warranted
to evaluate sustained efficacy and optimal patient selection.
"""

inputs = tokenizer("summarize: " + test_abstract, return_tensors="pt", max_length=1024, truncation=True)
outputs = model.generate(**inputs, max_length=256, num_beams=4)  # Note: longer max_length
summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Generated Summary:")
print(summary)