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
outputs = model.generate(**inputs, max_length=256, num_beams=4)
summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n" + "="*60)
print("COMPLEX EXAMPLE:")
print("="*60)
print("Generated Summary:")
print(summary)
print("="*60)