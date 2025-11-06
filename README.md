# MedicalPaperSummarizer

A machine learning project that fine-tunes a T5 model to generate structured summaries of medical research papers from PubMed abstracts.

## Overview

This project trains a sequence-to-sequence model to automatically generate comprehensive summaries of medical papers including:
- Plain-language summary
- Key findings
- Clinical relevance
- Methodology brief

## Requirements

- Python 3.12 (required - Python 3.13 has compatibility issues with PyTorch on macOS)
- PyTorch
- Transformers (Hugging Face)
- Datasets
- Other dependencies listed below

## Installation

### 1. Install Python 3.12

**Using Homebrew (recommended for macOS):**
```bash
brew install python@3.12
```

### 2. Create Virtual Environment

```bash
cd MedicalPaperSummarizer
/opt/homebrew/bin/python3.12 -m venv venv312
source venv312/bin/activate
```

### 3. Install Dependencies

```bash
pip install torch transformers datasets evaluate rouge-score sentencepiece accelerate
```

## Usage

### Training the Model

```bash
# Activate the virtual environment
source venv312/bin/activate

# Run the training script
python trainmodel_fixed.py
```

The script will:
1. Load PubMed articles from `pubmed_abstracts.json`
2. Process and structure the abstracts
3. Fine-tune the T5-small model
4. Save the trained model to `pubmed-summarizer-best/`

### Training Parameters

- Model: `google-t5/t5-small`
- Max steps: 5 (for quick testing)
- Batch size: 4
- Learning rate: 5e-5
- Device: CPU (configured for compatibility)

## Project Structure

```
MedicalPaperSummarizer/
├── trainmodel_fixed.py       # Main training script (Python 3.12 compatible)
├── trainmodel.py              # Original training script
├── trainmodel_v2.py           # Alternative version
├── pubmed_abstracts.json      # Input data (PubMed articles)
├── get_data.py                # Script to fetch PubMed data
├── venv312/                   # Python 3.12 virtual environment
├── pubmed-sum/                # Training outputs
└── pubmed-summarizer-best/    # Final trained model
```

## Known Issues & Solutions

### Python 3.13 Mutex Lock Issue

If you encounter `[mutex.cc : 452] RAW: Lock blocking` errors, you're likely using Python 3.13. This is a known PyTorch bug on macOS. **Solution: Use Python 3.12 as shown in the installation steps.**

### Overflow Error During Evaluation

Fixed in `trainmodel_fixed.py` by properly handling tensor conversions and clipping values to valid ranges.

## Data Format

The input data (`pubmed_abstracts.json`) should contain PubMed articles with structured abstracts including sections like:
- Background/Introduction
- Methods
- Results
- Conclusions

## Output Format

The model generates structured summaries in the following format:

```
# Plain-language summary
[3-sentence accessible summary]

# Key findings
- [Finding 1]
- [Finding 2]
- [Finding 3]
- [Finding 4]

# Clinical relevance
[Clinical implications and applications]

# Methodology brief
[Brief description of study methodology]
```

## Contributing

Feel free to submit issues and pull requests!

## License

MIT License

## Acknowledgments

- Built with Hugging Face Transformers
- Uses Google's T5 model
- PubMed data from NCBI
