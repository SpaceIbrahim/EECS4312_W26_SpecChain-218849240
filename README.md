# EECS4312_W26_SpecChain

Name: Muhammad Ibrahim  
Student #: 218849240  
Application: Calm - Sleep, Meditate, Relax

## Project Overview
This project analyzes user reviews for the Calm application and creates personas, requirements and tests using three different pipelines:

- Manual pipeline
- Automated pipeline
- Hybrid pipeline

The goal is to compare each pipeline on its persona, requirements and validation test creation and traceability.

## Dataset

### Data collection method
The raw review dataset was collected from the Google Play Store the `google_play_scraper` library.

### Original dataset
- File: `data/reviews_raw.jsonl`
- Source: Google Play Store
- Size: ~`6000`
- Collected: `5000`

### Final cleaned dataset
- File: `data/reviews_clean.jsonl`
- Final cleaned dataset size: **3930 reviews**

The cleaned dataset was produced by removing duplicates, empty reviews, and extremely short entries, then preprocessing the text by removing punctuation, special characters, emojis, and extra whitespace, converting text to lowercase, removing stop words, and lemmatizing the reviews.

## Repository Structure

- `data/`  
  Contains raw and cleaned datasets, dataset metadata, and review grouping files.

- `personas/`  
  Contains persona files for the manual, automated, and hybrid pipelines.

- `spec/`  
  Contains generated specifications for the manual, automated, and hybrid pipelines.

- `tests/`  
  Contains validation test files for the manual, automated, and hybrid pipelines.

- `metrics/`  
  Contains metrics for each pipeline and the summary comparison file.

- `src/`  
  Contains all python scripts used to collect, clean, generate, validate, and evaluate artifacts.

- `reflection/`  
  Contains the written reflection comparing the three pipelines.

## Files

### Data
- `data/reviews_raw.jsonl`
- `data/reviews_clean.jsonl`
- `data/dataset_metadata.json`
- `data/review_groups_manual.json`
- `data/review_groups_auto.json`
- `data/review_groups_hybrid.json`

### Personas
- `personas/personas_manual.json`
- `personas/personas_auto.json`
- `personas/personas_hybrid.json`

### Specifications
- `spec/spec_manual.md`
- `spec/spec_auto.md`
- `spec/spec_hybrid.md`

### Tests
- `tests/tests_manual.json`
- `tests/tests_auto.json`
- `tests/tests_hybrid.json`

### Metrics
- `metrics/metrics_manual.json`
- `metrics/metrics_auto.json`
- `metrics/metrics_hybrid.json`
- `metrics/metrics_summary.json`

### Source Code
- `src/00_validate_repo.py`
- `src/01_collect_or_import.py`
- `src/02_clean.py`
- `src/03_manual_coding_template.py`
- `src/04_personas_manual.py`
- `src/05_personas_auto.py`
- `src/06_spec_generate.py`
- `src/07_tests_generate.py`
- `src/08_metrics.py`
- `src/run_all.py`

### Prompts
- `prompts/prompt_auto.json`

### Reflection
- `reflection/reflection.md`

## How to Run

### 1. Validate repository structure
```bash
# Make sure you are in the repo directory
python3 src/00_validate_repo.py
````

### 2. Set your Groq API key

The automated pipeline requires a Groq API key.

#### macOS or Linux

```bash
export GROQ_API_KEY="your_key_here"
```
This is my free groq key. It expires in seven days.
```bash
GROQ_API_KEY="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"
```
#### Windows PowerShell

```powershell
$env:GROQ_API_KEY="your_key_here"
```

### 3. Run the full automated pipeline

```bash
python3 src/run_all.py
```

### 4. Open comparison results

Open `metrics/metrics_summary.json` for comparison results.

## Required Python Libraries

Before running the project, make sure these Python libraries are installed:

- groq
- scikit-learn
- numpy
- nltk
- google-play-scraper
- num2words
- sentence-transformers

You can install them with:

```bash
pip install groq scikit-learn numpy nltk spacy google-play-scraper num2words sentence-transformers