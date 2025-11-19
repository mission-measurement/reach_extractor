# Reach Extraction System

A program for extracting reach information from program descriptions using NLP and machine learning.

## Overview

This system automatically identifies how many people a program serves by analyzing text descriptions. It extracts numbers, types of beneficiaries (families, students, etc.), and provides confidence scores for each extraction.

## Features

- Extracts beneficiary counts from unstructured text
- Multi-stage validation for accuracy
- Filters out false positives (years, addresses, 501c3 references)
- Three processing modes (fast, balanced, full)
- Confidence scoring for each extraction

## Requirements

```
transformers==4.35.0
torch==2.0.1
sentence-transformers==2.2.2
spacy==3.6.1
nltk==3.8.1
scikit-learn==1.3.0
pandas==1.5.3
numpy==1.24.3
tqdm==4.66.1
```

## Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Usage

```python
from reach_extractor import MaxAccuracyExtractor
import pandas as pd

# Initialize
extractor = MaxAccuracyExtractor(validation_mode='balanced')

# Load data
df = pd.read_csv('programs.csv')

# Process
results = extractor.process_dataframe(
    df,
    description_column='program_description',
    id_column='program_id'
)

# View results
print(results[results['has_reach']])
```

## Processing Modes

- **Fast**: Pattern matching only (1000+ programs/min)
- **Balanced**: Patterns + AI validation (100-200 programs/min)
- **Full**: All validators (30-50 programs/min)

## How It Works

1. Pattern matching to find candidate numbers
2. AI validation using BART model
3. Semantic similarity checking
4. Context analysis of surrounding text
5. Syntactic parsing for grammar validation

## Performance

- F1 Score: ~85%
- Precision: 80-90%
- Recall: 75-85%


## License

MIT
