# PRACTIQ: Comprehensive Setup and Usage Guide

This is a guide for setting up and running the PRACTIQ (Ambiguous and Unanswerable Text-to-SQL) data generation pipeline.

Paper: [PRACTIQ: A Practical Conversational Text-to-SQL dataset with Ambiguous and Unanswerable Queries](https://aclanthology.org/2025.naacl-long.13/) (Dong et al., NAACL 2025)

## Table of Contents

1. [Overview](#overview)
2. [Python Environment Setup](#python-environment-setup)
3. [Project Structure](#project-structure)
4. [Running the Pipeline](#running-the-pipeline)
5. [Testing and Verification](#testing-and-verification)
6. [Advanced Usage](#advanced-usage)

---

## Python Environment Setup

⚠️ **Platform Support Notice:**
- **Ubuntu/Linux (Recommended)**: tested on Ubuntu 20.04 with Python 3.10
- **macOS**: may encounter compatibility issues with `typer` library command-line argument parsing. To troubleshoot, one may need to update the typer.Option to typer.Argument to make the scripts work.

**Requirements:**
- **Python Version**: 3.10.x (REQUIRED - Python 3.11+ will not work due to dependency compatibility)
- **Package Manager**: `uv` (recommended, 5x faster than pip) or `pip`
- **API Access**: AWS Bedrock access with bearer token authentication or modify to use other LLM providers or local ones via litellm (key file to edit is `src/litellm_helpers.py`).

**IMPORTANT**: The codebase requires Python 3.10 and was developed/tested on Ubuntu Linux.

### Step 1: Clone the repo

Example project structure after clone and setup and generation:

```
ambi-unans-text-to-sql/
├── .venv/                          # Python virtual environment
├── .vscode/
│   ├── .env                        # Authentication credentials (not in git)
│   ├── settings.json               # VS Code Python interpreter config
│   ├── combined_data_all/          # requires manual downloading of the SPIDER data
│   │   └── spider/                 # Spider dataset (dev.json, train.json, etc.)
│   │       ├── dev.json            # Dev set for answerable questions
│   │       ├── tables.json         # Database schema definitions
│   │       └── database/           # SQLite database files
│   └── output-YYYYMMDD_HHMMSS/     # Generated data (timestamped directories)
│       └── dev/                    # Per-category JSONL files
├── src/                            # Source code
│   ├── ambiguous/                  # 4 ambiguous categories
│   │   ├── ambiguous_SELECT_column/
│   │   ├── ambiguous_VALUES_within_column/
│   │   ├── ambiguous_VALUES_across_columns/
│   │   └── vague_filter_term/
│   ├── unanswerable/               # 4 unanswerable categories
│   │   ├── nonexistent_select_column/
│   │   ├── nonexistent_value/
│   │   ├── nonexistent_where_column/
│   │   └── unsupported_joins/
│   ├── experiment/                 # Evaluation and classification scripts
│   ├── litellm_helpers.py          # LiteLLM unified interface
│   ├── custom_sql_engine.py        # SQL execution engine
│   ├── combine_all_data_together.py      # Stage 2: Combine categories
│   ├── contextualize_and_explain_execution_results.py  # Stage 4: Finalize
│   └── utils.py                    # Utility functions
├── logs/                           # Execution logs (created during runs)
├── test/                           # Unit tests
├── test_per_category.sh            # Main test script for E2E pipeline
├── requirements.txt                # Python dependencies
└── README.md                       # Project overview
```

- **`test_per_category.sh`**: Orchestrates the entire pipeline (see next section)
- **`src/litellm_helpers.py`**: Configures LiteLLM with 8 model endpoints
- **`src/combine_all_data_together.py`**: Merges 8 categories + answerable questions
- **`src/contextualize_and_explain_execution_results.py`**: Adds natural language explanations

### Step 2: Authentication Configuration: Create .env File

**Security Note**: Ensure the `.vscode/.env` file is excluded from version control via `.gitignore`. Never commit authentication credentials to the repository.

The project uses AWS Bedrock for LLM API calls. Authentication is handled via bearer token stored in an environment file. You can modify the implementation to use other LLMs by changing `src/litellm_helpers.py`.

```bash
# Create the .vscode directory if it doesn't exist
mkdir -p .vscode

# Create the .env file
cat > .vscode/.env << 'EOF'
AWS_BEARER_TOKEN_BEDROCK=your-actual-bearer-token-here
EOF
```

Replace `your-actual-bearer-token-here` with your actual AWS bearer token.

### Step 3: Extract Spider Dataset

Download the public SPIDER dataset (https://drive.google.com/file/d/1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J/view?usp=sharing) to `.vscode/spider_data.zip` file and extract the proper location:

```bash
# Create the target directory
mkdir -p .vscode/combined_data_all

# Extract the Spider dataset
cd .vscode/combined_data_all
unzip ../spider_data.zip

# Return to project root
cd ../..
```

### Step 4: Install uv (if not already installed)

`uv` is a fast Python package installer written in Rust. It's significantly faster than pip.

### Step 5: Create Virtual Environment and Install Dependencies

**Note:** This project requires Python 3.10. The following instructions are for **Ubuntu/Linux systems** (the primary development environment).

**macOS users:** See the [macOS Installation Guide](#macos-installation-guide) at the end of this README if `uv` does NOT work for you.

#### Using uv (Recommended - Fast)

```bash
# Navigate to the project root
cd /path/to/ambi-unans-text-to-sql-cloned-repo

# Create virtual environment with Python 3.10
uv venv .venv --python 3.10 --seed

# Activate the virtual environment
source .venv/bin/activate

# Verify Python version in the environment
python --version  # Should show Python 3.10.x
python -c "import nltk; nltk.download('punkt')"  # install NLTK and download required resources
which python
which pip  # check that nltk is installed in the correct virtual env

# Install all dependencies using uv (much faster than pip)
uv pip install -r requirements.txt
```


On subsequent uses, simply activate the environment:
```bash
source .venv/bin/activate
```

### Step 6: Verify Installation

```bash
# Test basic imports
python -c "import litellm; import sqlglot; import pandas; import torch; print('✓ Basic imports successful!')"

# Test project module imports (all 8 categories)
export PYTHONPATH="$PWD/src:$PYTHONPATH"

# Test ambiguous categories (4)
python -c "from ambiguous.ambiguous_SELECT_column import ambiguous_select_column_main; print('✓ Ambiguous SELECT Column')"
python -c "from ambiguous.ambiguous_VALUES_within_column import ambiguous_values_within_column_main; print('✓ Ambiguous VALUES Within Column')"
python -c "from ambiguous.ambiguous_VALUES_across_columns import ambiguous_values_across_columns_main; print('✓ Ambiguous VALUES Across Columns')"
python -c "from ambiguous.vague_filter_term import vague_filter_term_main; print('✓ Vague Filter Term')"

# Test unanswerable categories (4)
python -c "from unanswerable.nonexistent_select_column import nonexistent_select_column_main; print('✓ Nonexistent SELECT Column')"
python -c "from unanswerable.nonexistent_value import nonexistent_value_main; print('✓ Nonexistent Value')"
python -c "from unanswerable.nonexistent_where_column import nonexistent_where_column_main; print('✓ Nonexistent WHERE Column')"
python -c "from unanswerable.unsupported_joins import unsupported_join_generation_main; print('✓ Unsupported Join')"
```

If all commands succeed, your environment is correctly set up!

---

## Running the Pipeline

The main entry point is `test_per_category.sh`, which runs the complete 4-stage pipeline.

### Pipeline Overview

```
Stage 1: Generate 8 Categories (parallel/sequential)
  ↓
Stage 2: Combine All Categories + Answerable
  ↓
Stage 3: Binary Classification (optional quality validation)
  ↓
Stage 4: Contextualize & Add Execution Explanations
```

### Usage

```bash
./test_per_category.sh [n_samples_per_db] [--parallel] [--with-classification]
```

Note that be default, previous LLM call response will be cached, to delete these, run:
```bash
rm -rf src/.vscode/.litellm_cache  # remove litellm cache
rm -rf .vscode/.litellm_cache  # remove litellm cache
clean_caches.sh  # remove simple cache
```

**Parameters**:
- `n_samples_per_db`: Number of questions to sample PER DATABASE (default: 3)
  - Spider has ~200 databases, so `n_samples=3` means ~600 total questions
  - Set to `0` to use ALL questions (no sampling) - full dataset
- `--parallel`: Run all 8 categories simultaneously (default: sequential)
- `--with-classification`: Enable binary classification for quality validation (default: disabled)

### Example Commands

**Quick Test** (3 questions per DB, sequential):
```bash
./test_per_category.sh 3
```

**Medium Test** (20 questions per DB, sequential):
```bash
./test_per_category.sh 20
```

**Parallel Execution** (3 questions per DB, all categories in parallel, may result in throttling exception if you do NOT have enough LLM API call quota):
```bash
./test_per_category.sh 3 --parallel
```

**Production Run** (full dataset, parallel, with classification for quality validation):
```bash
./test_per_category.sh 0 --parallel --with-classification
```

Note: `n_samples_per_db=0` means "use all questions" (no sampling)

### What Happens During Execution

1. **Stage 1: Category Generation** (8 categories)
   - For each category:
     - Import test: Verifies Python module can be imported
     - Generation: Runs the category script with Spider dataset
     - Output verification: Checks that JSONL file was created

   Generated files: `.vscode/output-{timestamp}/dev/{Category}.jsonl`

2. **Stage 2: Data Combination**
   - Merges all 8 category files + answerable questions from Spider dev.json
   - Standardizes category names and formats

   Output: `.vscode/output-{timestamp}/amb_unans_ans_combined_dev.jsonl`

3. **Stage 3: Binary Classification** (if `--with-classification` is used)
   - Uses Claude 3.5 Sonnet to classify each example
   - Filters out misclassified examples

   Output: `.vscode/output-{timestamp}/amb_unans_ans_combined_dev.jsonl.binary_classification___claude-3-5-sonnet___lexicalAndOracle.jsonl`

4. **Stage 4: Contextualization**
   - Rephrases templated responses to be more natural
   - Adds natural language explanations of SQL execution results

   Final output: `.vscode/output-{timestamp}/amb_unans_ans_combined_dev.jsonl.contextualize_and_execulation_explanation_v2.jsonl`


## Advanced Usage

### Running Individual Categories

Instead of running all categories via `test_per_category.sh`, you can run individual category scripts:

First generate each category of the unanswerable/ambiguous data.

```bash
# Set up environment
export PYTHONPATH="$PWD/src:$PYTHONPATH"
source .venv/bin/activate

# Run a specific category
PYTHON=".venv/bin/python"
SPIDER_DIR=".vscode/combined_data_all/spider"
OUTPUT_DIR=".vscode/output-manual-test"
mkdir -p $OUTPUT_DIR

# Example: Ambiguous SELECT Column
$PYTHON src/ambiguous/ambiguous_SELECT_column/ambiguous_select_column_main.py \
    --spider-data-root-dir $SPIDER_DIR \
    --output-dir $OUTPUT_DIR \
    --split dev \
    --n2sample 5
```

### Running Only Specific Pipeline Stages

```bash

# Combine existing category files
python src/combine_all_data_together.py \
    --answerable-fp .vscode/combined_data_all/spider/dev.json \
    --output-dir .vscode/output-20251229_153045 \
    --input-data-dir .vscode/output-20251229_153045/dev

# Binary classification
python src/experiment/amb_unans_classification.py \
    --infp .vscode/output-20251229_153045/amb_unans_ans_combined_dev.jsonl \
    --classification-type binary \
    --llm-model claude-3-5-sonnet \
    --cell-value-key lexicalAndOracle \
    --num-processes 12

# Contextualization
python src/contextualize_and_explain_execution_results.py \
    --spider-data-root-dir .vscode/combined_data_all/spider \
    --infp .vscode/output-20251229_153045/amb_unans_ans_combined_dev.jsonl \
    --n2sample 0
```

### Checking logs or Debugging Failed Runs

Check logs for detailed error messages:
```bash
# List all log files
ls -lht logs/*.log | head

# View most recent category log
tail -100 logs/category_1_ambiguous_SELECT_column_*.log

# Search for errors
grep -i "error\|exception\|failed" logs/category_*.log
```

---

## macOS Installation Guide

> **⚠️ Note for macOS Users:** This section is ONLY for users who want to run this project on macOS (especially Apple Silicon/ARM64). If you are using Ubuntu/Linux, please follow the main installation instructions above.

For macOS users, `requirements-macos.txt` has updated versions that have prebuilt wheels for torch, transformers, tokenizers, huggingface-hub, etc.

### Installation Steps

```bash
# Navigate to the project root
cd /path/to/ambi-unans-text-to-sql-cloned-repo

# Create a conda environment with Python 3.10
conda create -n ambi-text-to-sql python=3.10 -y

# Activate the conda environment
conda activate ambi-text-to-sql

# Verify Python version
python --version  # Should show Python 3.10.x

# Install dependencies using the macOS requirements file
pip install -r requirements-macos.txt
```

On subsequent uses:
```bash
conda activate ambi-text-to-sql
```

---

## Citation

If you find this work useful or use this codebase, please cite:

```bibtex
@inproceedings{dong-etal-2025-practiq,
    title = "{PRACTIQ}: A Practical Conversational Text-to-{SQL} dataset with Ambiguous and Unanswerable Queries",
    author = "Dong, Mingwen  and
      Ashok Kumar, Nischal  and
      etc",
    editor = "Chiruzzo, Luis  and
      Ritter, Alan  and
      Wang, Lu",
    booktitle = "Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = apr,
    year = "2025",
    address = "Albuquerque, New Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.naacl-long.13/",
    doi = "10.18653/v1/2025.naacl-long.13",
    pages = "255--273",
    ISBN = "979-8-89176-189-6"
}
```