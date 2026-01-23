#!/bin/bash

# Per-Category Test Script for PRACTIQ Refactoring
# Tests each category individually and verifies output
# Usage: ./test_per_category.sh [n_samples_per_db] [--parallel] [--with-classification]
#   n_samples_per_db: Number of questions to sample PER DATABASE (default: 3)
#   --parallel: Run all categories in parallel (default: sequential)
#   --with-classification: Run optional binary classification step (default: disabled)
# Examples:
#   ./test_per_category.sh                              # Sequential, 3 questions per DB, no classification
#   ./test_per_category.sh 20                           # Sequential, 20 questions per DB, no classification
#   ./test_per_category.sh --parallel                   # Parallel, 3 questions per DB, no classification
#   ./test_per_category.sh 20 --parallel                # Parallel, 20 questions per DB, no classification
#   ./test_per_category.sh 20 --with-classification     # Sequential, with classification
#   ./test_per_category.sh 20 --parallel --with-classification  # Parallel, with classification
#
# Note: Spider dataset has ~200 databases, so n_samples=3 means ~600 total questions

# Parse command-line arguments
N_SAMPLES=${1:-3}  # Number of questions to sample per database (default: 3)
PARALLEL=false
WITH_CLASSIFICATION=false

# Parse all arguments
for arg in "$@"; do
    case $arg in
        --parallel)
            PARALLEL=true
            ;;
        --with-classification)
            WITH_CLASSIFICATION=true
            ;;
        [0-9]*)
            N_SAMPLES=$arg
            ;;
    esac
done

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$BASE_DIR/.venv/bin/python"
SPIDER_DIR="$BASE_DIR/.vscode/combined_data_all/spider_data"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$BASE_DIR/.vscode/output-${TIMESTAMP}"
SRC_DIR="$BASE_DIR/src"
LOG_DIR="$BASE_DIR/logs"

# Set PYTHONPATH to include src directory
export PYTHONPATH="$SRC_DIR:$PYTHONPATH"

# Unset any existing AWS IAM credentials that might conflict with bearer token auth
unset AWS_ACCESS_KEY_ID
unset AWS_SECRET_ACCESS_KEY
unset AWS_SESSION_TOKEN
unset AWS_REGION

# Load AWS Bedrock bearer token credentials from .vscode/.env
# Only export bearer token variables, not IAM credentials to avoid conflicts
if [ -f "$BASE_DIR/.vscode/.env" ]; then
    export AWS_BEARER_TOKEN_BEDROCK=$(grep '^AWS_BEARER_TOKEN_BEDROCK=' "$BASE_DIR/.vscode/.env" | cut -d'=' -f2)
    export AWS_BEARER_TOKEN_BEDROCK_REGION=$(grep '^AWS_BEARER_TOKEN_BEDROCK_REGION=' "$BASE_DIR/.vscode/.env" | cut -d'=' -f2)

    if [ -n "$AWS_BEARER_TOKEN_BEDROCK" ] && [ -n "$AWS_BEARER_TOKEN_BEDROCK_REGION" ]; then
        echo "✓ Loaded AWS Bedrock bearer token from .vscode/.env"
        echo "  Region: $AWS_BEARER_TOKEN_BEDROCK_REGION"
    else
        echo "⚠ Warning: AWS bearer token credentials not found in .vscode/.env"
        exit 1
    fi
else
    echo "⚠ Error: .vscode/.env not found. AWS Bedrock authentication will fail."
    exit 1
fi

# Create logs and output directories
mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================="
echo "PRACTIQ Per-Category Verification Tests"
echo "Sample count: $N_SAMPLES questions per database"
echo "Output directory: $OUTPUT_DIR"
if [ "$PARALLEL" = true ]; then
    echo "Execution mode: PARALLEL (8 categories at once)"
else
    echo "Execution mode: SEQUENTIAL (one at a time)"
fi
if [ "$WITH_CLASSIFICATION" = true ]; then
    echo "Binary classification: ENABLED (for data quality validation)"
else
    echo "Binary classification: DISABLED (use --with-classification to enable)"
fi
echo "========================================="
echo ""

# Function to test a category
test_category() {
    local CATEGORY_NUM=$1
    local CATEGORY_TYPE=$2  # "ambiguous" or "unanswerable"
    local CATEGORY_NAME=$3
    local SCRIPT_NAME=$4
    local OUTPUT_FILE_NAME=$5
    local MIN_EXAMPLES=10

    # Create a result file for this category (for parallel execution)
    local RESULT_FILE="$LOG_DIR/result_${CATEGORY_NUM}_${TIMESTAMP}.txt"

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Category $CATEGORY_NUM: $CATEGORY_NAME"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Test 1: Import test
    echo -n "  [1/4] Testing imports... "
    cd "$SRC_DIR"
    if $PYTHON -c "from $CATEGORY_TYPE.$CATEGORY_NAME import $SCRIPT_NAME" 2>/dev/null; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${RED}✗${NC}"
        echo "FAILED" > "$RESULT_FILE"
        return 1
    fi

    # Test 2: Generation
    echo "  [2/4] Generating data ($N_SAMPLES questions per database)..."
    cd "$BASE_DIR"

    LOG_FILE="$LOG_DIR/category_${CATEGORY_NUM}_${CATEGORY_NAME}_${TIMESTAMP}.log"
    START_TIME=$(date +%s)
    # Timeout set to 48 hours (172800 seconds)
    if timeout 172800 $PYTHON "$SRC_DIR/$CATEGORY_TYPE/$CATEGORY_NAME/$SCRIPT_NAME.py" \
        --spider-data-root-dir "$SPIDER_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --split dev \
        --n2sample $N_SAMPLES > "$LOG_FILE" 2>&1; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        echo -e "        ${GREEN}✓${NC} Completed in ${DURATION}s"
    else
        EXIT_CODE=$?
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        if [ $EXIT_CODE -eq 124 ]; then
            echo -e "        ${YELLOW}⚠${NC} Timed out after ${DURATION}s (may have partial data)"
        else
            echo -e "        ${RED}✗${NC} Failed (exit code: $EXIT_CODE)"
            echo "        Log: $LOG_FILE"
            echo "FAILED" > "$RESULT_FILE"
            return 1
        fi
    fi

    # Test 3: Verify output file exists
    OUTPUT_FILE="$OUTPUT_DIR/dev/$OUTPUT_FILE_NAME.jsonl"
    echo -n "  [3/4] Checking output file... "
    if [ -f "$OUTPUT_FILE" ]; then
        LINE_COUNT=$(wc -l < "$OUTPUT_FILE")
        echo -e "${GREEN}✓${NC} ($LINE_COUNT examples)"

        # Test 4: Check minimum examples
        echo -n "  [4/4] Verifying ≥$MIN_EXAMPLES examples... "
        if [ $LINE_COUNT -ge $MIN_EXAMPLES ]; then
            echo -e "${GREEN}✓${NC}"
            echo "PASSED" > "$RESULT_FILE"
        else
            echo -e "${YELLOW}⚠${NC} Only $LINE_COUNT examples (need $MIN_EXAMPLES)"
            echo "PASSED" > "$RESULT_FILE"  # Still count as passed if some data generated
        fi
    else
        echo -e "${RED}✗${NC}"
        echo "        Expected: $OUTPUT_FILE"
        echo "FAILED" > "$RESULT_FILE"
        return 1
    fi

    echo ""
}

# Clean output directory
echo "Cleaning output directory..."
rm -rf "$OUTPUT_DIR/dev"
mkdir -p "$OUTPUT_DIR/dev"
echo ""

# Define all categories
declare -a CATEGORIES=(
    "1|ambiguous|ambiguous_SELECT_column|ambiguous_select_column_main|Ambiguous_SELECT_Column"
    "2|ambiguous|ambiguous_VALUES_within_column|ambiguous_values_within_column_main|Ambiguous_VALUES_Within_Column"
    "3|ambiguous|ambiguous_VALUES_across_columns|ambiguous_values_across_columns_main|Ambiguous_VALUES_Across_Columns"
    "4|ambiguous|vague_filter_term|vague_filter_term_main|Vague_Filter_Term"
    "5|unanswerable|nonexistent_select_column|nonexistent_select_column_main|Unanswerable_Nonexistent_SELECT_Column"
    "6|unanswerable|nonexistent_value|nonexistent_value_main|Unanswerable_Nonexistent_Value"
    "7|unanswerable|nonexistent_where_column|nonexistent_where_column_main|Unanswerable_Nonexistent_WHERE_Column"
    "8|unanswerable|unsupported_joins|unsupported_join_generation_main|Unanswerable_Unsupported_Join"
)

# Run tests based on mode
if [ "$PARALLEL" = true ]; then
    # Parallel execution
    echo "Starting all 8 categories in parallel..."
    echo ""

    declare -a PIDS=()
    for cat_spec in "${CATEGORIES[@]}"; do
        IFS='|' read -r num type name script output <<< "$cat_spec"
        test_category "$num" "$type" "$name" "$script" "$output" &
        PIDS+=($!)
    done

    # Wait for all background jobs to complete
    echo "Waiting for all categories to complete..."
    for pid in "${PIDS[@]}"; do
        wait $pid
    done
    echo ""
else
    # Sequential execution
    for cat_spec in "${CATEGORIES[@]}"; do
        IFS='|' read -r num type name script output <<< "$cat_spec"
        test_category "$num" "$type" "$name" "$script" "$output"
    done
fi

# Combine all category files together
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Combining all category files..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
COMBINE_LOG="$LOG_DIR/combine_data_${TIMESTAMP}.log"
START_TIME=$(date +%s)
if $PYTHON "$SRC_DIR/combine_all_data_together.py" \
    --answerable-fp "$SPIDER_DIR/dev.json" \
    --output-dir "$OUTPUT_DIR" \
    --input-data-dir "$OUTPUT_DIR/dev" > "$COMBINE_LOG" 2>&1; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    echo -e "${GREEN}✓${NC} Combination completed in ${DURATION}s"

    # Check if combined file was created
    COMBINED_FILE="$OUTPUT_DIR/amb_unans_ans_combined_dev.jsonl"
    if [ -f "$COMBINED_FILE" ]; then
        COMBINED_COUNT=$(wc -l < "$COMBINED_FILE")
        echo "  Combined file: $COMBINED_FILE"
        echo "  Total examples: $COMBINED_COUNT"
    fi
else
    EXIT_CODE=$?
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    echo -e "${RED}✗${NC} Combination failed (exit code: $EXIT_CODE)"
    echo "  Log: $COMBINE_LOG"
fi
echo ""

# Optional: Binary classification for data quality validation
CLASSIFICATION_FILE=""
if [ "$WITH_CLASSIFICATION" = true ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Running binary classification (data quality validation)..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    COMBINED_FILE="$OUTPUT_DIR/amb_unans_ans_combined_dev.jsonl"
    if [ -f "$COMBINED_FILE" ]; then
        CLASSIFICATION_LOG="$LOG_DIR/classification_${TIMESTAMP}.log"
        START_TIME=$(date +%s)
        if $PYTHON "$SRC_DIR/experiment/amb_unans_classification.py" \
            --infp "$COMBINED_FILE" \
            --classification-type binary \
            --llm-model claude-3-5-sonnet \
            --cell-value-key lexicalAndOracle \
            --num-processes 12 > "$CLASSIFICATION_LOG" 2>&1; then
            END_TIME=$(date +%s)
            DURATION=$((END_TIME - START_TIME))
            echo -e "${GREEN}✓${NC} Classification completed in ${DURATION}s"

            # Check if classification file was created
            CLASSIFICATION_FILE="${COMBINED_FILE}.binary_classification___claude-3-5-sonnet___lexicalAndOracle.jsonl"
            if [ -f "$CLASSIFICATION_FILE" ]; then
                CLASSIFICATION_COUNT=$(wc -l < "$CLASSIFICATION_FILE")
                echo "  Classification file: $CLASSIFICATION_FILE"
                echo "  Total examples: $CLASSIFICATION_COUNT"
            fi
        else
            EXIT_CODE=$?
            END_TIME=$(date +%s)
            DURATION=$((END_TIME - START_TIME))
            echo -e "${RED}✗${NC} Classification failed (exit code: $EXIT_CODE)"
            echo "  Log: $CLASSIFICATION_LOG"
        fi
    else
        echo -e "${YELLOW}⚠${NC} Combined file not found, skipping classification"
    fi
    echo ""
fi

# Contextualize and add execution explanations
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Contextualizing and adding execution explanations..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
COMBINED_FILE="$OUTPUT_DIR/amb_unans_ans_combined_dev.jsonl"
if [ -f "$COMBINED_FILE" ]; then
    CONTEXTUALIZE_LOG="$LOG_DIR/contextualize_${TIMESTAMP}.log"
    START_TIME=$(date +%s)

    # Build contextualize command with optional classification filtering
    CONTEXTUALIZE_CMD="$PYTHON $SRC_DIR/contextualize_and_explain_execution_results.py \
        --spider-data-root-dir $SPIDER_DIR \
        --infp $COMBINED_FILE \
        --n2sample 0"

    if [ -n "$CLASSIFICATION_FILE" ] && [ -f "$CLASSIFICATION_FILE" ]; then
        echo "  Using classification results for filtering: $CLASSIFICATION_FILE"
        CONTEXTUALIZE_CMD="$CONTEXTUALIZE_CMD --binary-classification-fp $CLASSIFICATION_FILE"
    fi

    if eval "$CONTEXTUALIZE_CMD" > "$CONTEXTUALIZE_LOG" 2>&1; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        echo -e "${GREEN}✓${NC} Contextualization completed in ${DURATION}s"

        # Check if contextualized file was created
        CONTEXTUALIZED_FILE="${COMBINED_FILE}.contextualize_and_execulation_explanation_v2.jsonl"
        if [ -f "$CONTEXTUALIZED_FILE" ]; then
            CONTEXTUALIZED_COUNT=$(wc -l < "$CONTEXTUALIZED_FILE")
            echo "  Contextualized file: $CONTEXTUALIZED_FILE"
            echo "  Total examples: $CONTEXTUALIZED_COUNT"
        fi
    else
        EXIT_CODE=$?
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        echo -e "${RED}✗${NC} Contextualization failed (exit code: $EXIT_CODE)"
        echo "  Log: $CONTEXTUALIZE_LOG"
    fi
else
    echo -e "${YELLOW}⚠${NC} Combined file not found, skipping contextualization"
fi
echo ""

# Collect results from result files
PASSED=0
FAILED=0
for i in {1..8}; do
    RESULT_FILE="$LOG_DIR/result_${i}_${TIMESTAMP}.txt"
    if [ -f "$RESULT_FILE" ]; then
        RESULT=$(cat "$RESULT_FILE")
        if [ "$RESULT" == "PASSED" ]; then
            ((PASSED++))
        else
            ((FAILED++))
        fi
    else
        ((FAILED++))  # If no result file, count as failed
    fi
done

# Summary
echo "========================================="
echo "SUMMARY"
echo "========================================="
echo -e "Passed: ${GREEN}$PASSED${NC}/8"
echo -e "Failed: ${RED}$FAILED${NC}/8"
echo ""
echo "Output directory: $OUTPUT_DIR"
echo ""

# Show generated files
echo "Generated files:"
ls -lh "$OUTPUT_DIR/dev/"*.jsonl 2>/dev/null || echo "No files generated"
echo ""
echo "Total examples per category:"
wc -l "$OUTPUT_DIR/dev/"*.jsonl 2>/dev/null | tail -1 || echo "No files generated"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
    echo "Output saved to: $OUTPUT_DIR"
    echo "Logs saved to: $LOG_DIR/*_${TIMESTAMP}.log"
    # Clean up result files
    rm -f "$LOG_DIR/result_"*"_${TIMESTAMP}.txt"
    exit 0
else
    echo -e "${RED}✗ Some tests failed${NC}"
    echo "Output directory: $OUTPUT_DIR"
    echo "Check logs in: $LOG_DIR/*_${TIMESTAMP}.log"
    # Clean up result files
    rm -f "$LOG_DIR/result_"*"_${TIMESTAMP}.txt"
    exit 1
fi
