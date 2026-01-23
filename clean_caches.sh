#!/bin/bash

# Clean all cache directories from the PRACTIQ project

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================="
echo "PRACTIQ Cache Cleanup Script"
echo "========================================="
echo ""

# Count caches before removal
CACHE_COUNT=$(find "$BASE_DIR/src" -type d \( -name "__cache__" -o -name "__pycache__" \) 2>/dev/null | wc -l)
LITELLM_CACHE_COUNT=$(find "$BASE_DIR" -type d -name ".litellm_cache" 2>/dev/null | wc -l)

echo "Found caches to remove:"
echo "  - Python/joblib caches (__cache__, __pycache__): $CACHE_COUNT"
echo "  - LiteLLM caches (.litellm_cache): $LITELLM_CACHE_COUNT"
echo ""

# Remove __cache__ directories (joblib cache from category modules)
echo "Removing __cache__ directories..."
find "$BASE_DIR/src" -type d -name "__cache__" -exec rm -rf {} + 2>/dev/null
echo "✓ Removed __cache__ directories"

# Remove __pycache__ directories (Python bytecode cache)
echo "Removing __pycache__ directories..."
find "$BASE_DIR/src" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
echo "✓ Removed __pycache__ directories"

# Remove .litellm_cache directories (LiteLLM API response cache)
echo "Removing .litellm_cache directories..."
rm -rf "$BASE_DIR/.vscode/.litellm_cache" 2>/dev/null
rm -rf "$BASE_DIR/src/.vscode/.litellm_cache" 2>/dev/null
echo "✓ Removed .litellm_cache directories"

# Remove pytest cache
echo "Removing pytest cache..."
rm -rf "$BASE_DIR/.pytest_cache" 2>/dev/null
echo "✓ Removed pytest cache"

echo ""
echo "========================================="
echo "Cache cleanup completed successfully!"
echo "========================================="
echo ""
echo "Note: Caches will be regenerated automatically on next run."
echo "      LiteLLM cache removal will force real API calls (no cached responses)."
