#!/bin/bash

# SEFER Vision — CLI Runner Script (Linux)
# Usage: ./run-cli.sh [options]

SCRIPT_DIR=$(dirname "$(realpath "$0")")
cd "$SCRIPT_DIR"

# Check virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -f "venv/bin/activate" ]; then
        echo "[→] Activating virtual environment..."
        source venv/bin/activate
    fi
fi

# Show help if no arguments
if [ $# -eq 0 ]; then
    echo "========================================"
    echo "  SEFER Vision — CLI Mode"
    echo "========================================"
    echo
    echo "Usage: ./run-cli.sh [OPTIONS]"
    echo
    echo "Examples:"
    echo "  ./run-cli.sh --input_dir data/raw --prefix SEFER"
    echo "  ./run-cli.sh --input_dir ./photos --limit 10 --visualize"
    echo "  ./run-cli.sh --conf 0.30 --ocr-conf 0.75 --clear-output"
    echo
    echo "Common Options:"
    echo "  --input_dir DIR          Input directory with images (default: data/raw)"
    echo "  --output_dir DIR         Output directory (default: output)"
    echo "  --prefix PREFIX          File name prefix (default: SEFER)"
    echo "  --limit N                Process only N images"
    echo "  --visualize              Save OCR visualizations"
    echo "  --clear-output           Clear output directory before processing"
    echo
    echo "Confidence Options:"
    echo "  --conf FLOAT             Detection confidence threshold (default: 0.2)"
    echo "  --ocr-conf FLOAT         OCR confidence threshold (default: 0.8)"
    echo "  --yolo-conf FLOAT        YOLO detection threshold (default: 0.1)"
    echo
    echo "For all options:"
    echo "  ./run-cli.sh --help"
    echo
fi

echo "[→] Starting SEFER Vision Pipeline..."
echo

python -m scripts.run_pipeline "$@"

EXIT_CODE=$?

echo
if [ $EXIT_CODE -eq 0 ]; then
    echo "[✓] Pipeline completed successfully"
else
    echo "[✗] Pipeline exited with code $EXIT_CODE"
fi