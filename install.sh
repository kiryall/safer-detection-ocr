#!/bin/bash

echo "========================================"
echo "  SEFER Vision — Installation Script"
echo "========================================"
echo

# Check Python version
echo "[→] Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')

echo "Found Python: $PYTHON_VERSION"

if [[ "$PYTHON_VERSION" != 3.11* ]]; then
    echo "[✗] Error: Python 3.11 is required, but found $PYTHON_VERSION"
    echo "[i] Please install Python 3.11 and try again."
    echo
    echo "Options:"
    echo "  1. Download from https://python.org/downloads/"
    echo "  2. Use conda: conda create -n safer python=3.11"
    echo
    exit 1
fi

echo "[✓] Python version OK: $PYTHON_VERSION"
echo

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "[✗] Error: pyproject.toml not found"
    echo "[i] Please run this script from the project root directory."
    exit 1
fi

# Check for virtual environment
if [ -n "$VIRTUAL_ENV" ]; then
    echo "[✓] Virtual environment already activated"
    echo "    Location: $VIRTUAL_ENV"
else
    echo "[→] Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "[✗] Failed to create virtual environment"
        exit 1
    fi
    echo "[✓] Virtual environment created"
    echo
    echo "[→] Activating virtual environment..."
    source venv/bin/activate
    if [ $? -ne 0 ]; then
        echo "[✗] Failed to activate virtual environment"
        exit 1
    fi
    echo "[✓] Virtual environment activated"
fi
echo

# Upgrade pip
echo "[→] Upgrading pip..."
pip install --upgrade pip setuptools wheel
if [ $? -ne 0 ]; then
    echo "[✗] Failed to upgrade pip"
    exit 1
fi
echo "[✓] pip upgraded"
echo

# Install dependencies
echo "[→] Installing dependencies..."
pip install -e .
if [ $? -ne 0 ]; then
    echo "[✗] Failed to install dependencies"
    exit 1
fi
echo "[✓] Dependencies installed"
echo

# Check for models
echo "[→] Checking for model files..."
MISSING_MODELS=false

if [ ! -f "models/yolo/v1/best.pt" ]; then
    echo "[✗] YOLO model not found: models/yolo/v1/best.pt"
    MISSING_MODELS=true
fi

if [ ! -f "models/ocr/digit_detector.pt" ]; then
    echo "[✗] OCR model not found: models/ocr/digit_detector.pt"
    MISSING_MODELS=true
fi

if [ "$MISSING_MODELS" = true ]; then
    echo
    echo "[⚠] Warning: Some model files are missing!"
    echo "    The application will not work without models."
    echo
    echo "    To download models:"
    echo "      1. Contact the project maintainer"
    echo "      2. Or train your own models using scripts/train_yolo.py"
    echo
else
    echo "[✓] All model files found"
fi
echo

# Create necessary directories
echo "[→] Creating project directories..."
mkdir -p data/raw output logs
echo "[✓] Directories created"
echo

# Installation complete
echo "========================================"
echo "  Installation Complete!"
echo "========================================"
echo
echo "  Quick start:"
echo
echo "    CLI mode:"
echo "      ./run-cli.sh --input_dir data/raw --prefix SEFER"
echo
echo "    GUI mode:"
echo "      ./run-gui.sh"
echo
echo "  Or manually:"
echo "    source venv/bin/activate"
echo "    python -m scripts.run_pipeline --help"
echo "    streamlit run ui/app.py"
echo

if [ -z "$VIRTUAL_ENV" ]; then
    echo "[i] Note: Virtual environment was created."
    echo "       Activate it with: source venv/bin/activate"
    echo
fi