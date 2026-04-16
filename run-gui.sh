#!/bin/bash

# SEFER Vision — GUI Runner Script (Linux)
# Usage: ./run-gui.sh [port]

SCRIPT_DIR=$(dirname "$(realpath "$0")")
cd "$SCRIPT_DIR"

# Default port
PORT=${1:-8501}

# Check virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -f "venv/bin/activate" ]; then
        echo "[→] Activating virtual environment..."
        source venv/bin/activate
    fi
fi

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "[✗] Streamlit not found"
    echo "    Please run ./install.sh first"
    exit 1
fi

echo "========================================"
echo "  SEFER Vision — GUI Mode"
echo "========================================"
echo
echo "[→] Starting Streamlit server..."
echo
echo "  Local URL: http://localhost:$PORT"
echo
echo "  Press Ctrl+C to stop"
echo

# Run Streamlit
streamlit run ui/app.py --server.port=$PORT --server.address=0.0.0.0