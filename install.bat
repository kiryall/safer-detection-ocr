@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

echo ========================================
echo   SEFER Vision — Installation Script
echo ========================================
echo.

:: Check Python version
echo [→] Checking Python version...
for /f "tokens=2" %%a in ('python --version 2^>^&1') do set PYTHON_VERSION=%%a

if not defined PYTHON_VERSION (
    echo [✗] Error: Python not found
    echo [i] Please install Python 3.11 and ensure it's in PATH.
    pause
    exit /b 1
)

echo Found Python: %PYTHON_VERSION%

if "%PYTHON_VERSION:~0,4%" neq "3.11" (
    echo [✗] Error: Python 3.11 is required, but found %PYTHON_VERSION%
    echo [i] Please install Python 3.11 and try again.
    echo.
    echo Options:
    echo   1. Download from https://python.org/downloads/
    echo   2. Use conda: conda create -n safer python=3.11
    pause
    exit /b 1
)

echo [✓] Python version OK: %PYTHON_VERSION%
echo.

:: Check if we're in the right directory
if not exist "pyproject.toml" (
    echo [✗] Error: pyproject.toml not found
    echo [i] Please run this script from the project root directory.
    pause
    exit /b 1
)

:: Check for virtual environment
if defined VIRTUAL_ENV (
    echo [✓] Virtual environment already activated
    echo    Location: %VIRTUAL_ENV%
) else (
    echo [→] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [✗] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [✓] Virtual environment created
    echo.
    echo [→] Activating virtual environment...
    call venv\Scripts\activate
    if errorlevel 1 (
        echo [✗] Failed to activate virtual environment
        pause
        exit /b 1
    )
    echo [✓] Virtual environment activated
)
echo.

:: Upgrade pip
echo [→] Upgrading pip...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo [✗] Failed to upgrade pip
    pause
    exit /b 1
)
echo [✓] pip upgraded
echo.

:: Install dependencies
echo [→] Installing dependencies...
pip install -e .
if errorlevel 1 (
    echo [✗] Failed to install dependencies
    pause
    exit /b 1
)
echo [✓] Dependencies installed
echo.

:: Check for models
echo [→] Checking for model files...
set MISSING_MODELS=false

if not exist "models\yolo\v1\best.pt" (
    echo [✗] YOLO model not found: models\yolo\v1\best.pt
    set MISSING_MODELS=true
)

if not exist "models\ocr\digit_detector.pt" (
    echo [✗] OCR model not found: models\ocr\digit_detector.pt
    set MISSING_MODELS=true
)

if "%MISSING_MODELS%"=="true" (
    echo.
    echo [⚠] Warning: Some model files are missing!
    echo     The application will not work without models.
    echo.
    echo     To download models:
    echo       1. Contact the project maintainer
    echo       2. Or train your own models using scripts\train_yolo.py
    echo.
) else (
    echo [✓] All model files found
)
echo.

:: Create necessary directories
echo [→] Creating project directories...
if not exist "data\raw" mkdir data\raw
if not exist "output" mkdir output
if not exist "logs" mkdir logs
echo [✓] Directories created
echo.

:: Installation complete
echo ========================================
echo   Installation Complete!
echo ========================================
echo.
echo   Quick start:
echo.
echo     CLI mode:
echo       run-cli.bat --input_dir data\raw --prefix SEFER
echo.
echo     GUI mode:
echo       run-gui.bat
echo.
echo   Or manually:
echo     venv\Scripts\activate
echo     python -m scripts.run_pipeline --help
echo     streamlit run ui\app.py
echo.

if not defined VIRTUAL_ENV (
    echo [i] Note: Virtual environment was created.
    echo        Activate it with: venv\Scripts\activate
    echo.
)

pause