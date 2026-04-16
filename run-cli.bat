@echo off
setlocal

:: SEFER Vision — CLI Runner Script (Windows)
:: Usage: run-cli.bat [options]

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

:: Check virtual environment
if not defined VIRTUAL_ENV (
    if exist "venv\Scripts\activate.bat" (
        echo [→] Activating virtual environment...
        call venv\Scripts\activate
    )
)

:: Show help if no arguments
if "%~1"=="" (
    echo ========================================
    echo   SEFER Vision — CLI Mode
    echo ========================================
    echo.
    echo Usage: run-cli.bat [OPTIONS]
    echo.
    echo Examples:
    echo   run-cli.bat --input_dir data\raw --prefix SEFER
    echo   run-cli.bat --input_dir .\photos --limit 10 --visualize
    echo   run-cli.bat --conf 0.30 --ocr-conf 0.75 --clear-output
    echo.
    echo Common Options:
    echo   --input_dir DIR          Input directory with images (default: data\raw)
    echo   --output_dir DIR         Output directory (default: output)
    echo   --prefix PREFIX          File name prefix (default: SEFER)
    echo   --limit N                Process only N images
    echo   --visualize              Save OCR visualizations
    echo   --clear-output           Clear output directory before processing
    echo.
    echo Confidence Options:
    echo   --conf FLOAT             Detection confidence threshold (default: 0.2)
    echo   --ocr-conf FLOAT         OCR confidence threshold (default: 0.8)
    echo   --yolo-conf FLOAT        YOLO detection threshold (default: 0.1)
    echo.
    echo For all options:
    echo   run-cli.bat --help
    echo.
)

echo [→] Starting SEFER Vision Pipeline...
echo.

python -m scripts.run_pipeline %*

set EXIT_CODE=%ERRORLEVEL%

echo.
if %EXIT_CODE%==0 (
    echo [✓] Pipeline completed successfully
) else (
    echo [✗] Pipeline exited with code %EXIT_CODE%
)

pause
exit /b %EXIT_CODE%