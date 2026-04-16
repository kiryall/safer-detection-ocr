@echo off
setlocal

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

:: SEFER Vision — GUI Runner Script (Windows)
:: Usage: run-gui.bat [port]

:: Default port
set PORT=%~1
if "%PORT%"=="" set PORT=8501

:: Check virtual environment
if not defined VIRTUAL_ENV (
    if exist "venv\Scripts\activate.bat" (
        echo [→] Activating virtual environment...
        call venv\Scripts\activate
    )
)

:: Check if streamlit is installed
where streamlit >nul 2>&1
if errorlevel 1 (
    echo [✗] Streamlit not found
    echo     Please run install.bat first
    pause
    exit /b 1
)

echo ========================================
echo   SEFER Vision — GUI Mode
echo ========================================
echo.
echo [→] Starting Streamlit server...
echo.
echo   Local URL: http://localhost:%PORT%
echo.
echo   Press Ctrl+C to stop
echo.

:: Run Streamlit
streamlit run ui\app.py --server.port=%PORT% --server.address=0.0.0.0