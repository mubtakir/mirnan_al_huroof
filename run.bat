@echo off
title Mirnan - Letter Physics Field Analyzer
cd /d "%~dp0"
set PYTHONPATH=%~dp0

echo.
echo ==================================================
echo        Mirnan - Letter Physics Field Analyzer
echo ==================================================
echo.
echo  1. Interactive Terminal (CLI)
echo  2. Web Interface (Web UI - http://127.0.0.1:8000)
echo  3. Run Automated Tests (pytest)
echo  4. Exit
echo.

set /p CHOICE="Select option (1-4): "

if "%CHOICE%"=="1" (
    echo.
    echo Starting CLI...
    python cli.py -i
    pause
    goto :eof
)
if "%CHOICE%"=="2" (
    echo.
    echo Starting FastAPI server at http://127.0.0.1:8000
    echo Press Ctrl+C to stop.
    echo.
    python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
    pause
    goto :eof
)
if "%CHOICE%"=="3" (
    echo.
    echo Running tests...
    python -m pytest tests/test_word_physics.py -v
    pause
    goto :eof
)
if "%CHOICE%"=="4" (
    exit /b
)

echo.
echo Invalid option.
pause