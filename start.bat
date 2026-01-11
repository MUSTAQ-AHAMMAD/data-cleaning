@echo off
REM AI CRM Data Cleaning - Windows Startup Script

echo ==================================================
echo    AI CRM - Data Cleaning System
echo ==================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed.
    echo Please install Python 3.8 or higher from https://www.python.org/
    pause
    exit /b 1
)

REM Recommend virtual environment
if not defined VIRTUAL_ENV (
    echo WARNING: Recommendation: Use a virtual environment
    echo    python -m venv venv
    echo    venv\Scripts\activate
    echo.
    set /p continue="Continue without virtual environment? (y/n): "
    if /i not "%continue%"=="y" exit /b 1
)

echo Starting the application...
echo.

REM Check if dependencies are installed
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements.txt
    echo.
)

REM Start Streamlit
echo Starting Streamlit application...
echo The application will open in your browser automatically.
echo.
echo If it doesn't open automatically, navigate to:
echo   http://localhost:8501
echo.
echo Press Ctrl+C to stop the application.
echo.

streamlit run app.py
pause
