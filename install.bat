@echo off
REM Installation script for MVP detector (Windows)

echo Installing MVP: Motion Vector Propagation for Zero-Shot Object Detection
echo ========================================================================

REM Check Python version
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://python.org
    pause
    exit /b 1
)

echo Python found ✓

REM Create virtual environment (optional)
set /p create_venv="Do you want to create a virtual environment? (y/n): "
if /i "%create_venv%"=="y" (
    echo Creating virtual environment...
    python -m venv mvp_env
    call mvp_env\Scripts\activate.bat
    echo Virtual environment created and activated.
)

REM Install requirements
echo Installing Python dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

REM Install motion vector extractor
echo Installing motion vector extractor...
cd mv-extractor
pip install -e .
cd ..

REM Install MVP package
echo Installing MVP package...
pip install -e .

echo.
echo Installation completed successfully!
echo.
echo To get started:
echo 1. Activate the virtual environment (if created): mvp_env\Scripts\activate.bat
echo 2. Run the basic example: python examples\basic_usage.py
echo 3. Check the README.md for more detailed usage instructions
echo.
echo For evaluation, make sure you have:
echo - Video files in your dataset directory
echo - Motion vectors extracted using the mv-extractor tool
echo - Ground truth annotations in the correct format
pause
