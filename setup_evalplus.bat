@echo off
REM Exit on any error
setlocal enabledelayedexpansion

echo ================================
echo Checking Python installation...
echo ================================

REM Check if Python is installed
python --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo Python is not installed. Please install Python before running this script.
    exit /b
)

REM Create a virtual environment in the current directory
echo Creating a Python virtual environment in the current directory...
python -m venv venv

REM Activate the virtual environment
echo Activating the virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
pip install --upgrade pip

REM Install required packages
echo Installing EvalPlus and dependencies...
pip install "evalplus[vllm]" --upgrade
pip install --upgrade "evalplus[vllm] @ git+https://github.com/evalplus/evalplus"

REM Set Hugging Face token (replace <your_huggingface_token> with your token)
echo Setting Hugging Face token...
set "HF_TOKEN=sample_token"

REM Run evalplus.codegen
echo Running EvalPlus code generation...
evalplus.codegen --model "mistralai/Mistral-7B-Instruct-v0.3" --greedy --root .\results\humaneval --dataset humaneval --backend hf

echo ================================
echo EvalPlus code generation completed!
echo ================================
pause
