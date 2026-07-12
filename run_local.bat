@echo off
setlocal

where python >nul 2>&1
if errorlevel 1 (
  echo Python nao foi encontrado. Instale o Python 3.11 ou 3.12.
  pause
  exit /b 1
)

if not exist .venv (
  python -m venv .venv
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py

pause
