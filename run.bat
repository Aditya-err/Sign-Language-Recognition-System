@echo off
title Indian Sign Language Recognition System V2
echo =====================================================================
echo   Starting Indian Sign Language Recognition System V2...
echo =====================================================================

set PYTHON_EXE=D:\Anaconda3\envs\ai_sign_lang\python.exe
if not exist "%PYTHON_EXE%" (
    echo [WARNING] Conda environment python not found at %PYTHON_EXE%
    echo Falling back to system python...
    set PYTHON_EXE=python
)

echo Using Python: %PYTHON_EXE%
"%PYTHON_EXE%" run_app.py %*

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Application exited with error code %ERRORLEVEL%.
    pause
)
