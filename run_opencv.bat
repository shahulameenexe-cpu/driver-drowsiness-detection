@echo off
title Sentinel Drive - OpenCV Driver Drowsiness Detection System
cd /d "%~dp0"
echo ====================================================================
echo   Starting OpenCV Driver Drowsiness & Fatigue Detection System
echo ====================================================================
echo.
echo Installing requirements (if needed)...
python -m pip install -r requirements.txt
echo.
echo Starting OpenCV Camera Stream...
python drowsiness_detector.py
pause
