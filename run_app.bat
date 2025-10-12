@echo off
title Deepfake Detection App
color 0A

echo.
echo ========================================
echo    Deepfake Detection App Launcher
echo ========================================
echo.

echo Installing requirements...
pip install -r requirements.txt

echo.
echo Starting Deepfake Detection App...
echo.
echo Access URLs:
echo   Local:  http://localhost:5000
echo   Mobile: http://YOUR_IP:5000
echo.
echo To find your IP address, run: ipconfig
echo.
echo Press Ctrl+C to stop the app
echo.

python web_app.py

pause
