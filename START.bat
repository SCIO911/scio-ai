@echo off
title SCIO AI-Workstation
color 0A

echo.
echo +======================================================================+
echo ^|   SCIO - Service Computer Intelligence Organization                  ^|
echo ^|   AI-WORKSTATION v2.0 - VOLLAUTOMATISCH                              ^|
echo ^|   Alle Abhaengigkeiten in C:\SCIO enthalten                          ^|
echo +======================================================================+
echo.

cd /d "C:\SCIO"

:: Use local venv Python - no external dependencies
if exist "C:\SCIO\venv\Scripts\python.exe" (
    echo [OK] Python venv gefunden
    echo [OK] Starte mit lokaler Python-Umgebung...
    echo.
    echo ========================================
    echo   Server wird gestartet...
    echo   http://localhost:5000
    echo   Admin: http://localhost:5000/admin
    echo ========================================
    echo.

    C:\SCIO\venv\Scripts\python.exe C:\SCIO\start.py
) else (
    echo [ERROR] Python venv nicht gefunden in C:\SCIO\venv
    echo.
    echo [INFO] Setup-Anleitung:
    echo   1. python -m venv C:\SCIO\venv
    echo   2. C:\SCIO\venv\Scripts\pip install -r C:\SCIO\requirements.txt
    echo.
)

pause
