@echo off
title SCIO - Autostart Installation
color 0B

echo.
echo ========================================
echo   SCIO AUTOSTART INSTALLATION
echo ========================================
echo.
echo Dieses Script richtet SCIO so ein,
echo dass es automatisch beim Windows-Start läuft.
echo.
echo [1] Windows-Aufgabe wird erstellt...

:: Create scheduled task
schtasks /create /tn "SCIO AI-Workstation" /tr "powershell.exe -ExecutionPolicy Bypass -WindowStyle Hidden -File C:\SCIO\START-AUTO.ps1" /sc onlogon /rl highest /f

if errorlevel 1 (
    echo.
    echo [FEHLER] Konnte Aufgabe nicht erstellen.
    echo Bitte als Administrator ausfuehren!
    pause
    exit /b 1
)

echo.
echo [2] Firewall-Regel wird erstellt...

:: Add firewall rule
netsh advfirewall firewall add rule name="SCIO AI-Workstation" dir=in action=allow protocol=tcp localport=5000 >nul 2>&1

echo.
echo ========================================
echo   INSTALLATION ABGESCHLOSSEN!
echo ========================================
echo.
echo SCIO startet jetzt automatisch
echo bei jedem Windows-Start.
echo.
echo Zum Deinstallieren:
echo   schtasks /delete /tn "SCIO AI-Workstation" /f
echo.

pause
