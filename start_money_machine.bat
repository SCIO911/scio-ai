@echo off
REM ============================================================
REM SCIO MONEY MACHINE - Automatisches Geldverdienen
REM ============================================================

echo.
echo ============================================================
echo   SCIO MONEY MACHINE
echo   Automatisches Geldverdienen aktivieren
echo ============================================================
echo.

cd /d C:\SCIO

REM 1. Backend starten
echo [1/3] Starte SCIO Backend...
start /B python backend/app.py > flask_out.txt 2>&1
timeout /t 10 /nobreak > nul
echo [OK] Backend gestartet

REM 2. Cloudflare Tunnel starten
echo [2/3] Starte Public API Tunnel...
if exist "%LOCALAPPDATA%\cloudflared\cloudflared.exe" (
    start /B "%LOCALAPPDATA%\cloudflared\cloudflared.exe" tunnel --url http://localhost:5000 > tunnel_log.txt 2>&1
    timeout /t 5 /nobreak > nul
    echo [OK] Tunnel gestartet
) else (
    echo [WARN] Cloudflared nicht installiert
    echo        Fuehre zuerst aus: powershell -File start_public_api.ps1
)

REM 3. Status anzeigen
echo.
echo ============================================================
echo   SCIO MONEY MACHINE AKTIV
echo ============================================================
echo.
echo   Lokale API:    http://localhost:5000
echo   Admin:         http://localhost:5000/admin
echo   Public API:    Siehe tunnel_log.txt
echo.
echo   Geldquellen:
echo   [1] Vast.ai GPU-Vermietung (wenn idle)
echo   [2] Bezahlte API-Aufrufe (Chat, Images, Audio, Code)
echo   [3] Stripe Zahlungen fuer Services
echo.
echo   Earnings:      http://localhost:5000/api/stats/earnings
echo.
echo ============================================================
echo.

REM Health Check
curl -s http://localhost:5000/health
echo.
echo.
echo Druecke eine Taste zum Beenden...
pause > nul
