# SCIO AI-Workstation - Vollautomatischer Start mit Auto-Restart
# Startet den Server automatisch neu bei Absturz
# ALLES IN C:\SCIO - Keine externen Abhaengigkeiten

$ErrorActionPreference = "Continue"
$Host.UI.RawUI.WindowTitle = "SCIO AI-Workstation"

$ProjectDir = "C:\SCIO"
$VenvPython = "$ProjectDir\venv\Scripts\python.exe"
$PythonScript = "$ProjectDir\start.py"
$LogFile = "$ProjectDir\data\logs\server.log"
$MaxRestarts = 10
$RestartDelay = 5

# Banner
Write-Host @"

+======================================================================+
|                                                                      |
|    SSSS   CCCC  III   OOO                                            |
|   S      C       I   O   O                                           |
|    SSS   C       I   O   O                                           |
|       S  C       I   O   O                                           |
|   SSSS    CCCC  III   OOO                                            |
|                                                                      |
|   Service Computer Intelligence Organization                         |
|   VOLLAUTOMATISCHER BETRIEB - Auto-Restart bei Absturz               |
|   Alle Abhaengigkeiten in C:\SCIO enthalten                          |
|                                                                      |
+======================================================================+

"@ -ForegroundColor Cyan

Set-Location $ProjectDir

# Create log directory
if (!(Test-Path "$ProjectDir\data\logs")) {
    New-Item -ItemType Directory -Path "$ProjectDir\data\logs" -Force | Out-Null
}

# Check if venv exists
if (!(Test-Path $VenvPython)) {
    Write-Host "[ERROR] Python venv nicht gefunden: $VenvPython" -ForegroundColor Red
    Write-Host "[INFO] Bitte erst 'python -m venv venv' und Dependencies installieren" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Druecke eine Taste zum Beenden..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}

Write-Host "[OK] Python venv gefunden: $VenvPython" -ForegroundColor Green
Write-Host "[OK] Alle Abhaengigkeiten in C:\SCIO enthalten" -ForegroundColor Green
Write-Host ""

$RestartCount = 0

while ($RestartCount -lt $MaxRestarts) {
    $StartTime = Get-Date

    Write-Host "`n[$(Get-Date -Format 'HH:mm:ss')] Server wird gestartet... (Versuch $($RestartCount + 1)/$MaxRestarts)" -ForegroundColor Green

    try {
        # Start Python process with local venv
        $process = Start-Process -FilePath $VenvPython -ArgumentList $PythonScript -NoNewWindow -PassThru -Wait

        $EndTime = Get-Date
        $Runtime = ($EndTime - $StartTime).TotalSeconds

        if ($process.ExitCode -eq 0) {
            Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Server normal beendet." -ForegroundColor Yellow
            break
        }
        else {
            Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Server abgestuerzt! Exit Code: $($process.ExitCode)" -ForegroundColor Red

            # Log crash
            $CrashLog = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') - Crash after ${Runtime}s - Exit: $($process.ExitCode)"
            Add-Content -Path $LogFile -Value $CrashLog

            # If ran for more than 60 seconds, reset restart counter
            if ($Runtime -gt 60) {
                $RestartCount = 0
            }
            else {
                $RestartCount++
            }

            if ($RestartCount -lt $MaxRestarts) {
                Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Neustart in $RestartDelay Sekunden..." -ForegroundColor Yellow
                Start-Sleep -Seconds $RestartDelay
            }
        }
    }
    catch {
        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Fehler: $_" -ForegroundColor Red
        $RestartCount++
        Start-Sleep -Seconds $RestartDelay
    }
}

if ($RestartCount -ge $MaxRestarts) {
    Write-Host "`n[FEHLER] Maximale Neustarts erreicht. Server wird nicht mehr gestartet." -ForegroundColor Red
    Write-Host "Bitte Logs pruefen: $LogFile" -ForegroundColor Yellow
}

Write-Host "`nDruecke eine Taste zum Beenden..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
