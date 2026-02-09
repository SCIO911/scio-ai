# ============================================================
# SCIO - Start Public API with Cloudflare Tunnel
# Macht SCIO API weltweit erreichbar
# ============================================================

Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "  SCIO PUBLIC API STARTER" -ForegroundColor Cyan
Write-Host "  API weltweit erreichbar machen" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Cyan

# 1. Prüfe ob cloudflared installiert ist
$cloudflared = Get-Command cloudflared -ErrorAction SilentlyContinue
if (-not $cloudflared) {
    Write-Host "`n[1/3] Installiere Cloudflare Tunnel (cloudflared)..." -ForegroundColor Yellow

    # Download cloudflared
    $url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
    $output = "$env:LOCALAPPDATA\cloudflared\cloudflared.exe"

    New-Item -ItemType Directory -Force -Path "$env:LOCALAPPDATA\cloudflared" | Out-Null
    Invoke-WebRequest -Uri $url -OutFile $output

    # Zu PATH hinzufügen
    $env:Path += ";$env:LOCALAPPDATA\cloudflared"

    Write-Host "[OK] Cloudflared installiert" -ForegroundColor Green
} else {
    Write-Host "`n[1/3] Cloudflared bereits installiert" -ForegroundColor Green
}

# 2. Starte SCIO Backend falls nicht läuft
$scioRunning = Get-NetTCPConnection -LocalPort 5000 -ErrorAction SilentlyContinue
if (-not $scioRunning) {
    Write-Host "`n[2/3] Starte SCIO Backend..." -ForegroundColor Yellow
    Start-Process -FilePath "python" -ArgumentList "backend/app.py" -WorkingDirectory "C:\SCIO" -WindowStyle Hidden
    Start-Sleep -Seconds 5
    Write-Host "[OK] SCIO Backend gestartet" -ForegroundColor Green
} else {
    Write-Host "`n[2/3] SCIO Backend läuft bereits" -ForegroundColor Green
}

# 3. Starte Cloudflare Tunnel
Write-Host "`n[3/3] Starte Cloudflare Tunnel..." -ForegroundColor Yellow
Write-Host "=" * 60 -ForegroundColor Cyan

# Quick Tunnel (keine Anmeldung nötig)
$tunnelProcess = Start-Process -FilePath "$env:LOCALAPPDATA\cloudflared\cloudflared.exe" `
    -ArgumentList "tunnel --url http://localhost:5000" `
    -PassThru -RedirectStandardOutput "C:\SCIO\tunnel_url.txt" `
    -WindowStyle Hidden

Write-Host "Warte auf Tunnel-URL..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# URL aus Log extrahieren
$tunnelLog = Get-Content "C:\SCIO\tunnel_url.txt" -ErrorAction SilentlyContinue
$tunnelUrl = $tunnelLog | Select-String -Pattern "https://.*\.trycloudflare\.com" | Select-Object -First 1

if ($tunnelUrl) {
    Write-Host "`n" + "=" * 60 -ForegroundColor Green
    Write-Host "  SCIO API IST JETZT OEFFENTLICH ERREICHBAR!" -ForegroundColor Green
    Write-Host "=" * 60 -ForegroundColor Green
    Write-Host "`nPublic URL: $($tunnelUrl.Matches[0].Value)" -ForegroundColor Cyan
    Write-Host "`nAPI Endpoints:" -ForegroundColor White
    Write-Host "  - Pricing:     $($tunnelUrl.Matches[0].Value)/api/v1/public/pricing"
    Write-Host "  - Create Key:  POST $($tunnelUrl.Matches[0].Value)/api/v1/public/keys"
    Write-Host "  - Chat:        POST $($tunnelUrl.Matches[0].Value)/api/v1/public/chat/completions"
    Write-Host "  - Images:      POST $($tunnelUrl.Matches[0].Value)/api/v1/public/images/generate"
    Write-Host "  - Health:      $($tunnelUrl.Matches[0].Value)/api/v1/public/health"

    # URL speichern
    $tunnelUrl.Matches[0].Value | Out-File "C:\SCIO\public_url.txt"

    Write-Host "`n[OK] URL gespeichert in: C:\SCIO\public_url.txt" -ForegroundColor Green
} else {
    Write-Host "[WARN] Tunnel-URL konnte nicht ermittelt werden" -ForegroundColor Yellow
    Write-Host "Pruefe C:\SCIO\tunnel_url.txt manuell" -ForegroundColor Yellow
}

Write-Host "`n" + "=" * 60 -ForegroundColor Cyan
Write-Host "Druecke Strg+C zum Beenden" -ForegroundColor Yellow
Write-Host "=" * 60 -ForegroundColor Cyan

# Warte auf Beendigung
Wait-Process -Id $tunnelProcess.Id
