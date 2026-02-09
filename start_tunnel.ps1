# SCIO - Start Cloudflare Tunnel
$localAppData = [Environment]::GetFolderPath('LocalApplicationData')
$cloudflared = Join-Path $localAppData 'cloudflared\cloudflared.exe'

Write-Host "============================================================"
Write-Host "  SCIO PUBLIC API TUNNEL" -ForegroundColor Cyan
Write-Host "============================================================"

if (Test-Path $cloudflared) {
    Write-Host "`nStarte Cloudflare Tunnel..."
    Write-Host "Die Public URL wird gleich angezeigt..." -ForegroundColor Yellow
    Write-Host ""

    # Start tunnel and show output
    & $cloudflared tunnel --url http://localhost:5000
} else {
    Write-Host "[FEHLER] Cloudflared nicht gefunden: $cloudflared" -ForegroundColor Red
}
