# Install cloudflared for SCIO
$localAppData = [Environment]::GetFolderPath('LocalApplicationData')
$cloudflaredDir = Join-Path $localAppData 'cloudflared'

Write-Host "Erstelle Verzeichnis: $cloudflaredDir"
New-Item -ItemType Directory -Force -Path $cloudflaredDir | Out-Null

$url = 'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe'
$output = Join-Path $cloudflaredDir 'cloudflared.exe'

Write-Host "Lade cloudflared herunter..."
Invoke-WebRequest -Uri $url -OutFile $output

if (Test-Path $output) {
    Write-Host "[OK] Cloudflared erfolgreich installiert: $output"
} else {
    Write-Host "[FEHLER] Installation fehlgeschlagen"
}
