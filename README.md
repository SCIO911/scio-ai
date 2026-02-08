# SCIO - Service Computer Intelligence Organization

AI-Workstation mit vollautomatischem Betrieb

## Quick Start

```bash
# Dependencies installieren
pip install -r requirements.txt

# Server starten
cd C:\SCIO
python backend/app.py

# Oder mit Auto-Restart
START.bat
```

Browser: http://localhost:5000

## Struktur

```
C:\SCIO\
├── backend\
│   ├── app.py              # Flask Server
│   ├── config.py           # Zentrale Konfiguration
│   ├── models\             # SQLAlchemy Models
│   ├── services\           # Business Logic
│   ├── workers\            # LLM/Image Workers
│   ├── integrations\       # Vast.ai, RunPod
│   ├── automation\         # Auto-Worker, Scheduler
│   └── routes\             # API Endpoints
├── frontend\
│   ├── index.html          # Kunden-Portal
│   ├── admin.html          # Admin Dashboard
│   └── docs.html           # API Dokumentation
├── data\
│   ├── orders\             # Bestellungen
│   ├── models\             # LLM Modelle
│   └── scio.db             # SQLite Datenbank
├── .env                    # Konfiguration
├── requirements.txt        # Dependencies
├── START.bat               # Start-Script
└── README.md               # Diese Datei
```

## Endpoints

| URL | Beschreibung |
|-----|--------------|
| http://localhost:5000 | Kunden-Portal |
| http://localhost:5000/admin | Admin Dashboard |
| http://localhost:5000/docs | API Dokumentation |
| http://localhost:5000/health | Health Check |

## API

OpenAI-kompatible API:

```bash
curl http://localhost:5000/api/v1/chat/completions \
  -H "Authorization: Bearer scio_xxx" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral-7b",
    "messages": [{"role": "user", "content": "Hallo!"}]
  }'
```

## Features

- LLM Inference (OpenAI-kompatibel)
- LLM Fine-Tuning (LoRA/QLoRA)
- Image Generation (SDXL, SD)
- Hardware Monitoring
- Job Queue mit Retry-Logik
- API Key Management
- Stripe Payments
- Vast.ai Integration (optional)
- RunPod Integration (optional)
- Vollautomatischer Betrieb

## Konfiguration

Bearbeite `.env` fuer Anpassungen:

```env
# Stripe
STRIPE_PUBLISHABLE_KEY=pk_live_xxx
STRIPE_SECRET_KEY=sk_live_xxx

# Preise (in Cent)
PRICE_LLAMA_7B=20000      # 200 EUR
PRICE_LLAMA_13B=40000     # 400 EUR
PRICE_LLAMA_70B=100000    # 1000 EUR

# Vast.ai (optional)
VASTAI_ENABLED=true
VASTAI_API_KEY=xxx

# RunPod (optional)
RUNPOD_ENABLED=true
RUNPOD_API_KEY=xxx

# Notifications (optional)
DISCORD_WEBHOOK=xxx
TELEGRAM_BOT_TOKEN=xxx
TELEGRAM_CHAT_ID=xxx
```

## Autostart

```bash
# Windows Autostart einrichten
install-autostart.bat
```

## Support

SCIO AI-Workstation v2.0

Lizenz: Proprietary
