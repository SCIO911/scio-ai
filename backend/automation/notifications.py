#!/usr/bin/env python3
"""
SCIO - Automatische Benachrichtigungen
Discord, Telegram, Email
"""

import os
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Optional
import threading

import requests

from backend.config import Config


class NotificationService:
    """
    Automatische Benachrichtigungen

    Sendet Alerts bei:
    - Neuen Bestellungen
    - Abgeschlossenen Jobs
    - Fehlern
    - Einnahmen-Updates
    - System-Warnungen
    """

    def __init__(self):
        self.discord_webhook = os.getenv('DISCORD_WEBHOOK')
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.email_enabled = bool(os.getenv('SMTP_HOST'))

    def _send_async(self, func, *args):
        """Sendet Nachricht asynchron"""
        thread = threading.Thread(target=func, args=args, daemon=True)
        thread.start()

    # ═══════════════════════════════════════════════════════════════
    # DISCORD
    # ═══════════════════════════════════════════════════════════════

    def send_discord(self, title: str, message: str, color: int = 0x3b82f6,
                     fields: list = None):
        """Sendet Discord Embed"""
        if not self.discord_webhook:
            return

        embed = {
            'title': title,
            'description': message,
            'color': color,
            'timestamp': datetime.utcnow().isoformat(),
            'footer': {'text': 'SCIO AI-Workstation'},
        }

        if fields:
            embed['fields'] = fields

        try:
            requests.post(self.discord_webhook, json={'embeds': [embed]}, timeout=10)
        except Exception as e:
            print(f"[ERROR] Discord Fehler: {e}")

    # ═══════════════════════════════════════════════════════════════
    # TELEGRAM
    # ═══════════════════════════════════════════════════════════════

    def send_telegram(self, message: str):
        """Sendet Telegram Nachricht"""
        if not self.telegram_token or not self.telegram_chat_id:
            return

        url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"

        try:
            requests.post(url, json={
                'chat_id': self.telegram_chat_id,
                'text': message,
                'parse_mode': 'HTML',
            }, timeout=10)
        except Exception as e:
            print(f"[ERROR] Telegram Fehler: {e}")

    # ═══════════════════════════════════════════════════════════════
    # EMAIL
    # ═══════════════════════════════════════════════════════════════

    def send_email(self, to: str, subject: str, body: str):
        """Sendet Email"""
        smtp_host = os.getenv('SMTP_HOST')
        smtp_port = int(os.getenv('SMTP_PORT', 587))
        smtp_user = os.getenv('SMTP_USER')
        smtp_pass = os.getenv('SMTP_PASS')
        from_email = os.getenv('SMTP_FROM', Config.SERVICE_EMAIL)

        if not all([smtp_host, smtp_user, smtp_pass]):
            return

        try:
            msg = MIMEMultipart()
            msg['From'] = from_email
            msg['To'] = to
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'html'))

            with smtplib.SMTP(smtp_host, smtp_port) as server:
                server.starttls()
                server.login(smtp_user, smtp_pass)
                server.send_message(msg)

        except Exception as e:
            print(f"[ERROR] Email Fehler: {e}")

    # ═══════════════════════════════════════════════════════════════
    # EVENT NOTIFICATIONS
    # ═══════════════════════════════════════════════════════════════

    def notify_new_order(self, order_id: str, email: str, amount: float,
                         model: str):
        """Benachrichtigt über neue Bestellung"""
        # Discord
        self._send_async(self.send_discord,
            "[MONEY] Neue Bestellung!",
            f"**{model.upper()}** Training",
            0x22c55e,  # Grün
            [
                {'name': 'Order-ID', 'value': order_id, 'inline': True},
                {'name': 'Betrag', 'value': f'{amount:.2f}€', 'inline': True},
                {'name': 'Kunde', 'value': email, 'inline': True},
            ]
        )

        # Telegram
        self._send_async(self.send_telegram,
            f"[MONEY] <b>Neue Bestellung!</b>\n\n"
            f"[JOB] Order: <code>{order_id}</code>\n"
            f"[AUTO] Model: {model.upper()}\n"
            f"[STRIPE] Betrag: {amount:.2f}€\n"
            f"📧 Kunde: {email}"
        )

    def notify_job_completed(self, job_id: str, job_type: str,
                            duration_seconds: float, user_email: str = None):
        """Benachrichtigt über abgeschlossenen Job"""
        duration = f"{duration_seconds/60:.1f} min" if duration_seconds > 60 else f"{duration_seconds:.0f}s"

        self._send_async(self.send_discord,
            "[OK] Job Abgeschlossen",
            f"**{job_type}** erfolgreich beendet",
            0x3b82f6,  # Blau
            [
                {'name': 'Job-ID', 'value': job_id[:16], 'inline': True},
                {'name': 'Dauer', 'value': duration, 'inline': True},
            ]
        )

        # Email an Kunden wenn vorhanden
        if user_email:
            self._send_async(self.send_email,
                user_email,
                f"SCIO - Ihr Job ist fertig!",
                f"""
                <h2>Ihr Job wurde erfolgreich abgeschlossen!</h2>
                <p><strong>Job-ID:</strong> {job_id}</p>
                <p><strong>Typ:</strong> {job_type}</p>
                <p><strong>Dauer:</strong> {duration}</p>
                <p>Sie können die Ergebnisse in Ihrem Dashboard abrufen.</p>
                <br>
                <p>Mit freundlichen Grüßen,<br>Ihr SCIO Team</p>
                """
            )

    def notify_job_failed(self, job_id: str, error: str):
        """Benachrichtigt über fehlgeschlagenen Job"""
        self._send_async(self.send_discord,
            "[ERROR] Job Fehlgeschlagen",
            f"```{error[:500]}```",
            0xef4444,  # Rot
            [
                {'name': 'Job-ID', 'value': job_id[:16], 'inline': True},
            ]
        )

    def notify_earnings_daily(self, today: float, week: float, month: float):
        """Täglicher Einnahmen-Report"""
        self._send_async(self.send_discord,
            "[STATS] Täglicher Einnahmen-Report",
            f"Stand: {datetime.now().strftime('%d.%m.%Y')}",
            0xa855f7,  # Lila
            [
                {'name': 'Heute', 'value': f'{today:.2f}€', 'inline': True},
                {'name': 'Diese Woche', 'value': f'{week:.2f}€', 'inline': True},
                {'name': 'Dieser Monat', 'value': f'{month:.2f}€', 'inline': True},
            ]
        )

        self._send_async(self.send_telegram,
            f"[STATS] <b>Täglicher Report</b>\n\n"
            f"[MONEY] Heute: {today:.2f}€\n"
            f"📅 Woche: {week:.2f}€\n"
            f"📆 Monat: {month:.2f}€"
        )

    def notify_system_warning(self, title: str, message: str):
        """System-Warnung"""
        self._send_async(self.send_discord,
            f"[WARN] {title}",
            message,
            0xeab308,  # Gelb
        )

        self._send_async(self.send_telegram,
            f"[WARN] <b>{title}</b>\n\n{message}"
        )

    def notify_system_error(self, title: str, message: str):
        """System-Fehler"""
        self._send_async(self.send_discord,
            f"🚨 {title}",
            message,
            0xef4444,  # Rot
        )

        self._send_async(self.send_telegram,
            f"🚨 <b>{title}</b>\n\n{message}"
        )

    def notify_gpu_rental_started(self, platform: str, price_per_hour: float):
        """GPU-Rental gestartet"""
        self._send_async(self.send_discord,
            f"🖥️ {platform} Rental Gestartet",
            f"Passives Einkommen aktiviert!",
            0x22c55e,
            [
                {'name': 'Preis/Stunde', 'value': f'${price_per_hour:.2f}', 'inline': True},
                {'name': 'Täglich', 'value': f'~${price_per_hour*24:.2f}', 'inline': True},
            ]
        )

    def notify_startup(self):
        """System-Start Benachrichtigung"""
        self._send_async(self.send_discord,
            "[LAUNCH] SCIO Gestartet",
            f"AI-Workstation ist online und bereit!",
            0x22c55e,
            [
                {'name': 'Zeit', 'value': datetime.now().strftime('%H:%M:%S'), 'inline': True},
                {'name': 'URL', 'value': Config.SERVICE_URL, 'inline': True},
            ]
        )

        self._send_async(self.send_telegram,
            f"[LAUNCH] <b>SCIO Gestartet</b>\n\n"
            f"[OK] AI-Workstation ist online!\n"
            f"[NET] {Config.SERVICE_URL}"
        )


# Singleton
_notification_service: Optional[NotificationService] = None


def get_notification_service() -> NotificationService:
    global _notification_service
    if _notification_service is None:
        _notification_service = NotificationService()
    return _notification_service
