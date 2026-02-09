#!/usr/bin/env python3
"""
SCIO - Paid API Service

Monetarisierung der SCIO Agents durch bezahlte API-Aufrufe.
Stripe-Integration für automatische Abrechnung.
"""

import os
import time
import json
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path

from flask import request, jsonify, g
from backend.config import Config

# Stripe
try:
    import stripe
    stripe.api_key = Config.STRIPE_SECRET_KEY
    STRIPE_AVAILABLE = True
except ImportError:
    STRIPE_AVAILABLE = False


# ============================================================
# PRICING - Preise pro API-Aufruf
# ============================================================

PRICING = {
    # LLM Services
    'chat_completion': {
        'name': 'Chat Completion',
        'price_per_call': 0.002,  # $0.002 pro Aufruf
        'price_per_1k_tokens': 0.001,  # $0.001 pro 1000 Tokens
        'description': 'GPT-ähnliche Chat-Vervollständigung',
    },
    'text_generation': {
        'name': 'Text Generation',
        'price_per_call': 0.001,
        'price_per_1k_tokens': 0.0008,
        'description': 'Freie Textgenerierung',
    },

    # Image Services
    'image_generation': {
        'name': 'Image Generation',
        'price_per_call': 0.02,  # $0.02 pro Bild
        'description': 'KI-Bildgenerierung (FLUX, SDXL)',
    },
    'image_upscale': {
        'name': 'Image Upscale',
        'price_per_call': 0.01,
        'description': 'Bild-Hochskalierung',
    },

    # Audio Services
    'speech_to_text': {
        'name': 'Speech to Text',
        'price_per_minute': 0.006,  # $0.006 pro Minute
        'description': 'Whisper STT',
    },
    'text_to_speech': {
        'name': 'Text to Speech',
        'price_per_1k_chars': 0.015,  # $0.015 pro 1000 Zeichen
        'description': 'Natürliche Sprachsynthese',
    },

    # Code Services
    'code_generation': {
        'name': 'Code Generation',
        'price_per_call': 0.003,
        'description': 'Code-Generierung und -Vervollständigung',
    },
    'code_review': {
        'name': 'Code Review',
        'price_per_call': 0.005,
        'description': 'Automatisches Code-Review',
    },

    # Vision Services
    'image_analysis': {
        'name': 'Image Analysis',
        'price_per_call': 0.005,
        'description': 'Bildanalyse und -beschreibung',
    },
    'ocr': {
        'name': 'OCR',
        'price_per_page': 0.002,
        'description': 'Texterkennung aus Bildern',
    },

    # Data Services
    'embedding': {
        'name': 'Text Embedding',
        'price_per_1k_tokens': 0.0001,
        'description': 'Text-Vektorisierung',
    },
    'data_analysis': {
        'name': 'Data Analysis',
        'price_per_call': 0.01,
        'description': 'Automatische Datenanalyse',
    },
}


@dataclass
class APIKey:
    """API Key für einen Benutzer"""
    key: str
    user_id: str
    email: str
    name: str
    created_at: datetime
    balance: float = 0.0  # Prepaid-Guthaben
    total_spent: float = 0.0
    total_calls: int = 0
    rate_limit: int = 100  # Calls pro Minute
    is_active: bool = True
    stripe_customer_id: Optional[str] = None


@dataclass
class UsageRecord:
    """Einzelner API-Aufruf"""
    api_key: str
    service: str
    cost: float
    timestamp: datetime
    request_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class PaidAPIService:
    """
    Service für bezahlte API-Aufrufe

    Features:
    - API Key Management
    - Usage Tracking
    - Stripe Billing
    - Rate Limiting
    - Prepaid & Postpaid
    """

    def __init__(self):
        self._api_keys: Dict[str, APIKey] = {}
        self._usage_records: List[UsageRecord] = []
        self._rate_limits: Dict[str, List[float]] = {}  # key -> timestamps

        # Persistenz
        self._data_file = Config.DATA_DIR / 'paid_api_data.json'
        self._load_data()

        print("[MONEY] Paid API Service initialisiert")

    def _load_data(self):
        """Lädt gespeicherte Daten"""
        try:
            if self._data_file.exists():
                with open(self._data_file, 'r') as f:
                    data = json.load(f)

                for key_data in data.get('api_keys', []):
                    key = APIKey(
                        key=key_data['key'],
                        user_id=key_data['user_id'],
                        email=key_data['email'],
                        name=key_data['name'],
                        created_at=datetime.fromisoformat(key_data['created_at']),
                        balance=key_data.get('balance', 0.0),
                        total_spent=key_data.get('total_spent', 0.0),
                        total_calls=key_data.get('total_calls', 0),
                        rate_limit=key_data.get('rate_limit', 100),
                        is_active=key_data.get('is_active', True),
                        stripe_customer_id=key_data.get('stripe_customer_id'),
                    )
                    self._api_keys[key.key] = key

                print(f"[OK] {len(self._api_keys)} API Keys geladen")
        except Exception as e:
            print(f"[WARN] API-Daten laden fehlgeschlagen: {e}")

    def _save_data(self):
        """Speichert Daten"""
        try:
            self._data_file.parent.mkdir(parents=True, exist_ok=True)

            data = {
                'api_keys': [
                    {
                        'key': k.key,
                        'user_id': k.user_id,
                        'email': k.email,
                        'name': k.name,
                        'created_at': k.created_at.isoformat(),
                        'balance': k.balance,
                        'total_spent': k.total_spent,
                        'total_calls': k.total_calls,
                        'rate_limit': k.rate_limit,
                        'is_active': k.is_active,
                        'stripe_customer_id': k.stripe_customer_id,
                    }
                    for k in self._api_keys.values()
                ],
                'last_updated': datetime.now().isoformat(),
            }

            with open(self._data_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[ERROR] API-Daten speichern fehlgeschlagen: {e}")

    def create_api_key(
        self,
        email: str,
        name: str,
        initial_balance: float = 0.0,
    ) -> APIKey:
        """Erstellt neuen API Key"""
        key = f"sk_scio_{secrets.token_hex(24)}"
        user_id = hashlib.sha256(email.encode()).hexdigest()[:16]

        api_key = APIKey(
            key=key,
            user_id=user_id,
            email=email,
            name=name,
            created_at=datetime.now(),
            balance=initial_balance,
        )

        # Stripe Customer erstellen
        if STRIPE_AVAILABLE and stripe.api_key:
            try:
                customer = stripe.Customer.create(
                    email=email,
                    name=name,
                    metadata={'scio_user_id': user_id},
                )
                api_key.stripe_customer_id = customer.id
            except Exception as e:
                print(f"[WARN] Stripe Customer erstellen fehlgeschlagen: {e}")

        self._api_keys[key] = api_key
        self._save_data()

        print(f"[API] Neuer API Key erstellt für {email}")
        return api_key

    def validate_api_key(self, key: str) -> Optional[APIKey]:
        """Validiert API Key"""
        api_key = self._api_keys.get(key)

        if not api_key:
            return None

        if not api_key.is_active:
            return None

        return api_key

    def check_rate_limit(self, key: str) -> bool:
        """Prüft Rate Limit"""
        api_key = self._api_keys.get(key)
        if not api_key:
            return False

        now = time.time()
        window = 60  # 1 Minute

        # Alte Einträge entfernen
        if key in self._rate_limits:
            self._rate_limits[key] = [
                ts for ts in self._rate_limits[key]
                if now - ts < window
            ]
        else:
            self._rate_limits[key] = []

        # Prüfen
        if len(self._rate_limits[key]) >= api_key.rate_limit:
            return False

        # Neuen Eintrag hinzufügen
        self._rate_limits[key].append(now)
        return True

    def charge_usage(
        self,
        key: str,
        service: str,
        units: float = 1.0,
        metadata: Dict[str, Any] = None,
    ) -> tuple[bool, float, str]:
        """
        Berechnet Kosten für API-Aufruf

        Returns:
            (success, cost, message)
        """
        api_key = self._api_keys.get(key)
        if not api_key:
            return False, 0, "Invalid API key"

        pricing = PRICING.get(service)
        if not pricing:
            return False, 0, f"Unknown service: {service}"

        # Kosten berechnen
        cost = 0.0
        if 'price_per_call' in pricing:
            cost = pricing['price_per_call']
        elif 'price_per_1k_tokens' in pricing:
            cost = (units / 1000) * pricing['price_per_1k_tokens']
        elif 'price_per_minute' in pricing:
            cost = units * pricing['price_per_minute']
        elif 'price_per_1k_chars' in pricing:
            cost = (units / 1000) * pricing['price_per_1k_chars']
        elif 'price_per_page' in pricing:
            cost = units * pricing['price_per_page']

        # Guthaben prüfen
        if api_key.balance < cost:
            return False, cost, f"Insufficient balance. Need ${cost:.4f}, have ${api_key.balance:.4f}"

        # Abbuchen
        api_key.balance -= cost
        api_key.total_spent += cost
        api_key.total_calls += 1

        # Usage Record
        record = UsageRecord(
            api_key=key,
            service=service,
            cost=cost,
            timestamp=datetime.now(),
            request_id=secrets.token_hex(8),
            metadata=metadata or {},
        )
        self._usage_records.append(record)

        # Speichern
        self._save_data()

        # MoneyMaker benachrichtigen
        try:
            from backend.automation.money_maker import get_money_maker
            money_maker = get_money_maker()
            money_maker._log_earning('api', cost, {
                'service': service,
                'user': api_key.email,
            })
        except Exception:
            pass

        return True, cost, "OK"

    def add_balance(
        self,
        key: str,
        amount: float,
        payment_method: str = 'stripe',
    ) -> bool:
        """Fügt Guthaben hinzu"""
        api_key = self._api_keys.get(key)
        if not api_key:
            return False

        api_key.balance += amount
        self._save_data()

        print(f"[MONEY] +${amount:.2f} Guthaben für {api_key.email}")
        return True

    def get_usage_stats(self, key: str) -> Dict[str, Any]:
        """Gibt Usage-Statistiken zurück"""
        api_key = self._api_keys.get(key)
        if not api_key:
            return {}

        # Letzte 30 Tage
        cutoff = datetime.now() - timedelta(days=30)
        recent_usage = [
            r for r in self._usage_records
            if r.api_key == key and r.timestamp >= cutoff
        ]

        # Nach Service gruppieren
        by_service = {}
        for record in recent_usage:
            if record.service not in by_service:
                by_service[record.service] = {'calls': 0, 'cost': 0}
            by_service[record.service]['calls'] += 1
            by_service[record.service]['cost'] += record.cost

        return {
            'balance': api_key.balance,
            'total_spent': api_key.total_spent,
            'total_calls': api_key.total_calls,
            'last_30_days': {
                'calls': len(recent_usage),
                'cost': sum(r.cost for r in recent_usage),
                'by_service': by_service,
            },
        }

    def get_pricing(self) -> Dict[str, Any]:
        """Gibt Preisliste zurück"""
        return PRICING

    def get_total_revenue(self) -> float:
        """Gibt Gesamteinnahmen zurück"""
        return sum(k.total_spent for k in self._api_keys.values())


# Singleton
_paid_api_service: Optional[PaidAPIService] = None


def get_paid_api_service() -> PaidAPIService:
    """Gibt Singleton zurück"""
    global _paid_api_service
    if _paid_api_service is None:
        _paid_api_service = PaidAPIService()
    return _paid_api_service


# ============================================================
# FLASK DECORATOR für bezahlte Endpoints
# ============================================================

def require_api_key(service: str, unit_calculator=None):
    """
    Decorator für bezahlte API-Endpoints

    Usage:
        @require_api_key('chat_completion', lambda req: req.json.get('max_tokens', 100))
        def chat_endpoint():
            ...
    """
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            # API Key aus Header
            auth = request.headers.get('Authorization', '')
            if not auth.startswith('Bearer '):
                return jsonify({'error': 'Missing API key'}), 401

            api_key = auth[7:]  # Remove 'Bearer '

            # Validieren
            paid_api = get_paid_api_service()
            key_obj = paid_api.validate_api_key(api_key)

            if not key_obj:
                return jsonify({'error': 'Invalid API key'}), 401

            # Rate Limit
            if not paid_api.check_rate_limit(api_key):
                return jsonify({'error': 'Rate limit exceeded'}), 429

            # Units berechnen
            units = 1.0
            if unit_calculator:
                try:
                    units = unit_calculator(request)
                except Exception:
                    units = 1.0

            # Kosten berechnen
            success, cost, message = paid_api.charge_usage(
                api_key,
                service,
                units,
                {'endpoint': request.path},
            )

            if not success:
                return jsonify({'error': message, 'cost': cost}), 402  # Payment Required

            # Request-Kontext setzen
            g.api_key = key_obj
            g.api_cost = cost

            # Endpoint ausführen
            response = f(*args, **kwargs)

            # Cost Header hinzufügen
            if hasattr(response, 'headers'):
                response.headers['X-API-Cost'] = str(cost)
                response.headers['X-API-Balance'] = str(key_obj.balance)

            return response

        return wrapped
    return decorator
