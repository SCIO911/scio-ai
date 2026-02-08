#!/usr/bin/env python3
"""
SCIO - API Key Service
Verwaltet API-Keys, Token-Zählung und Rate-Limiting
"""

import time
from datetime import datetime, timedelta
from typing import Optional, Tuple
from collections import defaultdict

from sqlalchemy.orm import Session
from backend.models import SessionLocal, APIKey
from backend.config import Config


class RateLimiter:
    """Token Bucket Rate Limiter"""

    def __init__(self):
        self._requests: dict = defaultdict(list)  # key_id -> [timestamps]
        self._tokens: dict = defaultdict(list)  # key_id -> [(timestamp, count)]

    def check_request_limit(self, key_id: int, limit_rpm: int) -> Tuple[bool, int]:
        """
        Prüft Request-Limit

        Returns:
            (allowed, remaining)
        """
        now = time.time()
        minute_ago = now - 60

        # Clean old entries
        self._requests[key_id] = [t for t in self._requests[key_id] if t > minute_ago]

        current_count = len(self._requests[key_id])
        remaining = limit_rpm - current_count

        if current_count >= limit_rpm:
            return False, 0

        self._requests[key_id].append(now)
        return True, remaining - 1

    def check_token_limit(self, key_id: int, limit_tpm: int, token_count: int) -> Tuple[bool, int]:
        """
        Prüft Token-Limit

        Returns:
            (allowed, remaining)
        """
        now = time.time()
        minute_ago = now - 60

        # Clean old entries
        self._tokens[key_id] = [(t, c) for t, c in self._tokens[key_id] if t > minute_ago]

        current_tokens = sum(c for _, c in self._tokens[key_id])
        remaining = limit_tpm - current_tokens

        if current_tokens + token_count > limit_tpm:
            return False, remaining

        self._tokens[key_id].append((now, token_count))
        return True, remaining - token_count


class APIKeyService:
    """
    API Key Service

    Verwaltet:
    - API Key Erstellung und Validierung
    - Token-Zählung und Abrechnung
    - Rate Limiting
    - Usage Tracking
    """

    def __init__(self):
        self._rate_limiter = RateLimiter()
        self._key_cache: dict = {}  # hash -> APIKey (cached for 5 min)
        self._cache_time: dict = {}  # hash -> timestamp

    def create_key(
        self,
        user_email: str,
        name: str = None,
        description: str = None,
        rate_limit_rpm: int = 60,
        rate_limit_tpm: int = 100000,
        monthly_token_limit: int = 1000000,
        credits_cents: int = 0,
        allowed_models: list = None,
        allowed_features: list = None,
        expires_days: int = None,
        is_admin: bool = False,
    ) -> Tuple[str, dict]:
        """
        Erstellt neuen API Key

        Returns:
            (full_key, key_info)
        """
        full_key, key_hash, key_prefix = APIKey.generate_key()

        db = SessionLocal()
        try:
            api_key = APIKey(
                key_hash=key_hash,
                key_prefix=key_prefix,
                name=name or f"Key für {user_email}",
                description=description,
                user_email=user_email,
                is_admin=is_admin,
                rate_limit_rpm=rate_limit_rpm,
                rate_limit_tpm=rate_limit_tpm,
                monthly_token_limit=monthly_token_limit,
                credits_cents=credits_cents,
                allowed_models=allowed_models,
                allowed_features=allowed_features or ['inference'],
                expires_at=datetime.utcnow() + timedelta(days=expires_days) if expires_days else None,
                reset_at=datetime.utcnow() + timedelta(days=30),
            )
            db.add(api_key)
            db.commit()
            db.refresh(api_key)

            print(f"[OK] API Key erstellt: {key_prefix} für {user_email}")

            return full_key, api_key.to_dict()

        except Exception as e:
            db.rollback()
            print(f"[ERROR] API Key erstellen fehlgeschlagen: {e}")
            raise
        finally:
            db.close()

    def validate_key(self, key: str) -> Tuple[bool, Optional[dict], str]:
        """
        Validiert API Key

        Returns:
            (valid, key_info, error_message)
        """
        if not key or not key.startswith('scio_'):
            return False, None, 'Invalid API key format'

        key_hash = APIKey.hash_key(key)

        # Check cache
        now = time.time()
        if key_hash in self._key_cache:
            if now - self._cache_time.get(key_hash, 0) < 300:  # 5 min cache
                cached = self._key_cache[key_hash]
                if cached.is_valid:
                    return True, cached.to_dict(), ''
                return False, None, 'API key is inactive or expired'

        # Query database
        db = SessionLocal()
        try:
            api_key = db.query(APIKey).filter(APIKey.key_hash == key_hash).first()

            if not api_key:
                return False, None, 'API key not found'

            # Update cache
            self._key_cache[key_hash] = api_key
            self._cache_time[key_hash] = now

            if not api_key.is_valid:
                if not api_key.is_active:
                    return False, None, 'API key is deactivated'
                if api_key.expires_at and datetime.utcnow() > api_key.expires_at:
                    return False, None, 'API key has expired'

            return True, api_key.to_dict(), ''

        finally:
            db.close()

    def check_rate_limits(
        self,
        key: str,
        estimated_tokens: int = 100,
    ) -> Tuple[bool, dict]:
        """
        Prüft Rate Limits

        Returns:
            (allowed, limits_info)
        """
        key_hash = APIKey.hash_key(key)

        db = SessionLocal()
        try:
            api_key = db.query(APIKey).filter(APIKey.key_hash == key_hash).first()
            if not api_key:
                return False, {'error': 'Key not found'}

            # Check request limit
            req_allowed, req_remaining = self._rate_limiter.check_request_limit(
                api_key.id, api_key.rate_limit_rpm
            )

            if not req_allowed:
                return False, {
                    'error': 'Rate limit exceeded (requests)',
                    'limit': api_key.rate_limit_rpm,
                    'remaining': 0,
                    'reset_in_seconds': 60,
                }

            # Check token limit
            tok_allowed, tok_remaining = self._rate_limiter.check_token_limit(
                api_key.id, api_key.rate_limit_tpm, estimated_tokens
            )

            if not tok_allowed:
                return False, {
                    'error': 'Rate limit exceeded (tokens)',
                    'limit': api_key.rate_limit_tpm,
                    'remaining': tok_remaining,
                    'reset_in_seconds': 60,
                }

            # Check monthly limit
            if api_key.monthly_tokens_used + estimated_tokens > api_key.monthly_token_limit:
                return False, {
                    'error': 'Monthly token limit exceeded',
                    'limit': api_key.monthly_token_limit,
                    'used': api_key.monthly_tokens_used,
                    'remaining': api_key.tokens_remaining,
                }

            return True, {
                'requests_remaining': req_remaining,
                'tokens_remaining_minute': tok_remaining,
                'tokens_remaining_month': api_key.tokens_remaining - estimated_tokens,
            }

        finally:
            db.close()

    def record_usage(
        self,
        key: str,
        tokens_input: int,
        tokens_output: int,
        cost_cents: int = None,
    ):
        """Zeichnet API-Nutzung auf"""
        key_hash = APIKey.hash_key(key)

        db = SessionLocal()
        try:
            api_key = db.query(APIKey).filter(APIKey.key_hash == key_hash).first()
            if not api_key:
                return

            total_tokens = tokens_input + tokens_output

            # Calculate cost if not provided
            if cost_cents is None:
                cost_cents = (
                    (tokens_input * Config.API_PRICES['input'] // 1000) +
                    (tokens_output * Config.API_PRICES['output'] // 1000)
                )

            # Update stats
            api_key.total_requests += 1
            api_key.total_tokens_input += tokens_input
            api_key.total_tokens_output += tokens_output
            api_key.monthly_tokens_used += total_tokens
            api_key.total_spent_cents += cost_cents
            api_key.last_used_at = datetime.utcnow()

            # Deduct from credits if applicable
            if api_key.credits_cents > 0:
                api_key.credits_cents = max(0, api_key.credits_cents - cost_cents)

            db.commit()

            # Update cache
            self._key_cache[key_hash] = api_key
            self._cache_time[key_hash] = time.time()

        except Exception as e:
            db.rollback()
            print(f"[ERROR] Usage aufzeichnen fehlgeschlagen: {e}")
        finally:
            db.close()

    def add_credits(self, key: str, amount_cents: int) -> bool:
        """Fügt Credits hinzu"""
        key_hash = APIKey.hash_key(key)

        db = SessionLocal()
        try:
            api_key = db.query(APIKey).filter(APIKey.key_hash == key_hash).first()
            if not api_key:
                return False

            api_key.credits_cents += amount_cents
            db.commit()

            print(f"[MONEY] {amount_cents/100:.2f}€ Credits hinzugefügt für {api_key.key_prefix}")
            return True

        except Exception as e:
            db.rollback()
            print(f"[ERROR] Credits hinzufügen fehlgeschlagen: {e}")
            return False
        finally:
            db.close()

    def deactivate_key(self, key: str) -> bool:
        """Deaktiviert API Key"""
        key_hash = APIKey.hash_key(key)

        db = SessionLocal()
        try:
            api_key = db.query(APIKey).filter(APIKey.key_hash == key_hash).first()
            if not api_key:
                return False

            api_key.is_active = False
            db.commit()

            # Clear cache
            if key_hash in self._key_cache:
                del self._key_cache[key_hash]

            print(f"[BLOCKED] API Key deaktiviert: {api_key.key_prefix}")
            return True

        except Exception as e:
            db.rollback()
            return False
        finally:
            db.close()

    def get_keys_for_user(self, user_email: str) -> list:
        """Gibt alle Keys eines Users zurück"""
        db = SessionLocal()
        try:
            keys = db.query(APIKey).filter(
                APIKey.user_email == user_email
            ).order_by(APIKey.created_at.desc()).all()

            return [key.to_dict() for key in keys]
        finally:
            db.close()

    def reset_monthly_usage(self):
        """Setzt monatliche Nutzung zurück (für Scheduler)"""
        db = SessionLocal()
        try:
            now = datetime.utcnow()
            keys_to_reset = db.query(APIKey).filter(
                APIKey.reset_at <= now
            ).all()

            for key in keys_to_reset:
                key.monthly_tokens_used = 0
                key.reset_at = now + timedelta(days=30)

            db.commit()
            print(f"[RETRY] {len(keys_to_reset)} API Keys zurückgesetzt")

        except Exception as e:
            db.rollback()
            print(f"[ERROR] Reset fehlgeschlagen: {e}")
        finally:
            db.close()


# Singleton Instance
_api_key_service: Optional[APIKeyService] = None


def get_api_key_service() -> APIKeyService:
    """Gibt Singleton-Instanz zurück"""
    global _api_key_service
    if _api_key_service is None:
        _api_key_service = APIKeyService()
    return _api_key_service
