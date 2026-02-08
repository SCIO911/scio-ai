#!/usr/bin/env python3
"""
SCIO - API Key Model
"""

import secrets
import hashlib
from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, JSON
from . import Base


class APIKey(Base):
    """API Key Datenmodell"""
    __tablename__ = 'api_keys'

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Key Info
    key_hash = Column(String(64), unique=True, nullable=False, index=True)
    key_prefix = Column(String(12), nullable=False)  # scio_xxx für Anzeige
    name = Column(String(255))
    description = Column(Text)

    # User Info
    user_email = Column(String(255), nullable=False, index=True)
    user_id = Column(String(64), index=True)

    # Status
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)

    # Limits
    rate_limit_rpm = Column(Integer, default=60)  # Requests per Minute
    rate_limit_tpm = Column(Integer, default=100000)  # Tokens per Minute
    monthly_token_limit = Column(Integer, default=1000000)  # 1M tokens/month
    monthly_tokens_used = Column(Integer, default=0)

    # Credit System
    credits_cents = Column(Integer, default=0)  # Prepaid Credits
    total_spent_cents = Column(Integer, default=0)

    # Usage Stats
    total_requests = Column(Integer, default=0)
    total_tokens_input = Column(Integer, default=0)
    total_tokens_output = Column(Integer, default=0)

    # Permissions (JSON Array)
    allowed_models = Column(JSON)  # None = all models
    allowed_features = Column(JSON)  # ['inference', 'training', 'images']

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used_at = Column(DateTime)
    expires_at = Column(DateTime)
    reset_at = Column(DateTime)  # Monthly reset

    @staticmethod
    def generate_key() -> tuple[str, str, str]:
        """
        Generiert neuen API Key

        Returns:
            (full_key, key_hash, key_prefix)
        """
        # Generiere zufälligen Key
        random_part = secrets.token_hex(24)
        full_key = f"scio_{random_part}"

        # Hash für Speicherung
        key_hash = hashlib.sha256(full_key.encode()).hexdigest()

        # Prefix für Anzeige
        key_prefix = f"scio_{random_part[:8]}..."

        return full_key, key_hash, key_prefix

    @staticmethod
    def hash_key(key: str) -> str:
        """Hasht einen API Key"""
        return hashlib.sha256(key.encode()).hexdigest()

    def to_dict(self, include_stats: bool = True) -> dict:
        """Konvertiert APIKey zu Dictionary"""
        data = {
            'id': self.id,
            'key_prefix': self.key_prefix,
            'name': self.name,
            'description': self.description,
            'user_email': self.user_email,
            'is_active': self.is_active,
            'is_admin': self.is_admin,
            'rate_limit_rpm': self.rate_limit_rpm,
            'rate_limit_tpm': self.rate_limit_tpm,
            'monthly_token_limit': self.monthly_token_limit,
            'credits_cents': self.credits_cents,
            'credits_eur': self.credits_cents / 100,
            'allowed_models': self.allowed_models,
            'allowed_features': self.allowed_features,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
        }

        if include_stats:
            data.update({
                'monthly_tokens_used': self.monthly_tokens_used,
                'total_requests': self.total_requests,
                'total_tokens_input': self.total_tokens_input,
                'total_tokens_output': self.total_tokens_output,
                'total_spent_cents': self.total_spent_cents,
                'total_spent_eur': self.total_spent_cents / 100,
                'last_used_at': self.last_used_at.isoformat() if self.last_used_at else None,
            })

        return data

    @property
    def is_valid(self) -> bool:
        """Prüft ob Key gültig ist"""
        if not self.is_active:
            return False
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        return True

    @property
    def tokens_remaining(self) -> int:
        """Verbleibende Tokens im Monat"""
        return max(0, self.monthly_token_limit - self.monthly_tokens_used)

    def has_credits(self, amount_cents: int = 1) -> bool:
        """Prüft ob genug Credits vorhanden"""
        return self.credits_cents >= amount_cents

    def __repr__(self):
        return f"<APIKey {self.key_prefix} ({self.user_email})>"
