#!/usr/bin/env python3
"""
SCIO - Earnings Model
"""

from datetime import datetime
from enum import Enum
from sqlalchemy import Column, Integer, String, Text, DateTime, Float, JSON, Enum as SQLEnum
from . import Base


class EarningSource(str, Enum):
    """Einnahmequelle"""
    WEBSITE = 'website'           # Stripe Zahlungen
    API = 'api'                   # API-Nutzung
    VASTAI = 'vastai'             # Vast.ai GPU Rental
    RUNPOD = 'runpod'             # RunPod
    FREELANCE = 'freelance'       # Manuelle Aufträge
    OTHER = 'other'


class EarningType(str, Enum):
    """Art der Einnahme/Ausgabe"""
    INCOME = 'income'
    EXPENSE = 'expense'
    REFUND = 'refund'


class Earning(Base):
    """Einnahmen-Tracking Datenmodell"""
    __tablename__ = 'earnings'

    id = Column(Integer, primary_key=True, autoincrement=True)
    earning_id = Column(String(64), unique=True, nullable=False, index=True)

    # Typ & Quelle
    earning_type = Column(SQLEnum(EarningType), default=EarningType.INCOME)
    source = Column(SQLEnum(EarningSource), nullable=False)

    # Betrag
    amount_cents = Column(Integer, nullable=False)
    currency = Column(String(3), default='EUR')
    net_amount_cents = Column(Integer)  # Nach Fees

    # Referenzen
    job_id = Column(String(64), index=True)
    order_id = Column(String(64), index=True)
    api_key_id = Column(Integer, index=True)
    external_id = Column(String(255))  # Stripe ID, Vast.ai ID, etc.

    # Details
    description = Column(Text)
    extra_data = Column(JSON)  # renamed from 'metadata' (reserved in SQLAlchemy)

    # Fees
    platform_fee_cents = Column(Integer, default=0)
    stripe_fee_cents = Column(Integer, default=0)

    # Status
    status = Column(String(32), default='completed')  # pending, completed, failed

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)

    def to_dict(self) -> dict:
        """Konvertiert Earning zu Dictionary"""
        return {
            'id': self.id,
            'earning_id': self.earning_id,
            'earning_type': self.earning_type.value if self.earning_type else None,
            'source': self.source.value if self.source else None,
            'amount_cents': self.amount_cents,
            'amount_eur': self.amount_cents / 100,
            'net_amount_cents': self.net_amount_cents,
            'net_amount_eur': (self.net_amount_cents or 0) / 100,
            'currency': self.currency,
            'job_id': self.job_id,
            'order_id': self.order_id,
            'external_id': self.external_id,
            'description': self.description,
            'platform_fee_cents': self.platform_fee_cents,
            'stripe_fee_cents': self.stripe_fee_cents,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
        }

    @property
    def amount_eur(self) -> float:
        """Betrag in Euro"""
        return self.amount_cents / 100

    @property
    def net_eur(self) -> float:
        """Netto-Betrag in Euro"""
        if self.net_amount_cents:
            return self.net_amount_cents / 100
        return self.amount_eur

    @property
    def total_fees_cents(self) -> int:
        """Gesamte Gebühren"""
        return (self.platform_fee_cents or 0) + (self.stripe_fee_cents or 0)

    def __repr__(self):
        return f"<Earning {self.earning_id} {self.amount_eur:.2f}€ [{self.source.value}]>"
