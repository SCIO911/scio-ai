#!/usr/bin/env python3
"""
SCIO - Memory System
Langzeit-Gedächtnis für kontinuierliche Verbesserung

Features:
- Event-Logging
- Erfahrungs-Speicherung
- Pattern-Erkennung
- Fehler-Analyse
- Erfolgs-Tracking
"""

import os
import json
import sqlite3
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum


class EventType(str, Enum):
    """Typen von Events"""
    SYSTEM = "system"
    EVOLUTION = "evolution"
    CODE_GENERATION = "code_generation"
    CODE_SAVED = "code_saved"
    TEST = "test"
    ERROR = "error"
    SUCCESS = "success"
    LEARNING = "learning"
    USER_FEEDBACK = "user_feedback"


@dataclass
class MemoryEvent:
    """Ein gespeichertes Event"""
    id: int
    timestamp: datetime
    event_type: str
    message: str
    data: Optional[Dict[str, Any]]
    tags: List[str]


@dataclass
class LearningEntry:
    """Ein Lern-Eintrag"""
    id: int
    created_at: datetime
    category: str
    pattern: str
    outcome: str  # success, failure, neutral
    confidence: float
    occurrences: int
    last_seen: datetime


class Memory:
    """
    SCIO Memory System

    Speichert und analysiert:
    - Events (was ist passiert)
    - Learnings (was wurde gelernt)
    - Patterns (wiederkehrende Muster)
    - Errors (Fehler und deren Ursachen)
    """

    def __init__(self):
        from backend.config import Config
        self.base_path = Path(getattr(Config, 'DATA_DIR', 'C:/SCIO/data'))
        self.db_path = self.base_path / "memory.db"
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()

    def initialize(self) -> bool:
        """Initialisiert die Datenbank"""
        try:
            self.base_path.mkdir(parents=True, exist_ok=True)

            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row

            # Create tables
            self._create_tables()

            print(f"[OK] Memory initialisiert: {self.db_path}")
            return True

        except Exception as e:
            print(f"[ERROR] Memory Init fehlgeschlagen: {e}")
            return False

    def _create_tables(self):
        """Erstellt Datenbank-Tabellen"""
        cursor = self._conn.cursor()

        # Events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                message TEXT NOT NULL,
                data TEXT,
                tags TEXT
            )
        ''')

        # Learnings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learnings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                category TEXT NOT NULL,
                pattern TEXT NOT NULL,
                outcome TEXT NOT NULL,
                confidence REAL DEFAULT 0.5,
                occurrences INTEGER DEFAULT 1,
                last_seen TEXT NOT NULL,
                UNIQUE(category, pattern)
            )
        ''')

        # Errors table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS errors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                error_type TEXT NOT NULL,
                message TEXT NOT NULL,
                context TEXT,
                resolved INTEGER DEFAULT 0,
                resolution TEXT
            )
        ''')

        # Statistics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS statistics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                metric TEXT NOT NULL,
                value REAL NOT NULL,
                UNIQUE(date, metric)
            )
        ''')

        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_learnings_category ON learnings(category)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_errors_resolved ON errors(resolved)')

        self._conn.commit()

    def log_event(
        self,
        event_type: str,
        message: str,
        data: Dict[str, Any] = None,
        tags: List[str] = None
    ) -> int:
        """
        Loggt ein Event

        Args:
            event_type: Typ des Events
            message: Nachricht
            data: Zusätzliche Daten
            tags: Tags für Kategorisierung

        Returns:
            Event ID
        """
        with self._lock:
            cursor = self._conn.cursor()

            cursor.execute('''
                INSERT INTO events (timestamp, event_type, message, data, tags)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                event_type,
                message,
                json.dumps(data) if data else None,
                json.dumps(tags) if tags else None,
            ))

            self._conn.commit()
            return cursor.lastrowid

    def get_events(
        self,
        event_type: str = None,
        since: datetime = None,
        limit: int = 100
    ) -> List[MemoryEvent]:
        """
        Ruft Events ab

        Args:
            event_type: Optionaler Filter nach Typ
            since: Optionaler Filter nach Zeit
            limit: Maximale Anzahl

        Returns:
            Liste von Events
        """
        with self._lock:
            cursor = self._conn.cursor()

            query = "SELECT * FROM events WHERE 1=1"
            params = []

            if event_type:
                query += " AND event_type = ?"
                params.append(event_type)

            if since:
                query += " AND timestamp >= ?"
                params.append(since.isoformat())

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)

            events = []
            for row in cursor.fetchall():
                events.append(MemoryEvent(
                    id=row['id'],
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    event_type=row['event_type'],
                    message=row['message'],
                    data=json.loads(row['data']) if row['data'] else None,
                    tags=json.loads(row['tags']) if row['tags'] else [],
                ))

            return events

    def learn(
        self,
        category: str,
        pattern: str,
        outcome: str,
        confidence: float = 0.5
    ):
        """
        Speichert ein Learning

        Args:
            category: Kategorie (z.B. "code_generation", "testing")
            pattern: Das erkannte Muster
            outcome: Ergebnis (success, failure, neutral)
            confidence: Konfidenz (0-1)
        """
        with self._lock:
            cursor = self._conn.cursor()
            now = datetime.now().isoformat()

            # Try to update existing
            cursor.execute('''
                UPDATE learnings
                SET occurrences = occurrences + 1,
                    last_seen = ?,
                    confidence = (confidence * occurrences + ?) / (occurrences + 1)
                WHERE category = ? AND pattern = ?
            ''', (now, confidence, category, pattern))

            if cursor.rowcount == 0:
                # Insert new
                cursor.execute('''
                    INSERT INTO learnings (created_at, category, pattern, outcome, confidence, last_seen)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (now, category, pattern, outcome, confidence, now))

            self._conn.commit()

    def get_learnings(
        self,
        category: str = None,
        min_confidence: float = 0.0,
        limit: int = 50
    ) -> List[LearningEntry]:
        """
        Ruft Learnings ab

        Args:
            category: Optionaler Filter nach Kategorie
            min_confidence: Minimale Konfidenz
            limit: Maximale Anzahl

        Returns:
            Liste von Learnings
        """
        with self._lock:
            cursor = self._conn.cursor()

            query = "SELECT * FROM learnings WHERE confidence >= ?"
            params = [min_confidence]

            if category:
                query += " AND category = ?"
                params.append(category)

            query += " ORDER BY occurrences DESC, confidence DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)

            learnings = []
            for row in cursor.fetchall():
                learnings.append(LearningEntry(
                    id=row['id'],
                    created_at=datetime.fromisoformat(row['created_at']),
                    category=row['category'],
                    pattern=row['pattern'],
                    outcome=row['outcome'],
                    confidence=row['confidence'],
                    occurrences=row['occurrences'],
                    last_seen=datetime.fromisoformat(row['last_seen']),
                ))

            return learnings

    def log_error(
        self,
        error_type: str,
        message: str,
        context: Dict[str, Any] = None
    ) -> int:
        """
        Loggt einen Fehler

        Args:
            error_type: Typ des Fehlers
            message: Fehlermeldung
            context: Kontext-Informationen

        Returns:
            Error ID
        """
        with self._lock:
            cursor = self._conn.cursor()

            cursor.execute('''
                INSERT INTO errors (timestamp, error_type, message, context)
                VALUES (?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                error_type,
                message,
                json.dumps(context) if context else None,
            ))

            self._conn.commit()

            # Also log as event
            self.log_event("error", message, {"type": error_type, "context": context})

            return cursor.lastrowid

    def resolve_error(self, error_id: int, resolution: str):
        """
        Markiert einen Fehler als gelöst

        Args:
            error_id: ID des Fehlers
            resolution: Wie wurde er gelöst
        """
        with self._lock:
            cursor = self._conn.cursor()

            cursor.execute('''
                UPDATE errors
                SET resolved = 1, resolution = ?
                WHERE id = ?
            ''', (resolution, error_id))

            self._conn.commit()

    def get_unresolved_errors(self, limit: int = 20) -> List[dict]:
        """Gibt ungelöste Fehler zurück"""
        with self._lock:
            cursor = self._conn.cursor()

            cursor.execute('''
                SELECT * FROM errors
                WHERE resolved = 0
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))

            errors = []
            for row in cursor.fetchall():
                errors.append({
                    "id": row['id'],
                    "timestamp": row['timestamp'],
                    "error_type": row['error_type'],
                    "message": row['message'],
                    "context": json.loads(row['context']) if row['context'] else None,
                })

            return errors

    def record_statistic(self, metric: str, value: float, date: datetime = None):
        """
        Speichert eine Statistik

        Args:
            metric: Name der Metrik
            value: Wert
            date: Datum (default: heute)
        """
        with self._lock:
            cursor = self._conn.cursor()
            date_str = (date or datetime.now()).strftime("%Y-%m-%d")

            cursor.execute('''
                INSERT OR REPLACE INTO statistics (date, metric, value)
                VALUES (?, ?, ?)
            ''', (date_str, metric, value))

            self._conn.commit()

    def get_statistics(
        self,
        metric: str,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Ruft Statistiken ab

        Args:
            metric: Name der Metrik
            days: Anzahl Tage zurück

        Returns:
            Liste von Statistik-Einträgen
        """
        with self._lock:
            cursor = self._conn.cursor()

            since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

            cursor.execute('''
                SELECT date, value FROM statistics
                WHERE metric = ? AND date >= ?
                ORDER BY date ASC
            ''', (metric, since))

            return [{"date": row['date'], "value": row['value']} for row in cursor.fetchall()]

    def get_summary(self) -> dict:
        """Gibt eine Zusammenfassung des Memory zurück"""
        with self._lock:
            cursor = self._conn.cursor()

            # Count events
            cursor.execute("SELECT COUNT(*) as count FROM events")
            total_events = cursor.fetchone()['count']

            # Count events by type
            cursor.execute('''
                SELECT event_type, COUNT(*) as count
                FROM events
                GROUP BY event_type
            ''')
            events_by_type = {row['event_type']: row['count'] for row in cursor.fetchall()}

            # Count learnings
            cursor.execute("SELECT COUNT(*) as count FROM learnings")
            total_learnings = cursor.fetchone()['count']

            # Count errors
            cursor.execute("SELECT COUNT(*) as count FROM errors WHERE resolved = 0")
            unresolved_errors = cursor.fetchone()['count']

            # Recent activity
            cursor.execute('''
                SELECT COUNT(*) as count FROM events
                WHERE timestamp >= ?
            ''', ((datetime.now() - timedelta(hours=24)).isoformat(),))
            events_24h = cursor.fetchone()['count']

            return {
                "total_events": total_events,
                "events_by_type": events_by_type,
                "total_learnings": total_learnings,
                "unresolved_errors": unresolved_errors,
                "events_last_24h": events_24h,
                "database_path": str(self.db_path),
            }

    def cleanup(self, days_to_keep: int = 90):
        """
        Bereinigt alte Einträge

        Args:
            days_to_keep: Wie viele Tage behalten
        """
        with self._lock:
            cursor = self._conn.cursor()
            cutoff = (datetime.now() - timedelta(days=days_to_keep)).isoformat()

            cursor.execute("DELETE FROM events WHERE timestamp < ?", (cutoff,))
            events_deleted = cursor.rowcount

            cursor.execute("DELETE FROM errors WHERE timestamp < ? AND resolved = 1", (cutoff,))
            errors_deleted = cursor.rowcount

            self._conn.commit()

            print(f"[CLEANUP] Gelöscht: {events_deleted} Events, {errors_deleted} Errors")

    def close(self):
        """Schließt die Datenbankverbindung"""
        if self._conn:
            self._conn.close()
            self._conn = None


# Singleton
_memory: Optional[Memory] = None

def get_memory() -> Memory:
    """Gibt Singleton-Instanz zurück"""
    global _memory
    if _memory is None:
        _memory = Memory()
    return _memory
