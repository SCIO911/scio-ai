#!/usr/bin/env python3
"""
SCIO - Job Model
"""

from datetime import datetime
from enum import Enum
from sqlalchemy import Column, Integer, String, Text, DateTime, Float, JSON, Enum as SQLEnum
from . import Base


class JobStatus(str, Enum):
    """Job Status Enum"""
    PENDING = 'pending'
    QUEUED = 'queued'
    RUNNING = 'running'
    COMPLETED = 'completed'
    FAILED = 'failed'
    CANCELLED = 'cancelled'


class JobType(str, Enum):
    """Job Type Enum - Alle verfügbaren AI-Tasks"""
    # LLM
    LLM_TRAINING = 'llm_training'
    LLM_INFERENCE = 'llm_inference'
    BATCH_INFERENCE = 'batch_inference'

    # Image
    IMAGE_GENERATION = 'image_generation'
    IMAGE_UPSCALE = 'image_upscale'
    FACE_RESTORE = 'face_restore'

    # Audio
    SPEECH_TO_TEXT = 'speech_to_text'
    TEXT_TO_SPEECH = 'text_to_speech'
    MUSIC_GENERATION = 'music_generation'

    # Video
    VIDEO_GENERATION = 'video_generation'
    IMAGE_TO_VIDEO = 'image_to_video'

    # Vision
    IMAGE_CAPTION = 'image_caption'
    VISUAL_QA = 'visual_qa'
    OCR = 'ocr'
    OBJECT_DETECTION = 'object_detection'

    # Code
    CODE_GENERATION = 'code_generation'
    CODE_COMPLETION = 'code_completion'
    CODE_REVIEW = 'code_review'
    CODE_FIX = 'code_fix'

    # Embeddings
    TEXT_EMBEDDING = 'text_embedding'
    IMAGE_EMBEDDING = 'image_embedding'
    SIMILARITY_SEARCH = 'similarity_search'

    # 3D
    TEXT_TO_3D = 'text_to_3d'
    IMAGE_TO_3D = 'image_to_3d'

    # Documents
    DOCUMENT_PARSE = 'document_parse'
    PDF_EXTRACT = 'pdf_extract'
    TEXT_CHUNK = 'text_chunk'


class Job(Base):
    """Job Datenmodell"""
    __tablename__ = 'jobs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(64), unique=True, nullable=False, index=True)
    order_id = Column(String(64), index=True)

    # Job Info
    job_type = Column(SQLEnum(JobType), nullable=False)
    status = Column(SQLEnum(JobStatus), default=JobStatus.PENDING)
    priority = Column(Integer, default=0)  # Higher = more priority

    # User Info
    user_email = Column(String(255))
    api_key_id = Column(Integer, index=True)

    # Job Data
    input_data = Column(JSON)  # Model, Parameters, Input
    output_data = Column(JSON)  # Result, Output Path
    error_message = Column(Text)

    # Resource Usage
    tokens_input = Column(Integer, default=0)
    tokens_output = Column(Integer, default=0)
    gpu_seconds = Column(Float, default=0)
    vram_peak_gb = Column(Float, default=0)

    # Billing
    cost_cents = Column(Integer, default=0)
    paid = Column(Integer, default=0)  # 0=no, 1=yes

    # Retry
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self) -> dict:
        """Konvertiert Job zu Dictionary"""
        return {
            'id': self.id,
            'job_id': self.job_id,
            'order_id': self.order_id,
            'job_type': self.job_type.value if self.job_type else None,
            'status': self.status.value if self.status else None,
            'priority': self.priority,
            'user_email': self.user_email,
            'input_data': self.input_data,
            'output_data': self.output_data,
            'error_message': self.error_message,
            'tokens_input': self.tokens_input,
            'tokens_output': self.tokens_output,
            'gpu_seconds': self.gpu_seconds,
            'vram_peak_gb': self.vram_peak_gb,
            'cost_cents': self.cost_cents,
            'paid': self.paid,
            'retry_count': self.retry_count,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
        }

    @property
    def duration_seconds(self) -> float:
        """Berechnet Job-Dauer in Sekunden"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return 0

    @property
    def cost_eur(self) -> float:
        """Kosten in Euro"""
        return self.cost_cents / 100

    def __repr__(self):
        return f"<Job {self.job_id} [{self.status.value}]>"
