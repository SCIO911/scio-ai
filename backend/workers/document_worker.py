#!/usr/bin/env python3
"""
SCIO - Document Worker
PDF Processing, Document Analysis, RAG Pipeline
Optimiert für RTX 5090 mit 24GB VRAM
"""

import os
import time
import uuid
import json
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

from .base_worker import BaseWorker, WorkerStatus, model_manager
from backend.config import Config

# PyTorch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# PDF Libraries
PYMUPDF_AVAILABLE = False
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    pass

PYPDF_AVAILABLE = False
try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    pass

# Document Parsing
UNSTRUCTURED_AVAILABLE = False
try:
    from unstructured.partition.auto import partition
    from unstructured.partition.pdf import partition_pdf
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    pass

# Docling (IBM)
DOCLING_AVAILABLE = False
try:
    from docling.document_converter import DocumentConverter
    DOCLING_AVAILABLE = True
except ImportError:
    pass

# PIL for images
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Markdown
try:
    import markdown
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False

# Text Chunking
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


class DocumentWorker(BaseWorker):
    """
    Document Worker - Handles all document processing tasks

    Features:
    - PDF Text Extraction
    - PDF to Images
    - Document Parsing (tables, images, text)
    - Text Chunking for RAG
    - Document Analysis
    - Multi-format support (PDF, DOCX, PPTX, etc.)
    """

    def __init__(self):
        super().__init__("Document Processing")
        self._device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        self._docling = None

    def initialize(self) -> bool:
        """Initialize the worker"""
        available_features = []

        if PYMUPDF_AVAILABLE:
            available_features.append("PyMuPDF")
        if PYPDF_AVAILABLE:
            available_features.append("PyPDF")
        if UNSTRUCTURED_AVAILABLE:
            available_features.append("Unstructured")
        if DOCLING_AVAILABLE:
            available_features.append("Docling")

        if not available_features:
            self._error_message = "No document libraries available"
            self.status = WorkerStatus.ERROR
            return False

        self.status = WorkerStatus.READY
        print(f"[OK] Document Worker bereit (Features: {', '.join(available_features)})")
        return True

    def extract_text_from_pdf(
        self,
        pdf_path: str,
        pages: List[int] = None,
    ) -> dict:
        """Extract text from PDF"""
        start_time = time.time()

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        text_content = []
        page_count = 0

        if PYMUPDF_AVAILABLE:
            doc = fitz.open(pdf_path)
            page_count = len(doc)

            for page_num in range(page_count):
                if pages and page_num not in pages:
                    continue
                page = doc[page_num]
                text = page.get_text()
                text_content.append({
                    "page": page_num + 1,
                    "text": text,
                })
            doc.close()

        elif PYPDF_AVAILABLE:
            reader = PdfReader(pdf_path)
            page_count = len(reader.pages)

            for page_num, page in enumerate(reader.pages):
                if pages and page_num not in pages:
                    continue
                text = page.extract_text()
                text_content.append({
                    "page": page_num + 1,
                    "text": text,
                })

        else:
            raise ValueError("No PDF library available")

        # Combine all text
        full_text = "\n\n".join([p["text"] for p in text_content])

        return {
            "text": full_text,
            "pages": text_content,
            "page_count": page_count,
            "char_count": len(full_text),
            "word_count": len(full_text.split()),
            "processing_seconds": time.time() - start_time,
        }

    def pdf_to_images(
        self,
        pdf_path: str,
        pages: List[int] = None,
        dpi: int = 150,
        output_dir: str = None,
    ) -> dict:
        """Convert PDF pages to images"""
        start_time = time.time()

        if not PYMUPDF_AVAILABLE:
            raise ValueError("PyMuPDF required for PDF to images")

        if output_dir is None:
            output_dir = str(Config.DATA_DIR / "generated" / f"pdf_{uuid.uuid4().hex[:8]}")

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        doc = fitz.open(pdf_path)
        image_paths = []

        zoom = dpi / 72  # 72 is default DPI
        mat = fitz.Matrix(zoom, zoom)

        for page_num in range(len(doc)):
            if pages and page_num not in pages:
                continue

            page = doc[page_num]
            pix = page.get_pixmap(matrix=mat)

            output_path = os.path.join(output_dir, f"page_{page_num + 1:04d}.png")
            pix.save(output_path)
            image_paths.append(output_path)

        doc.close()

        return {
            "output_dir": output_dir,
            "image_paths": image_paths,
            "page_count": len(image_paths),
            "dpi": dpi,
            "processing_seconds": time.time() - start_time,
        }

    def parse_document(
        self,
        file_path: str,
        extract_images: bool = True,
        extract_tables: bool = True,
    ) -> dict:
        """Parse document with advanced extraction"""
        start_time = time.time()

        if DOCLING_AVAILABLE:
            if self._docling is None:
                self._docling = DocumentConverter()

            result = self._docling.convert(file_path)

            return {
                "text": result.document.export_to_text(),
                "markdown": result.document.export_to_markdown(),
                "tables": [t.to_dict() for t in result.document.tables] if extract_tables else [],
                "images": len(result.document.pictures) if extract_images else 0,
                "processing_seconds": time.time() - start_time,
            }

        elif UNSTRUCTURED_AVAILABLE:
            elements = partition(file_path)

            text_content = []
            tables = []

            for element in elements:
                if hasattr(element, 'text'):
                    text_content.append(element.text)
                if extract_tables and element.category == "Table":
                    tables.append(str(element))

            return {
                "text": "\n\n".join(text_content),
                "elements": len(elements),
                "tables": tables,
                "processing_seconds": time.time() - start_time,
            }

        else:
            # Fallback to basic extraction
            if file_path.lower().endswith('.pdf'):
                return self.extract_text_from_pdf(file_path)
            else:
                raise ValueError("Advanced parsing not available")

    def chunk_text(
        self,
        text: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: List[str] = None,
    ) -> dict:
        """Split text into chunks for RAG"""
        start_time = time.time()

        if LANGCHAIN_AVAILABLE:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=separators or ["\n\n", "\n", ". ", " ", ""],
            )
            chunks = splitter.split_text(text)
        else:
            # Simple chunking fallback
            chunks = []
            for i in range(0, len(text), chunk_size - chunk_overlap):
                chunk = text[i:i + chunk_size]
                if chunk.strip():
                    chunks.append(chunk)

        # Add metadata
        chunk_data = []
        for i, chunk in enumerate(chunks):
            chunk_data.append({
                "id": i,
                "text": chunk,
                "char_count": len(chunk),
                "word_count": len(chunk.split()),
                "hash": hashlib.md5(chunk.encode()).hexdigest()[:8],
            })

        return {
            "chunks": chunk_data,
            "chunk_count": len(chunks),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "total_chars": len(text),
            "processing_seconds": time.time() - start_time,
        }

    def summarize_document(
        self,
        file_path: str = None,
        text: str = None,
    ) -> dict:
        """Generate document summary statistics"""
        if file_path:
            result = self.extract_text_from_pdf(file_path) if file_path.lower().endswith('.pdf') else self.parse_document(file_path)
            text = result.get("text", "")

        if not text:
            raise ValueError("No text to summarize")

        words = text.split()
        sentences = text.replace("!", ".").replace("?", ".").split(".")
        paragraphs = text.split("\n\n")

        return {
            "char_count": len(text),
            "word_count": len(words),
            "sentence_count": len([s for s in sentences if s.strip()]),
            "paragraph_count": len([p for p in paragraphs if p.strip()]),
            "avg_word_length": sum(len(w) for w in words) / len(words) if words else 0,
            "avg_sentence_length": len(words) / len(sentences) if sentences else 0,
        }

    def process(self, job_id: str, input_data: dict) -> dict:
        """Process a document job"""
        task_type = input_data.get("task", "extract_text")

        self.notify_progress(job_id, 0.1, f"Starting {task_type}")

        if task_type == "extract_text":
            file_path = input_data.get("file") or input_data.get("pdf_path")
            pages = input_data.get("pages")
            result = self.extract_text_from_pdf(file_path, pages=pages)

        elif task_type == "pdf_to_images":
            file_path = input_data.get("file") or input_data.get("pdf_path")
            pages = input_data.get("pages")
            dpi = input_data.get("dpi", 150)
            result = self.pdf_to_images(file_path, pages=pages, dpi=dpi)

        elif task_type == "parse":
            file_path = input_data.get("file")
            extract_images = input_data.get("extract_images", True)
            extract_tables = input_data.get("extract_tables", True)
            result = self.parse_document(file_path, extract_images, extract_tables)

        elif task_type == "chunk":
            text = input_data.get("text")
            if not text and input_data.get("file"):
                doc_result = self.parse_document(input_data.get("file"))
                text = doc_result.get("text", "")
            chunk_size = input_data.get("chunk_size", 1000)
            chunk_overlap = input_data.get("chunk_overlap", 200)
            result = self.chunk_text(text, chunk_size, chunk_overlap)

        elif task_type == "summarize":
            file_path = input_data.get("file")
            text = input_data.get("text")
            result = self.summarize_document(file_path, text)

        else:
            raise ValueError(f"Unknown task type: {task_type}")

        self.notify_progress(job_id, 1.0, "Complete")
        return result

    def cleanup(self):
        """Release resources"""
        self._docling = None
        print("[OK] Document Worker bereinigt")

    def get_supported_formats(self) -> List[str]:
        """Return supported document formats"""
        formats = [".pdf"]
        if UNSTRUCTURED_AVAILABLE or DOCLING_AVAILABLE:
            formats.extend([".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls", ".txt", ".md", ".html"])
        return formats


# Singleton Instance
_document_worker: Optional[DocumentWorker] = None


def get_document_worker() -> DocumentWorker:
    """Get singleton instance"""
    global _document_worker
    if _document_worker is None:
        _document_worker = DocumentWorker()
    return _document_worker
