"""
SCIO MultiModal Module

Einheitliche Verarbeitung aller Modalitäten:
- Text (Verstehen, Generieren, Übersetzen)
- Bild (Verstehen, Generieren, Bearbeiten)
- Audio (Verstehen, Generieren, Transkribieren)
- Video (Verstehen, Generieren, Bearbeiten)
- Code (Verstehen, Generieren, Debuggen)
- 3D (Verstehen, Generieren)
"""

from scio.multimodal.unified_engine import (
    UnifiedMultiModal,
    ModalityType,
    MultiModalConfig,
    MultiModalResult,
    get_multimodal,
)

__all__ = [
    "UnifiedMultiModal",
    "ModalityType",
    "MultiModalConfig",
    "MultiModalResult",
    "get_multimodal",
]
