"""
Data Protocol
=============

Standardisierte Datenformate und Serialisierung.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, Union, Callable
from enum import Enum, auto
from datetime import datetime
import json
import base64
import hashlib
import zlib


class DataFormat(Enum):
    """Unterstuetzte Datenformate"""
    JSON = "json"
    BINARY = "binary"
    CSV = "csv"
    NUMPY = "numpy"
    PANDAS = "pandas"
    TORCH = "torch"
    RAW = "raw"


class CompressionType(Enum):
    """Kompressionstypen"""
    NONE = "none"
    ZLIB = "zlib"
    GZIP = "gzip"
    LZ4 = "lz4"


@dataclass
class DataSchema:
    """Schema-Definition fuer Datenvalidierung"""

    name: str = ""
    version: str = "1.0"
    fields: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    required: List[str] = field(default_factory=list)
    description: str = ""

    def validate(self, data: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validiert Daten gegen Schema"""
        errors = []

        # Pruefe erforderliche Felder
        for req_field in self.required:
            if req_field not in data:
                errors.append(f"Erforderliches Feld fehlt: {req_field}")

        # Pruefe Feldtypen
        for field_name, field_def in self.fields.items():
            if field_name in data:
                expected_type = field_def.get("type")
                if expected_type and not self._check_type(data[field_name], expected_type):
                    errors.append(f"Falscher Typ fuer {field_name}: erwartet {expected_type}")

        return len(errors) == 0, errors

    def _check_type(self, value: Any, expected: str) -> bool:
        """Prueft Typ eines Werts"""
        type_map = {
            "string": str,
            "int": int,
            "float": (int, float),
            "bool": bool,
            "list": list,
            "dict": dict,
            "any": object,
        }
        expected_type = type_map.get(expected, object)
        return isinstance(value, expected_type)

    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert zu Dictionary"""
        return {
            "name": self.name,
            "version": self.version,
            "fields": self.fields,
            "required": self.required,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataSchema':
        """Erstellt Schema aus Dictionary"""
        return cls(
            name=data.get("name", ""),
            version=data.get("version", "1.0"),
            fields=data.get("fields", {}),
            required=data.get("required", []),
            description=data.get("description", ""),
        )


@dataclass
class DataPacket:
    """Container fuer Datenaustausch"""

    id: str = field(default_factory=lambda: hashlib.md5(str(datetime.now()).encode()).hexdigest()[:12])
    format: DataFormat = DataFormat.JSON
    data: Any = None
    schema: Optional[DataSchema] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    compression: CompressionType = CompressionType.NONE
    checksum: Optional[str] = None
    size_bytes: int = 0

    def __post_init__(self):
        if self.data is not None:
            self._update_size()

    def _update_size(self):
        """Aktualisiert Groesse"""
        try:
            if isinstance(self.data, (str, bytes)):
                self.size_bytes = len(self.data)
            elif isinstance(self.data, (list, dict)):
                self.size_bytes = len(json.dumps(self.data, default=str))
            else:
                self.size_bytes = len(str(self.data))
        except Exception:
            self.size_bytes = 0

    def compute_checksum(self) -> str:
        """Berechnet Pruefsumme"""
        if isinstance(self.data, bytes):
            data_bytes = self.data
        else:
            data_bytes = json.dumps(self.data, default=str, sort_keys=True).encode()

        self.checksum = hashlib.sha256(data_bytes).hexdigest()
        return self.checksum

    def verify_checksum(self) -> bool:
        """Verifiziert Pruefsumme"""
        if not self.checksum:
            return True

        computed = self.compute_checksum()
        return computed == self.checksum

    def compress(self, compression: CompressionType = CompressionType.ZLIB) -> 'DataPacket':
        """Komprimiert Daten"""
        if self.compression != CompressionType.NONE:
            return self

        if isinstance(self.data, bytes):
            data_bytes = self.data
        else:
            data_bytes = json.dumps(self.data, default=str).encode()

        if compression == CompressionType.ZLIB:
            compressed = zlib.compress(data_bytes)
        else:
            compressed = data_bytes

        return DataPacket(
            id=self.id,
            format=DataFormat.BINARY,
            data=base64.b64encode(compressed).decode(),
            schema=self.schema,
            metadata={**self.metadata, "original_format": self.format.value},
            compression=compression,
        )

    def decompress(self) -> 'DataPacket':
        """Dekomprimiert Daten"""
        if self.compression == CompressionType.NONE:
            return self

        data_bytes = base64.b64decode(self.data)

        if self.compression == CompressionType.ZLIB:
            decompressed = zlib.decompress(data_bytes)
        else:
            decompressed = data_bytes

        original_format = DataFormat(self.metadata.get("original_format", "json"))

        if original_format == DataFormat.JSON:
            data = json.loads(decompressed.decode())
        else:
            data = decompressed

        return DataPacket(
            id=self.id,
            format=original_format,
            data=data,
            schema=self.schema,
            metadata=self.metadata,
            compression=CompressionType.NONE,
        )

    def validate(self) -> tuple[bool, List[str]]:
        """Validiert gegen Schema"""
        if not self.schema:
            return True, []

        if not isinstance(self.data, dict):
            return False, ["Daten muessen ein Dictionary sein fuer Schema-Validierung"]

        return self.schema.validate(self.data)

    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert zu Dictionary"""
        return {
            "id": self.id,
            "format": self.format.value,
            "data": self.data,
            "schema": self.schema.to_dict() if self.schema else None,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "compression": self.compression.value,
            "checksum": self.checksum,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataPacket':
        """Erstellt DataPacket aus Dictionary"""
        return cls(
            id=data.get("id", ""),
            format=DataFormat(data.get("format", "json")),
            data=data.get("data"),
            schema=DataSchema.from_dict(data["schema"]) if data.get("schema") else None,
            metadata=data.get("metadata", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now(),
            compression=CompressionType(data.get("compression", "none")),
            checksum=data.get("checksum"),
            size_bytes=data.get("size_bytes", 0),
        )


def serialize(data: Any, format: DataFormat = DataFormat.JSON) -> bytes:
    """Serialisiert Daten"""
    if format == DataFormat.JSON:
        return json.dumps(data, default=str, ensure_ascii=False).encode('utf-8')

    elif format == DataFormat.BINARY:
        if isinstance(data, bytes):
            return data
        return str(data).encode('utf-8')

    elif format == DataFormat.NUMPY:
        try:
            import numpy as np
            import io
            buffer = io.BytesIO()
            np.save(buffer, data)
            return buffer.getvalue()
        except ImportError:
            raise ValueError("NumPy nicht installiert")

    elif format == DataFormat.TORCH:
        try:
            import torch
            import io
            buffer = io.BytesIO()
            torch.save(data, buffer)
            return buffer.getvalue()
        except ImportError:
            raise ValueError("PyTorch nicht installiert")

    elif format == DataFormat.PANDAS:
        try:
            import pandas as pd
            import io
            buffer = io.BytesIO()
            data.to_pickle(buffer)
            return buffer.getvalue()
        except ImportError:
            raise ValueError("Pandas nicht installiert")

    else:
        return str(data).encode('utf-8')


def deserialize(data: bytes, format: DataFormat = DataFormat.JSON) -> Any:
    """Deserialisiert Daten"""
    if format == DataFormat.JSON:
        return json.loads(data.decode('utf-8'))

    elif format == DataFormat.BINARY:
        return data

    elif format == DataFormat.NUMPY:
        try:
            import numpy as np
            import io
            buffer = io.BytesIO(data)
            return np.load(buffer, allow_pickle=True)
        except ImportError:
            raise ValueError("NumPy nicht installiert")

    elif format == DataFormat.TORCH:
        try:
            import torch
            import io
            buffer = io.BytesIO(data)
            return torch.load(buffer)
        except ImportError:
            raise ValueError("PyTorch nicht installiert")

    elif format == DataFormat.PANDAS:
        try:
            import pandas as pd
            import io
            buffer = io.BytesIO(data)
            return pd.read_pickle(buffer)
        except ImportError:
            raise ValueError("Pandas nicht installiert")

    else:
        return data.decode('utf-8')


__all__ = [
    'DataFormat',
    'CompressionType',
    'DataSchema',
    'DataPacket',
    'serialize',
    'deserialize',
]
