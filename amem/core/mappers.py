#!/usr/bin/env python3
"""
Data Mappers for converting between domain models and persistence/DTO formats.
"""

import json
from typing import Any, Dict, Protocol, TypedDict, Union

from amem.core.models.memory_note import MemoryNote
from amem.utils.utils import setup_logger

logger = setup_logger(__name__)


# Define the structure expected by ChromaDB Store
class ChromaPersistenceData(TypedDict):
    id: str
    document: str
    metadata: Dict[str, Union[str, int, float, bool]]
    # embedding is handled separately by the store/retriever


class IMemoryNoteMapper(Protocol):
    """Interface for mapping MemoryNote domain objects to/from persistence formats."""

    def to_persistence(self, note: MemoryNote) -> ChromaPersistenceData:
        """Convert a MemoryNote domain object to a ChromaDB-compatible dict."""
        ...

    def to_domain(self, data_id: str, document: str, metadata: Dict[str, Any]) -> MemoryNote:
        """Convert raw data retrieved from ChromaDB into a MemoryNote domain object."""
        ...


class MemoryNoteMapper(IMemoryNoteMapper):
    """Maps MemoryNote to/from ChromaDB data format."""

    def to_persistence(self, note: MemoryNote) -> ChromaPersistenceData:
        """Convert a MemoryNote domain object to a ChromaDB-compatible dict."""
        processed_metadata = {}
        for key, value in note.metadata.items():
            if isinstance(value, list):
                try:
                    processed_metadata[key] = json.dumps(value)
                except TypeError:
                    logger.warning(f"Mapper: Could not JSON serialize list for key '{key}'. Storing as string.")
                    processed_metadata[key] = str(value)
            elif value is None:
                processed_metadata[key] = ""  # Store None as empty string
            elif not isinstance(value, (str, int, float, bool)):
                # Ensure compatibility with ChromaDB basic types
                processed_metadata[key] = str(value)
            else:
                processed_metadata[key] = value

        # Ensure essential timestamps are in the processed metadata
        processed_metadata.setdefault("created_at", note.created_at)
        processed_metadata.setdefault("updated_at", note.updated_at)

        return {
            "id": note.id,
            "document": note.content,
            "metadata": processed_metadata,
        }

    def to_domain(self, data_id: str, document: str, metadata: Dict[str, Any]) -> MemoryNote:
        """Convert raw data retrieved from ChromaDB into a MemoryNote domain object."""
        deserialized_metadata = {}
        if metadata:  # Handle potentially None metadata from Chroma get
            for key, value in metadata.items():
                if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
                    try:
                        deserialized_metadata[key] = json.loads(value)
                    except json.JSONDecodeError:
                        deserialized_metadata[key] = value  # Keep as string if invalid JSON
                elif value == "":
                    # Keep empty strings as is for now, unless specific keys known to be None
                    deserialized_metadata[key] = value
                else:
                    deserialized_metadata[key] = value

        # The MemoryNote constructor handles setting defaults if keys are missing
        # It also uses setdefault for created_at/updated_at if they exist here
        return MemoryNote(id=data_id, content=document, **deserialized_metadata)
