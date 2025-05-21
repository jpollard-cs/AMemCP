#!/usr/bin/env python3
"""
Core domain model: MemoryNote.
Represents a single unit of memory with content and metadata.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, Optional


class MemoryNote:
    """A single memory note, focused on state and behavior, not persistence.

    Attributes:
        id (str): Unique identifier.
        content (str): The main text content of the memory.
        created_at (str): ISO 8601 timestamp of creation.
        updated_at (str): ISO 8601 timestamp of last update.
        metadata (Dict[str, Any]): Dictionary for storing various metadata fields.
            Common fields are set with defaults on initialization.
    """

    def __init__(self, content: str, id: Optional[str] = None, **metadata):
        """Initialize a memory note.

        Args:
            content: The content of the note.
            id: Optional ID for the note (generated if not provided).
            **metadata: Additional metadata. Timestamps and common fields will be
            populated with defaults if not provided here.
        """
        self.id = id or str(uuid.uuid4())
        self.content = content
        # Use provided metadata, ensuring it's a dict
        self.metadata = metadata or {}

        # Ensure timestamps exist, using provided or generating new
        now_iso = datetime.now().isoformat()
        self.created_at = self.metadata.setdefault("created_at", now_iso)
        self.updated_at = self.metadata.setdefault("updated_at", now_iso)

        # Ensure common metadata fields exist with defaults
        # These provide structure but can be extended via the metadata dict
        self.metadata.setdefault("keywords", [])
        self.metadata.setdefault("summary", "")
        self.metadata.setdefault("context", "General")
        self.metadata.setdefault("type", "general")
        self.metadata.setdefault("related_notes", [])
        self.metadata.setdefault("sentiment", "neutral")
        self.metadata.setdefault("importance", 0.5)
        self.metadata.setdefault("tags", [])
        # Note: Name is handled purely within metadata if desired
        # self.metadata.setdefault("name", None)

    def update_content(self, new_content: str):
        """Updates the content and the updated_at timestamp."""
        if self.content != new_content:
            self.content = new_content
            self._touch()

    def update_metadata(self, updates: Dict[str, Any], overwrite_all: bool = False):
        """Update metadata fields and the updated_at timestamp.

        Args:
            updates: Dictionary of metadata fields to add or update.
            overwrite_all: If True, replaces the entire metadata dict (excluding timestamps)
                           with the updates. Default is False (merge).
        """
        if overwrite_all:
            # Preserve essential timestamps even when overwriting
            created = self.metadata.get("created_at", self.created_at)
            self.metadata = updates
            self.metadata["created_at"] = created  # Restore created_at
        else:
            self.metadata.update(updates)

        self._touch()  # Update updated_at timestamp

    def add_tag(self, tag: str):
        """Adds a tag to the metadata tags list if not already present."""
        if "tags" not in self.metadata or not isinstance(self.metadata["tags"], list):
            self.metadata["tags"] = []
        if tag not in self.metadata["tags"]:
            self.metadata["tags"].append(tag)
            self._touch()

    def remove_tag(self, tag: str):
        """Removes a tag from the metadata tags list."""
        if "tags" in self.metadata and isinstance(self.metadata["tags"], list) and tag in self.metadata["tags"]:
            self.metadata["tags"].remove(tag)
            self._touch()

    def _touch(self):
        """Updates the updated_at timestamp in both the attribute and metadata dict."""
        self.updated_at = datetime.now().isoformat()
        self.metadata["updated_at"] = self.updated_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert the note to a dictionary for general serialization (e.g., API)."""
        # Includes core attributes and a copy of metadata for safety
        return {
            "id": self.id,
            "content": self.content,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata.copy(),
        }

    def __repr__(self) -> str:
        content_preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        note_type = self.metadata.get("type", "N/A")
        return f"MemoryNote(id={self.id}, type={note_type}, content='{content_preview}')"

    def __eq__(self, other: Any) -> bool:
        """Equality check based on ID."""
        if not isinstance(other, MemoryNote):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        """Hash based on ID."""
        return hash(self.id)
