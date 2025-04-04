#!/usr/bin/env python3
"""
Mock memory system implementation for testing the MCP server.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MemoryNote:
    """A single memory note in the memory system."""

    def __init__(self, content: str, id: Optional[str] = None, **metadata):
        """Initialize a memory note.

        Args:
            content: The content of the note
            id: Optional ID for the note (generated if not provided)
            **metadata: Additional metadata for the note
        """
        self.id = id or str(uuid.uuid4())
        self.content = content
        self.metadata = metadata or {}
        self.created_at = self.metadata.get("created_at", datetime.now().isoformat())
        self.updated_at = self.metadata.get("updated_at", datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert the note to a dictionary.

        Returns:
            Dictionary representation of the note
        """
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class AgenticMemorySystem:
    """A mock memory system for testing."""

    def __init__(self, project_name: str = "default", **kwargs):
        """Initialize the memory system.

        Args:
            project_name: Name of the project
            **kwargs: Additional configuration options
        """
        self.project_name = project_name
        self.memories = {}
        logger.info(f"Initialized mock memory system for project: {project_name}")
        # Log kwargs for debugging
        for key, value in kwargs.items():
            logger.info(f"  {key}: {value}")

    def create(self, content: str, **metadata) -> MemoryNote:
        """Create a new memory note.

        Args:
            content: Content of the note
            **metadata: Additional metadata

        Returns:
            The created memory note
        """
        # The mock system doesn't generate metadata automatically
        # It just stores what's passed in
        note = MemoryNote(content, **metadata)
        self.memories[note.id] = note
        logger.info(f"Created memory: {note.id}")
        return note

    def get(self, note_id: str) -> MemoryNote:
        """Get a memory note by ID.

        Args:
            note_id: ID of the note to retrieve

        Returns:
            The memory note

        Raises:
            ValueError: If the note doesn't exist
        """
        if note_id not in self.memories:
            raise ValueError(f"Memory not found: {note_id}")
        logger.info(f"Retrieved memory: {note_id}")
        return self.memories[note_id]

    def update(self, note_id: str, content: str, **metadata) -> MemoryNote:
        """Update a memory note.

        Args:
            note_id: ID of the note to update
            content: New content for the note
            **metadata: Additional metadata to update

        Returns:
            The updated memory note

        Raises:
            ValueError: If the note doesn't exist
        """
        if note_id not in self.memories:
            raise ValueError(f"Memory not found: {note_id}")

        note = self.memories[note_id]
        note.content = content
        if metadata:
            note.metadata.update(metadata)
        note.updated_at = datetime.now().isoformat()
        logger.info(f"Updated memory: {note_id}")
        return note

    def delete(self, note_id: str) -> bool:
        """Delete a memory note.

        Args:
            note_id: ID of the note to delete

        Returns:
            True if the note was deleted, False otherwise

        Raises:
            ValueError: If the note doesn't exist
        """
        if note_id not in self.memories:
            raise ValueError(f"Memory not found: {note_id}")

        del self.memories[note_id]
        logger.info(f"Deleted memory: {note_id}")
        return True

    def search(self, query: str, k: int = 5) -> List[MemoryNote]:
        """Search for memory notes (simple substring matching).

        Args:
            query: Search query
            k: Maximum number of results

        Returns:
            List of matching memory notes
        """
        results = []
        for memory in self.memories.values():
            if query.lower() in memory.content.lower():
                results.append(memory)

        logger.info(f"Mock Search for '{query}' found {len(results)} results")
        return results[:k]
