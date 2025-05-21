#!/usr/bin/env python3
"""
Simple in-memory cache implementation.
"""

from typing import Dict, List, Optional

from amem.core.interfaces import IMemoryCache
from amem.core.models.memory_note import MemoryNote


class InMemoryCache(IMemoryCache):
    """A basic in-memory cache using a dictionary."""

    def __init__(self):
        self._cache: Dict[str, MemoryNote] = {}
        # Optional: Add a lock for potential future multi-threaded access if needed,
        # but for pure async, it might not be strictly necessary if operations
        # on the dict are atomic enough for the use case.
        # self._lock = asyncio.Lock()

    async def get(self, note_id: str) -> Optional[MemoryNote]:
        """Get a memory note from the cache by ID."""
        # async with self._lock:
        return self._cache.get(note_id)

    async def put(self, note: MemoryNote) -> None:
        """Add or update a memory note in the cache."""
        # async with self._lock:
        if note and note.id:
            self._cache[note.id] = note

    async def delete(self, note_id: str) -> None:
        """Remove a memory note from the cache by ID."""
        # async with self._lock:
        if note_id in self._cache:
            del self._cache[note_id]

    async def get_all_ids(self) -> List[str]:
        """Get all note IDs currently in the cache."""
        # async with self._lock:
        return list(self._cache.keys())

    async def get_all_notes(self) -> List[MemoryNote]:
        """Get all MemoryNote objects currently in the cache."""
        # async with self._lock:
        return list(self._cache.values())

    async def clear(self) -> None:
        """Clear the entire cache."""
        # async with self._lock:
        self._cache.clear()

    def __len__(self) -> int:
        """Return the number of items in the cache."""
        return len(self._cache)
