#!/usr/bin/env python3
"""
Interfaces (Protocols) for the Agentic Memory System Components.
"""

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from amem.core.models.memory_note import MemoryNote  # Assuming MemoryNote is moved


@runtime_checkable
class IMemoryCache(Protocol):
    """Interface for in-memory caching of MemoryNote objects."""

    async def get(self, note_id: str) -> Optional[MemoryNote]:
        """Get a memory note from the cache by ID."""
        ...

    async def put(self, note: MemoryNote) -> None:
        """Add or update a memory note in the cache."""
        ...

    async def delete(self, note_id: str) -> None:
        """Remove a memory note from the cache by ID."""
        ...

    async def get_all_ids(self) -> List[str]:
        """Get all note IDs currently in the cache."""
        ...

    async def get_all_notes(self) -> List[MemoryNote]:
        """Get all MemoryNote objects currently in the cache."""
        ...

    async def clear(self) -> None:
        """Clear the entire cache."""
        ...


@runtime_checkable
class IEmbeddingProvider(Protocol):
    """Interface for generating embeddings for text."""

    async def get_embeddings(self, text: str, task_type: Optional[str] = None) -> List[float]:
        """Generate embeddings for the given text."""
        ...


@runtime_checkable
class ICompletionProvider(Protocol):
    """Interface for generating completions for text."""

    async def get_completion(self, prompt: str, config: Optional[Dict] = None, **kwargs) -> str:
        """Get completion from the LLM (async)"""
        ...


@runtime_checkable
class IContentAnalyzer(Protocol):
    """Interface for analyzing content properties."""

    async def analyze_content(self, content: str) -> Dict[str, Any]:
        """Analyze content to determine type, keywords, summary, sentiment, etc."""
        ...

    async def segment_content(self, content: str) -> List[Dict[str, Any]]:
        """Segment mixed content into distinct parts."""
        ...


@runtime_checkable
class IMemoryStore(Protocol):
    """Interface for persistent storage of MemoryNote objects."""

    async def initialize(self) -> None:
        """Initialize the store connection and resources."""
        ...

    async def shutdown(self) -> None:
        """Clean up store resources and connections."""
        ...

    async def add(self, note: MemoryNote, embedding: Optional[List[float]] = None) -> None:
        """Add a new memory note to the store."""
        ...

    async def update(self, note: MemoryNote, embedding: Optional[List[float]] = None) -> None:
        """Update an existing memory note in the store."""
        ...

    async def delete(self, note_id: str) -> bool:
        """Delete a memory note from the store by ID."""
        ...

    async def get_by_id(self, note_id: str) -> Optional[MemoryNote]:
        """Retrieve a memory note from the store by ID."""
        ...

    async def get_all(self) -> List[MemoryNote]:
        """Retrieve all memory notes from the store."""
        ...


# Search Result structure (could be refined)
class SearchResult:
    def __init__(self, note_id: str, score: float, content: Optional[str] = None, metadata: Optional[Dict] = None):
        self.note_id = note_id
        self.score = score
        self.content = content  # Optional, might be fetched later
        self.metadata = metadata  # Optional

    def __repr__(self):
        return f"SearchResult(note_id='{self.note_id}', score={self.score:.4f})"


@runtime_checkable
class IRetriever(Protocol):
    """Interface for retrieving memory notes based on a query."""

    async def search(
        self,
        query: str,
        top_k: int,
        embedding: Optional[List[float]] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search for relevant memory notes."""
        ...


@runtime_checkable
class IManagesOwnIndex(Protocol):  # Marker interface
    """Marker interface for retrievers that manage their own index
    and require explicit add/remove/rebuild calls.
    Implementers MUST provide add_document, remove_document, rebuild_index methods.
    """

    async def add_document(self, note_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a document to the retriever's index."""
        ...

    async def remove_document(self, note_id: str) -> None:
        """Remove a document from the retriever's index."""
        ...

    async def rebuild_index(self, notes: List[MemoryNote]) -> None:
        """Rebuild the retriever's index from a list of notes."""
        ...


@runtime_checkable
class IReranker(Protocol):
    """Interface for reranking search results."""

    async def rerank(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Rerank the provided search results based on the query."""
        ...


# --- Specific Retriever Types (Optional but can be useful) ---


@runtime_checkable
class IVectorRetriever(IRetriever, Protocol):
    """Interface specifically for vector-based retrieval."""

    # Inherits search from IRetriever, relies on embedding parameter


@runtime_checkable
class IKeywordRetriever(IRetriever, Protocol):
    """Interface specifically for keyword-based retrieval (e.g., BM25)."""

    # Inherits search from IRetriever, may not use embedding parameter
