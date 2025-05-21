#!/usr/bin/env python3
"""
BM25 Keyword Retriever Implementation.
"""

import asyncio
import threading
from typing import Any, Dict, List, Optional

from rank_bm25 import BM25Okapi

from amem.core.interfaces import IKeywordRetriever, IManagesOwnIndex, IRetriever, SearchResult
from amem.core.models.memory_note import MemoryNote

# from amem.utils.nlp import simple_preprocess
from amem.utils.utils import setup_logger

logger = setup_logger(__name__)


class BM25Retriever(IKeywordRetriever, IRetriever, IManagesOwnIndex):
    """Keyword-based retriever using BM25Okapi that manages its own index."""

    def __init__(self):
        self._bm25: Optional[BM25Okapi] = None
        self._corpus_tokens: List[List[str]] = []
        self._doc_id_map: Dict[str, int] = {}  # Map doc ID to index in corpus
        self._doc_ids: List[str] = []  # List of doc IDs in order
        # Use a lock for thread-safe index modifications, as BM25Okapi is not inherently thread-safe
        self._lock = threading.Lock()
        logger.info("Initialized BM25Retriever.")

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenizer."""
        # Consider using a more sophisticated tokenizer (e.g., spaCy, NLTK) for better results
        return text.lower().split()

    # --- Interface Methods ---

    async def search(
        self,
        query: str,
        top_k: int,
        embedding: Optional[List[float]] = None,  # Not used by BM25
        metadata_filter: Optional[Dict[str, Any]] = None,  # Not used by BM25
    ) -> List[SearchResult]:
        """Search for documents using BM25."""
        if not self._bm25:
            logger.warning("BM25 index is not built. Returning empty results.")
            return []
        if not self._corpus_tokens:
            logger.warning("BM25 corpus is empty. Returning empty results.")
            return []

        logger.debug(f"Performing BM25 search (top_k={top_k})")
        tokenized_query = self._tokenize(query)

        try:
            # BM25 scoring is CPU-bound, run in thread executor for async compatibility
            loop = asyncio.get_running_loop()
            with self._lock:  # Ensure read access is safe during potential index rebuilds
                if not self._bm25:
                    return []  # Check again inside lock
                scores = await loop.run_in_executor(
                    None, self._bm25.get_scores, tokenized_query  # Default thread pool executor
                )

            # Combine scores with doc IDs
            # Ensure doc_ids list length matches scores length
            num_docs = len(self._doc_ids)
            if len(scores) != num_docs:
                logger.error(f"BM25 score count ({len(scores)}) mismatch with document count ({num_docs}).")
                # Attempt recovery if possible, otherwise return empty
                # This state usually indicates an issue during index add/remove/rebuild
                return []

            id_score_pairs = list(zip(self._doc_ids, scores))

            # Sort by score descending
            id_score_pairs.sort(key=lambda x: x[1], reverse=True)

            # Format as SearchResult
            results = [SearchResult(note_id=pair[0], score=pair[1]) for pair in id_score_pairs[:top_k]]
            logger.info(f"BM25 search found {len(results)} results.")
            return results

        except Exception as e:
            logger.error(f"Error during BM25 search: {e}", exc_info=True)
            return []

    async def add_document(self, note_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a single document to the BM25 index."""
        logger.debug(f"Adding document {note_id} to BM25 index.")
        tokenized_doc = self._tokenize(content)

        # BM25 build/update is CPU-bound, run in executor
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._add_document_sync, note_id, tokenized_doc)

    def _add_document_sync(self, note_id: str, tokenized_doc: List[str]):
        """Synchronous part of adding a document (thread-safe)."""
        with self._lock:
            if note_id in self._doc_id_map:
                # Document already exists, update it (remove old, add new)
                logger.warning(f"Document {note_id} already exists in BM25 index. Updating.")
                self._remove_document_sync(note_id)

            # Add new document
            idx = len(self._corpus_tokens)
            self._corpus_tokens.append(tokenized_doc)
            self._doc_ids.append(note_id)
            self._doc_id_map[note_id] = idx

            # Rebuild the index (inefficient for single adds, but simple)
            # Consider batch updates or incremental indexing if performance is critical
            if self._corpus_tokens:
                self._bm25 = BM25Okapi(self._corpus_tokens)
                logger.info(f"Rebuilt BM25 index after adding {note_id}.")
            else:
                self._bm25 = None  # Handle case where corpus becomes empty

    async def remove_document(self, note_id: str) -> None:
        """Remove a document from the BM25 index."""
        logger.debug(f"Removing document {note_id} from BM25 index.")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._remove_document_sync, note_id)

    def _remove_document_sync(self, note_id: str):
        """Synchronous part of removing a document (thread-safe)."""
        with self._lock:
            if note_id not in self._doc_id_map:
                logger.warning(f"Document {note_id} not found in BM25 index for removal.")
                return

            idx_to_remove = self._doc_id_map[note_id]

            # Remove from lists and map
            del self._corpus_tokens[idx_to_remove]
            del self._doc_ids[idx_to_remove]
            del self._doc_id_map[note_id]

            # Update indices in the map for subsequent elements
            for i in range(idx_to_remove, len(self._doc_ids)):
                current_id = self._doc_ids[i]
                self._doc_id_map[current_id] = i

            # Rebuild the index
            if self._corpus_tokens:
                self._bm25 = BM25Okapi(self._corpus_tokens)
                logger.info(f"Rebuilt BM25 index after removing {note_id}.")
            else:
                self._bm25 = None
                logger.info("BM25 index is now empty.")

    async def rebuild_index(self, notes: List[MemoryNote]) -> None:
        """Rebuild the entire BM25 index from a list of MemoryNote objects."""
        logger.info(f"Rebuilding BM25 index with {len(notes)} notes.")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._rebuild_index_sync, notes)

    def _rebuild_index_sync(self, notes: List[MemoryNote]):
        """Synchronous part of rebuilding the index (thread-safe)."""
        with self._lock:
            self._corpus_tokens = [self._tokenize(note.content) for note in notes]
            self._doc_ids = [note.id for note in notes]
            self._doc_id_map = {note.id: i for i, note in enumerate(notes)}

            if self._corpus_tokens:
                self._bm25 = BM25Okapi(self._corpus_tokens)
                logger.info(f"Successfully rebuilt BM25 index with {len(self._corpus_tokens)} documents.")
            else:
                self._bm25 = None
                logger.info("Rebuilt BM25 index: index is empty.")
