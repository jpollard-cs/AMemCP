#!/usr/bin/env python3
"""
ChromaDB Vector Retriever Implementation.
"""

import json
from typing import Any, Dict, List, Optional

from chromadb import Collection, QueryResult

from amem.core.interfaces import IVectorRetriever, SearchResult
from amem.utils.utils import setup_logger

logger = setup_logger(__name__)


class ChromaVectorRetriever(IVectorRetriever):
    """Vector retrieval implementation using an existing ChromaDB Collection object."""

    # Note: This retriever *only* does search. It relies on an IMemoryStore
    # (like ChromaStore) to provide the initialized ChromaDB collection.

    def __init__(
        self,
        collection: Collection,  # Accept initialized collection
    ):
        self.collection: Collection = collection
        # No client needed here, collection object handles communication
        # No host/port needed
        self._initialized = True  # Considered initialized once collection is provided
        logger.info(f"ChromaVectorRetriever initialized with collection: {collection.name}")

    async def search(
        self,
        query: str,  # Query text might still be useful for logging or context
        top_k: int,
        embedding: Optional[List[float]] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search for documents in ChromaDB using embeddings."""
        # Check collection directly, _initialized is always True if constructor succeeded
        if not self.collection:
            # This case should technically not happen if constructor validation works
            raise ConnectionError("ChromaVectorRetriever has no valid collection.")
        if not embedding:
            logger.warning("Vector search called without providing an embedding.")
            return []

        logger.debug(f"Performing vector search in '{self.collection.name}' (top_k={top_k})")

        # Helper to process metadata filter for ChromaDB (needs string values for lists etc.)
        def _process_filter(filt: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
            if not filt:
                return None
            processed = {}
            for k, v in filt.items():
                if isinstance(v, list):
                    processed[k] = json.dumps(v)
                elif v is None:
                    processed[k] = ""  # Assume None stored as empty string
                elif not isinstance(v, (str, int, float, bool)):
                    processed[k] = str(v)
                else:
                    processed[k] = v
            return processed

        processed_filter = _process_filter(metadata_filter)

        try:
            # Perform vector query using the provided embedding
            results: QueryResult = await self.collection.query(
                query_embeddings=[embedding],
                n_results=top_k,
                where=processed_filter,  # Apply processed metadata filter
                include=["metadatas", "distances", "documents"],  # Include documents for context/reranking
            )

            formatted_results = []
            if results and results.get("ids") and results["ids"][0]:
                logger.info(f"Vector search retrieved {len(results['ids'][0])} candidates.")
                # Chroma returns lists of lists, even for single query
                ids = results["ids"][0]
                distances = results["distances"][0] if results.get("distances") else [0.0] * len(ids)
                results["metadatas"][0] if results.get("metadatas") else [{}] * len(ids)
                documents = results["documents"][0] if results.get("documents") else [""] * len(ids)

                for i, doc_id in enumerate(ids):
                    # Convert distance to score (higher is better, assuming cosine distance 0..2)
                    # Normalize distance (0..2) -> score (1..0)
                    score = 1.0 - (min(max(distances[i], 0.0), 2.0) / 2.0)

                    # Deserialize metadata (optional, could be done later)
                    # raw_metadata = metadatas[i]
                    # deserialized_metadata = self._deserialize_metadata_from_chroma(raw_metadata)

                    formatted_results.append(
                        SearchResult(
                            note_id=doc_id,
                            score=score,
                            content=documents[i],  # Pass content for potential reranking
                            # metadata=deserialized_metadata # Pass raw or deserialized
                        )
                    )
            else:
                logger.info("Vector search returned no results.")

            # Already sorted by distance/score by ChromaDB
            return formatted_results

        except Exception as e:
            logger.error(f"Error during ChromaDB vector search: {e}", exc_info=True)
            return []  # Return empty list on error

    # --- Mutation methods removed ---
    # add_document, remove_document, rebuild_index are not part of IRetriever
    # and not applicable to this class as indexing is handled by the Store.

    # --- Helper for deserializing metadata (if needed here) ---
    # def _deserialize_metadata_from_chroma(self, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    #     if not metadata:
    #         return {}
    #     deserialized = {}
    #     for key, value in metadata.items():
    #         if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
    #             try:
    #                 deserialized[key] = json.loads(value)
    #             except json.JSONDecodeError:
    #                 deserialized[key] = value
    #         else:
    #             deserialized[key] = value
    #     return deserialized
