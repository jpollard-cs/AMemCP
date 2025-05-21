#!/usr/bin/env python3
"""
Ensemble Retriever Implementation.
Combines results from multiple retrievers using weighted scoring.
"""

import asyncio
from typing import Any, Dict, List, Optional

from amem.core.interfaces import IManagesOwnIndex, IRetriever, SearchResult
from amem.utils.utils import setup_logger

logger = setup_logger(__name__)


class EnsembleRetriever(IRetriever, IManagesOwnIndex):
    """Combines results from multiple retrievers, manages sub-indexes if needed."""

    def __init__(self, retrievers: List[IRetriever], weights: List[float]):
        """Initialize ensemble retriever.

        Args:
            retrievers: List of retriever objects implementing IRetriever.
            weights: List of weights for each retriever (should sum to 1 ideally).
        """
        if len(retrievers) != len(weights):
            raise ValueError("Number of retrievers must match the number of weights.")
        if not retrievers:
            raise ValueError("At least one retriever must be provided.")
        # Optional: Check if weights sum close to 1
        if not abs(sum(weights) - 1.0) < 1e-6:
            logger.warning(
                f"Retriever weights sum to {sum(weights)}, not 1.0. Normalization might be needed depending on the fusion method."
            )

        self.retrievers = retrievers
        self.weights = weights
        logger.info(f"Initialized EnsembleRetriever with {len(retrievers)} retrievers.")

    # --- Interface Methods ---

    async def search(
        self,
        query: str,
        top_k: int,
        embedding: Optional[List[float]] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search using an ensemble of retrievers and combine results."""
        logger.debug(f"Performing ensemble search (top_k={top_k})")

        # Gather results from all retrievers concurrently
        # We ask each retriever for more results (e.g., top_k * 3) to get a good pool for fusion
        fetch_k = top_k * 3
        tasks = [retriever.search(query, fetch_k, embedding, metadata_filter) for retriever in self.retrievers]
        all_retriever_results: List[List[SearchResult]] = await asyncio.gather(*tasks, return_exceptions=True)

        # --- Reciprocal Rank Fusion (RRF) ---
        # RRF is generally preferred over simple weighted sum for combining ranks.
        # Score = sum( weight / (k + rank) ) for each document across lists.
        # k is a constant (e.g., 60) to mitigate the impact of high ranks.
        rrf_k = 60
        fused_scores: Dict[str, float] = {}
        doc_content_map: Dict[str, str] = {}  # Store content if available

        for i, results in enumerate(all_retriever_results):
            if isinstance(results, Exception):
                logger.error(f"Retriever {i} failed during ensemble search: {results}")
                continue  # Skip failed retriever
            if not isinstance(results, list):
                logger.error(f"Retriever {i} returned unexpected type: {type(results)}")
                continue

            weight = self.weights[i]
            for rank, result in enumerate(results):
                if not isinstance(result, SearchResult):
                    logger.warning(f"Retriever {i} returned non-SearchResult item: {result}")
                    continue

                doc_id = result.note_id
                # RRF score update
                score_update = weight / (rrf_k + rank + 1)  # rank is 0-based
                fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + score_update

                # Store content if we have it (useful if reranking later)
                if doc_id not in doc_content_map and result.content:
                    doc_content_map[doc_id] = result.content

        if not fused_scores:
            logger.warning("Ensemble search yielded no results after fusion.")
            return []

        # Sort documents by fused score
        sorted_doc_ids = sorted(fused_scores.keys(), key=lambda doc_id: fused_scores[doc_id], reverse=True)

        # Create final SearchResult list
        final_results = [
            SearchResult(
                note_id=doc_id,
                score=fused_scores[doc_id],
                content=doc_content_map.get(doc_id),  # Include content if available
            )
            for doc_id in sorted_doc_ids[:top_k]
        ]

        logger.info(f"Ensemble search fused {len(fused_scores)} candidates into {len(final_results)} results.")
        return final_results

    # --- Indexing Methods (Delegate to sub-retrievers that manage their own index) ---

    async def add_document(self, note_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add document to underlying retrievers that implement IManagesOwnIndex."""
        logger.debug(f"Processing add_document for {note_id} in ensemble.")
        tasks = []
        for retriever in self.retrievers:
            if isinstance(retriever, IManagesOwnIndex):
                logger.debug(f"Calling add_document on {type(retriever).__name__} for {note_id}.")
                tasks.append(retriever.add_document(note_id, content, metadata))
            # No else needed, skip if not IManagesOwnIndex
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # Log errors
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error adding document {note_id} to sub-retriever {i}: {result}")

    async def remove_document(self, note_id: str) -> None:
        """Remove document from underlying retrievers that implement IManagesOwnIndex."""
        logger.debug(f"Processing remove_document for {note_id} in ensemble.")
        tasks = []
        for retriever in self.retrievers:
            if isinstance(retriever, IManagesOwnIndex):
                logger.debug(f"Calling remove_document on {type(retriever).__name__} for {note_id}.")
                tasks.append(retriever.remove_document(note_id))
            # No else needed, skip if not IManagesOwnIndex
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error removing document {note_id} from sub-retriever {i}: {result}")
        else:
            logger.debug(f"No sub-retrievers requiring explicit removal for {note_id}.")

    async def rebuild_index(self, notes: List[Any]) -> None:
        """Rebuild index for underlying retrievers that implement IManagesOwnIndex."""
        logger.info(f"Processing rebuild_index for {len(self.retrievers)} ensemble retrievers.")
        tasks = []
        for retriever in self.retrievers:
            if isinstance(retriever, IManagesOwnIndex):
                logger.debug(f"Calling rebuild_index on {type(retriever).__name__}.")
                # Pass MemoryNote list if that's what the sub-retriever expects
                # This assumes notes is List[MemoryNote] as per IManagesOwnIndex spec
                tasks.append(retriever.rebuild_index(notes))
            # No else needed, skip if not IManagesOwnIndex
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error rebuilding index for sub-retriever {i}: {result}")
        else:
            logger.debug("No sub-retrievers requiring explicit index rebuild.")
