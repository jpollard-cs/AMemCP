import json
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import chromadb
import nltk
import numpy as np
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from utils import get_logger

# Configure logging for this module
logger = get_logger(__name__)


def simple_tokenize(text):
    try:
        return word_tokenize(text)
    except LookupError:
        # Fallback to a simple tokenizer if NLTK resources aren't available
        import re

        return re.findall(r"\b\w+\b", text.lower())


class SimpleEmbeddingRetriever:
    """Simple retriever using sentence embeddings"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = None

    def add_document(self, document: str):
        """Add a document to the retriever.

        Args:
            document: Text content to add
        """
        self.documents.append(document)
        # Update embeddings
        if len(self.documents) == 1:
            self.embeddings = self.model.encode([document])
        else:
            new_embedding = self.model.encode([document])
            self.embeddings = np.vstack([self.embeddings, new_embedding])

    def search(self, query: str, top_k: int = 5, task_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for similar documents.

        Args:
            query: Search query
            top_k: Number of results to return
            task_type: Optional task type for specialized embeddings (ignored in base implementation)

        Returns:
            List of dictionaries containing document content and similarity score
        """
        if not self.documents:
            return []

        # Get query embedding
        query_embedding = self.model.encode([query])

        # Calculate cosine similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]

        # Get top k results
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append({"content": self.documents[idx], "score": float(similarities[idx])})

        return results


class ChromaRetriever:
    """Vector database retrieval using ChromaDB with persistence and custom embeddings."""

    def __init__(
        self,
        collection_name: str = "memories",
        persist_directory: Optional[str] = None,
        embedding_function: Optional[Any] = None,
    ):
        """Initialize ChromaDB retriever.

        Args:
            collection_name: Name of the ChromaDB collection.
            persist_directory: Path to store/load the database. Defaults to None (in-memory).
            embedding_function: An embedding function/provider instance (like GeminiProvider).
                                  If None, Chroma uses its default.
        """
        if persist_directory:
            logger.info(f"Initializing ChromaDB client with persistence at: {persist_directory}")
            self.client = chromadb.PersistentClient(path=persist_directory, settings=Settings(allow_reset=True))
        else:
            logger.info("Initializing ChromaDB client in-memory.")
            self.client = chromadb.Client(Settings(allow_reset=True))

        # Use provided embedding function or Chroma's default
        if embedding_function:
            # If we pass the provider, we might need to wrap it or use a specific chroma interface
            # Option 1: Assume the provider has an `embed_documents` or similar method Chroma can use
            # chroma_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_function.embed_model) # Example if using SentenceTransformer directly

            # Option 2: If AgenticMemorySystem passes pre-computed embeddings,
            # set embedding_function=None here and handle embeddings in add_document.
            # This is generally safer when dealing with task-specific embeddings.
            chroma_ef = None
            self.external_embedding_function = (
                embedding_function  # Store for potential use? Unlikely needed if passing embeddings.
            )
            logger.info(f"ChromaDB configured to use externally computed embeddings.")
        else:
            chroma_ef = embedding_functions.DefaultEmbeddingFunction()
            self.external_embedding_function = None
            logger.warning(
                "ChromaDB using default embedding function. Custom LLM embeddings will not be used directly by Chroma."
            )

        logger.info(f"Getting or creating Chroma collection: {collection_name}")
        self.collection = self.client.get_or_create_collection(
            name=collection_name, embedding_function=chroma_ef  # Pass EF if using Chroma's embedding
        )

    def add_document(self, document: str, embedding: List[float], metadata: Dict, doc_id: str):
        """Add a document with pre-computed embedding to ChromaDB.

        Args:
            document: Text content to add.
            embedding: The pre-computed embedding vector for the document.
            metadata: Dictionary of metadata.
            doc_id: Unique identifier for the document.
        """
        # Metadata processing remains the same
        processed_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, list):
                # Ensure list items are strings for join
                processed_metadata[key] = ", ".join(map(str, value))
            elif not isinstance(value, (str, int, float, bool)):
                # Convert other types to string representation
                processed_metadata[key] = str(value)
            else:
                processed_metadata[key] = value

        try:
            self.collection.add(
                embeddings=[embedding],  # Pass the pre-computed embedding
                documents=[document],
                metadatas=[processed_metadata],
                ids=[doc_id],
            )
        except Exception as e:
            logger.error(f"Error adding document {doc_id} to Chroma: {e}", exc_info=True)
            # Potentially re-raise or handle error
            raise

    def delete_document(self, doc_id: str):
        """Delete a document from ChromaDB.

        Args:
            doc_id: ID of document to delete
        """
        try:
            self.collection.delete(ids=[doc_id])
        except Exception as e:
            logger.error(f"Error deleting document {doc_id} from Chroma: {e}", exc_info=True)
            # Potentially re-raise or handle error
            raise

    def search(self, query: str, k: int = 5, task_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for similar documents using externally computed query embedding.

        Args:
            query: Query text.
            k: Number of results to return.
            task_type: Optional task type for generating the query embedding.

        Returns:
            List of dicts containing document ID, content, metadata, and distance.
        """
        if not self.external_embedding_function:
            logger.error("Cannot perform search: ChromaRetriever not initialized with an external embedding function.")
            # Fallback to potentially less accurate text search? Or return empty?
            # results = self.collection.query(query_texts=[query], n_results=k)
            return []

        try:
            # Generate query embedding using the external provider and task_type
            query_embedding = self.external_embedding_function.get_embeddings(query, task_type=task_type)

            results = self.collection.query(
                query_embeddings=[query_embedding],  # Use the computed embedding
                n_results=k,
                include=["metadatas", "documents", "distances"],  # Include distances
            )
        except Exception as e:
            logger.error(f"Error during Chroma search for query '{query}': {e}", exc_info=True)
            return []  # Return empty list on error

        # Process results into a standardized format
        output_results = []
        if results and results.get("ids") and results["ids"][0]:
            for doc_id, doc, meta, dist in zip(
                results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]
            ):
                # Convert string metadata back to lists where appropriate (example)
                for key in ["keywords", "tags", "related_notes"]:
                    if key in meta and isinstance(meta.get(key), str):
                        meta[key] = [item.strip() for item in meta[key].split(",") if item.strip()]
                output_results.append(
                    {
                        "id": doc_id,
                        "content": doc,
                        "metadata": meta,
                        "score": 1.0 - dist,  # Convert distance to similarity score (0=identical, 1=distant)
                    }
                )

        return output_results


class BM25Retriever:
    """Sparse retrieval using BM25 algorithm"""

    def __init__(self):
        """Initialize BM25 retriever"""
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)

        self.documents = []
        self.doc_ids = []
        self.corpus = []
        self.bm25 = None

    def add_document(self, document: str, doc_id: Optional[str] = None):
        """Add a document to the retriever.

        Args:
            document: Text content to add
            doc_id: Optional document ID (defaults to index)
        """
        self.documents.append(document)
        self.doc_ids.append(doc_id if doc_id is not None else str(len(self.documents) - 1))

        # Tokenize document
        tokens = simple_tokenize(document.lower())
        self.corpus.append(tokens)

        # Reinitialize BM25
        self.bm25 = BM25Okapi(self.corpus)

    def search(self, query: str, top_k: int = 5, task_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for relevant documents using BM25.

        Args:
            query: Search query
            top_k: Number of results to return
            task_type: Optional task type for specialized embeddings (ignored in BM25)

        Returns:
            List of dictionaries containing document content and score
        """
        if not self.documents:
            return []

        # Tokenize query
        tokens = simple_tokenize(query.lower())

        # Get BM25 scores
        scores = self.bm25.get_scores(tokens)

        # Get top k results
        top_indices = np.argsort(scores)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include results with non-zero scores
                results.append({"id": self.doc_ids[idx], "content": self.documents[idx], "score": float(scores[idx])})

        return results


class EnsembleRetriever:
    """Ensemble retriever that combines results from multiple retrievers"""

    def __init__(self, retrievers: List[Any], weights: Optional[List[float]] = None):
        """Initialize ensemble retriever.

        Args:
            retrievers: List of retriever objects
            weights: Optional weights for each retriever (normalized internally)
        """
        self.retrievers = retrievers

        # Normalize weights if provided
        if weights:
            total = sum(weights)
            self.weights = [w / total for w in weights]
        else:
            # Equal weights by default
            self.weights = [1.0 / len(retrievers) for _ in retrievers]

    def add_document(self, document: str, metadata: Optional[Dict] = None, doc_id: Optional[str] = None):
        """Add a document to all retrievers.

        Args:
            document: Text content to add
            metadata: Optional metadata for the document
            doc_id: Optional document ID
        """
        for retriever in self.retrievers:
            try:
                # Different retrievers have different add_document signatures
                if hasattr(retriever, "add_document"):
                    if "metadata" in retriever.add_document.__code__.co_varnames:
                        retriever.add_document(document, metadata or {}, doc_id or str(uuid.uuid4()))
                    else:
                        retriever.add_document(document, doc_id or str(uuid.uuid4()))
            except Exception as e:
                print(f"Error adding document to retriever {type(retriever).__name__}: {e}")

    def search(self, query: str, top_k: int = 5, task_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search across all retrievers and combine results.

        Args:
            query: Search query
            top_k: Number of results to return
            task_type: Optional task type for specialized embeddings

        Returns:
            List of dictionaries containing combined search results
        """
        combined_results = []
        for i, retriever in enumerate(self.retrievers):
            try:
                # Pass task_type to retriever if supported
                if "task_type" in retriever.search.__code__.co_varnames:
                    results = retriever.search(
                        query, top_k * 2, task_type=task_type
                    )  # Get more results for better reranking
                else:
                    results = retriever.search(query, top_k * 2)  # Get more results for better reranking

                # Normalize and format results from different retrievers
                if isinstance(results, dict) and "ids" in results:  # ChromaDB format
                    for j, (doc_id, distance) in enumerate(zip(results["ids"][0], results["distances"][0])):
                        document = results["documents"][0][j] if results["documents"] else "Unknown"
                        combined_results.append(
                            {
                                "id": doc_id,
                                "content": document,
                                "score": distance * self.weights[i],
                                "retriever": type(retriever).__name__,
                            }
                        )
                else:  # Standardized retriever format
                    for result in results:
                        result_with_weight = result.copy()
                        result_with_weight["score"] = result.get("score", 0.5) * self.weights[i]
                        result_with_weight["retriever"] = type(retriever).__name__
                        combined_results.append(result_with_weight)
            except Exception as e:
                print(f"Error searching with retriever {type(retriever).__name__}: {e}")

        # Deduplicate results based on content
        deduplicated = {}
        for result in combined_results:
            content = result.get("content", "")
            if content:
                if content not in deduplicated or result["score"] > deduplicated[content]["score"]:
                    deduplicated[content] = result

        # Sort by score and take top_k
        final_results = sorted(deduplicated.values(), key=lambda x: x.get("score", 0), reverse=True)[:top_k]

        return final_results


class BaseRetriever(ABC):
    """Base class for all retrievers."""

    @abstractmethod
    async def search(
        self, query: str, top_k: int = 5, metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for documents similar to the query.
        Subclasses MUST implement this method.
        """
        logger.debug(f"BaseRetriever searching query: '{query[:50]}...'")
        raise NotImplementedError("Subclasses must implement the search method.")

    @abstractmethod
    async def get_all(self) -> Dict[str, Any]:
        """Retrieve all documents.
        Subclasses MUST implement this method.
        """
        raise NotImplementedError("Subclasses must implement the get_all method.")

    @abstractmethod
    async def add_document(
        self, document: str, doc_id: str, metadata: Dict[str, Any] = None, embedding: List[float] = None
    ):
        """Add a document.
        Subclasses MUST implement this method.
        """
        raise NotImplementedError("Subclasses must implement the add_document method.")

    @abstractmethod
    async def delete_document(self, doc_id: str):
        """Delete a document.
        Subclasses MUST implement this method.
        """
        raise NotImplementedError("Subclasses must implement the delete_document method.")

    @abstractmethod
    async def initialize(self):
        """Initialize the retriever.
        Subclasses MUST implement this method.
        """
        raise NotImplementedError("Subclasses must implement the initialize method.")


class AsyncChromaRetriever(BaseRetriever):
    async def search(self, query: str, top_k: int = 5, embedding: List[float] = None):
        """Search for documents in ChromaDB.

        Args:
            query: Search query text
            top_k: Maximum number of results
            embedding: Optional pre-computed query embedding

        Returns:
            List of search results with id, content, and metadata
        """
        results = None
        try:
            if embedding is not None:
                # Use pre-computed embedding
                logger.debug(f"Searching Chroma with pre-computed embedding, top_k={top_k}")
                results = await self.collection.query(
                    query_embeddings=[embedding], n_results=top_k, include=["documents", "metadatas", "distances"]
                )
            else:
                # Let ChromaDB compute embedding
                logger.debug(f"Searching Chroma with query text: '{query[:50]}...', top_k={top_k}")
                results = await self.collection.query(
                    query_texts=[query], n_results=top_k, include=["documents", "metadatas", "distances"]
                )

            formatted_results = []
            if results and results.get("ids") and results["ids"][0]:
                ids = results.get("ids", [[]])[0]
                docs = results.get("documents", [[]])[0]
                metadatas = results.get("metadatas", [[]])[0]
                distances = results.get("distances", [[]])[0]
                max_len = len(ids)

                docs.extend([""] * (max_len - len(docs)))
                metadatas.extend([{}] * (max_len - len(metadatas)))
                distances.extend([None] * (max_len - len(distances)))

                for i, doc_id in enumerate(ids):
                    doc_metadata = metadatas[i]
                    # Keep deserialization for reading data back
                    if doc_metadata:
                        for key in ["keywords", "tags", "related_notes"]:
                            if key in doc_metadata and isinstance(doc_metadata[key], str):
                                try:
                                    parsed_list = json.loads(doc_metadata[key])
                                    if isinstance(parsed_list, list):
                                        doc_metadata[key] = parsed_list
                                except json.JSONDecodeError:
                                    logger.warning(f"Could not JSON parse metadata key '{key}' for doc {doc_id}.")

                    formatted_results.append(
                        {"id": doc_id, "content": docs[i], "metadata": doc_metadata, "distance": distances[i]}
                    )

            return formatted_results
        except Exception as e:
            logger.error(f"❌ Error searching ChromaDB: {e}", exc_info=True)
            raise

    async def add_document(
        self, document: str, doc_id: str, metadata: Dict[str, Any] = None, embedding: List[float] = None
    ):
        """Add a document to ChromaDB. Assumes metadata values are compatible (str, int, float, bool)."""
        # Remove serialization logic - assume metadata is pre-processed
        # processed_metadata = {}
        # if metadata: ... (removed loop) ...
        # final_metadata = processed_metadata if processed_metadata else None
        final_metadata = metadata  # Use metadata directly as passed in

        try:
            logger.debug(f"Attempting upsert for ID: {doc_id} with final_metadata: {final_metadata}")

            await self.collection.upsert(
                ids=[doc_id],
                embeddings=[embedding] if embedding else None,
                documents=[document] if document else None,
                metadatas=[final_metadata] if final_metadata else None,  # Pass directly
            )
            logger.debug(f"📝 Added/Updated document {doc_id} in ChromaDB")
        except Exception as e:
            logger.error(f"❌ Error during ChromaDB upsert for {doc_id}: {e}", exc_info=True)
            # Re-raise the specific exception to allow AgenticMemorySystem to catch it
            raise ValueError(f"Failed to add document {doc_id} to Chroma: {e}") from e
