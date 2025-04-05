#!/usr/bin/env python3
"""
Agentic Memory System implementation with LLM integration, retrievers, and reranking.
Uses async I/O throughout for optimal performance.
"""

import asyncio
import json
import os
import subprocess
import threading
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sentence_transformers.cross_encoder import CrossEncoder

from amem.core.llm_content_analyzer import LLMContentAnalyzer
from amem.core.llm_controller import LLMController
from amem.utils.utils import (
    DEFAULT_COLLECTION_PREFIX,
    DEFAULT_GEMINI_EMBED_MODEL,
    DEFAULT_GEMINI_LLM_MODEL,
    DEFAULT_JINA_RERANKER_MODEL,
    DEFAULT_PERSIST_DIR,
    get_chroma_settings,
    setup_logger,
)

# Import async ChromaDB client

# Set up logger
logger = setup_logger(__name__)

# Global variable to track ChromaDB server process
_chroma_server_process = None
_chroma_server_started = False


def start_chroma_server(persist_directory: str = None, port: int = 8001):
    """Start a ChromaDB server in the background using the CLI.

    Args:
        persist_directory: Directory to persist data
        port: Port for the ChromaDB server (default: 8001)
    """
    global _chroma_server_process, _chroma_server_started

    if _chroma_server_started:
        logger.info("ChromaDB server already started, skipping")
        return

    def _run_server():
        global _chroma_server_process, _chroma_server_started
        logger.info(f"Attempting to start ChromaDB server via CLI on port {port}")

        # Base command
        cmd = ["chroma", "run", "--port", str(port)]

        # Add persistence path if provided
        if persist_directory:
            os.makedirs(persist_directory, exist_ok=True)
            cmd.extend(["--path", persist_directory])
            logger.info(f"Using persistence path: {persist_directory}")
        else:
            logger.info("Running ChromaDB server in-memory (no persistence)")

        # Add other settings via environment variables (if needed by CLI)
        os.environ["ANONYMIZED_TELEMETRY"] = "FALSE"
        os.environ["ALLOW_RESET"] = "TRUE"

        try:
            # Start the server process
            logger.info(f"Executing command: {' '.join(cmd)}")
            _chroma_server_process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True  # Decode stdout/stderr as text
            )
            _chroma_server_started = True
            logger.info(f"ChromaDB server process started (PID: {_chroma_server_process.pid})")

            # Optional: Monitor server output in a separate thread if needed
            # def monitor_output(pipe, log_func):
            #     for line in iter(pipe.readline, ''):
            #         log_func(f"[Chroma Server]: {line.strip()}")
            #     pipe.close()
            # threading.Thread(target=monitor_output, args=(_chroma_server_process.stdout, logger.info), daemon=True).start()
            # threading.Thread(target=monitor_output, args=(_chroma_server_process.stderr, logger.error), daemon=True).start()

        except FileNotFoundError:
            logger.error("❌ Error: 'chroma' command not found. Is ChromaDB installed and in PATH?")
            _chroma_server_started = False
        except Exception as e:
            logger.error(f"❌ Failed to start ChromaDB server process: {e}")
            _chroma_server_started = False

    # Start the server attempt in a background thread
    thread = threading.Thread(target=_run_server, daemon=True)
    thread.start()

    # Give server time to start up - slightly longer for CLI startup
    logger.info("Waiting for ChromaDB server to initialize...")
    time.sleep(5)


def stop_chroma_server():
    """Stop the ChromaDB server gracefully."""
    global _chroma_server_process, _chroma_server_started

    if _chroma_server_started:
        logger.info("Stopping ChromaDB server")

        # First try to terminate the server process if it exists
        if _chroma_server_process:
            try:
                _chroma_server_process.terminate()
                _chroma_server_process.wait(timeout=5)
            except Exception as e:
                logger.warning(f"Error while terminating ChromaDB process: {e}")

        # Also attempt to kill any uvicorn process running our server
        # This is a fallback for when we're using the direct uvicorn.run approach
        try:
            import os
            import signal

            import psutil

            # Find and terminate any process that might be running our server
            current_process = psutil.Process(os.getpid())
            for child in current_process.children(recursive=True):
                if "uvicorn" in child.name() or "chromadb" in child.name():
                    logger.info(f"Terminating child process: {child.name()} (PID: {child.pid})")
                    child.send_signal(signal.SIGTERM)
        except ImportError:
            logger.warning("psutil not available, cannot find server processes")
        except Exception as e:
            logger.warning(f"Error while finding server processes: {e}")

        _chroma_server_started = False
        _chroma_server_process = None
        logger.info("ChromaDB server stopped")
    else:
        logger.debug("No ChromaDB server was running, nothing to stop")


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
        # Ensure essential metadata exists
        self.metadata.setdefault("keywords", [])
        self.metadata.setdefault("summary", "")
        self.metadata.setdefault("context", "General")
        self.metadata.setdefault("type", "general")
        self.metadata.setdefault("related_notes", [])
        self.metadata.setdefault("sentiment", "neutral")
        self.metadata.setdefault("importance", 0.5)
        self.metadata.setdefault("tags", [])

    def to_dict(self) -> Dict[str, Any]:
        """Convert the note to a dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "MemoryNote":
        """Create a memory note from a dictionary.

        Args:
            data: Dictionary containing note data

        Returns:
            A new MemoryNote instance
        """
        # Extract required fields
        content = data.pop("content", "")
        note_id = data.pop("id", None)

        # Extract metadata if present, otherwise use remaining data as metadata
        metadata = data.pop("metadata", None)
        if metadata is None:
            # Use remaining data as metadata
            metadata = data

        # Create and return a new note
        return MemoryNote(content=content, id=note_id, **metadata)


class AsyncChromaRetriever:
    """Async wrapper for ChromaDB operations."""

    def __init__(self, collection_name: str, persist_directory: str, embedding_function: Any):
        """Initialize async ChromaDB client and collection.

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist data
            embedding_function: Function to generate embeddings
        """
        # Store parameters for later use during initialization
        self.settings = get_chroma_settings(persist_directory)
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        self.client = None
        self.collection = None

    async def initialize(self):
        """Initialize the collection asynchronously."""
        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                # Import inline to avoid circular dependencies
                import chromadb

                logger.info(f"Connecting to ChromaDB server (attempt {attempt+1}/{max_retries})")

                # Proper initialization of AsyncHttpClient for ChromaDB 1.0.0
                # The client factory function must be awaited
                self.client = await chromadb.AsyncHttpClient(
                    host="localhost",
                    port=8001,  # Using port 8001 to avoid conflict with our server on 8000
                    settings=self.settings,
                )

                # Add HNSW configuration as collection metadata
                # Docs: https://cookbook.chromadb.dev/core/configuration/#hnsw-configuration
                hnsw_config = {
                    # hnsw:space
                    # Description: Controls the distance metric. Use 'cosine' for text embeddings.
                    # Cannot be changed after creation. Default: 'l2'.
                    "hnsw:space": "cosine",
                    # hnsw:construction_ef
                    # Description: Controls # neighbours explored during index creation. Higher -> better quality,
                    # more memory. Cannot be changed after creation. Default: 100.
                    "hnsw:construction_ef": 100,
                    # hnsw:M
                    # Description: Max # neighbour connections per node.
                    # Higher -> denser graph, slower but more accurate search, more memory.
                    # Cannot be changed after creation.
                    # Default: 16.
                    "hnsw:M": 16,
                    # hnsw:search_ef
                    # Description: Controls # neighbours explored during search.
                    # Higher -> more accurate search, more memory.
                    # Can be changed after creation.
                    # Default: 10.
                    "hnsw:search_ef": 20,
                    # hnsw:batch_size
                    # Description: Controls size of in-memory index before transfer to HNSW.
                    # Must be < sync_threshold.
                    # Can be changed after creation.
                    # Default: 100.
                    "hnsw:batch_size": 100,
                    # hnsw:sync_threshold
                    # Description: Controls threshold when HNSW index is written to disk.
                    # Can be changed after creation.
                    # Default: 1000.
                    "hnsw:sync_threshold": 1000,
                    # Other parameters (Defaults are usually fine):
                    # hnsw:num_threads
                    # Description: Controls how many threads HNSW algo uses.
                    # Can be changed after creation.
                    # Default: CPU cores.
                    # "hnsw:num_threads": <number of CPU cores>
                    # hnsw:resize_factor
                    # Description: Controls graph growth rate when capacity is reached.
                    # Can be changed after creation.
                    # Default: 1.2.
                    # "hnsw:resize_factor": 1.2
                }

                # Now get or create the collection - this is an async call
                # Pass embedding_function=None because we provide embeddings manually
                logger.info(f"Getting or creating collection: {self.collection_name}")
                self.collection = await self.client.get_or_create_collection(
                    name=self.collection_name,
                    embedding_function=None,  # IMPORTANT: Set to None for client-side embeddings
                    metadata=hnsw_config,
                )

                logger.info(f"🔄 Initialized AsyncChromaRetriever collection: {self.collection_name}")
                logger.debug(f"HNSW config: {hnsw_config}")
                return
            except Exception as e:
                logger.error(f"❌ Error initializing AsyncChromaRetriever (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    raise

    async def add_document(
        self, document: str, doc_id: str, metadata: Dict[str, Any] = None, embedding: List[float] = None
    ):
        """Add a document to ChromaDB.

        Args:
            document: Text content to add
            doc_id: Unique document ID
            metadata: Optional metadata dict
            embedding: Optional pre-computed embedding
        """
        try:
            if embedding is not None:
                # Use pre-computed embedding - async call
                await self.collection.upsert(
                    ids=[doc_id],
                    embeddings=[embedding],
                    documents=[document],
                    metadatas=[metadata] if metadata else None,
                )
            else:
                # Let ChromaDB compute embedding - async call
                await self.collection.upsert(
                    ids=[doc_id], documents=[document], metadatas=[metadata] if metadata else None
                )
            logger.debug(f"📝 Added document {doc_id} to ChromaDB")
        except Exception as e:
            logger.error(f"❌ Error adding document {doc_id} to ChromaDB: {e}")
            raise

    async def delete_document(self, doc_id: str):
        """Delete a document from ChromaDB.

        Args:
            doc_id: ID of document to delete
        """
        try:
            # Async call
            await self.collection.delete(ids=[doc_id])
            logger.debug(f"🗑️ Deleted document {doc_id} from ChromaDB")
        except Exception as e:
            logger.error(f"❌ Error deleting document {doc_id} from ChromaDB: {e}")
            raise

    async def search(
        self,
        query: str,
        top_k: int = 5,
        embedding: List[float] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ):
        """Search for documents in ChromaDB.

        Args:
            query: Search query text
            top_k: Maximum number of results
            embedding: Optional pre-computed query embedding
            metadata_filter: Optional dictionary of metadata fields to filter by

        Returns:
            List of search results with id, content, and metadata
        """
        try:
            if embedding is not None:
                # Use pre-computed embedding - async call
                results = await self.collection.query(
                    query_embeddings=[embedding], n_results=top_k, include=["documents", "metadatas", "distances"]
                )
            else:
                # Let ChromaDB compute embedding - async call
                results = await self.collection.query(
                    query_texts=[query], n_results=top_k, include=["documents", "metadatas", "distances"]
                )

            # Format results
            formatted_results = []
            if results["ids"] and results["ids"][0]:  # Check if we have results
                for i, doc_id in enumerate(results["ids"][0]):
                    # Get metadata and deserialize any JSON string lists
                    metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                    processed_metadata = {}

                    # Process metadata to deserialize any JSON strings that represent lists
                    for key, value in metadata.items():
                        if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
                            # This looks like it might be a JSON list
                            try:
                                processed_metadata[key] = json.loads(value)
                            except json.JSONDecodeError:
                                # If it's not valid JSON, keep the original string
                                processed_metadata[key] = value
                        else:
                            processed_metadata[key] = value

                    # Apply metadata filtering if specified
                    if metadata_filter and processed_metadata:
                        if not all(
                            str(processed_metadata.get(key)) == str(value) for key, value in metadata_filter.items()
                        ):
                            continue

                    formatted_results.append(
                        {
                            "id": doc_id,
                            "content": results["documents"][0][i],
                            "metadata": processed_metadata,
                            "distance": results["distances"][0][i] if results["distances"] else None,
                        }
                    )

            return formatted_results
        except Exception as e:
            logger.error(f"❌ Error searching ChromaDB: {e}")
            raise

    async def get_all(self):
        """Get all documents from ChromaDB.

        Returns:
            Dictionary with ids, documents, and metadatas
        """
        try:
            # Async call
            return await self.collection.get(include=["metadatas", "documents"])
        except Exception as e:
            logger.error(f"❌ Error getting all documents from ChromaDB: {e}")
            raise

    async def update_metadata(self, doc_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Update metadata for an existing document.

        Args:
            doc_id: ID of the document to update
            metadata: New metadata to set

        Returns:
            True if successful, False otherwise
        """
        if not self.initialized:
            logger.warning("ChromaDB retriever not initialized")
            return False

        try:
            # Update metadata for the document
            await self.collection.update(ids=[doc_id], metadatas=[metadata])
            logger.debug(f"Updated metadata for document {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating metadata for document {doc_id}: {e}")
            return False


class AsyncBM25Retriever:
    """Async-compatible BM25 retriever."""

    def __init__(self):
        """Initialize BM25 retriever."""
        try:
            self.bm25 = None
            self.corpus = []
            self.doc_ids = []
            self.initialized = False
        except ImportError:
            logger.error("rank_bm25 not installed. Run: pip install rank-bm25")
            raise

    async def add_document(self, document: str, doc_id: str):
        """Add a document to the BM25 index.

        Args:
            document: Text content to add
            doc_id: Unique document ID
        """
        from rank_bm25 import BM25Okapi

        # Tokenize document
        tokenized_doc = document.lower().split()
        self.corpus.append(tokenized_doc)
        self.doc_ids.append(doc_id)

        # Rebuild BM25 index
        self.bm25 = BM25Okapi(self.corpus)
        self.initialized = True

    async def rebuild_index(self, documents: List[str], doc_ids: List[str]):
        """Rebuild the entire BM25 index.

        Args:
            documents: List of document texts
            doc_ids: List of document IDs
        """
        from rank_bm25 import BM25Okapi

        self.corpus = [doc.lower().split() for doc in documents]
        self.doc_ids = doc_ids
        self.bm25 = BM25Okapi(self.corpus)
        self.initialized = True

    async def search(self, query: str, top_k: int = 5):
        """Search for documents using BM25.

        Args:
            query: Search query text
            top_k: Maximum number of results

        Returns:
            List of search results with id, score
        """
        if not self.initialized:
            return []

        # Tokenize query
        tokenized_query = query.lower().split()

        # Get scores
        scores = self.bm25.get_scores(tokenized_query)

        # Create (id, score) pairs and sort by score
        id_score_pairs = list(zip(self.doc_ids, scores))
        id_score_pairs.sort(key=lambda x: x[1], reverse=True)

        # Return top_k results
        return [{"id": pair[0], "score": pair[1]} for pair in id_score_pairs[:top_k]]


class AsyncEnsembleRetriever:
    """Async ensemble retriever that combines multiple retrievers."""

    def __init__(self, retrievers: List[Any], weights: List[float]):
        """Initialize ensemble retriever.

        Args:
            retrievers: List of retriever objects
            weights: List of weights for each retriever (should sum to 1)
        """
        self.retrievers = retrievers
        self.weights = weights

        if len(retrievers) != len(weights):
            raise ValueError("Number of retrievers must match number of weights")

    async def search(self, query: str, top_k: int = 5, task_type: str = None):
        """Search using ensemble of retrievers.

        Args:
            query: Search query text
            top_k: Maximum number of results
            task_type: Optional task type for specialized embedding

        Returns:
            List of search results with id, content, score
        """
        all_results = {}
        final_results = []

        # Collect results from each retriever
        for i, retriever in enumerate(self.retrievers):
            weight = self.weights[i]

            try:
                # Different call signature for ChromaRetriever vs BM25Retriever
                if hasattr(retriever, "embedding_function") and task_type:  # ChromaRetriever
                    results = await retriever.search(query, top_k=top_k * 2)  # Get more for better ensemble
                else:  # BM25Retriever
                    results = await retriever.search(query, top_k=top_k * 2)

                # Update all_results with weighted scores
                for result in results:
                    doc_id = result["id"]
                    # ChromaRetriever returns distance (lower is better), convert to score
                    if "distance" in result:
                        # Convert distance to score (1 - normalized_distance)
                        score = 1.0 - min(result["distance"], 1.0)
                    else:
                        score = result.get("score", 0.5)  # Default or BM25 score

                    weighted_score = score * weight

                    if doc_id in all_results:
                        all_results[doc_id]["score"] += weighted_score
                    else:
                        all_results[doc_id] = {
                            "id": doc_id,
                            "score": weighted_score,
                            "content": result.get("content", ""),
                            "metadata": result.get("metadata", {}),
                        }
            except Exception as e:
                logger.error(f"Error in retriever {i}: {e}")

        # Sort by final score and return top_k
        final_results = list(all_results.values())
        final_results.sort(key=lambda x: x["score"], reverse=True)

        return final_results[:top_k]


class AgenticMemorySystem:
    """Fully async Agentic Memory System implementation."""

    def __init__(
        self,
        project_name: str = "default",
        llm_backend: str = "gemini",
        llm_model: Optional[str] = None,
        embed_model: Optional[str] = None,
        api_key: Optional[str] = None,
        persist_directory: Optional[str] = None,
        reranker_model: Optional[str] = None,
        jina_api_key: Optional[str] = None,  # Jina might require API key for some models/usage
        enable_llm_analysis: bool = False,
        enable_auto_segmentation: bool = False,
        **kwargs,
    ):
        """Initialize the Agentic Memory System.

        Args:
            project_name: Name of the project for organization
            llm_backend: LLM backend to use ("openai", "gemini", "ollama", "mock")
            llm_model: Name of the LLM model to use
            embed_model: Name of the embedding model to use
            api_key: API key for the LLM backend
            persist_directory: Directory to persist vector store data
            reranker_model: The Jina reranker model name.
            jina_api_key: Optional API key for Jina services.
            enable_llm_analysis: Whether to use LLM to analyze content and determine optimal storage
            enable_auto_segmentation: Whether to automatically segment mixed content into separate memories
            **kwargs: Additional configuration options.
        """
        self.project_name = project_name
        self.persist_directory = persist_directory or os.getenv("PERSIST_DIRECTORY", DEFAULT_PERSIST_DIR)

        # Ensure persist directory exists
        os.makedirs(self.persist_directory, exist_ok=True)

        # Use provided or default model names (prefer env vars if model args are None)
        _llm_model = llm_model or os.getenv("LLM_MODEL", DEFAULT_GEMINI_LLM_MODEL)
        _embed_model = embed_model or os.getenv("EMBED_MODEL", DEFAULT_GEMINI_EMBED_MODEL)
        _reranker_model = reranker_model or os.getenv("JINA_RERANKER_MODEL", DEFAULT_JINA_RERANKER_MODEL)
        jina_api_key or os.getenv("JINA_API_KEY")

        logger.info(f"Initializing async memory system for project: {project_name}")
        logger.info(f"  LLM Backend: {llm_backend}")
        logger.info(f"  LLM Model: {_llm_model}")
        logger.info(f"  Embedding Model: {_embed_model}")
        logger.info(f"  Reranker Model: {_reranker_model}")
        logger.info(f"  Persistence Directory: {self.persist_directory}")

        # Initialize LLM Controller and Analyzer
        self.llm_controller = LLMController(
            backend=llm_backend,
            model=_llm_model,
            embed_model=_embed_model,
            api_key=api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("OPENAI_API_KEY"),  # Pass relevant API key
        )
        self.content_analyzer = LLMContentAnalyzer(self.llm_controller)

        # Set up enhanced LLM features
        self.enable_llm_analysis = enable_llm_analysis
        self.enable_auto_segmentation = enable_auto_segmentation

        # Initialize Retrievers
        # Ensure unique collection name per project within the persist directory
        collection_name = f"{DEFAULT_COLLECTION_PREFIX}_{project_name.replace(' ', '_').lower()}"
        chroma_persist_path = os.path.join(self.persist_directory, project_name)
        os.makedirs(chroma_persist_path, exist_ok=True)
        logger.info(f"  Chroma Collection: {collection_name} in {chroma_persist_path}")

        self.chroma_retriever = AsyncChromaRetriever(
            collection_name=collection_name,
            persist_directory=chroma_persist_path,
            embedding_function=self.llm_controller.provider,  # Pass the provider for embeddings
        )
        self.bm25_retriever = AsyncBM25Retriever()
        self.ensemble_retriever = AsyncEnsembleRetriever(
            retrievers=[self.chroma_retriever, self.bm25_retriever],
            weights=[0.6, 0.4],  # Example weights, tune as needed
        )

        # Initialize Reranker
        try:
            self.reranker = CrossEncoder(
                _reranker_model,
                automodel_args={"torch_dtype": "auto"},  # Use auto for best available dtype
                trust_remote_code=True,  # Often needed for custom models like Jina's
            )
            logger.info(f"Initialized reranker: {_reranker_model}")
        except Exception as e:
            logger.error(f"Failed to initialize reranker model '{_reranker_model}': {e}. Reranking will be disabled.")
            self.reranker = None

        # In-memory cache for MemoryNote objects (IDs mapped to objects)
        self.memory_cache: Dict[str, MemoryNote] = {}

        # Flag to track initialization
        self.initialized = False

    async def initialize(self):
        """Initialize the memory system asynchronously."""
        if self.initialized:
            logger.debug("🔄 Memory system already initialized, skipping")
            return

        logger.info(f"🚀 Initializing async memory system for project: {self.project_name}")

        # Start ChromaDB server for AsyncHttpClient
        logger.info("⚙️ Starting ChromaDB server process...")
        chroma_server_path = os.path.join(
            self.persist_directory, "server_data"
        )  # Use a subdirectory for server's own data
        os.makedirs(chroma_server_path, exist_ok=True)
        start_chroma_server(persist_directory=chroma_server_path, port=8001)

        # Initialize ChromaRetriever (which will connect to the server)
        logger.info("📊 Initializing ChromaDB retriever")
        await self.chroma_retriever.initialize()

        # Load existing documents
        logger.info("📝 Loading existing documents into memory")
        await self._load_existing_documents()

        self.initialized = True
        logger.info("✅ Memory system initialization complete")

    async def _load_existing_documents(self):
        """Load existing documents from Chroma into BM25 and cache asynchronously."""
        try:
            logger.info("Loading existing documents for BM25 and cache...")
            # Get all documents from Chroma
            all_docs = await self.chroma_retriever.get_all()

            if all_docs and all_docs["ids"]:
                count = 0
                # Prepare documents for batch BM25 rebuild
                documents = []
                doc_ids = []

                for doc_id, content, metadata in zip(all_docs["ids"], all_docs["documents"], all_docs["metadatas"]):
                    # Add to document lists
                    documents.append(content)
                    doc_ids.append(doc_id)

                    # Process metadata to deserialize any JSON strings that represent lists
                    processed_metadata = {}
                    for key, value in metadata.items():
                        if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
                            # This looks like it might be a JSON list
                            try:
                                processed_metadata[key] = json.loads(value)
                            except json.JSONDecodeError:
                                # If it's not valid JSON, keep the original string
                                processed_metadata[key] = value
                        else:
                            processed_metadata[key] = value

                    # Add to in-memory cache with properly deserialized metadata
                    self.memory_cache[doc_id] = MemoryNote(content=content, id=doc_id, **processed_metadata)
                    count += 1

                # Rebuild BM25 index in one operation
                if documents:
                    await self.bm25_retriever.rebuild_index(documents, doc_ids)

                logger.info(f"Loaded {count} existing documents into BM25 and cache.")
            else:
                logger.info("No existing documents found to load.")
        except Exception as e:
            logger.error(f"Error loading existing documents: {e}")

    async def _analyze_content(self, content: str) -> Dict[str, Any]:
        """Helper method to analyze content (synchronously for now)."""
        try:
            logger.debug(f"Analyzing content snippet: '{content[:50]}...'")
            # Remove await: Call the synchronous analyzer method directly
            analysis_result = self.content_analyzer.analyze_content_type(content)
            logger.debug(f"Content analysis result: {analysis_result}")
            return analysis_result
        except Exception as e:
            logger.error(f"Error during content analysis: {e}", exc_info=True)
            # Return default analysis on error
            return {
                "primary_type": "general",
                "keywords": [],
                "summary": "",
                "sentiment": "neutral",
                "importance": 0.5,
                "storage_task_type": "RETRIEVAL_DOCUMENT",
                "confidence": 0.0,
                "has_mixed_content": False,
            }

    async def create(self, content: str, name: Optional[str] = None, **extra_metadata) -> MemoryNote:
        """Create a new memory note.

        Args:
            content: The content of the memory note
            name: Optional name for the memory
            **extra_metadata: Additional metadata fields

        Returns:
            The created MemoryNote
        """
        # ... (Initialization check is important here)
        if not self.initialized:
            logger.warning("Memory system not initialized. Call initialize() first.")
            # Handle appropriately - maybe raise error or auto-initialize?
            # For now, let's try auto-initializing, but this might hide issues.
            logger.info("⏳ Auto-initializing memory system in create()")
            await self.initialize()
            if not self.initialized:  # Check again after attempt
                raise RuntimeError("Memory system failed to initialize.")

        logger.info(f"🧠 Creating memory note. Name: '{name}'")

        # Apply LLM analysis if enabled
        if self.enable_llm_analysis:
            # Use the content analyzer to analyze the content
            analysis = await self._analyze_content(content)

            # Use detected content type if not provided
            if "type" not in extra_metadata:
                extra_metadata["type"] = analysis.get("primary_type", "general")

            # Get recommended storage task type
            storage_task_type = analysis.get("storage_task_type")

            # Add analysis results to metadata
            extra_metadata["analysis_results"] = {
                "primary_type": analysis.get("primary_type", "general"),
                "confidence": analysis.get("confidence", 0.0),
                "has_mixed_content": analysis.get("has_mixed_content", False),
                "type_proportions": analysis.get("types", {}),
            }
        else:
            # Default analysis without LLM
            analysis = await self._analyze_content(content)
            storage_task_type = analysis.get("storage_task_type")

        # Apply auto-segmentation if enabled and content is mixed
        if (
            self.enable_auto_segmentation
            and analysis.get("has_mixed_content", False)
            and len(content) > 500
            and hasattr(self, "content_analyzer")
            and self.content_analyzer
        ):
            # Use the content analyzer to segment the content
            segments = self.content_analyzer.segment_content(content)

            if len(segments) > 1:
                logger.info(f"Auto-segmenting content into {len(segments)} parts")

                # Create a parent memory for the full content
                parent_metadata = {
                    **extra_metadata,
                    "is_parent": True,
                    "segment_count": len(segments),
                }

                # 1. Generate embedding (if needed)
                parent_embedding = None
                if self.llm_controller and storage_task_type:
                    parent_embedding = self.llm_controller.get_embeddings(content, task_type=storage_task_type)

                # 2. Prepare metadata
                base_metadata = {
                    # Keywords default to list, will be serialized later
                    "keywords": parent_metadata.pop("keywords", []) or analysis.get("keywords", []),
                    "summary": parent_metadata.pop("summary", "") or analysis.get("summary", ""),
                    "context": parent_metadata.pop("context", "General"),
                    "type": analysis.get("primary_type", "general"),
                    # Related notes default to list, will be serialized later
                    "related_notes": parent_metadata.pop("related_notes", []),
                    "sentiment": parent_metadata.pop("sentiment", "neutral") or analysis.get("sentiment", "neutral"),
                    "importance": parent_metadata.pop("importance", 0.5) or analysis.get("importance", 0.5),
                    # Tags default to list, will be serialized later
                    "tags": parent_metadata.pop("tags", []),
                    "name": name or parent_metadata.pop("name", None),
                    "storage_task_type": storage_task_type,
                    "analysis_confidence": analysis.get("confidence", 0.0),
                    "has_mixed_content": analysis.get("has_mixed_content", False),
                    "is_parent": True,
                    "segment_count": len(segments),
                }

                # Merge base metadata with any remaining extra_metadata
                # Ensure timestamps are handled correctly
                now = datetime.now().isoformat()
                parent_metadata_to_store = {
                    **base_metadata,
                    **parent_metadata,  # Add remaining arbitrary metadata
                    "created_at": now,
                    "updated_at": now,
                }

                # Process metadata for storage
                processed_parent_metadata = self._process_metadata_for_chroma(parent_metadata_to_store)

                # 3. Store parent in retriever
                parent_id = str(uuid.uuid4())
                try:
                    await self.chroma_retriever.add_document(
                        document=content,
                        doc_id=parent_id,
                        metadata=processed_parent_metadata,
                        embedding=parent_embedding,
                    )
                    logger.info(f"📄 Added parent document {parent_id} to ChromaDB")
                except Exception as e:
                    logger.error(f"❌ Failed to add parent document {parent_id} to vector store: {e}", exc_info=True)
                    raise

                # Update BM25 index for the parent
                await self._update_bm25_index(parent_id, content)

                # Create segment IDs list to store in parent metadata
                segment_ids = []

                # Process each segment
                for i, segment in enumerate(segments):
                    segment_name = segment["metadata"].get(
                        "subtitle", f"{name} - Segment {i+1}" if name else f"Segment {i+1}"
                    )

                    # Prepare segment metadata
                    segment_metadata = {
                        **extra_metadata,  # Include original metadata
                        "parent_id": parent_id,
                        "segment_index": i,
                        "type": segment["type"],
                        "task_type": segment["task_type"],
                        "name": segment_name,
                    }

                    # Add segment metadata
                    if segment["metadata"]:
                        segment_metadata.update(segment["metadata"])

                    # Generate embedding for the segment
                    segment_embedding = None
                    if self.llm_controller and segment["task_type"]:
                        segment_embedding = self.llm_controller.get_embeddings(
                            segment["content"], task_type=segment["task_type"]
                        )

                    # Process metadata for storage
                    processed_segment_metadata = self._process_metadata_for_chroma(segment_metadata)

                    # Store segment in retriever
                    segment_id = str(uuid.uuid4())
                    try:
                        await self.chroma_retriever.add_document(
                            document=segment["content"],
                            doc_id=segment_id,
                            metadata=processed_segment_metadata,
                            embedding=segment_embedding,
                        )
                        logger.info(f"📄 Added segment document {segment_id} to ChromaDB")
                        segment_ids.append(segment_id)
                    except Exception as e:
                        logger.error(
                            f"❌ Failed to add segment document {segment_id} to vector store: {e}", exc_info=True
                        )
                        # Continue with other segments

                    # Update BM25 index for the segment
                    await self._update_bm25_index(segment_id, segment["content"])

                # Update parent metadata with segment IDs
                parent_metadata_to_store["segment_ids"] = segment_ids
                processed_parent_metadata = self._process_metadata_for_chroma(parent_metadata_to_store)

                # Update parent metadata in ChromaDB
                try:
                    # Direct update of metadata in ChromaDB
                    await self.chroma_retriever.update_metadata(parent_id, processed_parent_metadata)
                    logger.info(f"📝 Updated parent document {parent_id} with segment IDs")
                except Exception as e:
                    logger.error(f"❌ Failed to update parent document metadata: {e}", exc_info=True)

                # 5. Create MemoryNote object for parent
                parent_memory_note = MemoryNote(content=content, id=parent_id, **parent_metadata_to_store)

                # Add to memory cache
                self.memory_cache[parent_id] = parent_memory_note

                logger.info(f"✅ Parent memory note created with {len(segment_ids)} segments. ID: {parent_id}")
                return parent_memory_note

        # Create a regular memory note if not using segmentation
        # 1. Analyze content (use the result from above if LLM analysis was enabled)
        if not self.enable_llm_analysis:
            analysis = await self._analyze_content(content)

        primary_type = analysis.get("primary_type", "general")
        storage_task_type = analysis.get("storage_task_type")

        # 2. Generate embedding
        embedding = None
        if self.llm_controller:
            embedding = self.llm_controller.get_embeddings(content, task_type=storage_task_type)

        # 3. Prepare metadata
        base_metadata = {
            # Keywords default to list, will be serialized later
            "keywords": extra_metadata.pop("keywords", []) or analysis.get("keywords", []),
            "summary": extra_metadata.pop("summary", "") or analysis.get("summary", ""),
            "context": extra_metadata.pop("context", "General"),
            "type": primary_type,
            # Related notes default to list, will be serialized later
            "related_notes": extra_metadata.pop("related_notes", []),
            "sentiment": extra_metadata.pop("sentiment", "neutral") or analysis.get("sentiment", "neutral"),
            "importance": extra_metadata.pop("importance", 0.5) or analysis.get("importance", 0.5),
            # Tags default to list, will be serialized later
            "tags": extra_metadata.pop("tags", []),
            "name": name or extra_metadata.pop("name", None),
            "storage_task_type": storage_task_type,
            "analysis_confidence": analysis.get("confidence", 0.0),
            "has_mixed_content": analysis.get("has_mixed_content", False),
        }

        # Merge base metadata with any remaining extra_metadata
        # Ensure timestamps are handled correctly
        now = datetime.now().isoformat()
        metadata_to_store = {
            **base_metadata,
            **extra_metadata,  # Add remaining arbitrary metadata
            "created_at": now,
            "updated_at": now,
        }

        # Process metadata for storage
        processed_metadata = self._process_metadata_for_chroma(metadata_to_store)

        # 4. Store in retriever
        note_id = str(uuid.uuid4())
        try:
            await self.chroma_retriever.add_document(
                document=content, doc_id=note_id, metadata=processed_metadata, embedding=embedding
            )
            logger.info(f"📄 Added document {note_id} to ChromaDB")
        except ValueError as e:
            # Catch potential re-raised error from retriever
            logger.error(f"❌ Failed to add document {note_id} to vector store: {e}", exc_info=True)
            # Decide if we should still create the in-memory note or raise further
            raise
        except Exception as e:
            logger.error(f"❌ Unexpected error adding document {note_id} to vector store: {e}", exc_info=True)
            # Decide if we should still create the in-memory note or raise further
            raise

        # 5. Create MemoryNote object
        memory_note = MemoryNote(content=content, id=note_id, **metadata_to_store)

        # 6. Update cache and BM25 index
        # Add to memory cache
        self.memory_cache[note_id] = memory_note
        # Update BM25 index
        await self._update_bm25_index(note_id, content)

        logger.info(f"✅ Memory note created successfully. ID: {note_id}")
        return memory_note

    async def get(self, note_id: str) -> Optional[MemoryNote]:
        """Get a memory note by ID from the cache asynchronously.

        Args:
            note_id: ID of the note to retrieve.

        Returns:
            The MemoryNote object or None if not found.
        """
        # Ensure system is initialized
        if not self.initialized:
            await self.initialize()

        note = self.memory_cache.get(note_id)
        if note:
            logger.info(f"Retrieved memory from cache: {note_id}")
        else:
            logger.warning(f"Memory not found in cache: {note_id}")
        return note

    async def update(self, note_id: str, content: str, **metadata) -> Optional[MemoryNote]:
        """Update a memory note's content and metadata asynchronously."""
        # Ensure system is initialized
        if not self.initialized:
            logger.info("⏳ Auto-initializing memory system")
            await self.initialize()

        if note_id not in self.memory_cache:
            logger.error(f"❌ Cannot update memory: {note_id} not found in cache")
            return None

        content_preview = content[:30] + "..." if len(content) > 30 else content
        logger.info(f"🔄 Updating memory: {note_id}")
        logger.debug(f"Content preview: '{content_preview}'")

        original_note = self.memory_cache[note_id]
        original_note.content = content
        original_note.metadata.update(metadata)  # Update existing metadata dict
        original_note.updated_at = datetime.now().isoformat()

        # Re-analyze content and task type
        # TODO: Make content_analyzer.analyze_content_type async
        logger.info("🔍 Re-analyzing content")
        analysis = self.content_analyzer.analyze_content_type(content)
        storage_task_type = analysis.get("recommended_task_types", {}).get("storage", "RETRIEVAL_DOCUMENT")
        primary_type = analysis.get("primary_type", "general")
        original_note.metadata["storage_task_type"] = storage_task_type
        original_note.metadata["type"] = primary_type
        original_note.metadata["analysis_confidence"] = analysis.get("confidence", 0.0)
        original_note.metadata["has_mixed_content"] = analysis.get("has_mixed_content", False)
        logger.info(f"📊 Re-analyzed: type={primary_type}, task_type={storage_task_type}")

        # Update in ChromaDB
        try:
            # TODO: Make llm_controller.get_embeddings async
            logger.info("🧠 Generating updated embeddings")
            embedding = self.llm_controller.get_embeddings(content, task_type=storage_task_type)
            logger.info("💾 Updating in ChromaDB")
            await self.chroma_retriever.add_document(
                document=content, embedding=embedding, metadata=original_note.metadata, doc_id=note_id
            )
            logger.info(f"✅ Updated in Chroma with task_type: {storage_task_type}")
        except Exception as e:
            logger.error(f"❌ Failed to update document {note_id} in Chroma: {str(e)}")

        # Update in BM25
        try:
            # Collect all documents for BM25 rebuild
            logger.info("📚 Rebuilding BM25 index")
            documents = []
            doc_ids = []
            for nid, note_obj in self.memory_cache.items():
                documents.append(note_obj.content)
                doc_ids.append(nid)

            # Rebuild BM25 index
            await self.bm25_retriever.rebuild_index(documents, doc_ids)
            logger.info("✅ Rebuilt BM25 index after update")
        except Exception as e:
            logger.error(f"❌ Failed to update BM25 index for {note_id}: {str(e)}")

        logger.info(f"✅ Memory {note_id} updated successfully")
        return original_note

    async def delete(self, note_id: str) -> bool:
        """Delete a memory note asynchronously."""
        # Ensure system is initialized
        if not self.initialized:
            logger.info("⏳ Auto-initializing memory system")
            await self.initialize()

        if note_id not in self.memory_cache:
            logger.warning(f"⚠️ Memory not found for deletion: {note_id}")
            return False

        logger.info(f"🗑️ Deleting memory: {note_id}")

        # Delete from ChromaDB
        try:
            logger.info("🗑️ Removing from ChromaDB")
            await self.chroma_retriever.delete_document(note_id)
            logger.info("✅ Deleted from Chroma")
        except Exception as e:
            logger.error(f"❌ Failed to delete document {note_id} from Chroma: {str(e)}")

        # Delete from memory cache
        logger.info("🗑️ Removing from memory cache")
        del self.memory_cache[note_id]

        # Rebuild BM25 index
        try:
            # Collect all remaining documents
            logger.info("📚 Rebuilding BM25 index")
            documents = []
            doc_ids = []
            for nid, note_obj in self.memory_cache.items():
                documents.append(note_obj.content)
                doc_ids.append(nid)

            # Rebuild BM25 index
            await self.bm25_retriever.rebuild_index(documents, doc_ids)
            logger.info("✅ Rebuilt BM25 index after deletion")
        except Exception as e:
            logger.error(f"❌ Failed to update BM25 index after deleting {note_id}: {str(e)}")

        logger.info(f"✅ Memory {note_id} deleted from all systems")
        return True

    async def search(
        self,
        query: str,
        k: int = 5,
        use_reranker: bool = True,
        use_embeddings: bool = True,
        metadata_filter: Optional[Dict[str, Any]] = None,
        content_type: Optional[str] = None,
        task_type: Optional[str] = None,
    ) -> List[MemoryNote]:
        """Search for memory notes asynchronously.

        Args:
            query: The search query text
            k: Maximum number of results to return
            use_reranker: Whether to use the reranker for better results
            use_embeddings: Whether to use embeddings for search
            metadata_filter: Optional dictionary of metadata fields to filter by
            content_type: Optional content type to filter results by
            task_type: Optional embedding task type to use for the query

        Returns:
            List of MemoryNote objects matching the query
        """
        # Ensure system is initialized
        if not self.initialized:
            logger.info("⏳ Auto-initializing memory system")
            await self.initialize()

        log_params = f"top_k={k}, reranker={'enabled' if use_reranker else 'disabled'}, embeddings={'enabled' if use_embeddings else 'disabled'}"
        if metadata_filter:
            log_params += f", metadata_filter={metadata_filter}"

        logger.info(f"🔎 Searching memories for: '{query}' ({log_params})")

        # 1. Analyze query to get optimal task type (if task_type not specified and LLM analysis is enabled)
        query_task_type = task_type
        if not query_task_type and self.enable_llm_analysis:
            # Use LLM to analyze the query and determine the optimal task type
            query_analysis = self.content_analyzer.analyze_content_type(query)

            # Get recommended query task type from analysis
            query_task_type = query_analysis.get("recommended_task_types", {}).get("query")

            # If content type not specified, determine it from query analysis
            if not content_type and query_analysis:
                query_primary_type = query_analysis.get("primary_type")

                # For code-related queries, filter for code content
                if query_primary_type in ["code", "code_query"]:
                    content_type = "code"
                    logger.info(f"Using detected content type filter: {content_type}")

            logger.info(
                f"📊 Query analysis: task_type={query_task_type}, primary_type={query_analysis.get('primary_type')}"
            )
        else:
            # If not using LLM analysis or task_type already specified, use basic analysis
            basic_analysis = await self._analyze_content(query)
            if not query_task_type:
                query_task_type = basic_analysis.get("storage_task_type")

        # Update metadata filter to include content type if specified
        if content_type:
            if not metadata_filter:
                metadata_filter = {}
            metadata_filter["type"] = content_type

        # 2. Get initial candidates from retrievers
        initial_k = k * 5 if use_reranker and self.reranker else k
        try:
            logger.info(f"🔍 Retrieving initial candidates (top_{initial_k})")

            # Generate embedding for the query if using embeddings
            query_embedding = None
            if use_embeddings and self.llm_controller:
                try:
                    # Get embeddings using the LLM controller with the appropriate task type
                    query_embedding = self.llm_controller.get_embeddings(query, task_type=query_task_type)
                    logger.debug(f"Generated query embedding (dim={len(query_embedding)})")
                except Exception as e:
                    logger.error(f"❌ Error generating query embedding: {e}")
                    logger.warning("⚠️ Falling back to text search without embeddings")
                    use_embeddings = False

            candidates = []

            if use_embeddings:
                # Try vector search first if embeddings are enabled
                try:
                    vector_candidates = await self.chroma_retriever.search(
                        query=query, top_k=initial_k, embedding=query_embedding, metadata_filter=metadata_filter
                    )
                    candidates.extend(vector_candidates)
                    logger.debug(f"ChromaDB search found {len(vector_candidates)} results")
                except Exception as e:
                    logger.error(f"❌ Error in vector search: {str(e)}")

            # Always use BM25 search as fallback or additional source
            try:
                bm25_candidates = await self.bm25_retriever.search(query=query, top_k=initial_k)
                candidates.extend(bm25_candidates)
                logger.debug(f"BM25 search found {len(bm25_candidates)} results")
            except Exception as e:
                logger.error(f"❌ Error in BM25 search: {str(e)}")

            # Deduplicate candidates by ID
            seen_ids = set()
            unique_candidates = []
            for candidate in candidates:
                if candidate["id"] not in seen_ids:
                    seen_ids.add(candidate["id"])
                    unique_candidates.append(candidate)

            candidates = unique_candidates[:initial_k]
            logger.info(f"📋 Initial candidates retrieved: {len(candidates)}")

        except Exception as e:
            logger.error(f"❌ Error during search: {str(e)}")
            return []

        if not candidates:
            logger.warning("⚠️ No candidates found in search")
            return []

        # Prepare for reranking
        candidate_ids = [candidate["id"] for candidate in candidates]

        # With reranking
        if use_reranker and self.reranker and candidates:
            try:
                logger.info(f"⚖️ Reranking {len(candidates)} candidates")

                # Create query-document pairs for reranking
                rerank_pool = [(query, candidate["content"]) for candidate in candidates]

                # TODO: Make reranker.predict async
                scores = self.reranker.predict(rerank_pool)

                # Combine candidates with scores and sort
                reranked_candidates = sorted(zip(candidate_ids, scores), key=lambda item: item[1], reverse=True)
                logger.info("✅ Reranking completed")

                # Get top k reranked IDs
                final_ids = [item[0] for item in reranked_candidates[:k]]

            except Exception as e:
                logger.error(f"❌ Error during reranking: {str(e)}. Falling back to initial ranking")
                final_ids = candidate_ids[:k]
        else:
            # Use initial ranking if reranker is disabled or unavailable
            logger.info("📋 Using initial ranking (no reranking)")
            final_ids = candidate_ids[:k]
            if use_reranker and not self.reranker:
                logger.warning("⚠️ Reranker requested but not available/initialized")

        # 4. Process results
        # First, check if we need to handle segments/parents specially
        final_results = []
        seen_parent_ids = set()

        for note_id in final_ids:
            note = self.memory_cache.get(note_id)
            if not note:
                logger.warning(f"⚠️ Note ID {note_id} from search results not found in cache")
                continue

            # Apply metadata filtering if specified
            if metadata_filter and note.metadata:
                # Skip if doesn't match any filter criteria
                if not all(str(note.metadata.get(key)) == str(value) for key, value in metadata_filter.items()):
                    continue

            # Check if this is a segment with a parent
            parent_id = note.metadata.get("parent_id")

            # If this is a segment and we want to include its parent
            if parent_id and parent_id not in seen_parent_ids:
                # Get the parent memory
                parent = self.memory_cache.get(parent_id)
                if parent:
                    # Add the parent before the segment
                    final_results.append(parent)
                    seen_parent_ids.add(parent_id)

            # Add this result
            final_results.append(note)

        logger.info(f"✅ Search completed. Returning {len(final_results)} results")
        return final_results

    async def get_all(self) -> List[MemoryNote]:
        """Get all memory notes from the system."""
        if not self.initialized:
            logger.warning("Memory system not initialized. Call initialize() first.")
            return []

        # Get all data from retriever
        try:
            all_data = await self.chroma_retriever.get_all()
            memories = []
            # Check if data and IDs exist before iterating
            if all_data and all_data.get("ids"):
                ids = all_data.get("ids", [])
                docs = all_data.get("documents", [])
                metadatas = all_data.get("metadatas", [])

                # Pad shorter lists to match ids length
                max_len = len(ids)
                docs.extend([""] * (max_len - len(docs)))
                metadatas.extend([{}] * (max_len - len(metadatas)))

                # Create MemoryNote objects for each document
                for i, doc_id in enumerate(ids):
                    content = docs[i]
                    metadata = metadatas[i] if metadatas and i < len(metadatas) else {}

                    # Create note data dictionary with correct structure
                    note_data = {"id": doc_id, "content": content, "metadata": metadata}

                    # Create memory note using from_dict
                    memories.append(MemoryNote.from_dict(note_data))

            logger.info(f"Retrieved {len(memories)} total memories.")
            return memories
        except Exception as e:
            logger.error(f"Error retrieving all memories: {e}", exc_info=True)
            return []

    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory system asynchronously.

        Returns:
            Dictionary with statistics like total count and average content length.
        """
        # Ensure system is initialized
        if not self.initialized:
            await self.initialize()

        memories = self.memory_cache

        total = len(memories)
        if total == 0:
            return {"total_memories": 0, "average_content_length": 0.0}

        avg_length = sum(len(m.content) for m in memories.values()) / total

        return {"total_memories": total, "average_content_length": round(avg_length, 1)}

    async def _update_bm25_index(self, note_id: str, content: str):
        """Update the BM25 index with new content.

        Args:
            note_id: The ID of the document to add
            content: The content to index
        """
        try:
            logger.debug(f"Updating BM25 index for document: {note_id}")
            # Add to memory cache if not already present
            if note_id not in self.memory_cache and hasattr(self, "memory_cache"):
                # This shouldn't typically happen, but just in case
                logger.warning(f"Document {note_id} not found in memory cache during BM25 update")

            # Add to BM25 retriever
            await self.bm25_retriever.add_document(document=content, doc_id=note_id)
            logger.debug(f"✅ Updated BM25 index for document: {note_id}")
        except Exception as e:
            logger.error(f"❌ Error updating BM25 index for document {note_id}: {e}", exc_info=True)
            # Don't re-raise - just log the error as this is a secondary operation

    def __del__(self):
        """Clean up resources when the object is deleted."""
        try:
            # Stop the ChromaDB server when the memory system is destroyed
            logger.info("🧹 Cleaning up memory system, stopping ChromaDB server...")
            stop_chroma_server()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def _process_metadata_for_chroma(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process metadata to ensure it's compatible with ChromaDB.

        ChromaDB requires metadata values to be strings, numbers, or booleans.
        This method handles lists and other complex types by serializing them to JSON.

        Args:
            metadata: Dictionary of metadata to process

        Returns:
            Dictionary with processed metadata compatible with ChromaDB
        """
        processed_metadata = {}

        # Ensure all metadata values are ChromaDB compatible (str, int, float, bool)
        # Serialize lists to JSON strings
        for key, value in metadata.items():
            if isinstance(value, list):
                try:
                    processed_metadata[key] = json.dumps(value)
                except TypeError as e:
                    logger.warning(f"Could not JSON serialize list for key '{key}': {e}. Storing as string.")
                    processed_metadata[key] = str(value)  # Fallback
            elif value is None:  # Handle None explicitly (convert to empty string)
                processed_metadata[key] = ""
            elif not isinstance(value, (str, int, float, bool)):
                # Convert other potentially incompatible types
                processed_metadata[key] = str(value)
            else:
                # Value is already compatible
                processed_metadata[key] = value

        return processed_metadata
