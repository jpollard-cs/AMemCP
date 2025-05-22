from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from typing import Any, Dict, Optional

import psutil
import uvicorn
from fastmcp import Context, FastMCP
from starlette.applications import Starlette
from starlette.routing import Mount

from amem.core.cache import InMemoryCache
from amem.core.llm_content_analyzer import LLMContentAnalyzer
from amem.core.llm_controller import LLMController
from amem.core.mappers import MemoryNoteMapper
from amem.core.memory_system import AgenticMemorySystem, AgenticMemorySystemConfig
from amem.core.rerankers import JinaReranker
from amem.core.retrievers import BM25Retriever, ChromaVectorRetriever, EnsembleRetriever
from amem.core.stores import ChromaStore
from amem.utils.utils import DEFAULT_COLLECTION_PREFIX, DEFAULT_PERSIST_DIR, setup_logger

# ───────────────────────────── config ─────────────────────────────────── #


@dataclass
class ServerConfig:
    """Server configuration pulled from environment variables."""

    name: str = os.getenv("SERVER_NAME", "AMemCP")
    host: str = os.getenv("FASTMCP_HOST", "0.0.0.0")
    port: int = int(os.getenv("FASTMCP_PORT", 8010))
    log_level: str = os.getenv("FASTMCP_LOG_LEVEL", "info").lower()

    debug_monitor: bool = os.getenv("DEBUG_MONITOR", "0") == "1"
    monitor_port: int = int(os.getenv("DEBUG_MONITOR_PORT", 50101))

    # Memory system specifics
    project_name: str = os.getenv("PROJECT_NAME", "default")
    chroma_host: str = os.getenv("CHROMA_SERVER_HOST", "localhost")
    chroma_port: int = int(os.getenv("CHROMA_SERVER_HTTP_PORT", 8001))
    persist_directory: str = os.getenv("PERSIST_DIRECTORY", DEFAULT_PERSIST_DIR)

    llm_backend: str = os.getenv("LLM_BACKEND", "gemini")
    llm_model: Optional[str] = os.getenv("LLM_MODEL")
    embed_model: Optional[str] = os.getenv("EMBED_MODEL")
    api_key: Optional[str] = os.getenv("GOOGLE_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")

    enable_llm_analysis: bool = os.getenv("ENABLE_LLM_ANALYSIS", "true").lower() == "true"
    enable_auto_segmentation: bool = os.getenv("ENABLE_AUTO_SEGMENTATION", "false").lower() == "true"
    load_existing: bool = os.getenv("LOAD_EXISTING_ON_STARTUP", "true").lower() == "true"
    reranker_model: Optional[str] = os.getenv("RERANKER_MODEL")

    collection_name: str = field(init=False)

    def __post_init__(self):
        self.collection_name = f"{DEFAULT_COLLECTION_PREFIX}_{self.project_name.replace(' ', '_').lower()}"


# ─────────────────────────── main server ──────────────────────────── #


class AMemCPServer:
    """FastMCP wrapper that exposes the Agentic Memory System via SSE."""

    def __init__(self, cfg: ServerConfig):
        self.cfg = cfg
        # Adjust logger level based on debug_monitor flag
        log_level = logging.DEBUG if cfg.debug_monitor else cfg.log_level.upper()
        # Ensure log level is valid
        numeric_level = getattr(logging, str(log_level), logging.INFO)
        self.logger = setup_logger(cfg.name, level=numeric_level)
        self.mcp: FastMCP | None = None
        self.app = None
        self.memory_system: AgenticMemorySystem | None = None

    # ---------------------- Starlette startup/shutdown ---------------------- #
    async def _on_startup(self):
        """Initialize the AgenticMemorySystem once on app startup."""
        mapper = MemoryNoteMapper()
        store = ChromaStore(
            collection_name=self.cfg.collection_name,
            host=self.cfg.chroma_host,
            port=self.cfg.chroma_port,
            mapper=mapper,
        )
        await store.initialize()
        bm25 = BM25Retriever()
        vector = ChromaVectorRetriever(collection=store.collection)
        ensemble = EnsembleRetriever([vector, bm25], [0.5, 0.5])
        llm_ctrl = LLMController(
            backend=self.cfg.llm_backend,
            model=self.cfg.llm_model,
            embed_model=self.cfg.embed_model,
            api_key=self.cfg.api_key,
        )
        analyzer = LLMContentAnalyzer(llm_ctrl)

        # Instantiate the reranker
        reranker_model_name = self.cfg.reranker_model or "jinaai/jina-reranker-v2-base-multilingual"
        try:
            reranker = JinaReranker(model_name=reranker_model_name)
            if not reranker._initialized:  # Check if model loading failed in __init__
                self.logger.warning(
                    f"JinaReranker failed to initialize model '{reranker_model_name}'. Proceeding without reranker."
                )
                reranker = None
            else:
                self.logger.info(f"Initialized JinaReranker with model: {reranker_model_name}")
        except Exception as e:
            self.logger.error(f"Failed to instantiate JinaReranker: {e}", exc_info=True)
            reranker = None

        self.memory_system = AgenticMemorySystem(
            memory_store=store,
            cache=InMemoryCache(),
            embedder=llm_ctrl,
            analyzer=analyzer,
            retriever=ensemble,
            config=AgenticMemorySystemConfig(
                enable_llm_analysis=self.cfg.enable_llm_analysis,
                enable_auto_segmentation=self.cfg.enable_auto_segmentation,
            ),
            reranker=reranker,
        )
        await self.memory_system.initialize(load_existing=self.cfg.load_existing)
        self.logger.info("Memory system initialized on startup")

    async def _on_shutdown(self):
        """Shutdown the AgenticMemorySystem on app shutdown."""
        if self.memory_system:
            await self.memory_system.shutdown()
            self.logger.info("Memory system shutdown complete")

    # --------------------------- helpers --------------------------- #
    def _ms(self, ctx: Context) -> AgenticMemorySystem:
        """Return the global memory system created on startup."""
        if self.memory_system:
            return self.memory_system
        raise RuntimeError("Memory system not initialized")

    # ---------------------- tool registration ---------------------- #

    def _handle_tool_errors(self, func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                self.logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
                return {"error": f"Tool '{func.__name__}' failed: {str(e)}"}

        return wrapper

    def _register_tools(self):
        # Tools share a single memory system; context is retrieved internally

        # Apply the decorator to functions that need error handling

        @self._handle_tool_errors
        async def health_check():
            """Return process & service health information."""
            mem_mb = psutil.Process().memory_info().rss / 1024 / 1024
            return {"status": "ok", "memory_mb": round(mem_mb, 2), "ts": datetime.utcnow().isoformat()}

        @self._handle_tool_errors
        async def create_memory(content: str, name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
            """Create a memory note and return it as a dict."""
            ctx = self.mcp.get_context()
            ms = self._ms(ctx)
            meta = metadata.copy() if metadata else {}
            if name:
                meta["name"] = name
            note = await ms.create(content, **meta)
            return note.to_dict()

        @self._handle_tool_errors
        async def get_memory(memory_id: str):
            """Retrieve a memory by ID."""
            ctx = self.mcp.get_context()
            ms = self._ms(ctx)
            note = await ms.get(memory_id)
            if not note:
                return {"error": True, "message": "Memory not found", "id": memory_id}
            return note.to_dict()

        @self._handle_tool_errors
        async def update_memory(
            memory_id: str, content: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None
        ):
            """Update a memory's content and/or metadata."""
            if content is None and metadata is None:
                return {"error": True, "message": "Nothing to update", "id": memory_id}
            ctx = self.mcp.get_context()
            ms = self._ms(ctx)
            note = await ms.update(memory_id, content=content, **(metadata or {}))
            if not note:
                return {"error": True, "message": "Memory not found for update", "id": memory_id}
            return note.to_dict()

        @self._handle_tool_errors
        async def delete_memory(memory_id: str):
            """Delete a memory from the store."""
            ctx = self.mcp.get_context()
            ms = self._ms(ctx)
            success = await ms.delete(memory_id)
            return {"success": success, "id": memory_id}

        @self._handle_tool_errors
        async def search_memories(query: str, top_k: int = 5, use_reranker: bool = True):
            """Semantic search over stored memories (returns top-k)."""
            ctx = self.mcp.get_context()
            ms = self._ms(ctx)
            results = await ms.search(query, k=top_k, use_reranker=use_reranker)
            return {
                "count": len(results),
                "memories": [note.to_dict() for note in results],
            }

        @self._handle_tool_errors
        async def get_all_memories():
            """Return **all** memories currently stored."""
            ctx = self.mcp.get_context()
            ms = self._ms(ctx)
            try:
                notes = await ms.get_all_notes_from_store()
                return {
                    "count": len(notes),
                    "memories": [note.to_dict() for note in notes],
                }
            except Exception as e:
                self.logger.error(f"Error in get_all_memories tool: {e}", exc_info=True)
                return {"error": True, "message": f"Failed to retrieve all memories: {str(e)}"}

        @self._handle_tool_errors
        async def get_memory_stats():
            """Return summary statistics (total, average length, etc.)."""
            ctx = self.mcp.get_context()
            ms = self._ms(ctx)
            try:
                return await ms.get_stats()
            except Exception as e:
                self.logger.error(f"Error in get_memory_stats tool: {e}", exc_info=True)
                return {"error": True, "message": f"Failed to get memory stats: {str(e)}"}

        # Add tools using the decorated functions
        self.mcp.add_tool(health_check)
        self.mcp.add_tool(create_memory)
        self.mcp.add_tool(get_memory)
        self.mcp.add_tool(update_memory)
        self.mcp.add_tool(delete_memory)
        self.mcp.add_tool(search_memories)
        self.mcp.add_tool(get_all_memories)
        self.mcp.add_tool(get_memory_stats)

    # ------------------------- init / run -------------------------- #
    def init(self):
        """Create FastMCP instance, register tools, and mount under Starlette."""
        if self.mcp:  # already initialised
            return self
        self.logger.info(f"Initializing FastMCP server '{self.cfg.name}'...")
        self.mcp = FastMCP(self.cfg.name)
        self._register_tools()
        # Mount the MCP SSE app under a Starlette app with lifecycle handlers
        self.app = Starlette(
            debug=self.cfg.debug_monitor,
            routes=[Mount("/", app=self.mcp.sse_app())],
            on_startup=[self._on_startup],
            on_shutdown=[self._on_shutdown],
        )
        self.logger.info("Server initialization complete.")
        return self

    def run(self):
        """Run the server with uvicorn."""
        if not self.app:
            self.init()
        self.logger.info(f"Starting Uvicorn server at http://{self.cfg.host}:{self.cfg.port}")
        uvicorn.run(
            self.app,
            host=self.cfg.host,
            port=self.cfg.port,
            log_level=self.cfg.log_level,
        )
