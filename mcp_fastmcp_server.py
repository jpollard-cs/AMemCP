#!/usr/bin/env python3
"""
A-Mem MCP Server implementation using FastMCP from the Model Context Protocol Python SDK.
Handles initialization of the real Agentic Memory System.
"""

import json  # Import json for resource formatting
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.prompts import base
from starlette.applications import Starlette  # Import Starlette
from starlette.responses import JSONResponse, RedirectResponse  # Import response types
from starlette.routing import Mount, Route  # Import Mount and Route

# Import the *real* memory system
from memory_system import AgenticMemorySystem
from utils import setup_logger

# Import our SSE helper


print("--- mcp_fastmcp_server.py module execution started ---")  # Top-level diagnostic print

# Set up logger
logger = setup_logger(__name__, level=logging.DEBUG)


@asynccontextmanager
async def starlette_lifespan(app: Starlette) -> AsyncIterator[dict]:
    """Manage application lifecycle via Starlette, initializing AgenticMemorySystem."""
    # Track initialization
    initialization_complete = False
    memory_system = None

    try:
        # Initialize on startup - Fetch all config from environment variables
        project = os.getenv("PROJECT_NAME", "test_amemcp")  # Default to test_amemcp
        llm_backend = os.getenv("LLM_BACKEND", "gemini")
        llm_model = os.getenv("LLM_MODEL")
        embed_model = os.getenv("EMBED_MODEL")
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
        persist_directory = os.getenv("PERSIST_DIRECTORY")
        reranker_model = os.getenv("JINA_RERANKER_MODEL")
        jina_api_key = os.getenv("JINA_API_KEY")

        logger.info(f"Starlette Lifespan: Initializing REAL Agentic Memory System for project: {project}")

        # Create memory system
        memory_system = AgenticMemorySystem(
            project_name=project,
            llm_backend=llm_backend,
            llm_model=llm_model,
            embed_model=embed_model,
            api_key=api_key,
            persist_directory=persist_directory,
            reranker_model=reranker_model,
            jina_api_key=jina_api_key,
        )

        # Initialize memory system with timeout
        logger.info("Initializing memory system...")
        await memory_system.initialize()

        # Set initialization flag
        initialization_complete = True
        logger.info("Starlette Lifespan: Real Memory system initialized successfully")

        # Both yield the context dictionary and use app.state
        yield {"memory_system": memory_system, "initialization_complete": initialization_complete}
    except Exception as e:
        logger.error(f"❌ Error during memory system initialization: {e}", exc_info=True)
        # Still yield but with error status
        yield {"memory_system": None, "initialization_complete": False, "error": str(e)}
    finally:
        # Cleanup on shutdown
        logger.info("Starlette Lifespan: Shutting down memory system")
        if memory_system is not None:
            # Add explicit cleanup here if needed
            # The __del__ method should handle the ChromaDB server shutdown
            pass


# Create the MCP server instance (without lifespan)
mcp = FastMCP("AMemMCP", lifespan=starlette_lifespan)

# ------------------------
# Resources
# ------------------------

# TODO: add useful resources to enable clients to make the most of our MCP Server

# ------------------------
# Tools
# ------------------------


@mcp.tool()
async def create_memory(
    content: str, name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, ctx: Context = None
) -> Dict[str, Any]:
    """Create a new memory note in the Agentic Memory System.

    Creates a persistent memory note with the provided content and optional metadata.
    This memory can later be retrieved, updated, or searched.

    Args:
        content: The main text content of the memory to store. Required.
        name: Optional name/title for the memory. Helps with identification.
        metadata: Optional dictionary of additional metadata fields like tags, category, etc.

    Returns:
        A dictionary with the created memory, including generated ID and timestamps.

    Example:
        create_memory(
            content="The sky was particularly blue today with scattered clouds.",
            name="Sky observation",
            metadata={"tags": ["weather", "observation"], "importance": "medium"}
        )
    """

    logger.info("🧠 TOOL: create_memory - Starting memory creation")
    try:
        # Access yielded lifespan state dictionary
        lifespan_state = ctx.request_context.lifespan_context

        # Check if initialization completed successfully
        initialization_complete = lifespan_state.get("initialization_complete", False)
        if not initialization_complete:
            logger.error("❌ Memory system not fully initialized")
            return {
                "error": "Memory system not fully initialized yet. Please try again later.",
                "id": None,
                "content": content,
                "metadata": metadata,
                "created_at": None,
                "updated_at": None,
            }

        memory_system = lifespan_state.get("memory_system")
        logger.debug(f"Accessing memory system from lifespan context: {memory_system}")

        # Pass metadata correctly
        extra_metadata = metadata or {}

        # Truncate content for logging
        content_preview = content[:50] + "..." if len(content) > 50 else content
        logger.info(f"📝 Creating memory: '{content_preview}' with name: '{name}'")

        if not memory_system:
            logger.error("❌ Error accessing memory system from lifespan context")
            return {
                "error": "Memory system not found in lifespan context",
                "id": None,
                "content": content,
                "metadata": metadata,
                "created_at": None,
                "updated_at": None,
            }

        # Call the async method
        memory = await memory_system.create(content, name=name, **extra_metadata)
        logger.info(f"✅ Memory created successfully with ID: {memory.id}")
        return memory.to_dict()
    except Exception as e:
        logger.error(f"❌ Error creating memory: {str(e)}", exc_info=True)
        # Return an error structure compatible with expected dict output
        return {
            "error": str(e),
            "id": None,
            "content": content,
            "metadata": metadata,
            "created_at": None,
            "updated_at": None,
        }


@mcp.tool()
async def get_memory(memory_id: str, ctx: Context = None) -> Dict[str, Any]:
    """Retrieve a specific memory note by its ID.

    Fetches the complete memory record including all metadata and timestamps.

    Args:
        memory_id: The unique ID of the memory to retrieve. Required.

    Returns:
        A dictionary with the complete memory details if found, or an error message.

    Example:
        get_memory(memory_id="mem_12345")
    """
    logger.info(f"🔍 TOOL: get_memory - Retrieving memory with ID: {memory_id}")
    try:
        # Access yielded lifespan state dictionary
        lifespan_state = ctx.request_context.lifespan_context

        # Check if initialization completed successfully
        initialization_complete = lifespan_state.get("initialization_complete", False)
        if not initialization_complete:
            logger.error("❌ Memory system not fully initialized")
            return {"error": "Memory system not fully initialized yet. Please try again later."}

        memory_system = lifespan_state.get("memory_system")
        if not memory_system:
            logger.error("❌ Error accessing memory system from lifespan context")
            return {"error": "Memory system not available"}

        memory = await memory_system.get(memory_id)
        if memory:
            logger.info(f"✅ Memory found: {memory_id}")
            return memory.to_dict()
        else:
            logger.warning(f"⚠️ Memory not found: {memory_id}")
            return {"error": f"Memory not found: {memory_id}"}
    except Exception as e:
        logger.error(f"❌ Error retrieving memory: {str(e)}", exc_info=True)
        return {"error": str(e)}


@mcp.tool()
async def update_memory(
    memory_id: str, content: str, metadata: Optional[Dict[str, Any]] = None, ctx: Context = None
) -> Dict[str, Any]:
    """Update an existing memory note with new content or metadata.

    Modifies the content and/or metadata of an existing memory. The updated_at
    timestamp will be automatically updated to the current time.

    Args:
        memory_id: The unique ID of the memory to update. Required.
        content: The new text content for the memory. Required.
        metadata: Optional dictionary of metadata fields to update or add.

    Returns:
        A dictionary with the updated memory details, or an error message.

    Example:
        update_memory(
            memory_id="mem_12345",
            content="Updated observation: The sky turned gray later in the day.",
            metadata={"weather_changed": True}
        )
    """
    content_preview = content[:50] + "..." if len(content) > 50 else content
    logger.info(f"🔄 TOOL: update_memory - Updating memory {memory_id} with new content")
    logger.debug(f"New content preview: '{content_preview}'")

    try:
        # Access yielded lifespan state dictionary
        lifespan_state = ctx.request_context.lifespan_context

        # Check if initialization completed successfully
        initialization_complete = lifespan_state.get("initialization_complete", False)
        if not initialization_complete:
            logger.error("❌ Memory system not fully initialized")
            return {"error": "Memory system not fully initialized yet. Please try again later."}

        memory_system = lifespan_state.get("memory_system")
        if not memory_system:
            logger.error("❌ Error accessing memory system from lifespan context")
            return {"error": "Memory system not available", "id": memory_id}

        extra_metadata = metadata or {}
        memory = await memory_system.update(memory_id, content, **extra_metadata)
        if memory:
            logger.info(f"✅ Memory updated successfully: {memory_id}")
            return memory.to_dict()
        else:
            logger.warning(f"⚠️ Memory not found for update: {memory_id}")
            return {"error": f"Memory not found for update: {memory_id}"}
    except Exception as e:
        logger.error(f"❌ Error updating memory {memory_id}: {str(e)}", exc_info=True)
        return {"error": str(e), "id": memory_id}


@mcp.tool()
async def delete_memory(memory_id: str, ctx: Context = None) -> Dict[str, bool]:
    """Delete a memory note permanently from the system.

    Removes the specified memory from storage. This action cannot be undone.

    Args:
        memory_id: The unique ID of the memory to delete. Required.

    Returns:
        Dictionary with a success flag indicating whether deletion succeeded.

    Example:
        delete_memory(memory_id="mem_12345")
    """
    logger.info(f"🗑️ TOOL: delete_memory - Deleting memory with ID: {memory_id}")
    try:
        # Access yielded lifespan state dictionary
        lifespan_state = ctx.request_context.lifespan_context

        # Check if initialization completed successfully
        initialization_complete = lifespan_state.get("initialization_complete", False)
        if not initialization_complete:
            logger.error("❌ Memory system not fully initialized")
            return {"success": False, "error": "Memory system not fully initialized yet. Please try again later."}

        memory_system = lifespan_state.get("memory_system")
        if not memory_system:
            logger.error("❌ Error accessing memory system from lifespan context")
            return {"success": False, "error": "Memory system not available"}

        success = await memory_system.delete(memory_id)
        if success:
            logger.info(f"✅ Memory deleted successfully: {memory_id}")
        else:
            logger.warning(f"⚠️ Memory deletion failed: {memory_id}")
        return {"success": success}
    except Exception as e:
        logger.error(f"❌ Error deleting memory {memory_id}: {str(e)}", exc_info=True)
        return {"success": False, "error": str(e)}


@mcp.tool()
async def search_memories(
    query: str, top_k: int = 5, use_reranker: bool = True, ctx: Context = None
) -> List[Dict[str, Any]]:
    """Search for memory notes using semantic similarity.

    Performs a semantic search over all memories, finding those that match
    the meaning of your query rather than just keyword matching. Results are
    ranked by relevance.

    Args:
        query: The search query text to find relevant memories. Required.
        top_k: Maximum number of results to return (default: 5).
        use_reranker: Whether to use the reranker for improved ranking (default: True).

    Returns:
        List of matching memory notes as dictionaries, ordered by relevance.

    Example:
        search_memories(
            query="Observations about weather patterns",
            top_k=3,
            use_reranker=True
        )
    """
    logger.info(f"🔎 TOOL: search_memories - Searching with query: '{query}'")
    logger.debug(f"Search parameters: top_k={top_k}, use_reranker={use_reranker}")
    try:
        # Access yielded lifespan state dictionary
        lifespan_state = ctx.request_context.lifespan_context

        # Check if initialization completed successfully
        initialization_complete = lifespan_state.get("initialization_complete", False)
        if not initialization_complete:
            logger.error("❌ Memory system not fully initialized")
            return [{"error": "Memory system not fully initialized yet. Please try again later."}]

        memory_system = lifespan_state.get("memory_system")
        if not memory_system:
            logger.error("❌ Error accessing memory system from lifespan context")
            return [{"error": "Memory system not available"}]

        memories = await memory_system.search(query, k=top_k, use_reranker=use_reranker)
        logger.info(f"✅ Search complete. Found {len(memories)} results")
        return [m.to_dict() for m in memories]
    except Exception as e:
        logger.error(f"❌ Error searching memories for '{query}': {str(e)}", exc_info=True)
        return [{"error": str(e)}]


@mcp.tool()
async def get_all_memories(ctx: Context = None) -> Dict[str, Any]:
    """Retrieve all memories stored in the system.

    Returns a complete list of all memories in the system, with their
    content and metadata. Useful for overview and debugging.

    Args:
        No arguments required.

    Returns:
        Dictionary with count and list of all memories.

    Example:
        get_all_memories()
    """
    logger.info("📋 TOOL: get_all_memories - Retrieving all memories")
    try:
        # Access yielded lifespan state dictionary
        lifespan_state = ctx.request_context.lifespan_context

        # Check if initialization completed successfully
        initialization_complete = lifespan_state.get("initialization_complete", False)
        if not initialization_complete:
            logger.error("❌ Memory system not fully initialized")
            return {
                "count": 0,
                "memories": [],
                "error": "Memory system not fully initialized yet. Please try again later.",
            }

        memory_system = lifespan_state.get("memory_system")
        if not memory_system:
            logger.error("❌ Error accessing memory system from lifespan context")
            return {"count": 0, "memories": [], "error": "Memory system not available"}

        memories = await memory_system.get_all()
        mem_dicts = [m.to_dict() for m in memories]
        logger.info(f"✅ Retrieved {len(mem_dicts)} memories successfully")
        return {"count": len(mem_dicts), "memories": mem_dicts}
    except Exception as e:
        logger.error(f"❌ Error getting all memories: {str(e)}", exc_info=True)
        return {"count": 0, "memories": [], "error": str(e)}


@mcp.tool()
async def get_memory_stats(ctx: Context = None) -> Dict[str, Any]:
    """Get statistics about the memory system.

    Returns summary statistics about the memory system, such as
    total number of memories and average content length.

    Args:
        No arguments required.

    Returns:
        Dictionary with statistics about the memory system.

    Example:
        get_memory_stats()
    """
    logger.info("📊 TOOL: get_memory_stats - Getting memory system statistics")
    try:
        # Access yielded lifespan state dictionary
        lifespan_state = ctx.request_context.lifespan_context

        # Check if initialization completed successfully
        initialization_complete = lifespan_state.get("initialization_complete", False)
        if not initialization_complete:
            logger.error("❌ Memory system not fully initialized")
            return {
                "error": "Memory system not fully initialized yet. Please try again later.",
                "total_memories": 0,
                "average_content_length": 0.0,
            }

        memory_system = lifespan_state.get("memory_system")
        if not memory_system:
            logger.error("❌ Error accessing memory system from lifespan context")
            return {"error": "Memory system not available", "total_memories": 0, "average_content_length": 0.0}

        stats = await memory_system.get_stats()
        logger.info(f"✅ Retrieved memory stats: {stats}")
        return stats
    except Exception as e:
        logger.error(f"❌ Error getting memory stats: {str(e)}", exc_info=True)
        return {"error": str(e), "total_memories": 0, "average_content_length": 0.0}


# ------------------------
# Prompts (Keep existing or adapt as needed)
# ------------------------


@mcp.prompt()
def create_memory_prompt(content: str, name: Optional[str] = None) -> str:
    """Prompt to create a new memory."""
    prompt = "Please create a new memory with the following content:\n\n"
    prompt += f"{content}\n\n"

    if name:
        prompt += f"Name: {name}\n\n"

    prompt += "Analyze the content, generate appropriate metadata (keywords, summary), and confirm creation with the memory ID and generated metadata."
    return prompt


@mcp.prompt()
def search_memories_prompt(query: str) -> str:
    """Prompt to search for memories."""
    return f"Please search for memories matching the query: '{query}'. Use reranking for best results."


@mcp.prompt()
def summarize_memory(memory_id: str) -> List[base.Message]:
    """Prompt to summarize a memory's content."""
    return [
        base.UserMessage(f"Retrieve memory with ID: {memory_id} and provide a concise summary of its content."),
        base.UserMessage("Also include its key metadata like keywords and type."),
        base.AssistantMessage("Okay, I will retrieve the memory, summarize it, and include key metadata."),
    ]


# Root health check endpoint
async def health_check(request):
    """Simple health check endpoint at the root path."""
    # Check if memory system is initialized
    initialization_status = getattr(request.app.state, "initialization_complete", False)
    return JSONResponse(
        {
            "status": "ok" if initialization_status else "initializing",
            "service": "amem-mcp-server",
            "initialized": initialization_status,
        }
    )


# Detailed status endpoint
async def status_detail(request):
    """Detailed status endpoint with more information."""
    initialization_status = getattr(request.app.state, "initialization_complete", False)
    memory_system = getattr(request.app.state, "memory_system", None)

    response = {
        "status": "ok" if initialization_status else "initializing",
        "service": "amem-mcp-server",
        "initialized": initialization_status,
        "server_time": datetime.now().isoformat(),
    }

    # Add memory system stats if available
    if initialization_status and memory_system:
        try:
            stats = await memory_system.get_stats()
            response["memory_stats"] = stats
        except Exception as e:
            response["memory_stats_error"] = str(e)

    return JSONResponse(response)


# Redirect from root to healthcheck
async def root_redirect(request):
    """Redirect from root to healthcheck."""
    return RedirectResponse(url="/healthcheck")


# Create the Starlette application
print("Creating Starlette application with MCP SSE app...")
app = Starlette(
    routes=[
        Route("/healthcheck", health_check),
        Route("/status", status_detail),
        Mount("/", app=mcp.sse_app()),
    ]
)
print(f"Starlette application created with routes: /, /healthcheck, /status, /mcp/sse, /mcp/message")

# if __name__ == "__main__":
#     # This runs the server in stdio mode if executed directly
#     logger.info(f"Starting A-Mem MCP Server in stdio mode (use Uvicorn for network access)")
#     mcp.run()
