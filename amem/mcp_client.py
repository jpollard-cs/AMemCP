#!/usr/bin/env python3
"""
A-Mem MCP Client implementation using the Model Context Protocol Python SDK.
Connects to the MCP server via HTTP/SSE transport.
"""

import asyncio
import logging
import os
import uuid
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client

logger = logging.getLogger(__name__)


class AMemMCPClient:
    """Client for the A-Mem MCP Server (Network Connection - SSE)."""

    def __init__(self, server_url: Optional[str] = None, user_id: Optional[str] = None):
        """Initialize the client.

        Args:
            server_url: The HTTP URL of the MCP server endpoint (e.g., http://localhost:8000/mcp).
                        Reads from MCP_SERVER_URL environment variable if not provided.
            user_id: User ID to include in requests. Reads from MCP_USER_ID env var if not provided.
        """
        load_dotenv()

        # Default to http root URL with /mcp path
        default_url = "http://localhost:8000/mcp"
        raw_url = server_url or os.getenv("MCP_SERVER_URL", default_url)

        # Get user ID from param or env
        self.user_id = user_id or os.getenv("MCP_USER_ID", f"user-{uuid.uuid4().hex[:8]}")

        # Ensure it uses http/https
        if raw_url.startswith("ws"):
            raw_url = raw_url.replace("ws://", "http://", 1).replace("wss://", "https://", 1)

        # Ensure it ends with a trailing slash if needed
        if not raw_url.endswith("/"):
            raw_url = raw_url + "/"

        self.base_url = raw_url
        self.session = None
        self._connected = False
        self._client_context = None
        logger.info(f"AMemMCPClient configured for base URL: {self.base_url}, user_id: {self.user_id}")

    async def connect(self) -> bool:
        """Connect to the MCP server using SSE transport and initialize the session."""
        if self._connected:
            logger.info("Already connected")
            return True

        logger.info(f"Connecting to MCP server at: {self.base_url}")
        try:
            # Use the MCP library's SSE client function directly
            self._client_context = sse_client(f"{self.base_url}sse", f"{self.base_url}message")

            # Enter the context manager to get reader and writer streams
            logger.info("Initializing SSE connection...")
            streams = await self._client_context.__aenter__()

            # Create the MCP client session
            logger.info("Creating MCP client session...")
            self.session = ClientSession(streams[0], streams[1])

            # Initialize the session
            logger.info("Initializing MCP session...")
            init_options = await self.session.initialize()
            logger.info(
                f"MCP Session Initialized: Name={init_options.server_name}, Version={init_options.server_version}"
            )

            # Get available tools and prompts
            tools = await self.session.list_tools()
            prompts = await self.session.list_prompts()
            logger.info(f"Available tools: {', '.join(tool.name for tool in tools)}")
            logger.info(f"Available prompts: {', '.join(prompt.name for prompt in prompts)}")

            self._connected = True
            logger.info("MCP client session fully initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}", exc_info=True)
            await self.disconnect()
            return False

    async def disconnect(self) -> bool:
        """Disconnect from the MCP server."""
        logger.info("Disconnecting...")

        # First clean up the session
        self.session = None
        self._connected = False

        # Close the client context if it exists
        if self._client_context:
            try:
                await self._client_context.__aexit__(None, None, None)
                logger.info("Closed SSE client context")
            except Exception as e:
                logger.error(f"Error closing SSE client context: {e}")
            finally:
                self._client_context = None

        logger.info("Disconnected from MCP server")
        return True

    async def _ensure_connected(self) -> None:
        """Ensure the MCP session is initialized."""
        if not self._connected or not self.session:
            connected = await self.connect()
            if not connected:
                raise ConnectionError(f"Failed to connect and initialize MCP session")

    # --------------------
    # Tool methods
    # --------------------

    async def create_memory(
        self, content: str, name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a new memory note.

        Args:
            content: The content of the memory.
            name: Optional name/title for the memory.
            metadata: Optional dictionary of additional metadata fields.

        Returns:
            A dictionary representation of the created memory note.
        """
        await self._ensure_connected()
        params = {"content": content}
        if name is not None:
            params["name"] = name
        if metadata is not None:
            params["metadata"] = metadata
        result = await self.session.call_tool("create_memory", params)
        return result

    async def get_memory(self, memory_id: str) -> Dict[str, Any]:
        """Retrieve a memory note by ID.

        Args:
            memory_id: The ID of the memory to retrieve.

        Returns:
            A dictionary representation of the memory note.
        """
        await self._ensure_connected()
        result = await self.session.call_tool("get_memory", {"memory_id": memory_id})
        return result

    async def update_memory(
        self, memory_id: str, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Update an existing memory note.

        Args:
            memory_id: The ID of the memory to update.
            content: The new content for the memory.
            metadata: Optional dictionary of metadata fields to update or add.

        Returns:
            A dictionary representation of the updated memory note.
        """
        await self._ensure_connected()
        params = {"memory_id": memory_id, "content": content}
        if metadata is not None:
            params["metadata"] = metadata
        result = await self.session.call_tool("update_memory", params)
        return result

    async def delete_memory(self, memory_id: str) -> Dict[str, bool]:
        """Delete a memory note.

        Args:
            memory_id: The ID of the memory to delete.

        Returns:
            Dictionary indicating success or failure.
        """
        await self._ensure_connected()
        result = await self.session.call_tool("delete_memory", {"memory_id": memory_id})
        return result

    async def search_memories(self, query: str, top_k: int = 5, use_reranker: bool = True) -> List[Dict[str, Any]]:
        """Search for memory notes with optional reranking.

        Args:
            query: The search query.
            top_k: Maximum number of results (default: 5).
            use_reranker: Whether to use the reranker (default: True).

        Returns:
            List of matching memory notes as dictionaries.
        """
        await self._ensure_connected()
        result = await self.session.call_tool(
            "search_memories", {"query": query, "top_k": top_k, "use_reranker": use_reranker}
        )
        return result

    async def get_all_memories(self) -> Dict[str, Any]:
        """Get all memories as a dictionary containing a list.

        Returns:
            Dictionary with count and list of memories.
        """
        await self._ensure_connected()
        result = await self.session.call_tool("get_all_memories", {})
        return result

    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory system.

        Returns:
            Statistics including total count and average content length.
        """
        await self._ensure_connected()
        result = await self.session.call_tool("get_memory_stats", {})
        return result

    # --------------------
    # Prompt methods
    # --------------------

    async def get_create_memory_prompt(self, content: str, name: Optional[str] = None) -> str:
        """Get a prompt for creating a new memory."""
        await self._ensure_connected()
        params = {"content": content}
        if name is not None:
            params["name"] = name
        result = await self.session.get_prompt("create_memory_prompt", params)
        return result

    async def get_search_prompt(self, query: str) -> str:
        """Get a prompt for searching memories."""
        await self._ensure_connected()
        result = await self.session.get_prompt("search_memories_prompt", {"query": query})
        return result

    async def get_summarize_prompt(self, memory_id: str) -> List[Dict[str, str]]:
        """Get a prompt for summarizing a memory."""
        await self._ensure_connected()
        result = await self.session.get_prompt("summarize_memory", {"memory_id": memory_id})
        return result


async def main():
    """Test Function"""
    url = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp")
    user_id = os.getenv("MCP_USER_ID", f"test-user-{uuid.uuid4().hex[:8]}")

    print(f"Connecting to MCP server at: {url}")
    print(f"Using user ID: {user_id}")

    client = AMemMCPClient(server_url=url, user_id=user_id)
    try:
        # Connect and initialize MCP session
        print("Attempting to connect...")
        connected = await client.connect()
        if not connected:
            print("Connection and session initialization failed. Exiting.")
            return

        print("MCP Connection successful!")

        # Run some example tool calls
        print("\n--- Testing Tool Calls ---")
        stats = await client.get_memory_stats()
        print(f"Initial Server Stats: {stats}")

        print("\n--- Creating a memory ---")
        created = await client.create_memory("My first memory from MCP client!", name="MCP Test")
        print(f"Created Memory: {created}")
        mem_id = created.get("id")

        if mem_id:
            print(f"\n--- Getting memory {mem_id} ---")
            retrieved = await client.get_memory(mem_id)
            print(f"Retrieved Memory: {retrieved}")

            print("\n--- Searching for memories ---")
            results = await client.search_memories("memory")
            print(f"Search Results: {results}")

            print(f"\n--- Updating memory {mem_id} ---")
            updated = await client.update_memory(mem_id, "Updated memory content via MCP")
            print(f"Updated Memory: {updated}")

            print(f"\n--- Deleting memory {mem_id} ---")
            deleted = await client.delete_memory(mem_id)
            print(f"Delete Result: {deleted}")
        else:
            print("Memory ID not found after creation, skipping further tests.")

    except Exception as e:
        print(f"Error during testing: {e}")
    finally:
        print("\n--- Disconnecting ---")
        await client.disconnect()


if __name__ == "__main__":
    """Run test client function"""
    asyncio.run(main())
