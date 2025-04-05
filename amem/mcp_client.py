#!/usr/bin/env python3
"""
A-Mem MCP Client implementation using the Model Context Protocol Python SDK.
Connects to the MCP server via HTTP/SSE transport.
"""

import asyncio
import os
import time
import uuid
from typing import Any, Dict

# from mcp.client.sse import sse_client # Revert: Remove standard client
import aiohttp
from dotenv import load_dotenv
from mcp.client.session import ClientSession

from amem.custom_sse_client import custom_sse_client  # Re-add custom client
from amem.utils.utils import setup_logger

# Set up logger
logger = setup_logger(__name__)


class AMemMCPClient:
    """Client for the A-Mem MCP Server (Network Connection - SSE)."""

    def __init__(
        self,
        server_url: str = None,
        user_id: str = None,
    ):
        """Initialize the A-Mem MCP Client."""
        load_dotenv()  # Load environment variables from .env file if present
        self.logger = setup_logger(__name__)

        # Get the server URL, default to http://localhost:8000
        raw_url = server_url or os.getenv("MCP_SERVER_URL", "http://localhost:8000")

        # Ensure it uses http/https prefix
        if raw_url.startswith("ws"):
            raw_url = raw_url.replace("ws://", "http://", 1).replace("wss://", "https://", 1)
        elif not raw_url.startswith(("http://", "https://")):
            raw_url = "http://" + raw_url  # Assume http if no prefix

        # Ensure it ends with a trailing slash and remove any extra paths
        from urllib.parse import urlparse, urlunparse

        parsed_url = urlparse(raw_url)
        # Use the server root as the base URL
        self.base_url = urlunparse((parsed_url.scheme, parsed_url.netloc, "/", "", "", ""))
        # Health check URL is derived directly from base_url
        self.health_check_url = f"{self.base_url}healthcheck"

        self.user_id = user_id or os.getenv("MCP_USER_ID", "user-1")
        self.session = None
        self._connected = False
        self._client_context = None  # Will hold the custom client async generator
        self.logger.info(f"AMemMCPClient configured. Base URL: {self.base_url}, User ID: {self.user_id}")

    async def connect(self) -> bool:
        """Connect to the MCP server using the custom SSE transport."""
        if self._connected:
            self.logger.info("Already connected")
            return True

        self.logger.info(f"Connecting to MCP server at: {self.base_url} using CUSTOM sse_client")
        try:
            # First check if server is ready using a simple HTTP request
            self.logger.debug("Calling _wait_for_server_ready...")
            if not await self._wait_for_server_ready(timeout=15):
                self.logger.error("Server did not respond to basic health check ping")
                return False
            self.logger.debug("_wait_for_server_ready check passed.")

            # Use our custom SSE client implementation
            # Construct URLs relative to the base_url (root)
            sse_url = f"{self.base_url}sse"
            message_url = f"{self.base_url}message"

            self.logger.info(f"Initializing CUSTOM SSE connection to {sse_url} (message: {message_url})...")
            # Our custom client returns an async generator
            self._client_context = custom_sse_client(sse_url, message_url)
            self.logger.debug("Custom SSE client generator created. Getting reader/writer...")
            reader, writer = await anext(self._client_context)
            self.logger.debug("Custom SSE Reader/Writer obtained.")

            # Create the MCP client session
            self.logger.info("Creating MCP client session...")
            self.session = ClientSession(reader, writer)
            self.logger.debug("ClientSession created.")

            # Initialize the session
            self.logger.info("Initializing MCP session...")
            self.logger.debug("Calling session.initialize()...")
            init_options = await self.session.initialize()
            self.logger.debug(f"session.initialize() completed. Options: {init_options}")
            self.logger.info(
                f"MCP Session Initialized: Name={init_options.server_name}, Version={init_options.server_version}"
            )

            # Get available tools and prompts
            self.logger.debug("Listing tools...")
            tools = await self.session.list_tools()
            self.logger.debug(f"Tools listed: {len(tools)}")
            self.logger.debug("Listing prompts...")
            prompts = await self.session.list_prompts()
            self.logger.debug(f"Prompts listed: {len(prompts)}")

            self.logger.info(f"Available tools: {', '.join(tool.name for tool in tools)}")
            self.logger.info(f"Available prompts: {', '.join(prompt.name for prompt in prompts)}")

            self._connected = True
            self.logger.info("MCP client session fully initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to connect to MCP server: {e}", exc_info=True)
            await self.disconnect()
            return False

    async def disconnect(self) -> bool:
        """Disconnect from the MCP server."""
        self.logger.info("Disconnecting...")

        # First clean up the session
        self.session = None
        self._connected = False

        # Close the custom client context (async generator)
        if self._client_context:
            try:
                await self._client_context.aclose()  # Use aclose() for async generator
                self.logger.info("Closed custom SSE client context")
            except Exception as e:
                self.logger.error(f"Error closing custom SSE client context: {e}")
            finally:
                self._client_context = None

        self.logger.info("Disconnected from MCP server")
        return True

    # --------------------
    # Tool methods
    # --------------------

    async def create_memory(self, content: str, name: str = None, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a new memory note.

        Args:
            content: The content of the memory (required)
            name: Optional name for the memory
            metadata: Optional metadata dictionary

        Returns:
            The created memory note's data as a dictionary
        """
        if not self._connected:
            self.logger.error("Not connected to MCP server")
            return None

        params = {"content": content}
        if name:
            params["name"] = name
        if metadata:
            params["metadata"] = metadata

        try:
            self.logger.info(f"Creating memory with content: '{content[:50]}...'")
            result = await self.session.execute_tool("amem_create_memory", params)
            self.logger.debug(f"Create memory result: {result}")
            return result or {}
        except Exception as e:
            self.logger.error(f"Error creating memory: {e}")
            return {"error": str(e)}

    async def get_memory(self, memory_id: str) -> Dict[str, Any]:
        """Retrieve a memory by ID.

        Args:
            memory_id: The ID of the memory to retrieve

        Returns:
            The memory note data as a dictionary
        """
        if not self._connected:
            self.logger.error("Not connected to MCP server")
            return None

        try:
            self.logger.info(f"Retrieving memory with ID: {memory_id}")
            result = await self.session.execute_tool("amem_get_memory", {"memory_id": memory_id})
            self.logger.debug(f"Get memory result: {result}")
            return result or {}
        except Exception as e:
            self.logger.error(f"Error retrieving memory: {e}")
            return {"error": str(e)}

    async def update_memory(
        self,
        memory_id: str,
        content: str = None,
        metadata: Dict[str, Any] = None,
        name: str = None,
    ) -> Dict[str, Any]:
        """Update an existing memory.

        Args:
            memory_id: The ID of the memory to update
            content: Optional new content
            metadata: Optional new metadata
            name: Optional new name

        Returns:
            The updated memory note data
        """
        if not self._connected:
            self.logger.error("Not connected to MCP server")
            return None

        params = {"memory_id": memory_id}
        if content is not None:
            params["content"] = content
        if metadata is not None:
            params["metadata"] = metadata
        if name is not None:
            params["name"] = name

        try:
            self.logger.info(f"Updating memory with ID: {memory_id}")
            result = await self.session.execute_tool("amem_update_memory", params)
            self.logger.debug(f"Update memory result: {result}")
            return result or {}
        except Exception as e:
            self.logger.error(f"Error updating memory: {e}")
            return {"error": str(e)}

    async def delete_memory(self, memory_id: str) -> Dict[str, Any]:
        """Delete a memory by ID.

        Args:
            memory_id: The ID of the memory to delete

        Returns:
            Status of the deletion operation
        """
        if not self._connected:
            self.logger.error("Not connected to MCP server")
            return None

        try:
            self.logger.info(f"Deleting memory with ID: {memory_id}")
            result = await self.session.execute_tool("amem_delete_memory", {"memory_id": memory_id})
            self.logger.debug(f"Delete memory result: {result}")
            return result or {}
        except Exception as e:
            self.logger.error(f"Error deleting memory: {e}")
            return {"error": str(e)}

    async def search_memories(
        self, query: str, top_k: int = 5, metadata_filter: Dict[str, Any] = None, use_embeddings: bool = True
    ) -> Dict[str, Any]:
        """Search for memories using a text query.

        Args:
            query: The search query text
            top_k: Maximum number of results to return
            metadata_filter: Optional metadata filters
            use_embeddings: Whether to use embeddings for search (set to False if embeddings not configured)

        Returns:
            Search results containing matching memories
        """
        if not self._connected:
            self.logger.error("Not connected to MCP server")
            return None

        params = {"query": query, "top_k": top_k, "use_embeddings": use_embeddings}
        if metadata_filter:
            params["metadata_filter"] = metadata_filter

        try:
            self.logger.info(
                f"Searching memories with query: '{query}', top_k={top_k}, use_embeddings={use_embeddings}"
            )
            result = await self.session.execute_tool("amem_search_memories", params)
            self.logger.debug(f"Search memory result: {result}")
            return result or {"count": 0, "memories": []}
        except Exception as e:
            self.logger.error(f"Error searching memories: {e}")
            return {"error": str(e), "count": 0, "memories": []}

    async def get_all_memories(self) -> Dict[str, Any]:
        """Get all memory notes.

        Returns:
            Dictionary with count and array of memories
        """
        if not self._connected:
            self.logger.error("Not connected to MCP server")
            return None

        try:
            self.logger.info("Retrieving all memories")
            result = await self.session.execute_tool("amem_get_all_memories", {})
            self.logger.debug(f"Get all memories result: {result}")
            return result or {"count": 0, "memories": []}
        except Exception as e:
            self.logger.error(f"Error retrieving all memories: {e}")
            return {"error": str(e), "count": 0, "memories": []}

    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics.

        Returns:
            Dictionary with memory stats
        """
        if not self._connected:
            self.logger.error("Not connected to MCP server")
            return None

        try:
            self.logger.info("Retrieving memory stats")
            result = await self.session.execute_tool("amem_get_memory_stats", {})
            self.logger.debug(f"Get memory stats result: {result}")
            return result or {}
        except Exception as e:
            self.logger.error(f"Error retrieving memory stats: {e}")
            return {"error": str(e)}

    # --------------------
    # Prompt methods
    # --------------------

    async def get_create_memory_prompt(self, content: str) -> str:
        """Get the prompt for creating a memory.

        Args:
            content: The content to include in the prompt

        Returns:
            The formatted prompt text
        """
        if not self._connected:
            self.logger.error("Not connected to MCP server")
            return ""

        try:
            self.logger.info(f"Getting create memory prompt for: '{content[:50]}...'")
            result = await self.session.execute_prompt("amem_create_memory_prompt", {"content": content})
            return result.response or ""
        except Exception as e:
            self.logger.error(f"Error getting create memory prompt: {e}")
            return ""

    async def get_search_prompt(self, query: str) -> str:
        """Get the prompt for searching memories.

        Args:
            query: The search query

        Returns:
            The formatted prompt text
        """
        if not self._connected:
            self.logger.error("Not connected to MCP server")
            return ""

        try:
            self.logger.info(f"Getting search prompt for: '{query}'")
            result = await self.session.execute_prompt("amem_search_prompt", {"query": query})
            return result.response or ""
        except Exception as e:
            self.logger.error(f"Error getting search prompt: {e}")
            return ""

    async def get_summarize_prompt(self, memory_id: str) -> str:
        """Get the prompt for summarizing a memory.

        Args:
            memory_id: The ID of the memory to summarize

        Returns:
            The formatted prompt text
        """
        if not self._connected:
            self.logger.error("Not connected to MCP server")
            return ""

        try:
            self.logger.info(f"Getting summarize prompt for memory: '{memory_id}'")
            result = await self.session.execute_prompt("amem_summarize_prompt", {"memory_id": memory_id})
            return result.response or ""
        except Exception as e:
            self.logger.error(f"Error getting summarize prompt: {e}")
            return ""

    async def update_metadata(self, memory_id: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Update the metadata for a memory.

        Args:
            memory_id: The ID of the memory to update
            metadata: The new metadata dictionary

        Returns:
            The updated memory data
        """
        if not self._connected:
            self.logger.error("Not connected to MCP server")
            return None

        try:
            self.logger.info(f"Updating metadata for memory: '{memory_id}'")
            result = await self.session.execute_tool(
                "amem_update_memory", {"memory_id": memory_id, "metadata": metadata}
            )
            self.logger.debug(f"Update metadata result: {result}")
            return result or {}
        except Exception as e:
            self.logger.error(f"Error updating metadata: {e}")
            return {"error": str(e)}

    # --------------------
    # Private helper methods
    # --------------------

    def __del__(self):
        """Clean up resources on deletion."""
        if hasattr(self, "_connected") and self._connected:
            # Schedule disconnect using asyncio, but don't wait for it
            self.logger.info("Client being destroyed, scheduling disconnect...")
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.disconnect())
            except Exception as e:
                self.logger.error(f"Error during cleanup: {e}")

    async def _wait_for_server_ready(self, timeout: int = 15) -> bool:
        """Wait for the server process to be responding to HTTP requests."""
        start_time = time.time()
        # Use the dedicated health check URL
        health_url = self.health_check_url
        self.logger.info(f"Checking if MCP server is responding at: {health_url} (will wait up to {timeout}s)")
        request_timeout = aiohttp.ClientTimeout(total=3)  # Timeout for individual GET requests

        async with aiohttp.ClientSession() as session:  # Create session outside the loop
            while time.time() - start_time < timeout:
                try:
                    self.logger.debug(f"Pinging health endpoint: {health_url}")
                    async with session.get(health_url, timeout=request_timeout) as response:  # Apply request timeout
                        if response.status == 200:
                            self.logger.info(
                                f"Server responded to health check (Status: {response.status}). Assuming server process is up."
                            )
                            return True
                        else:
                            self.logger.warning(f"Server responded to health check with status {response.status}")
                except asyncio.TimeoutError:
                    self.logger.debug(f"Health check ping timed out after {request_timeout.total}s.")
                except aiohttp.ClientConnectorError as e:
                    self.logger.warning(f"Connection error during health check ping: {e}")
                except Exception as e:
                    self.logger.debug(f"Error during health check ping: {e}")

                # Wait before next check, unless total timeout exceeded
                if time.time() - start_time < timeout:
                    self.logger.debug(f"Waiting 0.5s before next health check ping...")
                    await asyncio.sleep(0.5)

        self.logger.error(f"Server did not respond to health check ping after {timeout} seconds")
        return False


async def main():
    """Test Function"""
    url = os.getenv("MCP_SERVER_URL", "http://localhost:8000")
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
