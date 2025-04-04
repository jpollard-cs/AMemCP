#!/usr/bin/env python3
"""
Comprehensive test script for the A-Mem MCP client.
Tests tools and prompts functionality via network connection.
"""

import asyncio
import json  # For potentially parsing results if needed
import logging
import sys
from pathlib import Path

# Add the project root directory to the path to find 'amem'
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from amem.mcp_client import AMemMCPClient

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def test_tools(client: AMemMCPClient):
    """Test all available tools."""
    logger.info("\n--- TESTING TOOLS ---")

    created_id = None
    try:
        # Create a memory
        logger.info("Creating a memory...")
        memory = await asyncio.wait_for(
            client.create_memory("This is a test memory for tools testing", metadata={"source": "test_client"}),
            timeout=10.0,  # Increased timeout for potential LLM calls
        )
        logger.info(f"Created memory: {memory}")
        created_id = memory.get("id")
        assert created_id is not None, "Memory creation failed or did not return ID"

        # Get the memory
        logger.info("Retrieving the memory...")
        retrieved = await asyncio.wait_for(client.get_memory(created_id), timeout=5.0)
        logger.info(f"Retrieved memory: {retrieved}")
        assert retrieved.get("id") == created_id
        assert retrieved.get("content") == "This is a test memory for tools testing"

        # Update the memory
        logger.info("Updating the memory...")
        updated = await asyncio.wait_for(
            client.update_memory(created_id, "This is an updated test memory content", metadata={"status": "updated"}),
            timeout=10.0,  # Increased timeout
        )
        logger.info(f"Updated memory: {updated}")
        assert updated.get("id") == created_id
        assert updated.get("content") == "This is an updated test memory content"
        assert updated.get("metadata", {}).get("status") == "updated"

        # Get stats
        logger.info("Getting memory system stats...")
        stats = await asyncio.wait_for(client.get_memory_stats(), timeout=5.0)  # Use renamed tool method
        logger.info(f"Memory system stats: {stats}")
        assert "total_memories" in stats

        # Search for the memory
        logger.info("Searching for the memory...")
        results = await asyncio.wait_for(
            client.search_memories("updated test", top_k=3, use_reranker=True),
            timeout=15.0,  # Increased timeout for search+rerank
        )
        logger.info(f"Search results: {results}")
        assert isinstance(results, list)
        # Check if the updated memory is in the results
        found = any(item.get("id") == created_id for item in results)
        assert found, "Updated memory not found in search results"

        # Get all memories tool
        logger.info("Getting all memories via tool...")
        all_mems_result = await asyncio.wait_for(client.get_all_memories(), timeout=5.0)  # Use tool method
        logger.info(f"All memories tool result: {all_mems_result}")
        assert "count" in all_mems_result
        assert "memories" in all_mems_result
        assert isinstance(all_mems_result["memories"], list)
        assert any(
            m.get("id") == created_id for m in all_mems_result["memories"]
        ), "Created memory not found in all memories list"

    except Exception as e:
        logger.error(f"Error during tool testing: {e}", exc_info=True)
        raise  # Re-raise exception to fail the test
    finally:
        # Delete the memory (if created)
        if created_id:
            logger.info(f"Deleting the test memory ({created_id})...")
            try:
                deleted = await asyncio.wait_for(client.delete_memory(created_id), timeout=5.0)
                logger.info(f"Deleted memory: {deleted}")
                assert deleted.get("success", False) is True
            except Exception as del_e:
                logger.error(f"Error deleting memory {created_id}: {del_e}")
                # Don't fail the whole test suite for cleanup failure, just log it


# Resource testing section removed as resources are no longer used directly
# async def test_resources(client: AMemMCPClient):
#    ... removed ...


async def test_prompts(client: AMemMCPClient):
    """Test all available prompts."""
    logger.info("\n--- TESTING PROMPTS ---")

    created_id = None
    try:
        # Create a test memory for prompts
        logger.info("Creating a test memory for prompts...")
        memory = await client.create_memory("This is a test memory for prompts testing")
        logger.info(f"Created memory: {memory}")
        created_id = memory.get("id")
        assert created_id is not None

        # Get create memory prompt
        logger.info("Getting create memory prompt...")
        create_prompt_text = await asyncio.wait_for(
            client.get_create_memory_prompt("New memory content", "Prompt Test Name"), timeout=5.0
        )
        logger.info(f"Create memory prompt text:\n{create_prompt_text}")
        assert "New memory content" in create_prompt_text
        assert "Prompt Test Name" in create_prompt_text

        # Get search prompt
        logger.info("Getting search prompt...")
        search_prompt_text = await asyncio.wait_for(client.get_search_prompt("test query for prompt"), timeout=5.0)
        logger.info(f"Search prompt text:\n{search_prompt_text}")
        assert "test query for prompt" in search_prompt_text

        # Get summarize prompt
        logger.info("Getting summarize prompt...")
        summarize_prompt_messages = await asyncio.wait_for(client.get_summarize_prompt(created_id), timeout=5.0)
        logger.info(f"Summarize prompt messages:")
        assert isinstance(summarize_prompt_messages, list)
        assert len(summarize_prompt_messages) > 0
        for msg in summarize_prompt_messages:
            logger.info(f"  {msg.get('role')}: {msg.get('content')}")
            assert "role" in msg
            assert "content" in msg

    except Exception as e:
        logger.error(f"Error during prompt testing: {e}", exc_info=True)
        raise
    finally:
        # Clean up
        if created_id:
            logger.info(f"Deleting prompt test memory ({created_id})...")
            try:
                deleted = await asyncio.wait_for(client.delete_memory(created_id), timeout=5.0)
                logger.info(f"Deleted memory: {deleted}")
                assert deleted.get("success", False) is True
            except Exception as del_e:
                logger.error(f"Error deleting memory {created_id}: {del_e}")


async def main():
    """Run comprehensive tests for the A-Mem MCP client (Network)."""
    logger.info("Starting A-Mem MCP client comprehensive test (Network)")

    # Create a client - reads URL from .env file
    client = AMemMCPClient()

    try:
        # Connect to the server with a timeout
        logger.info("Connecting to server...")
        connected = await asyncio.wait_for(client.connect(), timeout=10.0)
        if not connected:
            logger.error("Connection failed - server not responding or connection rejected.")
            return
        logger.info("Connected to server successfully")

        # Start MCP session
        logger.info("Starting MCP session...")
        mcp_started = await asyncio.wait_for(client.start_mcp_session(), timeout=10.0)
        if not mcp_started:
            logger.error("MCP session initialization failed.")
            return
        logger.info("MCP session started successfully")

        # Test tools functionality
        await test_tools(client)

        # Test prompts functionality
        await test_prompts(client)

        logger.info("\nAll tests completed successfully")
    except asyncio.TimeoutError:
        logger.error("Connection timeout - server not responding")
    except ConnectionError as e:
        logger.error(f"Connection error during tests: {e}")
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
    finally:
        # Disconnect from the server
        logger.info("Disconnecting...")
        try:
            await client.disconnect()
            logger.info("Disconnected from server")
        except Exception as e:
            logger.error(f"Error disconnecting: {e}")


if __name__ == "__main__":
    try:
        # Make sure the client .env file with MCP_SERVER_URL exists
        if not Path(".env").exists():
            print(
                "Client .env file not found. Please create it with MCP_SERVER_URL=ws://localhost:8000 (or appropriate URL)"
            )
        else:
            asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Unhandled exception in test runner: {e}", exc_info=True)
