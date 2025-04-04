#!/usr/bin/env python3
"""
Comprehensive test script for the A-Mem MCP client.
Tests tools, resources, and prompts functionality.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from amem.mcp_client import AMemMCPClient

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def test_tools(client):
    """Test all available tools."""
    logger.info("\n--- TESTING TOOLS ---")

    # Create a memory
    logger.info("Creating a memory...")
    memory = await asyncio.wait_for(client.create("This is a test memory for tools testing"), timeout=5.0)
    logger.info(f"Created memory: {memory}")

    # Get the memory
    logger.info("Retrieving the memory...")
    retrieved = await asyncio.wait_for(client.get(memory["id"]), timeout=5.0)
    logger.info(f"Retrieved memory: {retrieved}")

    # Update the memory
    logger.info("Updating the memory...")
    updated = await asyncio.wait_for(client.update(memory["id"], "This is an updated test memory content"), timeout=5.0)
    logger.info(f"Updated memory: {updated}")

    # Get stats
    logger.info("Getting memory system stats...")
    stats = await asyncio.wait_for(client.get_stats(), timeout=5.0)
    logger.info(f"Memory system stats: {stats}")

    # Search for the memory
    logger.info("Searching for the memory...")
    results = await asyncio.wait_for(client.search("updated test"), timeout=5.0)
    logger.info(f"Search results: {results}")

    # Delete the memory
    logger.info("Deleting the memory...")
    deleted = await asyncio.wait_for(client.delete(memory["id"]), timeout=5.0)
    logger.info(f"Deleted memory: {deleted}")

    return memory["id"]  # Return ID for cleanup just in case


async def test_resources(client):
    """Test all available resources."""
    logger.info("\n--- TESTING RESOURCES ---")

    # Create some test memories first
    logger.info("Creating test memories for resources...")
    memories = []
    for i in range(3):
        memory = await client.create(f"Resource test memory #{i+1}")
        memories.append(memory)
        logger.info(f"Created memory: {memory['id']}")

    # Get all memories as a resource
    logger.info("Getting all memories as a resource...")
    all_memories = await asyncio.wait_for(client.get_all_memories(), timeout=5.0)
    logger.info(f"All memories resource:\n{all_memories}")

    # Get a specific memory as a resource
    logger.info("Getting a specific memory as a resource...")
    memory_content = await asyncio.wait_for(client.get_memory_content(memories[0]["id"]), timeout=5.0)
    logger.info(f"Memory content resource: {memory_content}")

    # Get memory stats as a resource
    logger.info("Getting memory stats as a resource...")
    stats_resource = await asyncio.wait_for(client.get_memory_stats_resource(), timeout=5.0)
    logger.info(f"Memory stats resource:\n{stats_resource}")

    # Clean up
    for memory in memories:
        await client.delete(memory["id"])
        logger.info(f"Deleted memory: {memory['id']}")

    return [m["id"] for m in memories]  # Return IDs for cleanup just in case


async def test_prompts(client):
    """Test all available prompts."""
    logger.info("\n--- TESTING PROMPTS ---")

    # Create a test memory for prompts
    logger.info("Creating a test memory for prompts...")
    memory = await client.create("This is a test memory for prompts testing")
    logger.info(f"Created memory: {memory['id']}")

    # Get create memory prompt
    logger.info("Getting create memory prompt...")
    create_prompt = await asyncio.wait_for(
        client.get_create_memory_prompt("New memory content", "Prompt Test"), timeout=5.0
    )
    logger.info(f"Create memory prompt:\n{create_prompt}")

    # Get search prompt
    logger.info("Getting search prompt...")
    search_prompt = await asyncio.wait_for(client.get_search_prompt("test query"), timeout=5.0)
    logger.info(f"Search prompt:\n{search_prompt}")

    # Get summarize prompt
    logger.info("Getting summarize prompt...")
    summarize_prompt = await asyncio.wait_for(client.get_summarize_prompt(memory["id"]), timeout=5.0)
    logger.info(f"Summarize prompt messages:")
    for msg in summarize_prompt:
        logger.info(f"  {msg['role']}: {msg['content']}")

    # Clean up
    await client.delete(memory["id"])
    logger.info(f"Deleted memory: {memory['id']}")

    return [memory["id"]]  # Return ID for cleanup just in case


async def main():
    """Run comprehensive tests for the A-Mem MCP client."""
    logger.info("Starting A-Mem MCP client comprehensive test")

    # Create a client
    client = AMemMCPClient()

    # Track all created memory IDs for cleanup
    memory_ids = []

    try:
        # Connect to the server with a timeout
        logger.info("Connecting to server...")
        try:
            await asyncio.wait_for(client.connect(), timeout=10.0)
            logger.info("Connected to server successfully")
        except asyncio.TimeoutError:
            logger.error("Connection timeout - server not responding")
            return

        # Test tools functionality
        tool_memory_ids = await test_tools(client)
        if isinstance(tool_memory_ids, list):
            memory_ids.extend(tool_memory_ids)
        elif tool_memory_ids:
            memory_ids.append(tool_memory_ids)

        # Test resources functionality
        resource_memory_ids = await test_resources(client)
        if isinstance(resource_memory_ids, list):
            memory_ids.extend(resource_memory_ids)
        elif resource_memory_ids:
            memory_ids.append(resource_memory_ids)

        # Test prompts functionality
        prompt_memory_ids = await test_prompts(client)
        if isinstance(prompt_memory_ids, list):
            memory_ids.extend(prompt_memory_ids)
        elif prompt_memory_ids:
            memory_ids.append(prompt_memory_ids)

        logger.info("\nAll tests completed successfully")
    except Exception as e:
        logger.error(f"Test failed: {e}")
    finally:
        # Extra cleanup for any memories that might not have been deleted
        for memory_id in memory_ids:
            try:
                await client.delete(memory_id)
                logger.info(f"Cleaned up memory: {memory_id}")
            except:
                pass

        # Disconnect from the server
        try:
            await client.disconnect()
            logger.info("Disconnected from server")
        except Exception as e:
            logger.error(f"Error disconnecting: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
