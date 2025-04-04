#!/usr/bin/env python3
"""
Unit tests for the A-Mem MCP Client.
Tests connection to the MCP server using stdio transport.
"""

import sys
import unittest
from pathlib import Path

# Make sure the amem package is in the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from amem.mcp_client import AMemMCPClient

# Path to the server script
SERVER_PATH = str(Path(__file__).parent.parent / "mcp_fastmcp_server.py")


class TestAMemMCPClient(unittest.IsolatedAsyncioTestCase):
    """Test cases for the A-Mem MCP Client."""

    async def asyncSetUp(self):
        """Set up the test case."""
        self.client = AMemMCPClient(SERVER_PATH)
        await self.client.connect()

    async def asyncTearDown(self):
        """Clean up after the test case."""
        await self.client.disconnect()

    async def test_create_memory(self):
        """Test creating a memory."""
        memory = await self.client.create("Test memory content", "Test Memory")
        self.assertIsNotNone(memory)
        self.assertTrue("id" in memory)
        self.assertTrue("content" in memory)
        self.assertEqual(memory["content"], "Test memory content")

        # Clean up
        await self.client.delete(memory["id"])

    async def test_get_memory(self):
        """Test retrieving a memory."""
        # Create a memory first
        created = await self.client.create("Test memory for retrieval")

        # Retrieve it
        memory = await self.client.get(created["id"])
        self.assertIsNotNone(memory)
        self.assertEqual(memory["id"], created["id"])
        self.assertEqual(memory["content"], "Test memory for retrieval")

        # Clean up
        await self.client.delete(created["id"])

    async def test_update_memory(self):
        """Test updating a memory."""
        # Create a memory first
        created = await self.client.create("Original content")

        # Update it
        updated = await self.client.update(created["id"], "Updated content")
        self.assertIsNotNone(updated)
        self.assertEqual(updated["id"], created["id"])
        self.assertEqual(updated["content"], "Updated content")

        # Clean up
        await self.client.delete(created["id"])

    async def test_delete_memory(self):
        """Test deleting a memory."""
        # Create a memory first
        created = await self.client.create("Memory to delete")

        # Delete it
        result = await self.client.delete(created["id"])
        self.assertIsNotNone(result)
        self.assertTrue(result["success"])

    async def test_search_memories(self):
        """Test searching memories."""
        # Create some memories for searching
        mem1 = await self.client.create("Apple pie recipe")
        mem2 = await self.client.create("Orange juice benefits")

        # Search for them
        results = await self.client.search("apple")
        self.assertIsNotNone(results)
        self.assertIsInstance(results, list)
        self.assertTrue(any(m["id"] == mem1["id"] for m in results))

        # Clean up
        await self.client.delete(mem1["id"])
        await self.client.delete(mem2["id"])


if __name__ == "__main__":
    unittest.main()
