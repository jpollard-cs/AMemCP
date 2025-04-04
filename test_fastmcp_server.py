#!/usr/bin/env python3
"""
Test file for the A-Mem memory system using a simple MCP server
"""

import asyncio
import logging
import unittest
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from mcp.server.fastmcp import FastMCP

# Configure minimal logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_fastmcp")


# Mock MemoryNote
class MockMemoryNote:
    def __init__(self, content, id=None, **metadata):
        self.id = id or str(uuid.uuid4())
        self.content = content
        self.metadata = metadata or {}
        self.created_at = self.metadata.get("created_at", datetime.now().isoformat())
        self.updated_at = self.metadata.get("updated_at", datetime.now().isoformat())

    def to_dict(self):
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


# Mock AgenticMemorySystem
class MockMemorySystem:
    def __init__(self, project_name="default", **kwargs):
        self.project_name = project_name
        self.memories = {}
        self.next_id = 1
        logger.info(f"Initialized mock memory system for project: {project_name}")

    def create(self, content, **metadata):
        note_id = str(self.next_id)
        self.next_id += 1
        note = MockMemoryNote(content, id=note_id, **metadata)
        self.memories[note_id] = note
        return note

    def read(self, note_id):
        if note_id in self.memories:
            return self.memories[note_id]
        return None

    def update(self, note_id, content=None, **metadata):
        if note_id not in self.memories:
            return None
        note = self.memories[note_id]
        if content:
            note.content = content
        if metadata:
            note.metadata.update(metadata)
        note.updated_at = datetime.now().isoformat()
        return note

    def delete(self, note_id):
        if note_id in self.memories:
            del self.memories[note_id]
            return True
        return False

    def search(self, query, top_k=5, **kwargs):
        results = []
        for memory in self.memories.values():
            if query.lower() in memory.content.lower():
                results.append(memory)

        # Add dummy results if needed
        while len(results) < top_k and len(results) < 3:  # Max 3 dummy results
            dummy_id = f"dummy-{len(results)}"
            results.append(MockMemoryNote(f"Dummy result for {query}", id=dummy_id))

        return results[:top_k]

    def hybrid_search(self, query, top_k=5, **kwargs):
        return self.search(query, top_k, **kwargs)


# Helper function to run asyncio coroutines in sync context
def run_async(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestMCPMemorySystem(unittest.TestCase):
    """Test the A-Mem memory system with MCP integration"""

    def setUp(self):
        """Set up the test environment for each test"""
        # Create MCP instance
        self.mcp = FastMCP("A-Mem Memory System Test")

        # Create memory system
        self.memory_system = MockMemorySystem(project_name="test_project")

        # Define tools
        self.setup_tools()

    def setup_tools(self):
        """Set up MCP tools for interacting with the memory system"""

        @self.mcp.tool()
        async def create_memory(content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            """Create a new memory with the given content and metadata"""
            if metadata is None:
                metadata = {}
            memory = self.memory_system.create(content, **metadata)
            return memory.to_dict()

        @self.mcp.tool()
        async def get_memory(id: str) -> Dict[str, Any]:
            """Retrieve a memory by ID"""
            memory = self.memory_system.read(id)
            if not memory:
                raise ValueError(f"Memory not found: {id}")
            return memory.to_dict()

        @self.mcp.tool()
        async def update_memory(id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            """Update a memory with new content and/or metadata"""
            if metadata is None:
                metadata = {}
            memory = self.memory_system.update(id, content, **metadata)
            if not memory:
                raise ValueError(f"Memory not found: {id}")
            return memory.to_dict()

        @self.mcp.tool()
        async def delete_memory(id: str) -> Dict[str, bool]:
            """Delete a memory by ID"""
            success = self.memory_system.delete(id)
            if not success:
                raise ValueError(f"Memory not found: {id}")
            return {"success": success}

        @self.mcp.tool()
        async def search_memories(query: str, top_k: int = 5) -> Dict[str, Any]:
            """Search for memories based on a query"""
            results = self.memory_system.search(query, top_k=top_k)
            return {"results": [r.to_dict() for r in results], "count": len(results)}

        # Define a resource for system status
        @self.mcp.resource("amem://status")
        async def get_status() -> Dict[str, str]:
            """Get the status of the A-Mem memory system"""
            return {
                "status": "running",
                "project": "test_project",
                "version": "0.1.0",
                "llm_backend": "mock",
                "llm_model": "mock",
                "embedding_model": "mock",
            }

    async def async_test_direct_memory_operations(self):
        """Test memory operations directly without HTTP"""
        # Get the tool functions
        tools = await self.mcp.list_tools()
        create_memory_info = tools["create_memory"]
        get_memory_info = tools["get_memory"]
        update_memory_info = tools["update_memory"]
        delete_memory_info = tools["delete_memory"]

        # Create a memory
        content = "Test memory through direct call"
        memory = await create_memory_info["handler"](content, {"tag": "test"})

        self.assertEqual(memory["content"], content)
        self.assertEqual(memory["metadata"]["tag"], "test")
        self.assertIn("id", memory)

        # Get the memory
        memory_id = memory["id"]
        retrieved = await get_memory_info["handler"](memory_id)
        self.assertEqual(retrieved["content"], content)

        # Update the memory
        updated_content = "Updated content"
        updated = await update_memory_info["handler"](memory_id, updated_content, {"priority": "high"})
        self.assertEqual(updated["content"], updated_content)
        self.assertEqual(updated["metadata"]["priority"], "high")
        self.assertEqual(updated["metadata"]["tag"], "test")  # Original metadata preserved

        # Delete the memory
        result = await delete_memory_info["handler"](memory_id)
        self.assertTrue(result["success"])

        # Verify deletion
        with self.assertRaises(ValueError):
            await get_memory_info["handler"](memory_id)

    def test_direct_memory_operations(self):
        """Run the async test in a sync context"""
        run_async(self.async_test_direct_memory_operations())

    async def async_test_search_memories(self):
        """Test searching memories directly"""
        # Get the tool functions
        tools = await self.mcp.list_tools()
        create_memory_info = tools["create_memory"]
        search_memories_info = tools["search_memories"]

        # Create test memories
        contents = ["Memory about cats and dogs", "Memory about dogs and rabbits", "Memory about cats and parrots"]

        for content in contents:
            await create_memory_info["handler"](content)

        # Search for cats
        cat_results = await search_memories_info["handler"]("cats", 2, True)
        self.assertIn("results", cat_results)
        self.assertLessEqual(len(cat_results["results"]), 2)  # Should respect top_k

        # Search for dogs
        dog_results = await search_memories_info["handler"]("dogs", 2, False)
        self.assertIn("results", dog_results)

        # Search for something not in any memories
        no_results = await search_memories_info["handler"]("elephants", 2, True)
        self.assertEqual(len(no_results["results"]), 2)  # Should have 2 dummy results

    def test_search_memories(self):
        """Run the async test in a sync context"""
        run_async(self.async_test_search_memories())

    async def async_test_resource_access(self):
        """Test accessing MCP resources"""
        # Get the resource data
        status = await self.mcp.read_resource("amem://status")

        self.assertEqual(status["status"], "running")
        self.assertEqual(status["project"], "test_project")
        self.assertEqual(status["llm_backend"], "mock")

    def test_resource_access(self):
        """Run the async test in a sync context"""
        run_async(self.async_test_resource_access())


if __name__ == "__main__":
    unittest.main()
