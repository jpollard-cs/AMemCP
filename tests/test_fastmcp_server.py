#!/usr/bin/env python3
"""
Tests for the FastMCP server implementation
Tests both direct server functionality and MCP tools
"""

import os
import sys
import unittest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import server module
import mcp_fastmcp_server


class MockMemoryNote:
    """Mock memory note for testing"""

    def __init__(self, content, id=None, **metadata):
        import uuid

        self.id = id or str(uuid.uuid4())
        self.content = content
        self.metadata = metadata or {}
        self.created_at = self.metadata.get("created_at", "2024-03-31T00:00:00")
        self.updated_at = self.metadata.get("updated_at", "2024-03-31T00:00:00")

        # Mirror attributes from metadata for compatibility
        for key, value in self.metadata.items():
            setattr(self, key, value)

    def to_dict(self):
        """Convert to dictionary"""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class MockMemorySystem:
    """Mock memory system for testing"""

    def __init__(self, project_name="test", **kwargs):
        self.project_name = project_name
        self.memories = {}
        self.next_id = 1

    def create(self, content, **metadata):
        """Create a memory"""
        note_id = str(self.next_id)
        self.next_id += 1
        note = MockMemoryNote(content, id=note_id, **metadata)
        self.memories[note_id] = note
        return note

    def read(self, note_id):
        """Read a memory by ID"""
        if note_id in self.memories:
            return self.memories[note_id]
        return None

    def update(self, note_id, content=None, **metadata):
        """Update a memory"""
        if note_id not in self.memories:
            return False

        note = self.memories[note_id]
        if content is not None:
            note.content = content
        if metadata:
            note.metadata.update(metadata)
            # Mirror attributes from metadata for compatibility
            for key, value in metadata.items():
                setattr(note, key, value)
        note.updated_at = "2024-03-31T01:00:00"  # Simulate update time
        return True

    def delete(self, note_id):
        """Delete a memory"""
        if note_id not in self.memories:
            return False
        del self.memories[note_id]
        return True

    def search(self, query, k=5, **kwargs):
        """Search for memories"""
        results = []
        for memory in self.memories.values():
            if query.lower() in memory.content.lower():
                results.append(memory)

        # Add dummy results if needed
        while len(results) < k and len(results) < 3:  # Max 3 dummy results
            dummy_id = f"dummy-{len(results)}"
            results.append(MockMemoryNote(f"Dummy result for {query}", id=dummy_id))

        return results[:k]

    def hybrid_search(self, query, k=5, **kwargs):
        """Hybrid search implementation (same as search for mock)"""
        return self.search(query, k, **kwargs)


# Apply mocks before importing server code
sys.modules["memory_system"] = type(
    "MockMemorySystemModule",
    (),
    {
        "AgenticMemorySystem": MockMemorySystem,
        "MemoryNote": MockMemoryNote,
        "detect_content_type": lambda text: ("general", "RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"),
    },
)


class TestFastMCPServerDirect(unittest.TestCase):
    """Test the FastMCP server directly"""

    @classmethod
    def setUpClass(cls):
        """Set up the test environment once for all tests"""
        # Set up environment variables for testing
        os.environ["PROJECT_NAME"] = "test_project"
        os.environ["LLM_BACKEND"] = "mock"
        os.environ["USING_MOCKS"] = "1"

        # Override memory system
        mcp_fastmcp_server.memory_system = MockMemorySystem(project_name="test_project")
        mcp_fastmcp_server.project_name = "test_project"
        mcp_fastmcp_server.llm_backend = "mock"
        mcp_fastmcp_server.llm_model = "mock"
        mcp_fastmcp_server.embed_model = "mock"

    def test_create_memory(self):
        """Test creating a memory through the MCP handler"""
        # Create memory handler function
        handler = mcp_fastmcp_server.get_create_memory_handler()

        # Call the handler
        content = "Test memory through MCP"
        result = handler({"content": content, "metadata": {"tag": "test"}})

        # Verify the result
        self.assertIn("memory", result)
        memory = result["memory"]
        self.assertEqual(memory["content"], content)
        self.assertEqual(memory["metadata"]["tag"], "test")
        self.assertIn("id", memory)

        # Verify memory exists in the system
        memory_id = memory["id"]
        memory_obj = mcp_fastmcp_server.memory_system.read(memory_id)
        self.assertIsNotNone(memory_obj)
        self.assertEqual(memory_obj.content, content)

    def test_read_memory(self):
        """Test reading a memory through the MCP handler"""
        # Create a memory first
        create_handler = mcp_fastmcp_server.get_create_memory_handler()
        create_result = create_handler({"content": "Memory to read"})
        memory_id = create_result["memory"]["id"]

        # Read memory handler
        read_handler = mcp_fastmcp_server.get_get_memory_handler()
        read_result = read_handler({"id": memory_id})

        # Verify the result
        self.assertIn("memory", read_result)
        self.assertEqual(read_result["memory"]["id"], memory_id)
        self.assertEqual(read_result["memory"]["content"], "Memory to read")

    def test_update_memory(self):
        """Test updating a memory through the MCP handler"""
        # Create a memory first
        create_handler = mcp_fastmcp_server.get_create_memory_handler()
        create_result = create_handler({"content": "Original content"})
        memory_id = create_result["memory"]["id"]

        # Update memory handler
        update_handler = mcp_fastmcp_server.get_update_memory_handler()
        update_result = update_handler(
            {"id": memory_id, "content": "Updated content", "metadata": {"priority": "high"}}
        )

        # Verify the result
        self.assertTrue(update_result["success"])

        # Verify memory was updated
        read_handler = mcp_fastmcp_server.get_get_memory_handler()
        read_result = read_handler({"id": memory_id})
        self.assertEqual(read_result["memory"]["content"], "Updated content")
        self.assertEqual(read_result["memory"]["metadata"]["priority"], "high")

    def test_delete_memory(self):
        """Test deleting a memory through the MCP handler"""
        # Create a memory first
        create_handler = mcp_fastmcp_server.get_create_memory_handler()
        create_result = create_handler({"content": "Memory to delete"})
        memory_id = create_result["memory"]["id"]

        # Delete memory handler
        delete_handler = mcp_fastmcp_server.get_delete_memory_handler()
        delete_result = delete_handler({"id": memory_id})

        # Verify the result
        self.assertTrue(delete_result["success"])

        # Verify memory was deleted
        read_handler = mcp_fastmcp_server.get_get_memory_handler()
        read_result = read_handler({"id": memory_id})
        self.assertIsNone(read_result["memory"])

    def test_search_memories(self):
        """Test searching memories through the MCP handler"""
        # Create memories
        create_handler = mcp_fastmcp_server.get_create_memory_handler()
        contents = [
            "Memory about artificial intelligence",
            "Memory about machine learning",
            "Unrelated memory about cooking recipes",
        ]

        for content in contents:
            create_handler({"content": content})

        # Search memory handler
        search_handler = mcp_fastmcp_server.get_search_memories_handler()
        search_result = search_handler({"query": "artificial intelligence", "k": 2})

        # Verify the result
        self.assertIn("memories", search_result)
        self.assertGreater(len(search_result["memories"]), 0)
        self.assertLessEqual(len(search_result["memories"]), 2)  # Respect k parameter

        # Check content of first result
        self.assertIn("artificial intelligence", search_result["memories"][0]["content"].lower())


if __name__ == "__main__":
    unittest.main()
