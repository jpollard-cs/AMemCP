#!/usr/bin/env python3
"""
Unit tests for the mock memory system.
These tests validate the functionality without external dependencies.
"""

import os
import sys
import unittest
import uuid

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Mock classes for testing
class MockMemoryNote:
    """Mock memory note for testing"""

    def __init__(self, content, id=None, **metadata):
        self.id = id or str(uuid.uuid4())
        self.content = content
        self.keywords = metadata.get("keywords", [])
        self.context = metadata.get("context", "General")
        self.tags = metadata.get("tags", [])
        self.metadata = metadata


class MockMemorySystem:
    """Mock memory system for testing"""

    def __init__(self, project_name="test_project", **kwargs):
        self.project_name = project_name
        self.memories = {}

    def create(self, content, **metadata):
        """Create a memory"""
        note = MockMemoryNote(content, **metadata)
        self.memories[note.id] = note
        return note

    def read(self, memory_id):
        """Read a memory by ID"""
        return self.memories.get(memory_id)

    def update(self, memory_id, content=None, **metadata):
        """Update a memory"""
        if memory_id not in self.memories:
            return False

        memory = self.memories[memory_id]

        if content is not None:
            memory.content = content

        for key, value in metadata.items():
            setattr(memory, key, value)
            memory.metadata[key] = value

        return True

    def delete(self, memory_id):
        """Delete a memory"""
        if memory_id in self.memories:
            del self.memories[memory_id]
            return True
        return False

    def search(self, query, k=5):
        """Search for memories"""
        results = []
        for memory in self.memories.values():
            if query.lower() in memory.content.lower():
                results.append(memory)
        return results[:k]


# Import the memory system for tests that don't use mocks
# from memory_system import AgenticMemorySystem, MemoryNote


class TestAgenticMemorySystem(unittest.TestCase):
    """Test case for the AgenticMemorySystem class using mock implementation"""

    def setUp(self):
        """Set up before each test"""
        # Create a memory system with mock implementation
        self.memory_system = MockMemorySystem(project_name="test_project")

    def test_create_read_memory(self):
        """Test creating and retrieving a memory"""
        # Create a memory
        content = "This is a test memory."
        metadata = {"keywords": ["test", "memory"], "context": "Unit testing"}

        memory = self.memory_system.create(content, **metadata)

        # Verify memory was created successfully
        self.assertIsNotNone(memory)
        self.assertEqual(memory.content, content)
        self.assertEqual(memory.keywords, metadata["keywords"])
        self.assertEqual(memory.context, metadata["context"])

        # Read the memory
        retrieved_memory = self.memory_system.read(memory.id)

        # Verify retrieved memory matches created memory
        self.assertIsNotNone(retrieved_memory)
        self.assertEqual(retrieved_memory.content, content)
        self.assertEqual(retrieved_memory.keywords, metadata["keywords"])
        self.assertEqual(retrieved_memory.context, metadata["context"])

    def test_update_memory(self):
        """Test updating a memory"""
        # Create a memory
        content = "Original content"
        memory = self.memory_system.create(content)

        # Update the memory
        updated_content = "Updated content"
        updated_keywords = ["updated", "keywords"]

        success = self.memory_system.update(memory.id, content=updated_content, keywords=updated_keywords)

        # Verify update was successful
        self.assertTrue(success)

        # Read the updated memory
        updated_memory = self.memory_system.read(memory.id)

        # Verify memory was updated
        self.assertEqual(updated_memory.content, updated_content)
        self.assertEqual(updated_memory.keywords, updated_keywords)

    def test_delete_memory(self):
        """Test deleting a memory"""
        # Create a memory
        memory = self.memory_system.create("Memory to delete")

        # Delete the memory
        success = self.memory_system.delete(memory.id)

        # Verify deletion was successful
        self.assertTrue(success)

        # Try to read the deleted memory
        deleted_memory = self.memory_system.read(memory.id)

        # Verify memory was deleted
        self.assertIsNone(deleted_memory)

    def test_search_memory(self):
        """Test searching for memories"""
        # Create test memories
        memories = [
            self.memory_system.create("Memory about cats", keywords=["cats", "pets"]),
            self.memory_system.create("Memory about dogs", keywords=["dogs", "pets"]),
            self.memory_system.create("Memory about cars", keywords=["cars", "vehicles"]),
        ]

        # Search for pet-related memories
        results = self.memory_system.search("cats")

        # Verify search returned results
        self.assertGreater(len(results), 0)

        # Clean up
        for memory in memories:
            self.memory_system.delete(memory.id)


if __name__ == "__main__":
    unittest.main()
