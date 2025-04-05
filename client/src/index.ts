/**
 * MCP Client Test for AMemCP Server
 */
// Using direct paths to the SDK modules
import { Client } from "../node_modules/@modelcontextprotocol/sdk/dist/esm/client/index.js";
import { SSEClientTransport } from "../node_modules/@modelcontextprotocol/sdk/dist/esm/client/sse.js";
// Add explicit type import for Tool
import { type Tool } from "../node_modules/@modelcontextprotocol/sdk/dist/esm/types.js";

// Configuration
// We need to use the SSE endpoint which is at /sse
const SERVER_URL = "http://localhost:8000/sse";

// Memory interfaces
interface Memory {
  id: string;
  content: string;
  name?: string;
  metadata?: Record<string, any>;
  created_at: string;
  updated_at: string;
}

interface SearchResult {
  memories: Memory[];
  total?: number;
}

// Add a new sample mixed content for testing auto-segmentation
const MIXED_CONTENT_SAMPLE = `
# Algorithm Performance Analysis

This document provides a comparative analysis of sorting algorithms and their implementations.

## Quick Sort Implementation

\`\`\`python
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
\`\`\`

Quick sort has an average time complexity of O(n log n), but can degrade to O(n²) in worst cases.

## Merge Sort Implementation

\`\`\`python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result
\`\`\`

Merge sort has a consistent O(n log n) time complexity for all cases.

## Key Questions

1. When would you choose Quick Sort over Merge Sort?
2. How does memory usage differ between these algorithms?
3. What are the implications for parallel processing?
`;

async function main(): Promise<void> {
  console.log("Starting AMemCP TypeScript Client...");
  console.log(`Connecting to MCP server at: ${SERVER_URL}`);

  try {
    // Create SSE transport to connect to the server
    const transport = new SSEClientTransport(
      new URL(SERVER_URL),
      {
        // Enable redirects to handle any server routing redirects
        eventSourceInit: {
          withCredentials: true
        }
      }
    );

    console.log("Transport initialized. Creating client...");

    // Create client with capabilities needed
    const client = new Client(
      {
        name: "amem-ts-client",
        version: "1.0.0"
      },
      {
        capabilities: {
          tools: {},    // Enable tools capability
          prompts: {},  // Enable prompts capability
          resources: {} // Enable resources capability
        }
      }
    );

    console.log("Client created. Connecting to server...");

    // Connect to the server
    await client.connect(transport);
    console.log("Connected to server!");

    // List available tools
    console.log("Listing available tools...");
    const { tools = [] } = await client.listTools();
    console.log(`Found ${tools ? tools.length : 0} tools:`);

    // Check if tools is an array before iterating
    if (tools && Array.isArray(tools)) {
      tools.forEach((tool: any) => {
        console.log(`- ${tool.name}: ${tool.description}`);
      });

      // Test basic memory operations
      await testMemoryOperations(client);

      // Test new LLM analysis features
      console.log("\n=== Testing LLM Content Analysis and Auto-Segmentation Features ===\n");
      await testLLMContentAnalysis(client);
      await testAutoSegmentation(client);
      await testContentSpecificSearch(client);
    } else {
      console.log("No tools found or tools is not an array");
    }

    // Clean up - Note: There is no disconnect method; the connection will close
    // when the transport is closed
    console.log("Closing transport...");
    await transport.close();
    console.log("Transport closed.");
  } catch (error: unknown) {
    console.error("Error:", error);
  }
}

/**
 * Test memory operations using the client's tools
 */
async function testMemoryOperations(client: Client): Promise<void> {
  console.log("\n--- Testing Memory Operations ---\n");

  try {
    // 1. Create a memory
    console.log("Creating a test memory...");
    const memory = await createMemory(
      client,
      "This is a test memory created from the TypeScript client.",
      "Test Memory",
      { tags: ["test", "typescript"], source: "client-test" }
    );

    console.log("Memory created:", memory);

    if (!memory || !memory.id) {
      console.error("Failed to create memory with a valid ID");
      return;
    }

    console.log(`Successfully created memory with ID: ${memory.id}`);

    // 2. Retrieve the memory
    console.log("\nRetrieving memory...");
    const retrievedMemory = await getMemory(client, memory.id);
    console.log("Retrieved memory:", retrievedMemory);

    // 3. Search for memories
    console.log("\nSearching for memories with 'test'...");
    try {
      const searchResults = await searchMemories(client, "test");
      console.log(`Found ${searchResults.memories?.length || 0} memories:`);

      if (searchResults.memories && searchResults.memories.length > 0) {
        searchResults.memories.forEach((mem: Memory, index: number) => {
          console.log(`${index + 1}. ${mem.name || 'Unnamed'}: ${mem.content.substring(0, 50)}...`);
        });
      } else {
        console.log("No memories found matching the query");
      }
    } catch (error) {
      console.error("Search failed:", error);
    }

    // 4. Get all memories
    console.log("\nGetting all memories...");
    try {
      const allResult = await client.callTool({
        name: "get_all_memories",
        arguments: {}
      });

      console.log("Raw all memories result:", allResult);

      if (allResult.content && Array.isArray(allResult.content) && allResult.content.length > 0) {
        const allMemories = JSON.parse(allResult.content[0].text);
        console.log(`Total memories: ${allMemories.count || 0}`);

        if (allMemories.memories && allMemories.memories.length > 0) {
          allMemories.memories.forEach((mem: Memory, index: number) => {
            // Check for name in metadata or fall back to 'Unnamed'
            const memName = mem.metadata && mem.metadata.name ? mem.metadata.name : 'Unnamed';
            console.log(`${index + 1}. ${memName}: ${mem.id}`);
          });
        }
      }
    } catch (error) {
      console.error("Get all memories failed:", error);
    }

  } catch (error) {
    console.error("Error during memory operations:", error);
  }
}

/**
 * Create a memory using the create_memory tool
 */
async function createMemory(
  client: Client,
  content: string,
  name?: string,
  metadata?: Record<string, any>
): Promise<Memory> {
  // Log the parameters we're sending for debugging
  console.log("Creating memory with params:", { content, name, metadata });

  try {
    const result = await client.callTool({
      name: "create_memory",
      arguments: {
        content,
        ...(name ? { name } : {}),
        ...(metadata ? { metadata } : {})
      }
    });

    console.log("Raw result:", result);

    // Try to extract the result from wherever it is in the response
    if (result.content && Array.isArray(result.content) && result.content.length > 0) {
      if (result.content[0].text) {
        // The text field contains a JSON string that needs to be parsed
        try {
          return JSON.parse(result.content[0].text) as Memory;
        } catch (parseError) {
          console.error("Error parsing JSON response:", parseError);
          return result.content[0].text as unknown as Memory;
        }
      }
    }

    return result as unknown as Memory;
  } catch (error) {
    console.error("Error creating memory:", error);
    throw error;
  }
}

/**
 * Get a memory by ID using the get_memory tool
 */
async function getMemory(client: Client, memoryId: string): Promise<Memory> {
  console.log("Getting memory with ID:", memoryId);

  try {
    const result = await client.callTool({
      name: "get_memory",
      arguments: {
        memory_id: memoryId
      }
    });

    console.log("Raw get result:", result);

    // Extract the memory from the response
    if (result.content && Array.isArray(result.content) && result.content.length > 0) {
      if (result.content[0].text) {
        // The text field contains a JSON string that needs to be parsed
        try {
          return JSON.parse(result.content[0].text) as Memory;
        } catch (parseError) {
          console.error("Error parsing JSON response:", parseError);
          return result.content[0].text as unknown as Memory;
        }
      }
    }

    return result as unknown as Memory;
  } catch (error) {
    console.error("Error getting memory:", error);
    throw error;
  }
}

/**
 * Search for memories using the search_memories tool
 */
async function searchMemories(
  client: Client,
  query: string,
  metadataFilter?: Record<string, any>,
  topK: number = 5
): Promise<SearchResult> {
  return searchMemoriesWithFilter(client, query, metadataFilter, topK);
}

/**
 * Test LLM content analysis feature
 */
async function testLLMContentAnalysis(client: Client): Promise<void> {
  console.log("\n--- Testing LLM Content Analysis ---\n");

  const createdIds: string[] = [];

  try {
    // Test text content
    console.log("Creating text memory with LLM analysis...");
    const textMemory = await createMemory(
      client,
      "Neural networks are computational systems inspired by the biological neural networks in human brains.",
      "Neural Networks Description",
      { enable_llm_analysis: true }
    );

    if (textMemory && textMemory.id) {
      createdIds.push(textMemory.id);
      console.log("Text memory created:", textMemory);
      console.log("Content type:", textMemory.metadata?.type || "Type not detected");
    }

    // Test code content
    console.log("\nCreating code memory with LLM analysis...");
    const codeMemory = await createMemory(
      client,
      "def hello_world():\n    print('Hello, world!')\n\nhello_world()",
      "Hello World Function",
      { enable_llm_analysis: true }
    );

    if (codeMemory && codeMemory.id) {
      createdIds.push(codeMemory.id);
      console.log("Code memory created:", codeMemory);
      console.log("Content type:", codeMemory.metadata?.type || "Type not detected");
      console.log("Storage task type:", codeMemory.metadata?.storage_task_type || "Task type not detected");
    }

    // Test question content
    console.log("\nCreating question memory with LLM analysis...");
    const questionMemory = await createMemory(
      client,
      "What are the best practices for implementing a neural network from scratch?",
      "Neural Network Question",
      { enable_llm_analysis: true }
    );

    if (questionMemory && questionMemory.id) {
      createdIds.push(questionMemory.id);
      console.log("Question memory created:", questionMemory);
      console.log("Content type:", questionMemory.metadata?.type || "Type not detected");
    }
  } catch (error) {
    console.error("Error during LLM content analysis testing:", error);
  } finally {
    // Clean up - delete all created memories
    console.log("\nCleaning up created memories...");
    for (const id of createdIds) {
      try {
        await deleteMemory(client, id);
        console.log(`Deleted memory ${id}`);
      } catch (error) {
        console.error(`Error deleting memory ${id}:`, error);
      }
    }
  }
}

/**
 * Test auto-segmentation of mixed content
 */
async function testAutoSegmentation(client: Client): Promise<void> {
  console.log("\n--- Testing Auto-Segmentation ---\n");

  let parentId: string | null = null;
  const segmentIds: string[] = [];

  try {
    // Create memory with mixed content and enable auto-segmentation
    console.log("Creating mixed content memory with auto-segmentation...");
    const mixedMemory = await createMemory(
      client,
      MIXED_CONTENT_SAMPLE,
      "Sorting Algorithms Guide",
      {
        enable_llm_analysis: true,
        enable_auto_segmentation: true
      }
    );

    if (!mixedMemory || !mixedMemory.id) {
      console.error("Failed to create mixed content memory");
      return;
    }

    parentId = mixedMemory.id;
    console.log("Mixed content memory created:", mixedMemory);

    // Check if auto-segmentation occurred
    if (mixedMemory.metadata && mixedMemory.metadata.segment_ids) {
      let segments: string[] = [];

      if (typeof mixedMemory.metadata.segment_ids === 'string') {
        // Parse JSON string if needed
        try {
          segments = JSON.parse(mixedMemory.metadata.segment_ids);
        } catch (error) {
          console.error("Error parsing segment_ids:", error);
        }
      } else if (Array.isArray(mixedMemory.metadata.segment_ids)) {
        segments = mixedMemory.metadata.segment_ids;
      }

      console.log(`Content was segmented into ${segments.length} parts`);
      segmentIds.push(...segments);

      // Retrieve each segment to verify content
      for (let i = 0; i < segments.length; i++) {
        const segmentId = segments[i];
        try {
          const segment = await getMemory(client, segmentId);
          console.log(`\nSegment ${i+1}:`);
          console.log(`  Type: ${segment.metadata?.type || "unknown"}`);
          console.log(`  Name: ${segment.metadata?.name || "Unnamed"}`);
          console.log(`  Content preview: ${segment.content.substring(0, 100)}...`);

          // Verify segment has parent relationship
          if (segment.metadata?.parent_id === parentId) {
            console.log(`  Connected to parent: ✓`);
          } else {
            console.log(`  WARNING: Segment missing proper parent relationship`);
          }
        } catch (error) {
          console.error(`Error retrieving segment ${segmentId}:`, error);
        }
      }
    } else {
      console.log("Content was not segmented - this might indicate an issue with auto-segmentation");
    }

    // Check for mixed content flag
    if (mixedMemory.metadata?.has_mixed_content) {
      console.log("Mixed content flag correctly set: ✓");
    } else {
      console.log("Mixed content flag not set");
    }

  } catch (error) {
    console.error("Error during auto-segmentation testing:", error);
  } finally {
    // Clean up - delete parent and all segments
    console.log("\nCleaning up memories...");

    try {
      // Delete all segments first
      for (const segmentId of segmentIds) {
        try {
          await deleteMemory(client, segmentId);
          console.log(`Deleted segment ${segmentId}`);
        } catch (error) {
          console.error(`Error deleting segment ${segmentId}:`, error);
        }
      }

      // Then delete parent
      if (parentId) {
        await deleteMemory(client, parentId);
        console.log(`Deleted parent memory ${parentId}`);
      }
    } catch (error) {
      console.error("Error during cleanup:", error);
    }
  }
}

/**
 * Test content-specific search capabilities
 */
async function testContentSpecificSearch(client: Client): Promise<void> {
  console.log("\n--- Testing Content-Specific Search ---\n");

  const createdIds: string[] = [];

  try {
    // Create memories with different content types
    console.log("Creating test memories with different content types...");

    // Text memory
    const textMemory = await createMemory(
      client,
      "Python is a versatile programming language used for data science, web development, and automation.",
      "Python Overview",
      { enable_llm_analysis: true }
    );
    if (textMemory && textMemory.id) createdIds.push(textMemory.id);

    // Code memory
    const codeMemory = await createMemory(
      client,
      `def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1`,
      "Binary Search Algorithm",
      { enable_llm_analysis: true }
    );
    if (codeMemory && codeMemory.id) createdIds.push(codeMemory.id);

    // Mixed content memory
    const mixedMemory = await createMemory(
      client,
      MIXED_CONTENT_SAMPLE,
      "Sorting Algorithms Guide",
      { enable_llm_analysis: true, enable_auto_segmentation: true }
    );

    if (mixedMemory && mixedMemory.id) {
      createdIds.push(mixedMemory.id);

      // Add segment IDs to the cleanup list if segmentation occurred
      if (mixedMemory.metadata && mixedMemory.metadata.segment_ids) {
        let segments: string[] = [];
        if (typeof mixedMemory.metadata.segment_ids === 'string') {
          try {
            segments = JSON.parse(mixedMemory.metadata.segment_ids);
          } catch (error) {
            console.error("Error parsing segment_ids:", error);
          }
        } else if (Array.isArray(mixedMemory.metadata.segment_ids)) {
          segments = mixedMemory.metadata.segment_ids;
        }
        createdIds.push(...segments);
      }
    }

    console.log(`Created ${createdIds.length} test memories`);

    // Test different search queries
    const searchQueries = [
      "python programming language",
      "binary search implementation",
      "sorting algorithm time complexity",
      "quicksort vs merge sort"
    ];

    for (const query of searchQueries) {
      console.log(`\nSearching for: "${query}"`);

      try {
        const searchResults = await searchMemories(client, query, {
          enable_llm_analysis: true
        });

        console.log(`Found ${searchResults.memories?.length || 0} results:`);

        if (searchResults.memories && searchResults.memories.length > 0) {
          searchResults.memories.forEach((memory: Memory, index: number) => {
            const contentPreview = memory.content.length > 100
              ? memory.content.substring(0, 100) + "..."
              : memory.content;

            console.log(`${index + 1}. ${memory.name || memory.metadata?.name || 'Unnamed'}`);
            console.log(`   Type: ${memory.metadata?.type || "unknown"}`);
            console.log(`   Content: ${contentPreview}`);
          });
        } else {
          console.log("No results found");
        }

        // For code-related queries, try with content-type filter
        if (query.includes("search") || query.includes("algorithm")) {
          console.log(`\nSearching for "${query}" with code type filter:`);

          const codeResults = await searchMemoriesWithFilter(client, query, {
            type: "code"
          });

          console.log(`Found ${codeResults.memories?.length || 0} code results`);

          if (codeResults.memories && codeResults.memories.length > 0) {
            codeResults.memories.forEach((memory: Memory, index: number) => {
              console.log(`${index + 1}. ${memory.name || memory.metadata?.name || 'Unnamed'}`);
              console.log(`   Type: ${memory.metadata?.type || "unknown"}`);
            });
          }
        }

      } catch (error) {
        console.error(`Error searching for "${query}":`, error);
      }
    }

  } catch (error) {
    console.error("Error during content-specific search testing:", error);
  } finally {
    // Clean up all created memories
    console.log("\nCleaning up created memories...");
    for (const id of createdIds) {
      try {
        await deleteMemory(client, id);
        console.log(`Deleted memory ${id}`);
      } catch (error) {
        console.error(`Error deleting memory ${id}:`, error);
      }
    }
  }
}

/**
 * Delete a memory by ID
 */
async function deleteMemory(client: Client, memoryId: string): Promise<any> {
  try {
    const result = await client.callTool({
      name: "delete_memory",
      arguments: {
        memory_id: memoryId
      }
    });

    if (result.content && Array.isArray(result.content) && result.content.length > 0) {
      try {
        return JSON.parse(result.content[0].text);
      } catch (error) {
        return result.content[0].text;
      }
    }

    return result;
  } catch (error) {
    console.error(`Error deleting memory ${memoryId}:`, error);
    throw error;
  }
}

/**
 * Search memories with metadata filtering
 */
async function searchMemoriesWithFilter(
  client: Client,
  query: string,
  metadataFilter?: Record<string, any>,
  topK: number = 5
): Promise<SearchResult> {
  try {
    const args: any = {
      query,
      top_k: topK
    };

    if (metadataFilter) {
      args.metadata_filter = metadataFilter;
    }

    const result = await client.callTool({
      name: "search_memories",
      arguments: args
    });

    if (result.content && Array.isArray(result.content) && result.content.length > 0) {
      try {
        return JSON.parse(result.content[0].text) as SearchResult;
      } catch (error) {
        console.error("Error parsing search results:", error);
        return { memories: [] };
      }
    }

    return { memories: [] };
  } catch (error) {
    console.error("Error searching memories:", error);
    throw error;
  }
}

// Run the client
main().catch((error: unknown) => {
  console.error("Unhandled error:", error);
  process.exit(1);
});
