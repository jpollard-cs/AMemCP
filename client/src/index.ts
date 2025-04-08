/**
 * MCP Client Test for AMemCP Server
 */
// Using direct paths to the SDK modules
import { Client } from "../node_modules/@modelcontextprotocol/sdk/dist/esm/client/index.js";
import { SSEClientTransport } from "../node_modules/@modelcontextprotocol/sdk/dist/esm/client/sse.js";
// Add explicit type import for Tool
import { type Tool } from "../node_modules/@modelcontextprotocol/sdk/dist/esm/types.js";
// Import chalk for colorful logging
import chalk from 'chalk';

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

// Helper functions for functional approaches
const parseJsonSafely = <T>(text: string, defaultValue: T): T => {
  try {
    return JSON.parse(text) as T;
  } catch (error) {
    console.error(chalk.red("Error parsing JSON:"), error);
    return defaultValue;
  }
};

const extractTextContent = (result: any): string | null => {
  return result?.content?.[0]?.text || null;
};

// Logger with colors for better readability
const logger = {
  info: (message: string, ...args: any[]) => console.log(chalk.blue(`ℹ️  ${message}`), ...args),
  success: (message: string, ...args: any[]) => console.log(chalk.green(`✅  ${message}`), ...args),
  warn: (message: string, ...args: any[]) => console.log(chalk.yellow(`⚠️  ${message}`), ...args),
  error: (message: string, ...args: any[]) => console.error(chalk.red(`❌  ${message}`), ...args),
  highlight: (message: string, ...args: any[]) => console.log(chalk.cyan(`🔦  ${message}`), ...args),
  title: (message: string) => console.log(chalk.bold.magenta(`\n=== 🏷️ ${message} ===\n`)),
  result: (message: string, ...args: any[]) => console.log(chalk.green(`✉️  `), chalk.bold(message), ...args),
  code: (message: string) => console.log(chalk.gray(`  ${message}`)),

  // Add specialized contextual loggers
  memory: {
    create: (message: string, ...args: any[]) => console.log(chalk.blue(`🧠 📝  ${message}`), ...args),
    get: (message: string, ...args: any[]) => console.log(chalk.green(`🧠 🗂️  ${message}`), ...args),
    delete: (message: string, ...args: any[]) => console.log(chalk.blueBright(`🧠 🗑️  ${message}`), ...args),
    search: (message: string, ...args: any[]) => console.log(chalk.green(`🧠 🔍  ${message}`), ...args)
  },
  connection: {
    start: (message: string, ...args: any[]) => console.log(chalk.blue(`⚡️  ${message}`), ...args),
    success: (message: string, ...args: any[]) => console.log(chalk.green(`🔌 🟢  ${message}`), ...args),
    close: (message: string, ...args: any[]) => console.log(chalk.blueBright(`🔌 ✂️ 🔴  ${message}`), ...args),
  }
};

async function main(): Promise<void> {
  logger.title("Starting AMemCP TypeScript Client");
  logger.connection.start(`Connecting to MCP server at: ${chalk.cyan(SERVER_URL)}`);

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

    logger.info("Transport initialized. Creating client...");

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

    logger.info("Client created. Connecting to server...");

    // Connect to the server
    await client.connect(transport);
    logger.connection.success("Connected to server!");

    // List available tools
    logger.info("Listing available tools...");
    const { tools = [] } = await client.listTools();
    logger.highlight(`Found ${tools ? tools.length : 0} tools:`);

    // Check if tools is an array before iterating
    if (tools && Array.isArray(tools)) {
      tools.forEach((tool: any) => {
        logger.result(`${tool.name}: ${tool.description}`);
      });

      // Test basic memory operations
      await testMemoryOperations(client);

      // Test new LLM analysis features
      logger.title("Testing LLM Content Analysis and Auto-Segmentation Features");
      await testLLMContentAnalysis(client);
      await testAutoSegmentation(client);
      await testContentSpecificSearch(client);
    } else {
      logger.warn("No tools found or tools is not an array");
    }

    // Clean up - Note: There is no disconnect method; the connection will close
    // when the transport is closed
    logger.connection.close("Closing transport...");
    await transport.close();
    logger.success("Transport closed.");
  } catch (error: unknown) {
    logger.error("Error:", error);
  }
}

/**
 * Test memory operations using the client's tools
 */
async function testMemoryOperations(client: Client): Promise<void> {
  logger.title("Testing Memory Operations");

  try {
    // 1. Create a memory
    logger.memory.create("Creating a test memory...");
    const memory = await createMemory({
      client,
      content: "This is a test memory created from the TypeScript client.",
      name: "Test Memory",
      metadata: { tags: ["test", "typescript"], source: "client-test" }
    });

    logger.info("Memory created:", memory);

    if (!memory?.id) {
      logger.error("Failed to create memory with a valid ID");
      return;
    }

    const { id } = memory;
    logger.success(`Successfully created memory with ID: ${id}`);

    // 2. Retrieve the memory
    logger.memory.get("Retrieving memory...");
    const retrievedMemory = await getMemory({
      client,
      memoryId: id
    });
    logger.result("Retrieved memory:", retrievedMemory);

    // 3. Search for memories
    logger.memory.search("Searching for memories with 'test'...");
    try {
      const searchResults = await searchMemories({
        client,
        query: "test"
      });
      const { memories = [] } = searchResults;
      logger.highlight(`Found ${memories?.length || 0} memories:`);

      if (memories?.length > 0) {
        // Functional approach using map and forEach
        memories
          .map((mem, index) => {
            const { name = 'Unnamed', content } = mem;
            return `${index + 1}. ${chalk.bold(name)}: ${chalk.gray(content.substring(0, 50))}...`;
          })
          .forEach(formattedMemory => console.log(`  ${formattedMemory}`));
      } else {
        logger.warn("No memories found matching the query");
      }
    } catch (error) {
      logger.error("Search failed:", error);
    }

    // 4. Get all memories
    logger.memory.get("Getting all memories...");
    try {
      const allResult = await client.callTool({
        name: "get_all_memories",
        arguments: {}
      });

      logger.info("Raw all memories result:", allResult);

      const textContent = extractTextContent(allResult);
      if (textContent) {
        const allMemories = parseJsonSafely(textContent, { count: 0, memories: [] });
        const { count = 0, memories = [] } = allMemories;
        logger.highlight(`Total memories: ${count}`);

        if (memories.length > 0) {
          // Functional approach using map and forEach
          memories
            .map((mem, index) => {
              const { metadata = {}, id } = mem;
              // Fix for the type error by asserting the type of metadata
              const memName = (metadata as Record<string, any>)['name'] || 'Unnamed';
              return `${index + 1}. ${chalk.bold(memName)}: ${chalk.cyan(id)}`;
            })
            .forEach(formattedMemory => console.log(`  ${formattedMemory}`));
        }
      }
    } catch (error) {
      logger.error("Get all memories failed:", error);
    }

  } catch (error) {
    logger.error("Error during memory operations:", error);
  }
}

/**
 * Create a memory using the create_memory tool
 *
 * why this anonymous object pattern?
 * looser coupling and flexibility of parameter ordering -
 * definings interfaces based on actual usage rather than upfront design
 */
async function createMemory({
  client,
  content,
  name,
  metadata
}: {
  client: Client,
  content: string,
  name?: string,
  metadata?: Record<string, any>
}): Promise<Memory> {
  // Log the parameters we're sending for debugging
  logger.memory.create(`Creating memory "${name || 'Unnamed'}" with content length: ${content.length} chars`);

  try {
    // Build arguments object immutably
    const args = {
      content,
      ...(name && { name }),
      ...(metadata && { metadata })
    };

    const result = await client.callTool({
      name: "create_memory",
      arguments: args
    });

    logger.info("Raw result:", result);

    const textContent = extractTextContent(result);
    return textContent
      ? parseJsonSafely(textContent, result as unknown as Memory)
      : result as unknown as Memory;
  } catch (error) {
    logger.error("Error creating memory:", error);
    throw error;
  }
}

/**
 * Get a memory by ID using the get_memory tool
 */
async function getMemory({
  client,
  memoryId
}: {
  client: Client,
  memoryId: string
}): Promise<Memory> {
  logger.memory.get(`Getting memory with ID: ${memoryId}`);

  try {
    const result = await client.callTool({
      name: "get_memory",
      arguments: {
        memory_id: memoryId
      }
    });

    logger.info("Raw get result:", result);

    const textContent = extractTextContent(result);
    return textContent
      ? parseJsonSafely(textContent, result as unknown as Memory)
      : result as unknown as Memory;
  } catch (error) {
    logger.error("Error getting memory:", error);
    throw error;
  }
}

/**
 * Test LLM content analysis feature
 */
async function testLLMContentAnalysis(client: Client): Promise<void> {
  logger.title("Testing LLM Content Analysis");

  const createdIds: string[] = [];

  try {
    // Test text content
    logger.info("Creating text memory with LLM analysis...");
    const textMemory = await createMemory({
      client,
      content: "Neural networks are computational systems inspired by the biological neural networks in human brains.",
      name: "Neural Networks Description",
      metadata: { enable_llm_analysis: true }
    });

    if (textMemory?.id) {
      createdIds.push(textMemory.id);
      const { metadata = {} } = textMemory;
      logger.info("Text memory created:", textMemory);
      logger.info("Content type:", metadata.type || "Type not detected");
    }

    // Test code content
    logger.info("\nCreating code memory with LLM analysis...");
    const codeMemory = await createMemory({
      client,
      content: "def hello_world():\n    print('Hello, world!')\n\nhello_world()",
      name: "Hello World Function",
      metadata: { enable_llm_analysis: true }
    });

    if (codeMemory?.id) {
      createdIds.push(codeMemory.id);
      const { metadata = {} } = codeMemory;
      logger.info("Code memory created:", codeMemory);
      logger.info("Content type:", metadata.type || "Type not detected");
      logger.info("Storage task type:", metadata.storage_task_type || "Task type not detected");
    }

    // Test question content
    logger.info("\nCreating question memory with LLM analysis...");
    const questionMemory = await createMemory({
      client,
      content: "What are the best practices for implementing a neural network from scratch?",
      name: "Neural Network Question",
      metadata: { enable_llm_analysis: true }
    });

    if (questionMemory?.id) {
      createdIds.push(questionMemory.id);
      const { metadata = {} } = questionMemory;
      logger.info("Question memory created:", questionMemory);
      logger.info("Content type:", metadata.type || "Type not detected");
    }
  } catch (error) {
    logger.error("Error during LLM content analysis testing:", error);
  } finally {
    // Clean up - delete all created memories
    logger.info("\nCleaning up created memories...");
    for (const id of createdIds) {
      try {
        await deleteMemory({
          client,
          memoryId: id
        });
        logger.memory.delete(`Deleted memory ${id}`);
      } catch (error) {
        logger.error(`Error deleting memory ${id}:`, error);
      }
    }
  }
}

/**
 * Test auto-segmentation of mixed content
 */
async function testAutoSegmentation(client: Client): Promise<void> {
  logger.title("Testing Auto-Segmentation");

  let parentId: string | null = null;
  const segmentIds: string[] = [];

  try {
    // Create memory with mixed content and enable auto-segmentation
    logger.info("Creating mixed content memory with auto-segmentation...");
    const mixedMemory = await createMemory({
      client,
      content: MIXED_CONTENT_SAMPLE,
      name: "Sorting Algorithms Guide",
      metadata: {
        enable_llm_analysis: true,
        enable_auto_segmentation: true
      }
    });

    if (!mixedMemory?.id) {
      logger.error("Failed to create mixed content memory");
      return;
    }

    parentId = mixedMemory.id;
    logger.info("Mixed content memory created:", mixedMemory);

    // Check if auto-segmentation occurred
    if (mixedMemory.metadata?.segment_ids) {
      const { metadata } = mixedMemory;

      // Extract segment IDs, handling both string and array formats
      const segments: string[] = typeof metadata.segment_ids === 'string'
        ? parseJsonSafely(metadata.segment_ids, [])
        : Array.isArray(metadata.segment_ids) ? metadata.segment_ids : [];

      logger.highlight(`Content was segmented into ${segments.length} parts`);
      segmentIds.push(...segments);

      // Use Promise.all to process segments in parallel
      const segmentPromises = segments.map(async (segmentId, index) => {
        try {
          const segment = await getMemory({
            client,
            memoryId: segmentId
          });
          const { metadata, content } = segment;

          const segmentInfo = [
            `\nSegment ${index+1}:`,
            `  Type: ${metadata?.type || "unknown"}`,
            `  Name: ${metadata?.name || "Unnamed"}`,
            `  Content preview: ${content.substring(0, 100)}...`,
            `  Connected to parent: ${metadata?.parent_id === parentId ? '✓' : 'WARNING: Missing proper parent relationship'}`
          ].join('\n');

          return { success: true, info: segmentInfo };
        } catch (error) {
          return {
            success: false,
            info: `Error retrieving segment ${segmentId}: ${error}`
          };
        }
      });

      const segmentResults = await Promise.all(segmentPromises);

      // Output results
      segmentResults.forEach(result => {
        if (result.success) {
          logger.info(result.info);
        } else {
          logger.error(result.info);
        }
      });
    } else {
      logger.warn("Content was not segmented - this might indicate an issue with auto-segmentation");
    }

    // Check for mixed content flag
    if (mixedMemory.metadata?.has_mixed_content) {
      logger.info("Mixed content flag correctly set: ✓");
    } else {
      logger.info("Mixed content flag not set");
    }

  } catch (error) {
    logger.error("Error during auto-segmentation testing:", error);
  } finally {
    // Clean up - delete parent and all segments
    logger.info("\nCleaning up memories...");

    try {
      // Delete all segments first - using Promise.all for parallel operations
      await Promise.all(
        segmentIds.map(segmentId =>
          deleteMemory({
            client,
            memoryId: segmentId
          })
            .then(() => logger.memory.delete(`Deleted segment ${segmentId}`))
            .catch(error => logger.error(`Error deleting segment ${segmentId}:`, error))
        )
      );

      // Then delete parent
      if (parentId) {
        await deleteMemory({
          client,
          memoryId: parentId
        });
        logger.memory.delete(`Deleted parent memory ${parentId}`);
      }
    } catch (error) {
      logger.error("Error during cleanup:", error);
    }
  }
}

/**
 * Test content-specific search capabilities
 */
async function testContentSpecificSearch(client: Client): Promise<void> {
  logger.title("Testing Content-Specific Search");

  const createdIds: string[] = [];

  try {
    // Create memories with different content types
    logger.memory.create("Creating test memories with different content types...");

    // Define the type for memory configurations
    interface MemoryConfig {
      content: string;
      name: string;
      metadata: Record<string, any>;
    }

    // Create memories with functional array of configurations
    const memoryConfigs: MemoryConfig[] = [
      {
        content: "Python is a versatile programming language used for data science, web development, and automation.",
        name: "Python Overview",
        metadata: { enable_llm_analysis: true }
      },
      {
        content: `def binary_search(arr, target):
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
        name: "Binary Search Algorithm",
        metadata: { enable_llm_analysis: true }
      },
      {
        content: MIXED_CONTENT_SAMPLE,
        name: "Sorting Algorithms Guide",
        metadata: { enable_llm_analysis: true, enable_auto_segmentation: true }
      }
    ];

    // Create memories and collect IDs
    const memoryResults = await Promise.all(
      memoryConfigs.map(config =>
        createMemory({
          client,
          content: config.content,
          name: config.name,
          metadata: config.metadata
        })
      )
    );

    // Process results and collect IDs
    memoryResults.forEach(memory => {
      if (memory?.id) {
        createdIds.push(memory.id);

        // Add segment IDs to the cleanup list if segmentation occurred
        if (memory.metadata?.segment_ids) {
          const segments: string[] = typeof memory.metadata.segment_ids === 'string'
            ? parseJsonSafely(memory.metadata.segment_ids, [])
            : Array.isArray(memory.metadata.segment_ids) ? memory.metadata.segment_ids : [];

          createdIds.push(...segments);
        }
      }
    });

    logger.highlight(`Created ${createdIds.length} test memories`);

    // Test different search queries
    const searchQueries = [
      "python programming language",
      "binary search implementation",
      "sorting algorithm time complexity",
      "quicksort vs merge sort"
    ];

    // Process queries sequentially
    for (const query of searchQueries) {
      logger.memory.search(`Searching for: "${query}"`);

      try {
        const searchResults = await searchMemories({
          client,
          query,
          metadataFilter: {
            enable_llm_analysis: true
          }
        });

        const { memories = [] } = searchResults;
        logger.highlight(`Found ${memories.length || 0} results:`);

        if (memories.length > 0) {
          memories.forEach((memory, index) => {
            const { name, metadata = {}, content } = memory;
            const displayName = name || metadata.name || 'Unnamed';
            const contentPreview = content.length > 100
              ? content.substring(0, 100) + "..."
              : content;

            const memoryInfo = [
              `${index + 1}. ${displayName}`,
              `   Type: ${metadata.type || "unknown"}`,
              `   Content: ${contentPreview}`
            ].join('\n');

            logger.result(memoryInfo);
          });
        } else {
          logger.warn("No results found");
        }

        // For code-related queries, try with content-type filter
        if (query.includes("search") || query.includes("algorithm")) {
          logger.memory.search(`Searching for "${query}" with code type filter:`);

          const codeResults = await searchMemoriesWithFilter({
            client,
            query,
            metadataFilter: {
              type: "code"
            }
          });

          const { memories: codeMemories = [] } = codeResults;
          logger.highlight(`Found ${codeMemories.length || 0} code results 📝`);

          if (codeMemories.length > 0) {
            codeMemories
              .map((memory, index) => {
                const { name, metadata = {} } = memory;
                const displayName = name || metadata.name || 'Unnamed';
                return `${index + 1}. ${displayName}\n   Type: ${metadata.type || "unknown"}`;
              })
              .forEach(info => logger.result(info));
          }
        }

      } catch (error) {
        logger.error(`Error searching for "${query}":`, error);
      }
    }

  } catch (error) {
    logger.error("Error during content-specific search testing:", error);
  } finally {
    // Clean up all created memories with Promise.all for parallel deletion
    logger.info("\nCleaning up created memories...");

    await Promise.all(
      createdIds.map(id =>
        deleteMemory({
          client,
          memoryId: id
        })
          .then(() => logger.memory.delete(`Deleted memory ${id}`))
          .catch(error => logger.error(`Error deleting memory ${id}:`, error))
      )
    );
  }
}

/**
 * Delete a memory by ID
 */
async function deleteMemory({
  client,
  memoryId
}: {
  client: Client,
  memoryId: string
}): Promise<any> {
  try {
    logger.memory.delete(`Deleting memory ${memoryId}`);

    const result = await client.callTool({
      name: "delete_memory",
      arguments: {
        memory_id: memoryId
      }
    });

    const textContent = extractTextContent(result);
    return textContent
      ? parseJsonSafely(textContent, result)
      : result;
  } catch (error) {
    logger.error(`Error deleting memory ${memoryId}:`, error);
    throw error;
  }
}

/**
 * Search for memories using the search_memories tool
 */
async function searchMemories({
  client,
  query,
  metadataFilter,
  topK = 5
}: {
  client: Client,
  query: string,
  metadataFilter?: Record<string, any>,
  topK?: number
}): Promise<SearchResult> {
  return searchMemoriesWithFilter({
    client,
    query,
    metadataFilter,
    topK
  });
}

/**
 * Search memories with metadata filtering
 */
async function searchMemoriesWithFilter({
  client,
  query,
  metadataFilter,
  topK = 5
}: {
  client: Client,
  query: string,
  metadataFilter?: Record<string, any>,
  topK?: number
}): Promise<SearchResult> {
  try {
    // Build arguments object immutably
    const args = {
      query,
      top_k: topK,
      ...(metadataFilter && { metadata_filter: metadataFilter })
    };

    const result = await client.callTool({
      name: "search_memories",
      arguments: args
    });

    const textContent = extractTextContent(result);
    return textContent
      ? parseJsonSafely(textContent, { memories: [] })
      : { memories: [] };
  } catch (error) {
    logger.error("Error searching memories:", error);
    throw error;
  }
}

// Run the client
main().catch((error: unknown) => {
  logger.error("Unhandled error:", error);
  process.exit(1);
});
