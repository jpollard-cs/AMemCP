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

      // Test memory operations
      await testMemoryOperations(client);
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
 * Search for memories using the search_memories tool (if available)
 */
async function searchMemories(
  client: Client, 
  query: string, 
  limit?: number
): Promise<SearchResult> {
  console.log("Searching memories with query:", query, limit ? `(limit: ${limit})` : '');
  
  try {
    const searchParams: Record<string, any> = { query };
    if (limit) searchParams.top_k = limit; // Using top_k as per the tool description
    
    const result = await client.callTool({
      name: "search_memories",
      arguments: searchParams
    });
    
    console.log("Raw search result:", result);
    
    // Extract the search results from the response
    if (result.content && Array.isArray(result.content) && result.content.length > 0) {
      if (result.content[0].text) {
        // The text field contains a JSON string that needs to be parsed
        try {
          return JSON.parse(result.content[0].text) as SearchResult;
        } catch (parseError) {
          console.error("Error parsing JSON response:", parseError);
          return result.content[0].text as unknown as SearchResult;
        }
      }
    }
    
    return result as unknown as SearchResult;
  } catch (error) {
    // If search_memories tool doesn't exist
    console.error("Search memories tool not available:", error);
    throw new Error("Search memories tool not available");
  }
}

// Run the client
main().catch((error: unknown) => {
  console.error("Unhandled error:", error);
  process.exit(1);
});
