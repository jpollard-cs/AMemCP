/**
 * Declaration file for MCP SDK modules
 */

declare module '@modelcontextprotocol/sdk/client/index' {
  export class Client<Req = any, Res = any, Meta = any> {
    constructor(
      info: { name: string; version: string },
      opts?: { capabilities: { tools?: {}; prompts?: {}; resources?: {} } }
    );
    connect(transport: any): Promise<void>;
    listTools(): Promise<Tool[]>;
    // Add other methods as needed
  }
}

declare module '@modelcontextprotocol/sdk/client/sse' {
  export class SSEClientTransport {
    constructor(url: URL, opts?: any);
    close(): Promise<void>;
    start(): Promise<void>;
    send(message: any): Promise<void>;
    onclose?: () => void;
    onerror?: (error: Error) => void;
    onmessage?: (message: any) => void;
  }
}

declare module '@modelcontextprotocol/sdk/types' {
  export interface Tool {
    name: string;
    description: string;
    parameters?: any;
    // Add other properties as needed
  }
}
