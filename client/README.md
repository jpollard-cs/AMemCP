# AMemCP TypeScript Client

A TypeScript client built for **testing** the AMemCP server.

## Requirements

- Node.js 22.14.0 or later (use NVM to install)
- npm

## Setup

1. Install dependencies:
   ```
   npm install
   ```

2. Build the client:
   ```
   npm run build
   ```

3. Run the client:
   ```
   npm start
   ```

Or you can use the convenient build script that does all of the above:
```
./build.sh
```

## Development

- `npm run dev` - Build and run in one command
- Source code is in the `src/` directory
- Built JavaScript files are output to the `dist/` directory

## Features

This client uses the Model Context Protocol to communicate with the AMemCP server:

- Connects to the server using Server-Sent Events (SSE)
- Lists available tools
- Can execute tool calls
- Handles server redirects and status responses

## SDK Information

This client uses the Model Context Protocol (MCP) TypeScript SDK:

- Package: `@modelcontextprotocol/sdk`
- Components Used:
  - `Client` - Core MCP client for communicating with the server
  - `SSEClientTransport` - Transport layer for Server-Sent Events
  - `Tool` - Interface for working with server tools

For more information on the SDK, visit the [Model Context Protocol documentation](https://github.com/Chainlit/model-context-protocol).

## Related Components

- Python AMemCP server in the parent directory
- Python test client in `tests/test_client.py`
- Example scripts in the `examples/` directory
