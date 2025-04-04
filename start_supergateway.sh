#!/bin/bash

# Default port
PORT=8000

# Check for custom port in arguments
for arg in "$@"; do
  if [[ $arg =~ ^--port=([0-9]+)$ ]]; then
    PORT="${BASH_REMATCH[1]}"
  fi
done

echo "Using port: $PORT"

# Kill any existing Supergateway instances
pkill -f supergateway || true

# Check if the port is in use and kill the process if needed
if command -v lsof >/dev/null 2>&1; then
  # If lsof is available
  PORT_PID=$(lsof -i tcp:$PORT -t 2>/dev/null)
  if [ ! -z "$PORT_PID" ]; then
    echo "Port $PORT is in use by PID: $PORT_PID - attempting to kill..."
    kill -9 $PORT_PID 2>/dev/null || true
    sleep 1
  fi
fi

# Start Supergateway with our simple MCP server
npx supergateway \
  --stdio "python ./simple_mcp_server.py" \
  --port $PORT \
  --cors \
  --logLevel info

# Alternative options:
# Using ssePath explicitly
# npx supergateway \
#   --stdio "python ./simple_mcp_server.py" \
#   --port $PORT \
#   --cors \
#   --logLevel info \
#   --ssePath / \
#   --messagePath /message \
#   --healthEndpoint /health

# To connect to a remote MCP server:
# npx supergateway \
#   --sse "https://your-remote-mcp-server-url" \
#   --header "Authorization: Bearer your-auth-token" \
#   --outputTransport stdio
