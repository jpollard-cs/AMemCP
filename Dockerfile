FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variable for MPS fallback
ENV PYTORCH_ENABLE_MPS_FALLBACK=1

# Copy Python files
COPY *.py .
COPY *.env* .

# Create directory for Chroma database
RUN mkdir -p /app/chroma_db
VOLUME /app/chroma_db

# Expose port
EXPOSE 8000

# Healthcheck to verify the server is running
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/ || exit 1

# Set environment variables (these will be overridden by docker-compose)
ENV LLM_MODEL=""
ENV EMBED_MODEL=""
ENV OPENAI_API_KEY=""
ENV GOOGLE_API_KEY=""
ENV API_KEY=""

# Command to run the MCP server
CMD ["python", "mcp_fastmcp_server.py"]
