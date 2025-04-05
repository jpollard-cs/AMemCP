#!/usr/bin/env python3
"""
Custom SSE client implementation for the MCP Client, fixing header issues.
"""

import asyncio
import json
from typing import Any, AsyncGenerator, Dict, Optional

import aiohttp

from amem.utils.utils import setup_logger

# Set up logger
logger = setup_logger(__name__)

class WriteStream:
    """Implements the writer interface expected by MCP client."""
    
    def __init__(self, initial_url: str, session: aiohttp.ClientSession):
        # Start with the initial base message URL, might be updated
        self._url = initial_url 
        self.session = session
        self.logger = setup_logger(f"{__name__}.WriteStream")
        self._lock = asyncio.Lock() # Lock for updating URL

    async def update_url(self, new_path: str):
        """Update the target URL using the path received from the server."""
        async with self._lock:
            from urllib.parse import urlparse, urlunparse, urljoin
            # Get the base (scheme://netloc) from the current/initial URL
            parsed_current = urlparse(self._url)
            base = urlunparse((parsed_current.scheme, parsed_current.netloc, '', '', '', ''))
            # Join the base with the new path (e.g., /message?sessionId=...)
            self._url = urljoin(base, new_path)
            self.logger.info(f"WriteStream URL updated to: {self._url}")

    @property
    def url(self):
        # Property to safely get the current URL
        return self._url
    
    async def send(self, message: Any) -> None:
        """Send a message to the server using the current URL."""
        async with self._lock: # Ensure URL doesn't change mid-send
            current_url = self._url
        
        if not current_url:
            self.logger.error("WriteStream URL not set, cannot send message.")
            return
            
        try:
            self.logger.debug(f"Sending message to {current_url}: {message}")
            
            # Handle various message types
            if hasattr(message, 'model_dump_json'):  # Pydantic v2 model
                payload = message.model_dump_json()
                self.logger.debug(f"Used model_dump_json")
            elif hasattr(message, 'json'):  # Pydantic v1 or similar
                payload = message.json()
                self.logger.debug(f"Used json method")
            elif hasattr(message, 'to_json'):  # MCP specific method
                payload = message.to_json()
                self.logger.debug(f"Used to_json method")
            elif not isinstance(message, str):
                # For other data types, attempt direct JSON serialization
                try:
                    # If this is a dict-like object, convert to dict first
                    if hasattr(message, 'to_dict'):
                        message_dict = message.to_dict()
                        payload = json.dumps(message_dict)
                    elif hasattr(message, '__dict__'):
                        message_dict = vars(message)
                        # Attempt to clean the dict
                        for k in list(message_dict.keys()):
                            if k.startswith('_'):
                                del message_dict[k]
                        payload = json.dumps(message_dict)
                    else:
                        # Direct serialization attempt
                        payload = json.dumps(message)
                except TypeError as e:
                    self.logger.error(f"Cannot serialize object of type {type(message).__name__}: {e}")
                    # Fallback to string representation as a last resort
                    payload = json.dumps({"data": str(message), "error": "TypeError", "original_type": str(type(message))})
            else:
                # Already a string
                payload = message
                
            # Ensure the final payload is a properly encoded string
            if not isinstance(payload, str):
                payload = str(payload)
                
            self.logger.debug(f"Sending payload: {payload[:100]}...")
            
            # Use a shorter timeout for the HTTP request
            async with self.session.post(
                current_url, 
                data=payload,
                headers={"Content-Type": "application/json"},
                timeout=30  # 30 second timeout
            ) as response:
                if response.status != 200:
                    text = await response.text()
                    self.logger.error(f"Failed to send message: {response.status}, {text}")
                    
        except Exception as e:
            self.logger.error(f"Error in send: {e}", exc_info=True)
            raise

class ReadStream:
    """Implements the reader interface expected by MCP client."""
    
    def __init__(self, queue: asyncio.Queue):
        self.queue = queue
        self.logger = setup_logger(f"{__name__}.ReadStream")
    
    async def __call__(self, timeout: float = 30.0) -> Dict[str, Any]:
        """Read a message from the queue with timeout."""
        try:
            # Wait for the next message with a timeout
            self.logger.debug(f"Waiting for message with timeout {timeout}s, queue size: {self.queue.qsize()}")
            start_time = asyncio.get_event_loop().time()
            message = await asyncio.wait_for(self.queue.get(), timeout)
            elapsed = asyncio.get_event_loop().time() - start_time
            self.logger.debug(f"Got message after {elapsed:.2f}s: {str(message)[:100]}")
            self.queue.task_done()
            return message
        except asyncio.TimeoutError:
            # If we time out, return an empty message
            self.logger.debug(f"Timeout after {timeout}s waiting for message")
            return {}
        except Exception as e:
            self.logger.error(f"Error in read: {e}", exc_info=True)
            raise

async def custom_sse_client(
    sse_url: str, 
    message_url: str, # This is now the *initial* message URL
    headers: Dict[str, str] = None
) -> AsyncGenerator[tuple[ReadStream, WriteStream], None]:
    """
    Custom SSE client implementation that fixes header issues.
    
    Args:
        sse_url: URL for the SSE endpoint
        message_url: URL for the message endpoint
        headers: Optional additional headers
        
    Returns:
        AsyncGenerator yielding a tuple of (reader, writer)
    """
    logger.info(f"Setting up custom SSE client for SSE={sse_url}, Initial Message={message_url}")
    
    # Set up basic headers
    base_headers = {
        "Accept": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
    }
    
    # Merge with any provided headers
    if headers:
        base_headers.update(headers)
    
    # Queue for processing SSE events
    event_queue = asyncio.Queue()
    
    # Create a shared aiohttp session
    session = aiohttp.ClientSession()
    
    endpoint_received = asyncio.Event() # Event to signal endpoint URL is set
    
    # Create WriteStream early so process_sse_events can update it
    write_stream = WriteStream(message_url, session)
    
    async def process_sse_events():
        """Process SSE events from the server."""
        nonlocal write_stream # Allow modification of the outer scope variable
        try:
            logger.info(f"Connecting to SSE endpoint: {sse_url}")
            
            # We use an explicit timeout to avoid the connection being closed too early
            timeout = aiohttp.ClientTimeout(total=None)  # No timeout for SSE connection
            
            logger.debug(f"Opening SSE connection with headers: {base_headers}")
            async with session.get(sse_url, headers=base_headers, timeout=timeout) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Failed to connect to SSE endpoint: {response.status}, {error_text}")
                    endpoint_received.set() # Signal failure if connect fails
                    return
                
                logger.info(f"Connected to SSE endpoint: {sse_url} with status {response.status}")
                logger.debug(f"SSE response headers: {response.headers}")
                
                # Process the response
                buffer = ""
                logger.debug("Starting to read chunks from SSE stream")
                chunk_count = 0
                async for chunk in response.content.iter_chunked(1024):
                    chunk_count += 1
                    chunk_text = chunk.decode('utf-8')
                    logger.debug(f"Received chunk #{chunk_count}: {len(chunk_text)} chars")
                    buffer += chunk_text
                    
                    # Process complete events
                    events_processed = 0
                    while "\n\n" in buffer:
                        event_str, buffer = buffer.split("\n\n", 1)
                        events_processed += 1
                        logger.debug(f"Raw event #{events_processed}: {event_str[:100]}")
                        lines = event_str.strip().split("\n")
                        
                        sse_event_type = None
                        sse_event_data = ""
                        for line in lines:
                            if line.startswith("event:"):
                                sse_event_type = line[len("event:"):].strip()
                            elif line.startswith("data:"):
                                sse_event_data = line[len("data:"):].strip()
                            # Add handling for id: and retry: if needed later
                        
                        logger.debug(f"Parsed Event: type='{sse_event_type}', data='{sse_event_data[:100]}'")
                        
                        if sse_event_type == "endpoint" and sse_event_data:
                            logger.info(f"Received endpoint event, data: {sse_event_data}")
                            await write_stream.update_url(sse_event_data) # Update WriteStream URL
                            endpoint_received.set() # Signal that the endpoint is ready
                            logger.info("WriteStream URL updated and endpoint_received event set.")
                        elif sse_event_data: # Default to message event if type is not specified
                            try:
                                parsed_data = json.loads(sse_event_data)
                                logger.debug(f"Parsed message data: {str(parsed_data)[:100]}")
                                await event_queue.put(parsed_data)
                                logger.debug(f"Added message to queue, size: {event_queue.qsize()}")
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to parse message data: {sse_event_data}")
                        else:
                             logger.warning(f"Received SSE event with no data or unknown type: type={sse_event_type}")
                    
                    if events_processed > 0:
                        logger.debug(f"Processed {events_processed} events, buffer size: {len(buffer)}")
                
                logger.warning("SSE stream ended normally, which should not happen")
        except asyncio.CancelledError:
            logger.info("SSE event processor cancelled")
        except Exception as e:
            logger.error(f"Error in SSE event processor: {e}", exc_info=True)
            endpoint_received.set() # Signal failure on exception
        finally:
            logger.info("SSE event processor stopped")
    
    # Start the SSE event processor
    processor_task = asyncio.create_task(process_sse_events())
    
    # Wait for the endpoint event to be received (or failure/timeout)
    endpoint_wait_timeout = 15.0 # Wait up to 15 seconds for the endpoint event
    logger.info(f"Waiting up to {endpoint_wait_timeout}s for endpoint event from server...")
    try:
        await asyncio.wait_for(endpoint_received.wait(), timeout=endpoint_wait_timeout)
        logger.info("Endpoint event received or processing failed/cancelled. Proceeding.")
    except asyncio.TimeoutError:
        logger.error(f"Timed out waiting for endpoint event after {endpoint_wait_timeout}s.")
        # Ensure the processor task is cancelled if we time out
        processor_task.cancel()
        # Close the session explicitly on timeout
        await session.close() 
        # Raise an exception to stop the connection process in the main client
        raise TimeoutError(f"Server did not send endpoint information within {endpoint_wait_timeout}s")
        
    # Check if the endpoint URL was actually set (it might be set on failure too)
    # We rely on WriteStream having a valid URL now. A more robust check
    # could involve passing status back from process_sse_events.
    
    # Create the reader stream (writer was created earlier)
    read_stream = ReadStream(event_queue)
    
    try:
        # Yield the streams only after endpoint is confirmed
        yield read_stream, write_stream
    finally:
        # Clean up
        logger.info("Closing SSE client")
        processor_task.cancel()
        await session.close() 