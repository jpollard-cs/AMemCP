#!/usr/bin/env python3
"""
LLM Providers module
Provides a unified interface for different LLM backends, including OpenAI and Google Gemini
"""

import json
import os
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

# Langchain imports
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from amem.utils.utils import setup_logger

# Configure logging
logger = setup_logger(__name__)


class BaseLLMProvider(ABC):
    """Base class for LLM providers."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize base provider with API key."""
        self.api_key = api_key

    @abstractmethod
    def get_completion(self, prompt: str, response_format: Optional[Dict] = None) -> str:
        """Get completion from LLM"""

    @abstractmethod
    def get_embeddings(self, text: str, task_type: Optional[str] = None) -> List[float]:
        """Get embeddings for text

        Args:
            text: The text to embed
            task_type: Optional task type for specialized embeddings

        Returns:
            List of floats representing the embedding vector
        """


class OpenAIProvider(BaseLLMProvider):
    """OpenAI provider implementation"""

    def __init__(
        self, model: str = "gpt-4o", embed_model: str = "text-embedding-ada-002", api_key: Optional[str] = None
    ):
        """Initialize OpenAI provider"""
        # Initialize base class with API key
        super().__init__(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

        # Store the model name for LLMController
        self.model = model
        self.embed_model = embed_model

        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        # Set API key in environment
        os.environ["OPENAI_API_KEY"] = self.api_key

        # Initialize langchain clients
        self.chat_client = ChatOpenAI(model=model, api_key=self.api_key, temperature=0.2)

        self.embedding_client = OpenAIEmbeddings(model=embed_model, api_key=self.api_key)

        logger.info(f"Initialized OpenAI provider with model: {model}")

    def get_completion(self, prompt: str, response_format: Optional[Dict] = None) -> str:
        """Get completion from OpenAI"""
        try:
            messages = [HumanMessage(content=prompt)]

            # Handle JSON response format
            response_format_dict = None
            if response_format and response_format.get("type") == "json_schema":
                response_format_dict = {"type": "json_object"}
                logger.debug("Requesting JSON response format from OpenAI")

            # Create a new client with the response format if needed
            if response_format_dict:
                temp_client = ChatOpenAI(
                    model=self.model, api_key=self.api_key, temperature=0.2, response_format=response_format_dict
                )
                response = temp_client.invoke(messages)
            else:
                response = self.chat_client.invoke(messages)

            return response.content

        except Exception as e:
            logger.error(f"Error getting completion from OpenAI: {e}")
            raise

    def get_embeddings(self, text: str, task_type: Optional[str] = None) -> List[float]:
        """Get embeddings from OpenAI"""
        try:
            # Add timer for performance monitoring
            import time

            start_time = time.time()

            logger.info(f"Generating OpenAI embeddings for text of length {len(text)}")

            # Try with a timeout to avoid hanging
            try:
                from concurrent.futures import ThreadPoolExecutor, TimeoutError

                with ThreadPoolExecutor() as executor:
                    # Create a future for the embedding task
                    future = executor.submit(self.embedding_client.embed_query, text)

                    # Wait for the result with a timeout
                    embedding = future.result(timeout=15)  # 15 second timeout

                    elapsed_time = time.time() - start_time
                    logger.info(f"Generated OpenAI embedding in {elapsed_time:.2f}s (length: {len(embedding)})")
                    return embedding

            except (TimeoutError, Exception) as e:
                # If OpenAI embeddings time out, try direct API call as fallback
                logger.warning(f"OpenAI embedding failed after {time.time() - start_time:.2f}s: {str(e)[:200]}")
                logger.warning("Falling back to direct OpenAI API call for embeddings")

                # Try direct API call as fallback
                import openai

                response = openai.embeddings.create(model=self.embed_model, input=[text], encoding_format="float")
                embedding = response.data[0].embedding

                elapsed_time = time.time() - start_time
                logger.info(
                    f"Generated OpenAI embedding via direct API in {elapsed_time:.2f}s (length: {len(embedding)})"
                )
                return embedding

        except Exception as e:
            logger.error(
                f"Error getting embeddings from OpenAI after {time.time() - start_time:.2f}s: {e}", exc_info=True
            )
            # Last resort - return a mock embedding to prevent system failure
            logger.error("Returning mock embedding as last resort")
            import numpy as np

            return list(np.random.rand(1536))  # Standard size for OpenAI embeddings


class GeminiProvider(BaseLLMProvider):
    """Google Gemini provider implementation"""

    def __init__(
        self,
        model: str = "gemini-2.5-pro-exp-03-25",
        embed_model: str = "gemini-embedding-exp-03-07",
        api_key: Optional[str] = None,
    ):
        """Initialize Gemini provider"""
        # Initialize base class with API key
        super().__init__(api_key=api_key or os.environ.get("GOOGLE_API_KEY"))

        # Store the model name for LLMController and internal use
        self.model = model  # For compatibility with LLMController
        self.llm_model_name = model  # Keep for direct API calls if needed

        # Store the embedding model name for LLMController
        self.embed_model = embed_model  # For compatibility with LLMController

        # Correct format for embedding model names in different contexts
        self.gemini_embed_model_name = f"models/{embed_model}"

        if not self.api_key:
            raise ValueError("Google API key is required")

        # Initialize langchain clients
        self.chat_client = ChatGoogleGenerativeAI(model=model, api_key=self.api_key, temperature=0.2)

        # Initialize embedding client with a longer timeout
        self.embedding_client = GoogleGenerativeAIEmbeddings(
            model=self.gemini_embed_model_name,
            api_key=self.api_key,
            task_type="RETRIEVAL_DOCUMENT",  # Default task type
        )

        logger.info(f"Initialized Gemini provider with model: {model}, embedding model: {embed_model}")

    def get_completion(self, prompt: str, response_format: Optional[Dict] = None) -> str:
        """Get completion from Gemini"""
        try:
            messages = [HumanMessage(content=prompt)]

            # Handle JSON response format
            generation_config = {}
            if response_format and response_format.get("type") == "json_schema":
                generation_config = {
                    "response_mime_type": "application/json",
                }
                logger.debug("Requesting JSON response format from Gemini")

            # Create a temp client with the generation config
            temp_client = ChatGoogleGenerativeAI(
                model=self.model,
                api_key=self.api_key,
                temperature=0.2,
                generation_config=generation_config if generation_config else None,
            )

            logger.debug(f"Sending completion request to Gemini model: {self.llm_model_name}")
            response = temp_client.invoke(messages)

            content = response.content

            # Validate JSON content if requested
            if generation_config.get("response_mime_type") == "application/json":
                try:
                    # Strip potential markdown fences before parsing JSON
                    if content.startswith("```json"):
                        content = content.strip("```json\n ")
                    elif content.startswith("```"):
                        content = content.strip("```\n ")

                    json.loads(content)  # Validate JSON
                    logger.debug("Received valid JSON response from Gemini")
                except json.JSONDecodeError as json_err:
                    logger.error(f"Gemini response was not valid JSON despite request: {json_err}")
                    logger.error(f"Raw response content: {content[:500]}...")
                    raise ValueError(f"Invalid JSON response from Gemini: {content[:100]}...") from json_err

            return content

        except Exception as e:
            logger.error(f"Error getting completion from Gemini via langchain: {e}", exc_info=True)

    def get_embeddings(self, text: str, task_type: Optional[str] = None) -> List[float]:
        """Get embeddings from Gemini using the langchain wrapper with task type support."""
        try:
            effective_task_type = task_type or "RETRIEVAL_DOCUMENT"

            with ThreadPoolExecutor() as executor:
                # Create a future for the embedding task
                if effective_task_type != "RETRIEVAL_DOCUMENT":
                    temp_embedding_client = GoogleGenerativeAIEmbeddings(
                        model=self.gemini_embed_model_name, api_key=self.api_key, task_type=effective_task_type
                    )
                    future = executor.submit(temp_embedding_client.embed_query, text)
                else:
                    future = executor.submit(self.embedding_client.embed_query, text)

                embedding = future.result(timeout=30)
                logger.info(f"Generated embedding via gemini {self.gemini_embed_model_name}")
                return embedding
        except Exception:
            logger.error(
                f"Error getting embeddings from Gemini API with model {self.gemini_embed_model_name}", exc_info=True
            )


def get_provider(
    backend: str, model: str = None, embed_model: str = None, api_key: str = None, **kwargs
) -> BaseLLMProvider:
    """Get the appropriate LLM provider based on backend name."""
    logger.info(f"Fetching LLM provider for backend: {backend} with model: {model}, embed_model: {embed_model}")

    # Ensure API key is available if needed
    if backend in ["openai", "gemini"] and not api_key:
        api_key = os.getenv("GOOGLE_API_KEY" if backend == "gemini" else "OPENAI_API_KEY") or os.getenv("API_KEY")
        if not api_key:
            logger.warning(f"API key not explicitly provided and not found in environment for {backend}")
            # Allow proceeding for now, the provider init will raise ValueError if required

    if backend.lower() == "openai":
        logger.info(f"Initializing OpenAI provider with model: {model}, embed_model: {embed_model}")
        return OpenAIProvider(model=model, embed_model=embed_model, api_key=api_key)
    elif backend.lower() == "gemini":
        logger.info(f"Initializing Gemini provider with model: {model}, embed_model: {embed_model}")
        return GeminiProvider(model=model, embed_model=embed_model, api_key=api_key)
    else:
        raise ValueError(f"Unsupported LLM backend: {backend}")
