#!/usr/bin/env python3
"""
LLM Providers module
Provides a unified interface for different LLM backends, including OpenAI and Google Gemini
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import google.generativeai as genai
import litellm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseLLMProvider(ABC):
    """Base class for LLM providers"""

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
        self, model: str = "gpt-4", embed_model: str = "text-embedding-ada-002", api_key: Optional[str] = None
    ):
        """Initialize OpenAI provider"""
        self.model = model
        self.embed_model = embed_model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        # Set API key
        os.environ["OPENAI_API_KEY"] = self.api_key
        logger.info(f"Initialized OpenAI provider with model: {model}, embedding model: {embed_model}")

    def get_completion(self, prompt: str, response_format: Optional[Dict] = None) -> str:
        """Get completion from OpenAI"""
        try:
            messages = [{"role": "user", "content": prompt}]

            response = litellm.completion(model=self.model, messages=messages, response_format=response_format)

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error getting completion from OpenAI: {e}")
            raise

    def get_embeddings(self, text: str, task_type: Optional[str] = None) -> List[float]:
        """Get embeddings from OpenAI"""
        try:
            response = litellm.embedding(model=self.embed_model, input=[text])
            return response["data"][0]["embedding"]
        except Exception as e:
            logger.error(f"Error getting embeddings from OpenAI: {e}")
            raise


class GeminiProvider(BaseLLMProvider):
    """Google Gemini provider implementation"""

    def __init__(
        self,
        model: str = "gemini-2.5-pro-exp-03-25",
        embed_model: str = "gemini-embedding-exp-03-07",
        api_key: Optional[str] = None,
    ):
        """Initialize Gemini provider"""
        # Store the model name for LLMController and internal use
        self.model = model  # For compatibility with LLMController
        self.llm_model_name = model  # Keep for direct API calls if needed

        # Store the embedding model name for LLMController
        self.embed_model = embed_model  # For compatibility with LLMController

        # Correct format for direct genai embedding API calls
        self.genai_embed_model_name = f"models/{embed_model}"
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")

        if not self.api_key:
            raise ValueError("Google API key is required")

        # Configure Google AI for direct embeddings
        genai.configure(api_key=self.api_key)

        logger.info(f"Initialized Gemini provider with model: {model}, embedding model: {embed_model}")

    def get_completion(self, prompt: str, response_format: Optional[Dict] = None) -> str:
        """Get completion from Gemini"""
        try:
            messages = [{"role": "user", "content": prompt}]

            generation_config = {}
            if response_format and response_format.get("type") == "json_schema":
                generation_config = {
                    "response_mime_type": "application/json",
                }
                logger.debug("Requesting JSON response format from Gemini")

            response = litellm.completion(
                model=f"gemini/{self.llm_model_name}",
                messages=messages,
                api_key=self.api_key,
                temperature=0.2,
                generation_config=generation_config if generation_config else None,
            )

            if response and response.choices and response.choices[0].message:
                content = response.choices[0].message.content
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
            else:
                raise ValueError("Invalid or empty response from litellm Gemini completion")

        except Exception as e:
            logger.error(f"Error getting completion from Gemini via litellm: {e}", exc_info=True)
            try:
                logger.warning("Falling back to direct genai API call for completion")
                model = genai.GenerativeModel(self.llm_model_name)
                gen_config_direct = genai.types.GenerationConfig(**generation_config) if generation_config else None
                response_direct = model.generate_content(prompt, generation_config=gen_config_direct)
                if gen_config_direct and gen_config_direct.response_mime_type == "application/json":
                    json.loads(response_direct.text)
                    return response_direct.text
                else:
                    return response_direct.text
            except Exception as fallback_e:
                logger.error(f"Fallback to direct API also failed: {fallback_e}", exc_info=True)
                raise

    def get_embeddings(self, text: str, task_type: Optional[str] = None) -> List[float]:
        """Get embeddings from Gemini using the top-level function."""
        try:
            effective_task_type = task_type or "RETRIEVAL_DOCUMENT"
            # Use the top-level genai.embed_content function
            result = genai.embed_content(
                model=self.genai_embed_model_name,  # Use the models/ prefixed name
                content=text,  # Pass text directly
                task_type=effective_task_type,
            )
            logger.debug(f"Generated embedding with task_type: {effective_task_type}")
            return result["embedding"]
        except Exception as e:
            logger.error(f"Error getting embeddings from Gemini API: {e}", exc_info=True)
            raise


class OllamaProvider(BaseLLMProvider):
    """Ollama provider implementation for local models"""

    def __init__(self, model: str = "llama3", embed_model: str = "llama3", base_url: str = "http://localhost:11434"):
        """Initialize Ollama provider"""
        self.model = model
        self.embed_model = embed_model
        self.base_url = base_url
        logger.info(f"Initialized Ollama provider with model: {model}")

    def get_completion(self, prompt: str, response_format: Optional[Dict] = None) -> str:
        """Get completion from Ollama"""
        try:
            messages = [{"role": "user", "content": prompt}]

            response = litellm.completion(model=f"ollama/{self.model}", messages=messages, api_base=self.base_url)

            # Handle JSON format requests
            if response_format and response_format.get("type") == "json_schema":
                # Try to ensure response is valid JSON
                content = response.choices[0].message.content
                try:
                    # Check if it's already valid JSON
                    json.loads(content)
                    return content
                except json.JSONDecodeError:
                    # Try to extract JSON from response if wrapped in backticks
                    if "```json" in content and "```" in content:
                        json_content = content.split("```json")[1].split("```")[0].strip()
                        return json_content
                    elif "```" in content:
                        json_content = content.split("```")[1].split("```")[0].strip()
                        return json_content
                    # Return as-is if we can't extract JSON
                    return content

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error getting completion from Ollama: {e}")
            # Default mock response for testing
            if response_format and response_format.get("type") == "json_schema":
                return json.dumps(
                    {
                        "keywords": ["test", "mock", "memory"],
                        "context": "Test content for mocking memory operations",
                        "tags": ["test", "mock"],
                    }
                )
            return "Mock Ollama response"

    def get_embeddings(self, text: str, task_type: Optional[str] = None) -> List[float]:
        """Get embeddings from Ollama"""
        try:
            import requests

            response = requests.post(
                f"{self.base_url}/api/embeddings", json={"model": self.embed_model, "prompt": text}
            )
            response.raise_for_status()
            data = response.json()
            return data["embedding"]
        except Exception as e:
            logger.error(f"Error getting embeddings from Ollama: {e}")
            # Return mock embeddings for testing
            import numpy as np

            return list(np.random.rand(384))


class MockProvider(BaseLLMProvider):
    """Mock provider for testing"""

    def __init__(self, model: str = "mock", embed_model: str = "mock"):
        """Initialize mock provider"""
        self.model = model
        self.embed_model = embed_model
        logger.info(f"Initialized Mock provider with model: {model}")

    def get_completion(self, prompt: str, response_format: Optional[Dict] = None) -> str:
        """Get mock completion"""
        if response_format and response_format.get("type") == "json_schema":
            return json.dumps(
                {
                    "keywords": ["test", "mock", "memory"],
                    "context": "Test content for mocking memory operations",
                    "tags": ["test", "mock"],
                }
            )
        return "Mock LLM response"

    def get_embeddings(self, text: str, task_type: Optional[str] = None) -> List[float]:
        """Get mock embeddings"""
        import numpy as np

        return list(np.random.rand(384))


def get_provider(
    backend: str, model: str = None, embed_model: str = None, api_key: str = None, **kwargs
) -> BaseLLMProvider:
    """
    Factory function to get the appropriate LLM provider

    Args:
        backend: Provider backend (openai, gemini, ollama, mock)
        model: Model name for text generation
        embed_model: Model name for embeddings
        api_key: API key for the provider
        **kwargs: Additional provider-specific arguments

    Returns:
        BaseLLMProvider: The appropriate LLM provider
    """
    backend = backend.lower()

    if backend == "openai":
        return OpenAIProvider(
            model=model or "gpt-4o", embed_model=embed_model or "text-embedding-ada-002", api_key=api_key
        )
    elif backend == "gemini":
        return GeminiProvider(
            model=model or "gemini-1.5-flash-latest",
            embed_model=embed_model or "text-embedding-004",
            api_key=api_key,
        )
    elif backend == "ollama":
        return OllamaProvider(
            model=model or "llama3",
            embed_model=embed_model or "llama3",
            base_url=kwargs.get("base_url", "http://localhost:11434"),
        )
    elif backend == "mock":
        return MockProvider(model=model or "mock", embed_model=embed_model or "mock")
    else:
        raise ValueError(f"Unsupported provider: {backend}")
