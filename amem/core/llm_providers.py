"""
Provides a standard provider interface
"""

from __future__ import annotations

import asyncio
import os
import random
from abc import ABC, abstractmethod
from time import perf_counter
from typing import Dict, List, Optional

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import AsyncOpenAI

from amem.utils.utils import setup_logger

logger = setup_logger(__name__)

# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #


async def _with_timeout_and_retry(
    coro_factory, timeout_seconds: int, task: str, max_retries: int = 2, backoff_factor: float = 1.5
):
    """
    Run coroutine with timeout, retry logic, and exponential backoff.

    Args:
        coro_factory: Function that returns a fresh coroutine for each attempt
        timeout_seconds: Timeout for each individual attempt
        task: Description of the task for error messages
        max_retries: Maximum number of retry attempts (0 = no retries)
        backoff_factor: Multiplier for delay between retries
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            coro = coro_factory() if callable(coro_factory) else coro_factory
            return await asyncio.wait_for(coro, timeout_seconds)
        except asyncio.TimeoutError as e:
            last_exception = e
            if attempt < max_retries:
                delay = backoff_factor**attempt + random.uniform(0, 0.1)  # Add jitter
                logger.warning(f"{task} timeout on attempt {attempt + 1}/{max_retries + 1}, retrying in {delay:.2f}s")
                await asyncio.sleep(delay)
            else:
                break
        except Exception as e:
            # For non-timeout errors, don't retry
            logger.error(f"{task} failed with non-timeout error: {e}")
            raise

    raise RuntimeError(
        f"{task} failed after {max_retries + 1} attempts (timeout: {timeout_seconds}s)"
    ) from last_exception


async def _with_timeout(coro, seconds: int, task: str):
    """Simple timeout wrapper for backward compatibility."""
    return await _with_timeout_and_retry(lambda: coro, seconds, task, max_retries=0)


# --------------------------------------------------------------------------- #
# base class                                                                  #
# --------------------------------------------------------------------------- #


class BaseLLMProvider(ABC):
    """Async‑first provider base with common functionality."""

    def __init__(self, api_key: Optional[str], model: str, embed_model: str, provider_name: str):
        self.api_key = api_key
        self.model = model
        self.embed_model = embed_model
        self.provider_name = provider_name

        if not self.api_key:
            raise ValueError(f"{provider_name} API key missing")

        logger.info(f"{provider_name} provider ready (%s / %s)", model, embed_model)

    # async interface ------------------------------------------------------- #
    @abstractmethod
    async def get_completion(self, prompt: str, config: Optional[Dict] = None, **kwargs) -> str: ...

    @abstractmethod
    async def get_embeddings(self, text: str, task_type: Optional[str] = None) -> List[float]: ...


# --------------------------------------------------------------------------- #
# OpenAI                                                                      #
# --------------------------------------------------------------------------- #


class OpenAIProvider(BaseLLMProvider):
    def __init__(
        self,
        model: str = "gpt-4o",
        embed_model: str = "text-embedding-ada-002",
        api_key: Optional[str] = None,
    ):
        super().__init__(
            api_key=api_key or os.getenv("OPENAI_API_KEY"), model=model, embed_model=embed_model, provider_name="OpenAI"
        )

        # LangChain async‑ready clients
        self.chat = ChatOpenAI(model=model, api_key=self.api_key, temperature=0.2)
        self.embed = OpenAIEmbeddings(model=embed_model, api_key=self.api_key)

        # raw async client for last‑resort fallback
        self._aio_openai = AsyncOpenAI(api_key=self.api_key)

    # --------------------------------------------------------------------- #

    async def get_completion(self, prompt: str, config: Optional[Dict] = None, **kwargs) -> str:
        """Async completion using OpenAI, supporting JSON mode via config."""
        messages = [HumanMessage(content=prompt)]

        # Check config for OpenAI-specific JSON mode request
        openai_response_format = None
        if (
            config
            and config.get("response_format")
            and isinstance(config["response_format"], dict)
            and config["response_format"].get("type") == "json_object"
        ):

            openai_response_format = config["response_format"]

        client = (
            ChatOpenAI(
                model=self.model,
                api_key=self.api_key,
                temperature=0.2,
                # Pass the extracted response_format if JSON mode was requested
                response_format=openai_response_format,
            )
            if openai_response_format
            else self.chat
        )

        resp = await _with_timeout(client.ainvoke(messages), 30, "OpenAI completion")
        return resp.content

    # --------------------------------------------------------------------- #

    async def get_embeddings(self, text: str, task_type: Optional[str] = None) -> List[float]:
        t0 = perf_counter()
        try:
            emb = await _with_timeout_and_retry(
                lambda: self.embed.aembed_query(text), 15, "OpenAI embedding", max_retries=1
            )
            logger.info("OpenAI embedding in %.2fs", perf_counter() - t0)
            return emb
        except Exception as exc:
            logger.warning("LangChain embedding failed: %s – trying raw OpenAI", exc)

        resp = await _with_timeout_and_retry(
            lambda: self._aio_openai.embeddings.create(
                model=self.embed_model,
                input=[text],
                encoding_format="float",
            ),
            15,
            "raw OpenAI embedding",
            max_retries=1,
        )
        if not resp.data or len(resp.data) == 0:
            raise RuntimeError("No embedding data returned from OpenAI")

        return resp.data[0].embedding


# --------------------------------------------------------------------------- #
# Gemini                                                                      #
# --------------------------------------------------------------------------- #


class GeminiProvider(BaseLLMProvider):
    def __init__(
        self,
        model: str = "gemini-2.5-flash-preview-05-20",
        embed_model: str = "gemini-embedding-exp-03-07",
        api_key: Optional[str] = None,
    ):
        super().__init__(
            api_key=api_key or os.getenv("GOOGLE_API_KEY"), model=model, embed_model=embed_model, provider_name="Gemini"
        )

        # Gemini needs the model prefix for embeddings
        self.embed_model = f"models/{embed_model}"

        self.chat = ChatGoogleGenerativeAI(model=model, api_key=self.api_key, temperature=0.2)
        self.embed_default = GoogleGenerativeAIEmbeddings(
            model=self.embed_model, api_key=self.api_key, task_type="RETRIEVAL_DOCUMENT"
        )

    # --------------------------------------------------------------------- #

    async def get_completion(self, prompt: str, config: Optional[Dict] = None, **kwargs) -> str:
        """Get completion using the Gemini API, supporting structured output via config."""
        messages = [HumanMessage(content=prompt)]

        # Use the passed config directly if available
        cfg = config

        # Instantiate the client without generation_config
        chat_instance = self.chat

        # Pass the config dictionary directly to ainvoke
        resp = await _with_timeout(chat_instance.ainvoke(messages, generation_config=cfg), 30, "Gemini completion")
        content = resp.content

        # Removed manual JSON check - rely on API handling based on config['response_mime_type']
        # if cfg and cfg.get('response_mime_type') == 'application/json':
        #     if content.startswith("```"):
        #         content = content.strip("` \n")
        #     json.loads(content)

        return content

    # --------------------------------------------------------------------- #

    async def get_embeddings(self, text: str, task_type: Optional[str] = None) -> List[float]:
        emb_client = (
            GoogleGenerativeAIEmbeddings(model=self.embed_model, api_key=self.api_key, task_type=task_type)
            if task_type and task_type != "RETRIEVAL_DOCUMENT"
            else self.embed_default
        )
        return await _with_timeout(emb_client.aembed_query(text), 30, "Gemini embedding")


# --------------------------------------------------------------------------- #
# factory                                                                     #
# --------------------------------------------------------------------------- #


def get_provider(
    backend: str,
    model: Optional[str] = None,
    embed_model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> BaseLLMProvider:
    backend = backend.lower()
    if backend == "openai":
        return OpenAIProvider(model or "gpt-4o", embed_model or "text-embedding-ada-002", api_key)
    if backend == "gemini":
        return GeminiProvider(
            model or "gemini-2.5-flash-preview-05-20", embed_model or "gemini-embedding-exp-03-07", api_key
        )
    raise ValueError(f"Unsupported backend: {backend}")
