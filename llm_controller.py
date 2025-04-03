#!/usr/bin/env python3
"""
LLM Controller Module
Handles interactions with various LLM backends
"""

import os
import json
import logging
from typing import Dict, List, Any, Union, Optional
from abc import ABC, abstractmethod

from llm_providers import get_provider, BaseLLMProvider

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseLLMController(ABC):
    """Base class for LLM Controllers"""
    
    @abstractmethod
    def get_empty_values(self, current_time: str) -> Dict[str, Any]:
        """Get empty values for analysis"""
        pass
    
    @abstractmethod
    def get_completion(self, 
                      prompt: str, 
                      response_format: Optional[Dict] = None, 
                      **kwargs) -> str:
        """Get completion from the LLM"""
        pass


class LLMController(BaseLLMController):
    """Controller for memory metadata generation"""
    
    def __init__(self, 
                backend: str = "openai", 
                model: str = None, 
                embed_model: str = None,
                api_key: str = None, 
                **kwargs):
        """Initialize LLM controller"""
        # Get the appropriate provider
        self.provider = get_provider(
            backend=backend, 
            model=model, 
            embed_model=embed_model, 
            api_key=api_key,
            **kwargs
        )
        
        # Store provider details for reference
        self.backend = backend
        self.model = self.provider.model
        self.embed_model = self.provider.embed_model
        
        logger.info(f"Initialized LLM Controller with provider: {backend}, model: {self.model}")
        
    def get_empty_values(self, current_time: str) -> Dict[str, Any]:
        """Get empty values for the note metadata"""
        return {
            "keywords": [],
            "context": "",
            "summary": "",
            "type": "general",
            "created_at": current_time,
            "updated_at": current_time,
            "related_notes": [],
            "sentiment": "neutral",
            "importance": 0.5,
            "tags": []
        }
        
    def get_completion(self, 
                      prompt: str, 
                      response_format: Optional[Dict] = None, 
                      **kwargs) -> str:
        """Get completion from the provider"""
        try:
            response = self.provider.get_completion(prompt=prompt, response_format=response_format)
            return response
        except Exception as e:
            logger.error(f"Error getting completion from provider: {e}")
            # Return empty JSON if response format is JSON
            if response_format and response_format.get("type") == "json_schema":
                return "{}"
            return ""
            
    def get_embeddings(self, text: str, task_type: Optional[str] = None) -> List[float]:
        """Get embeddings from the provider
        
        Args:
            text: The text to embed
            task_type: Optional task type for specialized embeddings (specific to provider)
        
        Returns:
            List of floats representing the embedding vector
        """
        try:
            embeddings = self.provider.get_embeddings(text=text, task_type=task_type)
            return embeddings
        except Exception as e:
            logger.error(f"Error getting embeddings from provider: {e}")
            # Return mock embeddings on error
            import numpy as np
            return list(np.random.rand(384))


# Legacy controller classes maintained for backward compatibility
class OpenAIController(LLMController):
    """OpenAI controller implementation (legacy)"""
    
    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None):
        """Initialize OpenAI controller"""
        super().__init__(
            backend="openai",
            model=model,
            api_key=api_key
        )


class OllamaController(LLMController):
    """Ollama controller implementation (legacy)"""
    
    def __init__(self, model: str = "llama3", base_url: str = "http://localhost:11434"):
        """Initialize Ollama controller"""
        super().__init__(
            backend="ollama",
            model=model,
            base_url=base_url
        )
