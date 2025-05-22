#!/usr/bin/env python3
"""
LLM Content Analyzer
Provides advanced content analysis and segmentation using LLMs.
"""

import json
from typing import Any, Dict, List

# Import the interface and controller
from amem.core.interfaces import IContentAnalyzer
from amem.core.llm_controller import LLMController  # Type hint for controller
from amem.utils.utils import setup_logger

# Configure logging
logger = setup_logger(__name__)


# Implement the interface
class LLMContentAnalyzer(IContentAnalyzer):
    """
    Advanced content analyzer that uses LLMs to:
    1. Classify content types with high precision
    2. Segment mixed content into coherent parts
    3. Determine the best embedding task types for each segment

    Provides more sophisticated analysis than rule-based methods,
    especially for content with mixed types or ambiguous structure.
    """

    def __init__(self, llm_controller: LLMController):
        """
        Initialize the content analyzer with an LLM controller

        Args:
            llm_controller: Instance of LLMController for making LLM calls
        """
        self.llm_controller = llm_controller

    def _get_json_config(self) -> Dict[str, Any]:
        """
        Get provider-specific JSON configuration for structured output.

        Returns:
            Configuration dict appropriate for the current LLM provider
        """
        provider_backend = getattr(self.llm_controller, "backend", "unknown").lower()

        if provider_backend == "gemini":
            return {"response_mime_type": "application/json"}
        elif provider_backend == "openai":
            return {"response_format": {"type": "json_object"}}
        else:
            # Fallback - try Gemini format first, then OpenAI
            return {"response_mime_type": "application/json"}

    # Make analyze_content async
    async def analyze_content(self, text: str) -> Dict[str, Any]:
        """
        Analyze content to determine its type and characteristics (async)

        Uses LLM to classify content more precisely than rule-based methods.
        Handles mixed content, ambiguous queries, and contextual nuances.

        Args:
            text: The text content to analyze

        Returns:
            Dictionary with detailed content analysis:
            - primary_type: The main content type
            - confidence: Confidence score for classification
            - types: Dict of all detected types with their proportions
            - recommended_task_types: Dict with recommended embedding task types
            - has_mixed_content: Whether content contains multiple types
            - keywords: List of extracted keywords
            - summary: Generated summary
            - sentiment: Detected sentiment
            - importance: Estimated importance score
        """
        # Combine analysis into a single prompt for efficiency
        prompt = """
        Analyze the following content comprehensively.

        Content:
        ```
        {content}
        ```

        Determine:
        1.  Primary content type (e.g., code, question, documentation, discussion, mixed, general).
        2.  If mixed, proportions of each type.
        3.  Confidence in classification (0.0 to 1.0).
        4.  Recommended embedding task types (storage and query). MUST be one of:
            - Storage: RETRIEVAL_DOCUMENT, SEMANTIC_SIMILARITY, CLASSIFICATION, CLUSTERING
            - Query: RETRIEVAL_QUERY, SEMANTIC_SIMILARITY, CLASSIFICATION, CLUSTERING
            Choose RETRIEVAL_DOCUMENT/RETRIEVAL_QUERY for general text intended for search.
        5.  Whether content seems mixed.
        6.  Extract 5-10 relevant keywords.
        7.  Generate a concise one-sentence summary.
        8.  Overall sentiment (positive, negative, neutral).
        9.  Estimated importance score (0.0 to 1.0, based on likely usefulness).

        Return ONLY the analysis as a single JSON object:
        {{
            "primary_type": "string",
            "confidence": float,
            "types": {{ "type1": float, ... }},
            "recommended_task_types": {{ "storage": "string", "query": "string" }},
            "has_mixed_content": boolean,
            "keywords": ["string", ...],
            "summary": "string",
            "sentiment": "positive|negative|neutral",
            "importance": float
        }}
        """.format(
            content=text[:8000]
        )  # Limit context

        default_analysis = {
            "primary_type": "general",
            "confidence": 0.5,
            "types": {"other": 1.0},
            "recommended_task_types": {"storage": "RETRIEVAL_DOCUMENT", "query": "RETRIEVAL_QUERY"},
            "has_mixed_content": False,
            "keywords": [],
            "summary": "",
            "sentiment": "neutral",
            "importance": 0.5,
        }

        try:
            # Use the async get_completion method with provider-agnostic JSON config
            json_config = self._get_json_config()
            response_str = await self.llm_controller.get_completion(
                prompt=prompt,
                config=json_config,
            )
            analysis = json.loads(response_str)

            # Basic validation/cleanup
            analysis.setdefault("keywords", [])
            analysis.setdefault("summary", "")
            analysis.setdefault("sentiment", "neutral")
            analysis.setdefault("importance", 0.5)
            analysis.setdefault("recommended_task_types", default_analysis["recommended_task_types"])

            return analysis
        except json.JSONDecodeError as json_err:
            logger.error(f"Error decoding LLM analysis response: {json_err}. Response: '{response_str[:200]}...'")
            return default_analysis
        except Exception as e:
            logger.error(f"Error analyzing content type: {e}")
            return default_analysis

    # Make segment_content async
    async def segment_content(self, text: str) -> List[Dict[str, Any]]:
        """
        Segment mixed content into coherent chunks by type (async)

        Uses LLM to identify natural segment boundaries in mixed content,
        preserving semantic coherence while separating different content types.

        Args:
            text: The mixed content text to segment

        Returns:
            List of dictionaries, each representing a segment:
            - content: The segment text
            - type: Content type of the segment
            - task_type: Recommended embedding task type
            - order: Original position in the document
            - metadata: Additional segment attributes
        """
        # First check if segmentation is needed (reuse async analyze_content)
        analysis = await self.analyze_content(text)

        # Stricter check: only segment if clearly mixed and confidence is high
        if not analysis.get("has_mixed_content", False) or analysis.get("confidence", 0) < 0.8:
            logger.debug("Content not considered mixed or confidence too low, returning single segment.")
            return [
                {
                    "content": text,
                    "type": analysis.get("primary_type", "general"),
                    "task_type": analysis.get("recommended_task_types", {}).get("storage", "RETRIEVAL_DOCUMENT"),
                    "order": 0,
                    "metadata": {"is_complete_document": True, "analysis": analysis},
                }
            ]

        logger.debug("Content is mixed, attempting segmentation.")
        # Content is mixed, use LLM to segment it
        prompt = """
        Segment the following mixed content into coherent parts based on type.
        Identify natural boundaries between different content types (e.g., code, questions, documentation).

        Content:
        ```
        {content}
        ```

        For each segment, provide:
        1. The exact segment text (preserve formatting).
        2. The primary content type.
        3. The recommended embedding task type for storage.
        4. Segment's position (order).
        5. Optional metadata (e.g., subtitle, language).

        Return ONLY the segmentation as a JSON array:
        [
            {{
                "content": "string",
                "type": "string",
                "task_type": "string",
                "order": integer,
                "metadata": {{ "subtitle": "string", "language": "string", ... }}
            }},
            ...
        ]

        Guidelines:
        - Ensure segments are semantically coherent.
        - Do not split code blocks unnecessarily.
        - Preserve all original content.
        - Assign a logical type and task_type to each segment.
        """.format(
            content=text[:8000]
        )  # Limit context

        default_segment = [
            {
                "content": text,
                "type": analysis.get("primary_type", "general"),
                "task_type": analysis.get("recommended_task_types", {}).get("storage", "RETRIEVAL_DOCUMENT"),
                "order": 0,
                "metadata": {"is_complete_document": True, "analysis": analysis, "error": "Segmentation failed"},
            }
        ]

        try:
            # Use async get_completion with provider-agnostic JSON config
            json_config = self._get_json_config()
            response_str = await self.llm_controller.get_completion(
                prompt=prompt,
                config=json_config,
            )
            # The response might be the JSON array directly, or nested in a key.
            # Handle both common cases.
            try:
                segments = json.loads(response_str)
                if isinstance(segments, list):
                    # Already a list, use directly
                    pass
                elif isinstance(segments, dict) and len(segments) == 1:
                    # Check if it's nested like { "segments": [...] }
                    potential_list = next(iter(segments.values()))
                    if isinstance(potential_list, list):
                        segments = potential_list
                    else:
                        raise ValueError("Expected a list or a dict containing a list.")
                else:
                    raise ValueError("Unexpected JSON structure for segments.")

            except (json.JSONDecodeError, ValueError) as parse_err:
                logger.error(f"Error parsing segmentation response: {parse_err}. Response: '{response_str[:200]}...'")
                return default_segment

            # Basic validation and sorting
            if not isinstance(segments, list) or not all(isinstance(s, dict) for s in segments):
                logger.error(f"LLM segmentation did not return a list of dicts. Response: '{response_str[:200]}...'")
                return default_segment

            # Ensure required keys and sort
            for i, seg in enumerate(segments):
                seg.setdefault("order", i)  # Assign order if missing
                seg.setdefault("type", "general")
                seg.setdefault("task_type", "RETRIEVAL_DOCUMENT")
                seg.setdefault("metadata", {})

            segments.sort(key=lambda x: x["order"])

            # Optional: Verify content reconstruction
            # reconstructed = "".join(s['content'] for s in segments)
            # if reconstructed != text:
            #    logger.warning("Segmented content differs from original.")

            logger.debug(f"Successfully segmented content into {len(segments)} parts.")
            return segments

        except Exception as e:
            logger.error(f"Error during content segmentation: {e}")
            return default_segment
