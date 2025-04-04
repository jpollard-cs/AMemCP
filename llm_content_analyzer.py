#!/usr/bin/env python3
"""
LLM Content Analyzer
Provides advanced content analysis and segmentation using LLMs.
"""

import json
import logging
from typing import Any, Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMContentAnalyzer:
    """
    Advanced content analyzer that uses LLMs to:
    1. Classify content types with high precision
    2. Segment mixed content into coherent parts
    3. Determine the best embedding task types for each segment

    Provides more sophisticated analysis than rule-based methods,
    especially for content with mixed types or ambiguous structure.
    """

    def __init__(self, llm_controller):
        """
        Initialize the content analyzer with an LLM controller

        Args:
            llm_controller: Instance of LLMController for making LLM calls
        """
        self.llm_controller = llm_controller

    def analyze_content_type(self, text: str) -> Dict[str, Any]:
        """
        Analyze content to determine its type and characteristics

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
        """
        prompt = """
        Analyze the following content and classify its type.

        Content:
        ```
        {content}
        ```

        Determine:
        1. The primary content type (code, question, documentation, mixed, or other)
        2. If mixed, the proportion of each type (e.g., 70% code, 30% explanation)
        3. Confidence in this classification (0.0 to 1.0)
        4. The most appropriate embedding task types:
           - For storage: CODE_RETRIEVAL_DOCUMENT, RETRIEVAL_DOCUMENT, QUESTION_ANSWERING
           - For queries: CODE_RETRIEVAL_QUERY, RETRIEVAL_QUERY

        Return your analysis in the following JSON format:
        {{
            "primary_type": "code|question|documentation|mixed|other",
            "confidence": 0.0-1.0,
            "types": {{
                "code": 0.0-1.0,
                "question": 0.0-1.0,
                "documentation": 0.0-1.0,
                "other": 0.0-1.0
            }},
            "recommended_task_types": {{
                "storage": "CODE_RETRIEVAL_DOCUMENT|RETRIEVAL_DOCUMENT|QUESTION_ANSWERING",
                "query": "CODE_RETRIEVAL_QUERY|RETRIEVAL_QUERY"
            }},
            "has_mixed_content": true|false
        }}
        """.format(
            content=text[:8000]
        )  # Limit content size for LLM context window

        try:
            response = self.llm_controller.get_completion(
                prompt=prompt,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "type": "object",
                        "properties": {
                            "primary_type": {"type": "string"},
                            "confidence": {"type": "number"},
                            "types": {
                                "type": "object",
                                "properties": {
                                    "code": {"type": "number"},
                                    "question": {"type": "number"},
                                    "documentation": {"type": "number"},
                                    "other": {"type": "number"},
                                },
                            },
                            "recommended_task_types": {
                                "type": "object",
                                "properties": {"storage": {"type": "string"}, "query": {"type": "string"}},
                            },
                            "has_mixed_content": {"type": "boolean"},
                        },
                    },
                },
            )
            return json.loads(response)
        except Exception as e:
            logger.error(f"Error analyzing content type: {e}")
            # Return default analysis on error
            return {
                "primary_type": "general",
                "confidence": 0.5,
                "types": {"code": 0.0, "question": 0.0, "documentation": 0.0, "other": 1.0},
                "recommended_task_types": {"storage": "RETRIEVAL_DOCUMENT", "query": "RETRIEVAL_QUERY"},
                "has_mixed_content": False,
            }

    def segment_content(self, text: str) -> List[Dict[str, Any]]:
        """
        Segment mixed content into coherent chunks by type

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
        # First check if segmentation is needed
        analysis = self.analyze_content_type(text)

        if not analysis.get("has_mixed_content", False) or analysis.get("confidence", 0) < 0.7:
            # If content is not mixed or classification confidence is low, return as single segment
            return [
                {
                    "content": text,
                    "type": analysis.get("primary_type", "general"),
                    "task_type": analysis.get("recommended_task_types", {}).get("storage", "RETRIEVAL_DOCUMENT"),
                    "order": 0,
                    "metadata": {"is_complete_document": True, "confidence": analysis.get("confidence", 0.5)},
                }
            ]

        # Content is mixed, use LLM to segment it
        prompt = """
        Segment the following mixed content into coherent parts based on type.
        Identify natural boundaries between different content types (code, questions, documentation, etc.)

        Content:
        ```
        {content}
        ```

        For each segment, provide:
        1. The segment text, preserving all original formatting
        2. The content type of the segment
        3. The appropriate embedding task type
        4. The segment's position in the document

        Return the segmentation as a JSON array:
        [
            {{
                "content": "segment text here",
                "type": "code|question|documentation|other",
                "task_type": "CODE_RETRIEVAL_DOCUMENT|RETRIEVAL_DOCUMENT|QUESTION_ANSWERING|RETRIEVAL_QUERY",
                "order": 0,
                "metadata": {{
                    "subtitle": "optional segment title",
                    "language": "language if code",
                    "confidence": 0.0-1.0
                }}
            }},
            // additional segments...
        ]

        Ensure that:
        - Each segment is semantically coherent
        - Code segments include complete syntactic blocks (don't split functions or classes)
        - Question segments include the full question context
        - Explanatory text is grouped logically by topic
        - All original content is preserved across all segments
        """.format(
            content=text[:8000]
        )  # Limit content size for LLM context window

        try:
            response = self.llm_controller.get_completion(
                prompt=prompt,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string"},
                                "type": {"type": "string"},
                                "task_type": {"type": "string"},
                                "order": {"type": "integer"},
                                "metadata": {
                                    "type": "object",
                                    "properties": {
                                        "subtitle": {"type": "string"},
                                        "language": {"type": "string"},
                                        "confidence": {"type": "number"},
                                    },
                                },
                            },
                            "required": ["content", "type", "task_type", "order"],
                        },
                    },
                },
            )
            segments = json.loads(response)

            # Sort segments by order
            segments.sort(key=lambda x: x.get("order", 0))

            return segments

        except Exception as e:
            logger.error(f"Error segmenting content: {e}")
            # Return single segment on error
            return [
                {
                    "content": text,
                    "type": "general",
                    "task_type": "RETRIEVAL_DOCUMENT",
                    "order": 0,
                    "metadata": {"is_complete_document": True, "confidence": 0.5, "error": str(e)},
                }
            ]

    def get_optimal_task_type(self, text: str, is_query: bool = False) -> str:
        """
        Determine the optimal embedding task type for the given content

        Args:
            text: The text to analyze
            is_query: Whether this text is a search query

        Returns:
            The optimal task type for embeddings
        """
        analysis = self.analyze_content_type(text)

        # Get recommended task type based on whether this is a query or document
        key = "query" if is_query else "storage"
        recommended_type = analysis.get("recommended_task_types", {}).get(key)

        # Use appropriate default if recommendation is missing
        if not recommended_type:
            if is_query:
                if analysis.get("primary_type") == "code":
                    return "CODE_RETRIEVAL_QUERY"
                else:
                    return "RETRIEVAL_QUERY"
            else:
                if analysis.get("primary_type") == "code":
                    return "CODE_RETRIEVAL_DOCUMENT"
                elif analysis.get("primary_type") == "question":
                    return "QUESTION_ANSWERING"
                else:
                    return "RETRIEVAL_DOCUMENT"

        return recommended_type
