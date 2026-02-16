"""
Memory module for Atlas Brain.

Provides unified memory management:
- RAG client for graphiti-wrapper integration
- MemoryService for context aggregation
- User profile management
"""

from .feedback import FeedbackContext, FeedbackService, SourceCitation, get_feedback_service
from .quality import MemoryQualityDetector, QualitySignal, get_quality_detector
from .query_classifier import ClassificationResult, QueryClassifier, get_query_classifier
from .rag_client import RAGClient, get_rag_client
from .service import MemoryContext, MemoryService, get_memory_service
from .token_estimator import TokenBudget, TokenEstimator, TokenUsage, get_token_estimator

__all__ = [
    "ClassificationResult",
    "FeedbackContext",
    "FeedbackService",
    "get_feedback_service",
    "get_quality_detector",
    "get_query_classifier",
    "get_rag_client",
    "get_memory_service",
    "get_token_estimator",
    "MemoryContext",
    "MemoryQualityDetector",
    "MemoryService",
    "QualitySignal",
    "QueryClassifier",
    "RAGClient",
    "SourceCitation",
    "TokenBudget",
    "TokenEstimator",
    "TokenUsage",
]
