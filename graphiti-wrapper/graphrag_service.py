"""
GraphRAG Service Orchestrator.

High-level service that enhances prompts with document context.
Coordinates search, filtering, and context injection.
"""

import os
import re
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any

logger = logging.getLogger(__name__)


def get_env_bool(key: str, default: bool) -> bool:
    """Get boolean from environment variable."""
    val = os.environ.get(key, "").lower()
    if val in ("true", "1", "yes"):
        return True
    if val in ("false", "0", "no"):
        return False
    return default


def get_env_int(key: str, default: int) -> int:
    """Get integer from environment variable."""
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def get_env_float(key: str, default: float) -> float:
    """Get float from environment variable."""
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class SearchSource:
    """A source from search results."""
    entity: str
    relation: str
    fact: str
    confidence: float
    source_description: str = ""


@dataclass
class Citation:
    """A formatted citation."""
    source: str
    content: str
    confidence: float


@dataclass
class EnhancedPrompt:
    """Result of prompt enhancement."""
    prompt: str
    context_used: bool
    sources: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievalMetadata:
    """Metadata about the retrieval process."""
    graph_used: bool = False
    nodes_retrieved: int = 0
    context_chunks_used: int = 0
    retrieval_time_ms: float = 0.0
    context_relevance_score: float = 0.0
    answer_grounded_in_graph: bool = False
    search_method: str = "standard"
    decomposed: bool = False
    sub_query_count: int = 0
    traversal_used: bool = False


# ============================================================================
# Relationship Pattern Detection
# ============================================================================

RELATIONSHIP_PATTERN = re.compile(
    r"\b(relationship|connected|path between|how.*related|link between)\b",
    re.IGNORECASE
)

COMMON_WORDS = frozenset([
    "What", "How", "Why", "When", "Where", "Who",
    "Is", "Are", "The", "A", "An", "Can", "Could",
    "Would", "Should", "Do", "Does", "Did", "Has", "Have",
])


# ============================================================================
# GraphRAG Service
# ============================================================================

class GraphRAGService:
    """
    High-level GraphRAG service for prompt enhancement.

    Integrates document search with chat for context-aware responses.
    """

    def __init__(self):
        self.enabled = get_env_bool("GRAPHRAG_ENABLED", True)
        self.default_max_sources = get_env_int("GRAPHRAG_MAX_SOURCES", 5)
        self.default_min_confidence = get_env_float("GRAPHRAG_MIN_CONFIDENCE", 0.3)
        self.default_max_context_length = get_env_int("GRAPHRAG_MAX_CONTEXT_LENGTH", 4000)

    def is_relationship_query(self, query: str) -> bool:
        """Check if query is asking about relationships between entities."""
        return bool(RELATIONSHIP_PATTERN.search(query))

    def extract_entities_from_query(self, query: str) -> list:
        """Extract potential entity names from a query."""
        pattern = re.compile(r"\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\b")
        matches = pattern.findall(query)
        entities = [m for m in matches if m not in COMMON_WORDS and len(m) > 1]
        return list(dict.fromkeys(entities))

    def format_context(
        self,
        sources: list,
        compress: bool = False,
    ) -> str:
        """
        Format search results into readable context.

        Args:
            sources: List of SearchSource or dicts
            compress: Enable compression by grouping facts by entity

        Returns:
            Formatted context string
        """
        if not sources:
            return ""

        if compress and len(sources) > 1:
            return self._compress_context(sources)

        parts = []
        for idx, source in enumerate(sources):
            if isinstance(source, dict):
                fact = source.get("fact", "")
            else:
                fact = source.fact
            parts.append(f"[{idx + 1}] {fact}")

        return "\n\n".join(parts)

    def _compress_context(self, sources: list) -> str:
        """Compress context by grouping facts by entity."""
        if len(sources) <= 1:
            if sources:
                s = sources[0]
                return s.get("fact", "") if isinstance(s, dict) else s.fact
            return ""

        grouped = {}
        for source in sources:
            if isinstance(source, dict):
                entity = source.get("entity", "General") or "General"
                fact = source.get("fact", "")
            else:
                entity = source.entity or "General"
                fact = source.fact

            if entity not in grouped:
                grouped[entity] = []
            grouped[entity].append(fact)

        parts = []
        idx = 1
        for entity, facts in grouped.items():
            if len(facts) == 1:
                parts.append(f"[{idx}] {entity}: {facts[0]}")
            else:
                fact_list = "; ".join(facts)
                parts.append(f"[{idx}] {entity}: {fact_list}")
            idx += 1

        return "\n\n".join(parts)

    def inject_context(self, user_message: str, context: str) -> str:
        """
        Inject context into user message with LLM-friendly instructions.

        Args:
            user_message: Original user query
            context: Formatted context from search

        Returns:
            Enhanced prompt with context and instructions
        """
        return f"""You have access to the user's documents and knowledge base. Use this context to provide accurate, well-sourced answers.

CONTEXT FROM USER'S DOCUMENTS:
{context}

USER'S QUESTION:
{user_message}

INSTRUCTIONS:
1. ALWAYS cite your sources using [1], [2], etc. matching the context numbers above
2. Be explicit about what you found - start responses with "Based on your documents..." or "According to [source]..."
3. Quote specific facts, numbers, and details from the context
4. If multiple sources provide information, reference each one

For questions the context doesn't answer:
- Clearly state "Your documents don't contain information about X"
- Then optionally provide general knowledge, clearly marked as such"""

    def format_citations(self, sources: list) -> list:
        """Format sources as citations."""
        citations = []
        for source in sources:
            if isinstance(source, dict):
                citations.append(Citation(
                    source=source.get("source_description", source.get("entity", "Unknown")),
                    content=source.get("fact", ""),
                    confidence=source.get("confidence", 0.0),
                ))
            else:
                citations.append(Citation(
                    source=source.source_description or source.entity or "Unknown",
                    content=source.fact,
                    confidence=source.confidence,
                ))
        return citations

    def filter_sources(
        self,
        sources: list,
        min_confidence: Optional[float] = None,
        max_sources: Optional[int] = None,
    ) -> list:
        """Filter sources by confidence and limit count."""
        filtered = sources

        threshold = min_confidence if min_confidence is not None else self.default_min_confidence
        if threshold > 0:
            filtered = [
                s for s in filtered
                if (s.get("confidence", 0) if isinstance(s, dict) else s.confidence) >= threshold
            ]

        limit = max_sources if max_sources is not None else self.default_max_sources
        if limit > 0:
            filtered = filtered[:limit]

        return filtered

    def calculate_avg_relevance(self, sources: list) -> float:
        """Calculate average relevance score from sources."""
        if not sources:
            return 0.0

        total = 0.0
        for s in sources:
            conf = s.get("confidence", 0) if isinstance(s, dict) else s.confidence
            total += conf

        return total / len(sources)

    def build_metadata(
        self,
        sources: list,
        retrieval_time_ms: float,
        search_method: str = "standard",
        decomposed: bool = False,
        sub_query_count: int = 0,
        traversal_used: bool = False,
    ) -> dict:
        """Build retrieval metadata dictionary."""
        return {
            "graph_used": len(sources) > 0,
            "nodes_retrieved": len(sources),
            "context_chunks_used": len(sources),
            "retrieval_time_ms": retrieval_time_ms,
            "context_relevance_score": self.calculate_avg_relevance(sources),
            "answer_grounded_in_graph": len(sources) > 0,
            "search_method": search_method,
            "decomposed": decomposed,
            "sub_query_count": sub_query_count,
            "traversal_used": traversal_used,
        }

    async def enhance_prompt(
        self,
        user_message: str,
        sources: list,
        retrieval_time_ms: float = 0.0,
        min_confidence: Optional[float] = None,
        max_sources: Optional[int] = None,
        max_context_length: Optional[int] = None,
        compress_context: bool = False,
        include_metadata: bool = True,
        search_method: str = "standard",
    ) -> EnhancedPrompt:
        """
        Enhance user prompt with document context.

        Args:
            user_message: Original user query
            sources: Search results (list of dicts or SearchSource)
            retrieval_time_ms: Time taken for search
            min_confidence: Minimum confidence threshold
            max_sources: Maximum sources to include
            max_context_length: Maximum context length
            compress_context: Enable context compression
            include_metadata: Include retrieval metadata
            search_method: Search method used

        Returns:
            EnhancedPrompt with context and metadata
        """
        if not self.enabled:
            logger.debug("GraphRAG disabled, skipping enhancement")
            return EnhancedPrompt(
                prompt=user_message,
                context_used=False,
            )

        if not user_message or not user_message.strip():
            return EnhancedPrompt(
                prompt=user_message,
                context_used=False,
            )

        filtered = self.filter_sources(sources, min_confidence, max_sources)

        if not filtered:
            logger.debug("No sources after filtering, skipping enhancement")
            return EnhancedPrompt(
                prompt=user_message,
                context_used=False,
            )

        context = self.format_context(filtered, compress=compress_context)

        max_len = max_context_length or self.default_max_context_length
        if max_len > 0 and len(context) > max_len:
            context = context[:max_len]

        enhanced = self.inject_context(user_message, context)

        metadata = {}
        if include_metadata:
            metadata = self.build_metadata(
                sources=filtered,
                retrieval_time_ms=retrieval_time_ms,
                search_method=search_method,
            )

        logger.info("Enhanced prompt with %d sources", len(filtered))

        return EnhancedPrompt(
            prompt=enhanced,
            context_used=True,
            sources=filtered,
            metadata=metadata,
        )


graphrag_service = GraphRAGService()
