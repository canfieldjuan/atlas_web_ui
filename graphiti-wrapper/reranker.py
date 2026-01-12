"""
Rerankers for GraphRAG search results.

Provides:
- HeuristicReranker: Fast text matching (no external dependencies)
- CrossEncoderReranker: Neural reranking via external API

All weights configurable via environment variables.
"""

import os
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)


def get_env_float(key: str, default: float) -> float:
    """Get float from environment variable."""
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
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


def get_env_str(key: str, default: str) -> str:
    """Get string from environment variable."""
    return os.environ.get(key, default)


@dataclass
class RerankCandidate:
    """A candidate for reranking."""
    text: str
    score: float
    metadata: dict


@dataclass
class RerankResult:
    """Result of reranking."""
    original_index: int
    score: float
    text: str
    metadata: dict


class HeuristicReranker:
    """
    Fast heuristic reranker using text matching.

    Boosts results based on:
    - Exact phrase match
    - Token overlap
    - Recency (if timestamp available)
    """

    def __init__(self, top_k: int = 10):
        self.top_k = top_k
        self.exact_match_boost = get_env_float("GRAPHRAG_RERANK_EXACT_MATCH_BOOST", 0.2)
        self.token_overlap_weight = get_env_float("GRAPHRAG_RERANK_TOKEN_OVERLAP_WEIGHT", 0.15)
        self.recency_max_boost = get_env_float("GRAPHRAG_RERANK_RECENCY_MAX_BOOST", 0.1)
        self.recency_decay_days = get_env_int("GRAPHRAG_RERANK_RECENCY_DECAY_DAYS", 365)

    def rerank(self, query: str, candidates: list) -> list:
        """
        Rerank candidates based on query relevance.

        Args:
            query: Search query
            candidates: List of RerankCandidate or dicts with text, score, metadata

        Returns:
            List of RerankResult sorted by final score
        """
        if not candidates:
            return []

        query_lower = query.lower()
        query_tokens = self._tokenize(query_lower)

        scored = []
        for idx, candidate in enumerate(candidates):
            if isinstance(candidate, dict):
                text = candidate.get("text", "")
                score = candidate.get("score", 0.0)
                metadata = candidate.get("metadata", {})
            else:
                text = candidate.text
                score = candidate.score
                metadata = candidate.metadata

            text_lower = text.lower()

            exact_match = self.exact_match_boost if query_lower in text_lower else 0.0

            token_score = self._calculate_token_overlap(query_tokens, text_lower)

            recency_score = self._calculate_recency_score(metadata)

            final_score = score + exact_match + token_score + recency_score

            scored.append(RerankResult(
                original_index=idx,
                score=final_score,
                text=text,
                metadata=metadata,
            ))

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:self.top_k]

    def _tokenize(self, text: str) -> list:
        """Tokenize text into words."""
        tokens = text.split()
        return [t for t in tokens if len(t) > 2]

    def _calculate_token_overlap(self, query_tokens: list, text: str) -> float:
        """Calculate score based on token overlap."""
        if not query_tokens:
            return 0.0

        matching = sum(1 for token in query_tokens if token in text)
        return (matching / len(query_tokens)) * self.token_overlap_weight

    def _calculate_recency_score(self, metadata: dict) -> float:
        """Calculate recency boost based on timestamp."""
        timestamp = metadata.get("created_at") or metadata.get("createdAt")
        if not timestamp or not isinstance(timestamp, str):
            return 0.0

        try:
            if "T" in timestamp:
                created_date = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            else:
                created_date = datetime.strptime(timestamp, "%Y-%m-%d")
        except ValueError:
            return 0.0

        now = datetime.now(created_date.tzinfo) if created_date.tzinfo else datetime.now()
        age_seconds = (now - created_date).total_seconds()
        days_since_creation = age_seconds / (60 * 60 * 24)

        decay = 1 - (days_since_creation / self.recency_decay_days)
        return max(0.0, self.recency_max_boost * decay)


def deduplicate_by_content(items: list, content_key: str = "fact") -> list:
    """
    Deduplicate items by content fingerprint.

    Args:
        items: List of dicts with content
        content_key: Key to use for content (default: "fact")

    Returns:
        Deduplicated list
    """
    if len(items) <= 1:
        return items

    unique = []
    seen = set()

    for item in items:
        if isinstance(item, dict):
            content = item.get(content_key, "")
        else:
            content = getattr(item, content_key, "")

        fingerprint = content[:100].lower().replace(" ", "").strip()

        if fingerprint not in seen:
            unique.append(item)
            seen.add(fingerprint)

    return unique


# ============================================================================
# Cross-Encoder Reranker (Neural)
# ============================================================================

class CrossEncoderReranker:
    """
    Neural reranker using sentence-transformers CrossEncoder.

    Loads model directly in-process - no external server needed.
    Falls back to original order if model fails to load.
    """

    _model = None
    _model_name = None

    def __init__(
        self,
        top_k: int = 10,
        model: Optional[str] = None,
    ):
        self.top_k = top_k
        self.model_name = model or get_env_str(
            "GRAPHRAG_CROSSENCODER_MODEL",
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    def _get_model(self):
        """Lazy load the cross-encoder model."""
        if CrossEncoderReranker._model is None or CrossEncoderReranker._model_name != self.model_name:
            try:
                from sentence_transformers import CrossEncoder
                logger.info("Loading CrossEncoder model: %s", self.model_name)
                CrossEncoderReranker._model = CrossEncoder(self.model_name)
                CrossEncoderReranker._model_name = self.model_name
            except ImportError:
                logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
                return None
            except Exception as e:
                logger.error("Failed to load CrossEncoder model: %s", e)
                return None
        return CrossEncoderReranker._model

    def rerank(self, query: str, candidates: list) -> list:
        """
        Rerank candidates using cross-encoder model.

        Args:
            query: Search query
            candidates: List of dicts with text, score, metadata

        Returns:
            List of RerankResult sorted by neural relevance score
        """
        if not candidates:
            return []

        model = self._get_model()
        if model is None:
            logger.warning("CrossEncoder model unavailable, using fallback order")
            return self._fallback_order(candidates)

        try:
            documents = []
            for c in candidates:
                if isinstance(c, dict):
                    documents.append(c.get("text", ""))
                else:
                    documents.append(c.text)

            pairs = [[query, doc] for doc in documents]
            scores = model.predict(pairs)

            scored = []
            for idx, score in enumerate(scores):
                candidate = candidates[idx]
                if isinstance(candidate, dict):
                    text = candidate.get("text", "")
                    metadata = candidate.get("metadata", {})
                else:
                    text = candidate.text
                    metadata = candidate.metadata

                scored.append(RerankResult(
                    original_index=idx,
                    score=float(score),
                    text=text,
                    metadata=metadata,
                ))

            scored.sort(key=lambda x: x.score, reverse=True)
            return scored[:self.top_k]

        except Exception as e:
            logger.warning("CrossEncoder reranking failed: %s", e)
            return self._fallback_order(candidates)

    def _fallback_order(self, candidates: list) -> list:
        """Return candidates in original order as fallback."""
        results = []
        for idx, c in enumerate(candidates[:self.top_k]):
            if isinstance(c, dict):
                text = c.get("text", "")
                score = c.get("score", 0.0)
                metadata = c.get("metadata", {})
            else:
                text = c.text
                score = c.score
                metadata = c.metadata

            results.append(RerankResult(
                original_index=idx,
                score=score,
                text=text,
                metadata=metadata,
            ))
        return results


# ============================================================================
# Reranker Factory
# ============================================================================

def create_reranker(
    reranker_type: str = "heuristic",
    top_k: int = 10,
    **kwargs,
):
    """
    Create a reranker instance.

    Args:
        reranker_type: "heuristic", "cross-encoder", or "none"
        top_k: Maximum results to return
        **kwargs: Additional arguments for specific reranker

    Returns:
        Reranker instance or None
    """
    reranker_type = reranker_type.lower()

    if reranker_type == "none":
        return None
    elif reranker_type == "cross-encoder" or reranker_type == "crossencoder":
        return CrossEncoderReranker(
            top_k=top_k,
            model=kwargs.get("model"),
        )
    else:
        return HeuristicReranker(top_k=top_k)
