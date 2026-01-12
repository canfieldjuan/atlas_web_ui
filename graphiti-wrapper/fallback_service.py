"""
Fallback Service for GraphRAG.

Provides alternative search methods when graph search returns insufficient results.
Uses PostgreSQL full-text search as the fallback mechanism.
"""

import os
import re
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


def get_env_str(key: str, default: str) -> str:
    """Get string from environment variable."""
    return os.environ.get(key, default)


def get_env_int(key: str, default: int) -> int:
    """Get integer from environment variable."""
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def get_env_bool(key: str, default: bool) -> bool:
    """Get boolean from environment variable."""
    val = os.environ.get(key, "").lower()
    if val in ("true", "1", "yes"):
        return True
    if val in ("false", "0", "no"):
        return False
    return default


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class FallbackSource:
    """A source from fallback search."""
    entity: str
    relation: str
    fact: str
    confidence: float
    source_description: str


@dataclass
class FallbackResult:
    """Result from fallback search."""
    sources: list = field(default_factory=list)
    context: str = ""
    strategy: str = ""
    query_time_ms: float = 0.0


# ============================================================================
# Stop Words for Keyword Extraction
# ============================================================================

STOP_WORDS = frozenset([
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "can", "this",
    "that", "these", "those", "what", "which", "who", "whom",
    "how", "when", "where", "why", "and", "or", "but", "if",
    "then", "else", "for", "of", "to", "from", "in", "on", "at",
    "by", "with", "about", "into", "through", "during", "before",
    "after", "above", "below", "between", "under", "again", "further",
    "once", "here", "there", "all", "each", "few", "more", "most",
    "other", "some", "such", "no", "not", "only", "own", "same",
    "than", "too", "very", "just", "also", "now", "so", "any",
])


# ============================================================================
# Fallback Service
# ============================================================================

class FallbackService:
    """
    Provides fallback search when GraphRAG returns insufficient results.

    Strategies:
    - vector: PostgreSQL full-text search
    - keyword: ILIKE pattern matching
    - cascade: Try vector first, then keyword
    """

    def __init__(self):
        self.enabled = get_env_bool("GRAPHRAG_FALLBACK_ENABLED", True)
        self.strategy = get_env_str("GRAPHRAG_FALLBACK_STRATEGY", "cascade")
        self.min_results_threshold = get_env_int("GRAPHRAG_FALLBACK_MIN_RESULTS", 3)

        self._pg_host = get_env_str("GRAPHRAG_FALLBACK_PG_HOST", "")
        self._pg_port = get_env_int("GRAPHRAG_FALLBACK_PG_PORT", 5432)
        self._pg_database = get_env_str("GRAPHRAG_FALLBACK_PG_DATABASE", "")
        self._pg_user = get_env_str("GRAPHRAG_FALLBACK_PG_USER", "")
        self._pg_password = get_env_str("GRAPHRAG_FALLBACK_PG_PASSWORD", "")
        self._pg_table = get_env_str("GRAPHRAG_FALLBACK_PG_TABLE", "graphrag_documents")

        self._pool = None

    def is_configured(self) -> bool:
        """Check if PostgreSQL is configured."""
        return bool(self._pg_host and self._pg_database and self._pg_user)

    async def _get_pool(self):
        """Get or create connection pool."""
        if self._pool is None and self.is_configured():
            try:
                import asyncpg
                self._pool = await asyncpg.create_pool(
                    host=self._pg_host,
                    port=self._pg_port,
                    database=self._pg_database,
                    user=self._pg_user,
                    password=self._pg_password,
                    min_size=1,
                    max_size=5,
                )
                logger.info("PostgreSQL fallback pool created")
            except Exception as e:
                logger.error("Failed to create PostgreSQL pool: %s", e)
                self._pool = None
        return self._pool

    async def close(self):
        """Close the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    def should_trigger_fallback(self, primary_result_count: int) -> bool:
        """Check if fallback should be triggered."""
        if not self.enabled:
            return False
        return primary_result_count < self.min_results_threshold

    async def execute_fallback(
        self,
        query: str,
        group_id: Optional[str] = None,
        limit: int = 5,
    ) -> FallbackResult:
        """Execute fallback search."""
        start_time = datetime.now()

        logger.debug("Executing fallback search: strategy=%s, query=%s",
                     self.strategy, query[:50])

        try:
            if self.strategy == "vector":
                result = await self._vector_fallback(query, group_id, limit, start_time)
            elif self.strategy == "keyword":
                result = await self._keyword_fallback(query, group_id, limit, start_time)
            else:
                result = await self._cascade_fallback(query, group_id, limit, start_time)

            return result

        except Exception as e:
            logger.error("Fallback search failed: %s", e)
            elapsed = (datetime.now() - start_time).total_seconds() * 1000
            return FallbackResult(
                sources=[],
                context="",
                strategy=self.strategy,
                query_time_ms=elapsed,
            )

    async def _cascade_fallback(
        self,
        query: str,
        group_id: Optional[str],
        limit: int,
        start_time: datetime,
    ) -> FallbackResult:
        """Cascade: try vector first, then keyword."""
        vector_result = await self._vector_fallback(query, group_id, limit, start_time)
        if len(vector_result.sources) >= self.min_results_threshold:
            return vector_result

        keyword_result = await self._keyword_fallback(query, group_id, limit, start_time)

        combined = self._deduplicate_sources(
            vector_result.sources + keyword_result.sources
        )

        elapsed = (datetime.now() - start_time).total_seconds() * 1000
        return FallbackResult(
            sources=combined,
            context=self._build_context(combined),
            strategy="cascade",
            query_time_ms=elapsed,
        )

    async def _vector_fallback(
        self,
        query: str,
        group_id: Optional[str],
        limit: int,
        start_time: datetime,
    ) -> FallbackResult:
        """PostgreSQL full-text search fallback."""
        pool = await self._get_pool()
        if pool is None:
            elapsed = (datetime.now() - start_time).total_seconds() * 1000
            return FallbackResult(
                sources=[],
                context="",
                strategy="vector",
                query_time_ms=elapsed,
            )

        try:
            ts_query = self._build_tsquery(query)
            if not ts_query:
                elapsed = (datetime.now() - start_time).total_seconds() * 1000
                return FallbackResult(sources=[], context="", strategy="vector",
                                       query_time_ms=elapsed)

            sql = f"""
                SELECT id, filename, content, metadata,
                       ts_rank(to_tsvector('english', content), to_tsquery('english', $1)) as rank
                FROM {self._pg_table}
                WHERE to_tsvector('english', content) @@ to_tsquery('english', $1)
            """
            params = [ts_query]

            if group_id:
                sql += " AND (metadata->>'group_id' = $2 OR metadata->>'group_id' IS NULL)"
                params.append(group_id)

            sql += " ORDER BY rank DESC LIMIT $" + str(len(params) + 1)
            params.append(limit)

            async with pool.acquire() as conn:
                rows = await conn.fetch(sql, *params)

            sources = self._map_rows_to_sources(rows, "vector_fallback")
            elapsed = (datetime.now() - start_time).total_seconds() * 1000

            return FallbackResult(
                sources=sources,
                context=self._build_context(sources),
                strategy="vector",
                query_time_ms=elapsed,
            )

        except Exception as e:
            logger.warning("Vector fallback failed: %s", e)
            elapsed = (datetime.now() - start_time).total_seconds() * 1000
            return FallbackResult(sources=[], context="", strategy="vector",
                                   query_time_ms=elapsed)

    async def _keyword_fallback(
        self,
        query: str,
        group_id: Optional[str],
        limit: int,
        start_time: datetime,
    ) -> FallbackResult:
        """ILIKE keyword search fallback."""
        pool = await self._get_pool()
        if pool is None:
            elapsed = (datetime.now() - start_time).total_seconds() * 1000
            return FallbackResult(sources=[], context="", strategy="keyword",
                                   query_time_ms=elapsed)

        try:
            keywords = self._extract_keywords(query)
            if not keywords:
                elapsed = (datetime.now() - start_time).total_seconds() * 1000
                return FallbackResult(sources=[], context="", strategy="keyword",
                                       query_time_ms=elapsed)

            pattern = "%" + "%".join(keywords) + "%"

            sql = f"""
                SELECT id, filename, content, metadata
                FROM {self._pg_table}
                WHERE content ILIKE $1
            """
            params = [pattern]

            if group_id:
                sql += " AND (metadata->>'group_id' = $2 OR metadata->>'group_id' IS NULL)"
                params.append(group_id)

            sql += " LIMIT $" + str(len(params) + 1)
            params.append(limit)

            async with pool.acquire() as conn:
                rows = await conn.fetch(sql, *params)

            sources = self._map_rows_to_sources(rows, "keyword_fallback")
            elapsed = (datetime.now() - start_time).total_seconds() * 1000

            return FallbackResult(
                sources=sources,
                context=self._build_context(sources),
                strategy="keyword",
                query_time_ms=elapsed,
            )

        except Exception as e:
            logger.warning("Keyword fallback failed: %s", e)
            elapsed = (datetime.now() - start_time).total_seconds() * 1000
            return FallbackResult(sources=[], context="", strategy="keyword",
                                   query_time_ms=elapsed)

    def _build_tsquery(self, query: str) -> str:
        """Build PostgreSQL tsquery from search query."""
        keywords = self._extract_keywords(query)
        if not keywords:
            return ""
        return " & ".join(keywords)

    def _extract_keywords(self, query: str) -> list:
        """Extract meaningful keywords from query."""
        cleaned = re.sub(r"[^\w\s]", "", query.lower())
        words = cleaned.split()
        keywords = [w for w in words if len(w) > 2 and w not in STOP_WORDS]
        return list(dict.fromkeys(keywords))

    def _map_rows_to_sources(self, rows: list, source_type: str) -> list:
        """Map database rows to FallbackSource objects."""
        sources = []
        for row in rows:
            content = row.get("content", "") or ""
            filename = row.get("filename", "") or "unknown"
            snippet = self._extract_snippet(content, 500)

            sources.append(FallbackSource(
                entity=filename,
                relation=source_type,
                fact=snippet,
                confidence=0.5,
                source_description=f"Fallback ({source_type}): {filename}",
            ))
        return sources

    def _extract_snippet(self, content: str, max_length: int) -> str:
        """Extract snippet from content."""
        if not content:
            return ""
        if len(content) <= max_length:
            return content
        return content[:max_length] + "..."

    def _build_context(self, sources: list) -> str:
        """Build context string from sources."""
        if not sources:
            return ""

        facts = []
        for idx, source in enumerate(sources):
            facts.append(f"{idx + 1}. {source.fact}\n   Source: {source.source_description}")

        return "Fallback search results:\n\n" + "\n\n".join(facts)

    def _deduplicate_sources(self, sources: list) -> list:
        """Deduplicate sources by content fingerprint."""
        seen = set()
        unique = []

        for source in sources:
            fingerprint = source.fact[:100].lower().replace(" ", "").strip()
            if fingerprint not in seen:
                unique.append(source)
                seen.add(fingerprint)

        return unique

    def is_enabled(self) -> bool:
        """Check if fallback is enabled."""
        return self.enabled

    def get_config(self) -> dict:
        """Get current configuration."""
        return {
            "enabled": self.enabled,
            "strategy": self.strategy,
            "min_results_threshold": self.min_results_threshold,
            "pg_configured": self.is_configured(),
        }


fallback_service = FallbackService()
