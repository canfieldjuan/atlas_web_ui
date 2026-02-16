"""
Memory client for atlas-memory (graphiti-wrapper).

Provides conversation and knowledge storage via the GraphRAG API.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import httpx

from ...config import settings as app_settings
from ...memory.query_classifier import get_query_classifier

logger = logging.getLogger("atlas.services.memory")


@dataclass
class SearchResult:
    """A search result from the knowledge graph."""

    uuid: str
    name: str
    fact: str
    score: float
    source_description: Optional[str] = None


@dataclass
class EnhancedSearchResult:
    """Result from enhanced search with processing metadata."""

    results: list
    skipped: bool = False
    skip_reason: Optional[str] = None
    query_expanded: bool = False
    original_query: str = ""
    search_query: str = ""


@dataclass
class EpisodeResult:
    """Result from adding an episode."""

    episode_id: str
    entities_created: int = 0
    relations_created: int = 0


class MemoryClient:
    """
    Client for atlas-memory GraphRAG service.

    Handles storing conversations and retrieving relevant context.
    """

    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def config(self):
        """Get memory config from app settings."""
        return app_settings.memory

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                timeout=self.config.timeout,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def health_check(self) -> bool:
        """Check if the memory service is available."""
        try:
            client = await self._get_client()
            response = await client.get("/healthcheck")
            if response.status_code == 200:
                data = response.json()
                return data.get("status") == "healthy"
            return False
        except Exception as e:
            logger.warning("Memory service health check failed: %s", e)
            return False

    async def add_conversation_turn(
        self,
        role: str,
        content: str,
        session_id: str,
        speaker_name: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> bool:
        """
        Store a conversation turn in the knowledge graph via /messages.

        Args:
            role: "user" or "assistant"
            content: The message content
            session_id: Session identifier
            speaker_name: Optional speaker name
            timestamp: Optional timestamp (defaults to now)

        Returns:
            True if accepted, False otherwise
        """
        try:
            client = await self._get_client()

            ts = timestamp or datetime.now(timezone.utc)
            source = "atlas-voice-%s" % role
            if speaker_name:
                source = "%s:%s" % (source, speaker_name)

            message = {
                "content": content,
                "role_type": role,
                "role": speaker_name,
                "source_description": source,
                "timestamp": ts.isoformat() + "Z",
            }
            payload = {
                "group_id": self.config.group_id,
                "messages": [message],
            }

            response = await client.post("/messages", json=payload)

            if response.status_code not in (200, 202):
                logger.warning(
                    "add_conversation_turn failed (%d): %s",
                    response.status_code,
                    response.text[:200],
                )
                return False

            data = response.json()
            return data.get("success", False)

        except Exception as e:
            logger.error("Failed to add conversation turn: %s", e)
            return False

    async def send_messages(
        self,
        messages: list[dict],
        group_id: str | None = None,
    ) -> dict:
        """
        Send a batch of conversation messages for GraphRAG extraction.

        Uses the /messages endpoint which provides full LLM-powered
        entity/relationship extraction across the conversation context.

        Args:
            messages: List of message dicts with keys:
                content, role_type ("user"|"assistant"), role (optional), timestamp (ISO8601)
            group_id: Optional group ID (defaults to config group_id)

        Returns:
            Response dict from Graphiti, or empty dict on failure
        """
        try:
            client = await self._get_client()

            payload = {
                "group_id": group_id or self.config.group_id,
                "messages": messages,
            }

            response = await client.post(
                "/messages",
                json=payload,
                timeout=300.0,  # batch extraction can be slow
            )
            response.raise_for_status()
            return response.json()

        except Exception as e:
            logger.error("Failed to send messages batch: %s", e)
            return {}

    async def search(
        self,
        query: str,
        num_results: int = 5,
        group_id: Optional[str] = None,
    ) -> list[SearchResult]:
        """
        Search the knowledge graph for relevant context.

        Args:
            query: Search query
            num_results: Maximum results to return
            group_id: Optional group ID filter

        Returns:
            List of search results
        """
        try:
            client = await self._get_client()

            gid = group_id or self.config.group_id
            payload = {
                "query": query,
                "group_ids": [gid] if gid else None,
                "max_facts": num_results,
                "search_methods": ["cosine_similarity"],
                "reranker": "rrf",
            }

            response = await client.post("/search", json=payload)
            response.raise_for_status()

            data = response.json()
            results = []

            for fact in data.get("facts", []):
                results.append(SearchResult(
                    uuid=fact.get("uuid", ""),
                    name=fact.get("name", ""),
                    fact=fact.get("fact", ""),
                    score=0.0,
                    source_description=fact.get("source_description"),
                ))

            return results

        except Exception as e:
            logger.error("Failed to search memory: %s", e)
            return []

    async def enhanced_search(
        self,
        query: str,
        num_results: int = 5,
        group_id: Optional[str] = None,
        use_expansion: bool = True,
        use_reranking: bool = True,
        use_deduplication: bool = True,
    ) -> EnhancedSearchResult:
        """
        Enhanced search - delegates to basic search.

        The graphiti-service does not expose an /search/enhanced endpoint.
        This method wraps the basic search for API compatibility.

        Args:
            query: Search query
            num_results: Maximum results to return
            group_id: Optional group ID filter
            use_expansion: Unused (kept for signature compatibility)
            use_reranking: Unused (kept for signature compatibility)
            use_deduplication: Unused (kept for signature compatibility)

        Returns:
            EnhancedSearchResult with results
        """
        try:
            results = await self.search(
                query=query,
                num_results=num_results,
                group_id=group_id,
            )

            return EnhancedSearchResult(
                results=results,
                original_query=query,
                search_query=query,
            )

        except Exception as e:
            logger.error("Failed enhanced search: %s", e)
            return EnhancedSearchResult(results=[], original_query=query)

    async def get_context_for_query(
        self,
        query: str,
        num_results: int = 3,
        use_enhanced: bool = False,
    ) -> str:
        """
        Get formatted context string for a query.

        Uses query classification to skip RAG for device commands
        and simple queries, improving latency.

        Args:
            query: User query
            num_results: Number of results to include
            use_enhanced: Use enhanced search (currently disabled)

        Returns:
            Formatted context string
        """
        # Classify the query first
        classifier = get_query_classifier()
        classification = classifier.classify(query)

        if not classification.use_rag:
            logger.debug(
                "Skipping RAG (category=%s): %s",
                classification.category,
                query[:50],
            )
            return ""

        # Use basic search (enhanced search endpoint has issues)
        results = await self.search(query, num_results=num_results)

        if not results:
            return ""

        context_parts = []
        for result in results:
            if result.fact:
                context_parts.append(f"- {result.fact}")

        if not context_parts:
            return ""

        return "Relevant context:\n" + "\n".join(context_parts)

    async def add_fact(
        self,
        fact: str,
        source: str = "atlas-learned",
        timestamp: Optional[datetime] = None,
    ) -> bool:
        """
        Store a learned fact in the knowledge graph via /messages.

        Args:
            fact: The fact to store
            source: Source description
            timestamp: Optional timestamp

        Returns:
            True if accepted, False otherwise
        """
        try:
            client = await self._get_client()

            ts = timestamp or datetime.now(timezone.utc)

            message = {
                "content": fact,
                "role_type": "system",
                "role": None,
                "source_description": source,
                "timestamp": ts.isoformat() + "Z",
            }
            payload = {
                "group_id": self.config.group_id,
                "messages": [message],
            }

            response = await client.post("/messages", json=payload)

            if response.status_code not in (200, 202):
                logger.warning(
                    "add_fact failed (%d): %s",
                    response.status_code,
                    response.text[:200],
                )
                return False

            data = response.json()
            return data.get("success", False)

        except Exception as e:
            logger.error("Failed to add fact: %s", e)
            return False


_memory_client: Optional[MemoryClient] = None


def get_memory_client() -> MemoryClient:
    """Get or create the global memory client."""
    global _memory_client
    if _memory_client is None:
        _memory_client = MemoryClient()
    return _memory_client
