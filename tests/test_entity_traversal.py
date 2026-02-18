"""
Tests for entity-focused graph traversal.

Covers:
1. QueryClassifier entity name extraction
2. RAGClient.get_entity_edges()
3. Parallel search + traversal in retrieve_memory node
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from atlas_brain.memory.query_classifier import ClassificationResult, QueryClassifier
from atlas_brain.memory.rag_client import RAGClient, SearchResult, SearchSource


# ---------------------------------------------------------------------------
# 1. QueryClassifier entity extraction
# ---------------------------------------------------------------------------


class TestEntityExtraction:
    """Tests for _extract_entity_name and its wiring into classify()."""

    def setup_method(self):
        self.classifier = QueryClassifier()

    def test_what_do_you_know_about(self):
        result = self.classifier.classify("What do you know about Juan?")
        assert result.use_rag is True
        assert result.entity_name == "Juan"

    def test_tell_me_about(self):
        result = self.classifier.classify("Tell me about the living room")
        assert result.use_rag is True
        # "the" should be stripped
        assert result.entity_name == "Living Room"

    def test_who_is(self):
        result = self.classifier.classify("Who is Maria?")
        assert result.use_rag is True
        assert result.entity_name == "Maria"

    def test_what_is(self):
        result = self.classifier.classify("What is Atlas?")
        assert result.use_rag is True
        assert result.entity_name == "Atlas"

    def test_facts_about(self):
        result = self.classifier.classify("facts about the weather station")
        assert result.use_rag is True
        assert result.entity_name == "Weather Station"

    def test_no_entity_for_device_command(self):
        result = self.classifier.classify("Turn on the kitchen lights")
        assert result.use_rag is False
        assert result.entity_name is None

    def test_no_entity_for_greeting(self):
        result = self.classifier.classify("Hello")
        assert result.use_rag is False
        assert result.entity_name is None

    def test_no_entity_for_generic_question(self):
        """A knowledge query that doesn't match entity patterns."""
        result = self.classifier.classify("How does the weather work?")
        assert result.use_rag is True
        assert result.entity_name is None

    def test_strips_trailing_punctuation(self):
        result = self.classifier.classify("What do you know about Juan!!")
        assert result.entity_name == "Juan"

    def test_strips_leading_article_a(self):
        result = self.classifier.classify("Tell me about a new project")
        assert result.entity_name == "New Project"

    def test_general_category_with_entity(self):
        """Longer queries that fall to 'general' category can still extract entities."""
        result = self.classifier.classify(
            "Do you know anything about the security system configuration?"
        )
        assert result.use_rag is True
        assert result.entity_name == "Security System Configuration"

    def test_empty_entity_returns_none(self):
        """Edge case: pattern matches but capture group is empty after stripping."""
        name = self.classifier._extract_entity_name("tell me about the   ")
        assert name is None


# ---------------------------------------------------------------------------
# 2. RAGClient.get_entity_edges()
# ---------------------------------------------------------------------------


class TestGetEntityEdges:
    """Tests for RAGClient.get_entity_edges()."""

    @pytest.fixture
    def mock_settings(self):
        with patch("atlas_brain.memory.rag_client.settings") as mock:
            mock.memory.enabled = True
            mock.memory.base_url = "http://localhost:8003"
            mock.memory.timeout = 10.0
            mock.memory.group_id = "test-group"
            yield mock

    @pytest.mark.asyncio
    async def test_returns_search_result_with_facts(self, mock_settings):
        client = RAGClient(base_url="http://localhost:8003")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "edges": [
                {
                    "uuid": "edge-1",
                    "name": "lives_at",
                    "fact": "Juan lives at 123 Main St",
                    "score": 0.9,
                    "created_at": "2025-01-01T00:00:00Z",
                    "expired_at": None,
                },
                {
                    "uuid": "edge-2",
                    "name": "prefers",
                    "fact": "Juan prefers dark mode",
                    "score": 0.8,
                },
            ],
        }

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_resp)
        mock_http.is_closed = False
        client._client = mock_http

        result = await client.get_entity_edges("Juan")

        assert isinstance(result, SearchResult)
        assert len(result.facts) == 2
        assert result.facts[0].uuid == "edge-1"
        assert result.facts[0].fact == "Juan lives at 123 Main St"
        assert result.facts[0].confidence == 0.9
        assert result.facts[1].uuid == "edge-2"

        # Verify URL encoding
        mock_http.get.assert_called_once()
        call_args = mock_http.get.call_args
        assert "Juan" in call_args[0][0]

        await client.close()

    @pytest.mark.asyncio
    async def test_returns_empty_when_disabled(self, mock_settings):
        mock_settings.memory.enabled = False
        client = RAGClient()
        result = await client.get_entity_edges("Juan")
        assert result.facts == []

    @pytest.mark.asyncio
    async def test_handles_connection_error(self, mock_settings):
        import httpx

        client = RAGClient(base_url="http://localhost:8003")
        mock_http = AsyncMock()
        mock_http.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
        mock_http.is_closed = False
        client._client = mock_http

        result = await client.get_entity_edges("Juan")
        assert result.facts == []

        await client.close()

    @pytest.mark.asyncio
    async def test_handles_http_error(self, mock_settings):
        import httpx

        client = RAGClient(base_url="http://localhost:8003")
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not Found", request=MagicMock(), response=mock_resp,
        )

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_resp)
        mock_http.is_closed = False
        client._client = mock_http

        result = await client.get_entity_edges("Juan")
        assert result.facts == []

        await client.close()

    @pytest.mark.asyncio
    async def test_respects_max_edges(self, mock_settings):
        client = RAGClient(base_url="http://localhost:8003")
        edges = [{"uuid": f"e-{i}", "name": "rel", "fact": f"fact {i}"} for i in range(30)]
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"edges": edges}

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_resp)
        mock_http.is_closed = False
        client._client = mock_http

        result = await client.get_entity_edges("Juan", max_edges=5)
        assert len(result.facts) == 5

        await client.close()

    @pytest.mark.asyncio
    async def test_url_encodes_entity_name(self, mock_settings):
        client = RAGClient(base_url="http://localhost:8003")
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"edges": []}

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_resp)
        mock_http.is_closed = False
        client._client = mock_http

        await client.get_entity_edges("Living Room")

        call_url = mock_http.get.call_args[0][0]
        assert "Living%20Room" in call_url

        await client.close()


# ---------------------------------------------------------------------------
# 3. Parallel retrieval in retrieve_memory
# ---------------------------------------------------------------------------


def _make_source(uuid: str, fact: str) -> SearchSource:
    return SearchSource(uuid=uuid, name="test", fact=fact, confidence=0.9)


def _patch_config():
    """Patch atlas_brain.config.settings for retrieve_memory tests."""
    return patch("atlas_brain.config.settings")


def _configure_memory_settings(mock_cfg):
    """Set standard memory settings on a mocked config."""
    mock_cfg.memory.enabled = True
    mock_cfg.memory.retrieve_context = True
    mock_cfg.memory.context_results = 5
    mock_cfg.memory.context_timeout = 5.0
    mock_cfg.intent_router.conversation_confidence_threshold = 0.8


class TestParallelRetrieval:
    """Tests for parallel search + traversal in retrieve_memory node."""

    def _make_state(self, input_text: str, action_type: str = "conversation"):
        return {
            "input_text": input_text,
            "action_type": action_type,
            "confidence": 0.9,
            "session_id": "test-session",
            "runtime_context": {},
        }

    @pytest.mark.asyncio
    async def test_entity_query_runs_parallel(self):
        """When entity_name is detected, both search and traversal should run."""
        search_facts = [_make_source("s1", "Juan is a developer")]
        traversal_facts = [_make_source("t1", "Juan lives in Texas")]

        mock_search = AsyncMock(return_value=SearchResult(facts=search_facts))
        mock_edges = AsyncMock(return_value=SearchResult(facts=traversal_facts))

        with _patch_config() as mock_cfg:
            _configure_memory_settings(mock_cfg)

            mock_client = MagicMock()
            mock_client.search = mock_search
            mock_client.get_entity_edges = mock_edges

            with patch(
                "atlas_brain.memory.rag_client.get_rag_client",
                return_value=mock_client,
            ), patch(
                "atlas_brain.memory.query_classifier.get_query_classifier",
            ) as mock_clf:
                mock_clf.return_value.classify.return_value = ClassificationResult(
                    use_rag=True,
                    category="knowledge",
                    reason="test",
                    entity_name="Juan",
                )

                from atlas_brain.agents.graphs.atlas import retrieve_memory

                state = self._make_state("What do you know about Juan?")
                result = await retrieve_memory(state)

                # Both should have been called
                mock_search.assert_called_once()
                mock_edges.assert_called_once_with("Juan")

                # Result should contain merged facts
                sources = result.update.get("retrieved_sources", [])
                assert len(sources) == 2
                uuids = {s.uuid for s in sources}
                assert "s1" in uuids
                assert "t1" in uuids

    @pytest.mark.asyncio
    async def test_no_entity_skips_traversal(self):
        """When no entity_name, only search runs (no traversal)."""
        search_facts = [_make_source("s1", "Some fact")]
        mock_search = AsyncMock(return_value=SearchResult(facts=search_facts))
        mock_edges = AsyncMock()

        with _patch_config() as mock_cfg:
            _configure_memory_settings(mock_cfg)

            mock_client = MagicMock()
            mock_client.search = mock_search
            mock_client.get_entity_edges = mock_edges

            with patch(
                "atlas_brain.memory.rag_client.get_rag_client",
                return_value=mock_client,
            ), patch(
                "atlas_brain.memory.query_classifier.get_query_classifier",
            ) as mock_clf:
                mock_clf.return_value.classify.return_value = ClassificationResult(
                    use_rag=True,
                    category="knowledge",
                    reason="test",
                    entity_name=None,
                )

                from atlas_brain.agents.graphs.atlas import retrieve_memory

                state = self._make_state("How does weather work?")
                result = await retrieve_memory(state)

                mock_search.assert_called_once()
                mock_edges.assert_not_called()

    @pytest.mark.asyncio
    async def test_deduplication_by_uuid(self):
        """Duplicate UUIDs across search and traversal should be merged."""
        shared_uuid = "shared-1"
        search_facts = [_make_source(shared_uuid, "Juan is a developer")]
        traversal_facts = [
            _make_source(shared_uuid, "Juan is a developer"),
            _make_source("t2", "Juan lives in Texas"),
        ]

        mock_search = AsyncMock(return_value=SearchResult(facts=search_facts))
        mock_edges = AsyncMock(return_value=SearchResult(facts=traversal_facts))

        with _patch_config() as mock_cfg:
            _configure_memory_settings(mock_cfg)

            mock_client = MagicMock()
            mock_client.search = mock_search
            mock_client.get_entity_edges = mock_edges

            with patch(
                "atlas_brain.memory.rag_client.get_rag_client",
                return_value=mock_client,
            ), patch(
                "atlas_brain.memory.query_classifier.get_query_classifier",
            ) as mock_clf:
                mock_clf.return_value.classify.return_value = ClassificationResult(
                    use_rag=True,
                    category="knowledge",
                    reason="test",
                    entity_name="Juan",
                )

                from atlas_brain.agents.graphs.atlas import retrieve_memory

                state = self._make_state("What do you know about Juan?")
                result = await retrieve_memory(state)

                sources = result.update.get("retrieved_sources", [])
                # shared-1 appears once, t2 is unique = 2 total
                assert len(sources) == 2

    @pytest.mark.asyncio
    async def test_traversal_failure_preserves_search(self):
        """If traversal fails, search results are still returned."""
        search_facts = [_make_source("s1", "Juan is a developer")]

        mock_search = AsyncMock(return_value=SearchResult(facts=search_facts))
        mock_edges = AsyncMock(side_effect=Exception("traversal boom"))

        with _patch_config() as mock_cfg:
            _configure_memory_settings(mock_cfg)

            mock_client = MagicMock()
            mock_client.search = mock_search
            mock_client.get_entity_edges = mock_edges

            with patch(
                "atlas_brain.memory.rag_client.get_rag_client",
                return_value=mock_client,
            ), patch(
                "atlas_brain.memory.query_classifier.get_query_classifier",
            ) as mock_clf:
                mock_clf.return_value.classify.return_value = ClassificationResult(
                    use_rag=True,
                    category="knowledge",
                    reason="test",
                    entity_name="Juan",
                )

                from atlas_brain.agents.graphs.atlas import retrieve_memory

                state = self._make_state("What do you know about Juan?")
                result = await retrieve_memory(state)

                sources = result.update.get("retrieved_sources", [])
                assert len(sources) == 1
                assert sources[0].uuid == "s1"

    @pytest.mark.asyncio
    async def test_search_failure_preserves_traversal(self):
        """If search fails, traversal results are still returned."""
        traversal_facts = [_make_source("t1", "Juan lives in Texas")]

        mock_search = AsyncMock(side_effect=Exception("search boom"))
        mock_edges = AsyncMock(return_value=SearchResult(facts=traversal_facts))

        with _patch_config() as mock_cfg:
            _configure_memory_settings(mock_cfg)

            mock_client = MagicMock()
            mock_client.search = mock_search
            mock_client.get_entity_edges = mock_edges

            with patch(
                "atlas_brain.memory.rag_client.get_rag_client",
                return_value=mock_client,
            ), patch(
                "atlas_brain.memory.query_classifier.get_query_classifier",
            ) as mock_clf:
                mock_clf.return_value.classify.return_value = ClassificationResult(
                    use_rag=True,
                    category="knowledge",
                    reason="test",
                    entity_name="Juan",
                )

                from atlas_brain.agents.graphs.atlas import retrieve_memory

                state = self._make_state("What do you know about Juan?")
                result = await retrieve_memory(state)

                sources = result.update.get("retrieved_sources", [])
                assert len(sources) == 1
                assert sources[0].uuid == "t1"
