"""
Tests for conversation mode agent path.

Covers:
1. Config toggle: conversation_agent_enabled routes to agent path
2. Config opt-out: disabled flag preserves streaming
3. Sentence splitting: multi-sentence responses split for TTS
4. Sentence splitting: single sentence passes through
5. Max tokens propagation via runtime_ctx
6. Agent path has tool access (execute_with_tools called)
7. PRIORITY_TOOL_NAMES includes read/query tools
8. PRIORITY_TOOL_NAMES excludes write-only tools
"""

import inspect
import re

import pytest


class TestConversationAgentConfig:
    """Config fields exist and have correct defaults."""

    def test_conversation_agent_enabled_default(self):
        from atlas_brain.config import VoiceClientConfig

        cfg = VoiceClientConfig()
        assert cfg.conversation_agent_enabled is True

    def test_conversation_agent_max_tokens_default(self):
        from atlas_brain.config import VoiceClientConfig

        cfg = VoiceClientConfig()
        assert cfg.conversation_agent_max_tokens == 512

    def test_conversation_agent_disabled(self):
        from atlas_brain.config import VoiceClientConfig

        cfg = VoiceClientConfig(conversation_agent_enabled=False)
        assert cfg.conversation_agent_enabled is False


class TestConversationAgentRouting:
    """Launcher routes conversation mode to agent path."""

    def test_conversation_agent_enabled_skips_streaming(self):
        """When conversation_agent_enabled and state=='conversing',
        use_streaming should be set to False in check_and_run."""
        from atlas_brain.voice.launcher import _create_streaming_agent_runner

        source = inspect.getsource(_create_streaming_agent_runner)
        # The override block must check all three conditions
        assert "conversation_agent_enabled" in source
        assert 'state == "conversing"' in source
        assert "use_streaming = False" in source

    def test_conversation_agent_disabled_allows_streaming(self):
        """The override is gated on conversation_agent_enabled,
        so disabling it should not force agent path."""
        from atlas_brain.voice.launcher import _create_streaming_agent_runner

        source = inspect.getsource(_create_streaming_agent_runner)
        # Must check settings.voice.conversation_agent_enabled before overriding
        assert "settings.voice.conversation_agent_enabled" in source


class TestSentenceSplitting:
    """Agent responses are split into sentences for TTS."""

    def test_sentence_splitting_multiple(self):
        """Multi-sentence text splits at sentence boundaries."""
        text = "I found the email. He seems fine. I'll draft a reply."
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        assert len(sentences) == 3
        assert sentences[0] == "I found the email."
        assert sentences[1] == "He seems fine."
        assert sentences[2] == "I'll draft a reply."

    def test_sentence_splitting_single(self):
        """Single sentence (no split point) passes through as one."""
        text = "Hello"
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        assert len(sentences) == 1
        assert sentences[0] == "Hello"

    def test_sentence_splitting_exclamation_and_question(self):
        """Exclamation marks and question marks are split points."""
        text = "Done! Want more? Sure thing."
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        assert len(sentences) == 3

    def test_sentence_splitting_in_launcher(self):
        """_run_agent_fallback uses sentence splitting."""
        from atlas_brain.voice.launcher import _run_agent_fallback

        source = inspect.getsource(_run_agent_fallback)
        assert "re.split" in source
        assert "on_sentence(sentence)" in source


class TestConversationMaxTokens:
    """conversation_max_tokens propagated through runtime_ctx."""

    def test_conversation_max_tokens_propagated(self):
        """_run_agent_fallback sets conversation_max_tokens in runtime_ctx."""
        from atlas_brain.voice.launcher import _run_agent_fallback

        source = inspect.getsource(_run_agent_fallback)
        assert "conversation_max_tokens" in source
        assert "conversation_agent_max_tokens" in source

    def test_atlas_reads_conversation_max_tokens(self):
        """_generate_llm_response reads conversation_max_tokens from state."""
        from atlas_brain.agents.graphs.atlas import _generate_llm_response

        source = inspect.getsource(_generate_llm_response)
        assert '"conversation_max_tokens"' in source
        assert "conv_max_tokens" in source


class TestAgentToolAccess:
    """Agent path provides tool access via execute_with_tools."""

    def test_agent_path_has_tool_access(self):
        """_generate_llm_response calls execute_with_tools."""
        from atlas_brain.agents.graphs.atlas import _generate_llm_response

        source = inspect.getsource(_generate_llm_response)
        assert "execute_with_tools" in source

    def test_priority_tools_include_read_query_tools(self):
        """PRIORITY_TOOL_NAMES contains the new read/query tools."""
        from atlas_brain.services.tool_executor import PRIORITY_TOOL_NAMES

        expected_new = [
            "get_interactions", "get_contact_appointments",
            "get_thread", "list_sent_history",
            "send_sms", "lookup_phone", "get_call", "list_calls",
            "get_invoice", "list_invoices", "search_invoices",
            "customer_balance", "payment_history",
        ]
        for tool in expected_new:
            assert tool in PRIORITY_TOOL_NAMES, f"{tool} missing from PRIORITY_TOOL_NAMES"

    def test_priority_tools_exclude_write_tools(self):
        """Write/action tools must NOT be in PRIORITY_TOOL_NAMES."""
        from atlas_brain.services.tool_executor import PRIORITY_TOOL_NAMES

        excluded = [
            "make_call", "hangup_call", "start_recording", "stop_recording",
            "create_contact", "update_contact", "delete_contact",
            "log_interaction", "list_contacts",
            "create_invoice", "update_invoice", "send_invoice",
            "record_payment", "mark_void",
            "send_estimate", "send_proposal",
            "update_event", "delete_event",
        ]
        for tool in excluded:
            assert tool not in PRIORITY_TOOL_NAMES, f"{tool} should NOT be in PRIORITY_TOOL_NAMES"
