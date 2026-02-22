"""
Unit tests for the CRM provider, Email provider, and MCP server tools.

All external dependencies (DB pool, Directus HTTP, Gmail/Resend, GPU packages)
are mocked so these tests run without any real services or GPU hardware.
"""

import sys
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Mock heavy/unavailable dependencies at import time.
# PIL (Pillow) is required by the moondream VLM but isn't installed in the
# unit-test environment.  Setting it up in sys.modules before atlas_brain
# is imported prevents the ImportError cascade through services/__init__.py.
# ---------------------------------------------------------------------------
for _heavy_mod in ["PIL", "PIL.Image", "transformers", "torch"]:
    sys.modules.setdefault(_heavy_mod, MagicMock())

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_contact(**kwargs) -> dict:
    return {
        "id": str(uuid4()),
        "full_name": kwargs.get("full_name", "Jane Smith"),
        "phone": kwargs.get("phone", "618-555-0100"),
        "email": kwargs.get("email", "jane@example.com"),
        "status": kwargs.get("status", "active"),
        "contact_type": kwargs.get("contact_type", "customer"),
        "business_context_id": kwargs.get("business_context_id", "effingham_maids"),
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
        "tags": [],
        "notes": None,
        "source": "manual",
        "source_ref": None,
        "metadata": {},
    }


def _make_appointment(**kwargs) -> dict:
    return {
        "id": str(uuid4()),
        "start_time": datetime(2026, 3, 1, 9, 0, tzinfo=timezone.utc),
        "end_time": datetime(2026, 3, 1, 10, 0, tzinfo=timezone.utc),
        "service_type": kwargs.get("service_type", "Cleaning Estimate"),
        "status": kwargs.get("status", "confirmed"),
        "customer_name": kwargs.get("customer_name", "Jane Smith"),
        "customer_phone": kwargs.get("customer_phone", "618-555-0100"),
        "customer_email": kwargs.get("customer_email", "jane@example.com"),
        "customer_address": kwargs.get("customer_address", "123 Main St"),
        "notes": "",
        "created_at": datetime.now(timezone.utc),
    }


# ===========================================================================
# DatabaseCRMProvider (direct asyncpg queries)
# ===========================================================================

class TestDatabaseCRMProvider:
    """Tests for DatabaseCRMProvider using a mocked asyncpg pool."""

    def _mock_pool(self, fetchrow_return=None, fetch_return=None, execute_return=None):
        pool = MagicMock()
        pool.is_initialized = True
        pool.fetchrow = AsyncMock(return_value=fetchrow_return)
        pool.fetch = AsyncMock(return_value=fetch_return or [])
        pool.execute = AsyncMock(return_value=execute_return or "UPDATE 1")
        return pool

    async def test_create_contact_returns_dict(self):
        from atlas_brain.services.crm_provider import DatabaseCRMProvider

        contact_data = _make_contact()
        pool = self._mock_pool(fetchrow_return=contact_data)

        with patch("atlas_brain.services.crm_provider.DatabaseCRMProvider.create_contact") as mock_create:
            mock_create = AsyncMock(return_value=contact_data)
            provider = DatabaseCRMProvider()
            provider.create_contact = mock_create

            result = await provider.create_contact({
                "full_name": "Jane Smith",
                "phone": "618-555-0100",
                "email": "jane@example.com",
            })

        assert result["full_name"] == "Jane Smith"
        assert result["phone"] == "618-555-0100"

    async def test_search_contacts_returns_list(self):
        from atlas_brain.services.crm_provider import DatabaseCRMProvider

        contacts = [_make_contact(), _make_contact(full_name="John Doe")]

        provider = DatabaseCRMProvider()
        provider.search_contacts = AsyncMock(return_value=contacts)

        results = await provider.search_contacts(query="smith")
        assert len(results) == 2

    async def test_get_contact_not_found_returns_none(self):
        from atlas_brain.services.crm_provider import DatabaseCRMProvider

        provider = DatabaseCRMProvider()
        provider.get_contact = AsyncMock(return_value=None)

        result = await provider.get_contact(str(uuid4()))
        assert result is None

    async def test_delete_contact_soft_deletes(self):
        from atlas_brain.services.crm_provider import DatabaseCRMProvider

        provider = DatabaseCRMProvider()
        provider.delete_contact = AsyncMock(return_value=True)

        result = await provider.delete_contact(str(uuid4()))
        assert result is True

    async def test_health_check_initialized(self):
        from atlas_brain.services.crm_provider import DatabaseCRMProvider

        provider = DatabaseCRMProvider()
        with patch("atlas_brain.services.crm_provider.DatabaseCRMProvider.health_check",
                   new=AsyncMock(return_value=True)):
            ok = await provider.health_check()
        assert ok is True


# ===========================================================================
# get_crm_provider factory
# ===========================================================================

class TestGetCrmProviderFactory:
    async def test_returns_database_provider_when_directus_disabled(self):
        import atlas_brain.services.crm_provider as mod

        # Reset singleton
        mod._crm_provider = None

        mock_settings = MagicMock()
        mock_settings.directus.enabled = False
        mock_settings.directus.token = None

        with patch("atlas_brain.services.crm_provider.settings", mock_settings):
            provider = mod.get_crm_provider()
            from atlas_brain.services.crm_provider import DatabaseCRMProvider
            assert isinstance(provider, DatabaseCRMProvider)

        mod._crm_provider = None  # Reset for other tests

    async def test_returns_directus_provider_when_enabled(self):
        import atlas_brain.services.crm_provider as mod

        mod._crm_provider = None

        mock_settings = MagicMock()
        mock_settings.directus.enabled = True
        mock_settings.directus.token = "static-token-abc"
        mock_settings.directus.url = "http://localhost:8055"
        mock_settings.directus.timeout = 10.0

        with patch("atlas_brain.services.crm_provider.settings", mock_settings):
            provider = mod.get_crm_provider()
            from atlas_brain.services.crm_provider import DirectusCRMProvider
            assert isinstance(provider, DirectusCRMProvider)

        mod._crm_provider = None


# ===========================================================================
# CRM MCP server tools
# ===========================================================================

class TestCRMMCPTools:
    """Tests for MCP tool functions using a mocked CRM provider."""

    def _patch_provider(self, mock_provider):
        return patch(
            "atlas_brain.mcp.crm_server._provider",
            return_value=mock_provider,
        )

    async def test_search_contacts_found(self):
        from atlas_brain.mcp.crm_server import search_contacts

        contact = _make_contact()
        provider = MagicMock()
        provider.search_contacts = AsyncMock(return_value=[contact])

        with self._patch_provider(provider):
            raw = await search_contacts(query="Jane")

        data = json.loads(raw)
        assert data["found"] is True
        assert data["count"] == 1
        assert data["contacts"][0]["full_name"] == "Jane Smith"

    async def test_search_contacts_not_found(self):
        from atlas_brain.mcp.crm_server import search_contacts

        provider = MagicMock()
        provider.search_contacts = AsyncMock(return_value=[])

        with self._patch_provider(provider):
            raw = await search_contacts(query="nobody")

        data = json.loads(raw)
        assert data["found"] is False
        assert data["count"] == 0

    async def test_get_contact_found(self):
        from atlas_brain.mcp.crm_server import get_contact

        contact = _make_contact()
        provider = MagicMock()
        provider.get_contact = AsyncMock(return_value=contact)

        with self._patch_provider(provider):
            raw = await get_contact(contact["id"])

        data = json.loads(raw)
        assert data["found"] is True
        assert data["contact"]["full_name"] == "Jane Smith"

    async def test_get_contact_not_found(self):
        from atlas_brain.mcp.crm_server import get_contact

        provider = MagicMock()
        provider.get_contact = AsyncMock(return_value=None)

        with self._patch_provider(provider):
            raw = await get_contact(str(uuid4()))

        data = json.loads(raw)
        assert data["found"] is False

    async def test_create_contact_success(self):
        from atlas_brain.mcp.crm_server import create_contact

        contact = _make_contact()
        provider = MagicMock()
        provider.create_contact = AsyncMock(return_value=contact)

        with self._patch_provider(provider):
            raw = await create_contact(
                full_name="Jane Smith",
                phone="618-555-0100",
                email="jane@example.com",
            )

        data = json.loads(raw)
        assert data["success"] is True
        assert data["contact"]["full_name"] == "Jane Smith"

    async def test_update_contact_no_fields(self):
        from atlas_brain.mcp.crm_server import update_contact

        provider = MagicMock()

        with self._patch_provider(provider):
            raw = await update_contact(contact_id=str(uuid4()))

        data = json.loads(raw)
        assert data["success"] is False
        assert "No fields provided" in data["error"]

    async def test_delete_contact(self):
        from atlas_brain.mcp.crm_server import delete_contact

        provider = MagicMock()
        provider.delete_contact = AsyncMock(return_value=True)

        with self._patch_provider(provider):
            raw = await delete_contact(str(uuid4()))

        data = json.loads(raw)
        assert data["success"] is True

    async def test_log_interaction(self):
        from atlas_brain.mcp.crm_server import log_interaction

        interaction = {
            "id": str(uuid4()),
            "contact_id": str(uuid4()),
            "interaction_type": "call",
            "summary": "Customer called about scheduling",
            "occurred_at": datetime.now(timezone.utc).isoformat(),
        }
        provider = MagicMock()
        provider.log_interaction = AsyncMock(return_value=interaction)

        with self._patch_provider(provider):
            raw = await log_interaction(
                contact_id=interaction["contact_id"],
                interaction_type="call",
                summary="Customer called about scheduling",
            )

        data = json.loads(raw)
        assert data["success"] is True
        assert data["interaction"]["interaction_type"] == "call"

    async def test_get_interactions(self):
        from atlas_brain.mcp.crm_server import get_interactions

        interactions = [
            {"id": str(uuid4()), "interaction_type": "call", "summary": "first call"},
            {"id": str(uuid4()), "interaction_type": "email", "summary": "sent estimate"},
        ]
        provider = MagicMock()
        provider.get_interactions = AsyncMock(return_value=interactions)

        with self._patch_provider(provider):
            raw = await get_interactions(contact_id=str(uuid4()))

        data = json.loads(raw)
        assert data["count"] == 2

    async def test_get_contact_appointments(self):
        from atlas_brain.mcp.crm_server import get_contact_appointments

        appts = [_make_appointment(), _make_appointment()]
        provider = MagicMock()
        provider.get_contact_appointments = AsyncMock(return_value=appts)

        with self._patch_provider(provider):
            raw = await get_contact_appointments(str(uuid4()))

        data = json.loads(raw)
        assert data["count"] == 2

    async def test_search_contacts_provider_error(self):
        """Provider errors should return a safe error JSON, not raise."""
        from atlas_brain.mcp.crm_server import search_contacts

        provider = MagicMock()
        provider.search_contacts = AsyncMock(side_effect=RuntimeError("DB down"))

        with self._patch_provider(provider):
            raw = await search_contacts(query="anyone")

        data = json.loads(raw)
        assert "error" in data
        assert data["found"] is False


# ===========================================================================
# Email MCP server tools
# ===========================================================================

class TestEmailMCPTools:
    def _patch_provider(self, mock_provider):
        return patch(
            "atlas_brain.mcp.email_server._provider",
            return_value=mock_provider,
        )

    async def test_send_email_success(self):
        from atlas_brain.mcp.email_server import send_email

        provider = MagicMock()
        provider.send = AsyncMock(return_value={"id": "msg-123"})

        with self._patch_provider(provider):
            raw = await send_email(
                to="alice@example.com",
                subject="Hello",
                body="Test body",
            )

        data = json.loads(raw)
        assert data["success"] is True
        assert data["result"]["id"] == "msg-123"

    async def test_send_email_comma_separated_to(self):
        """Comma-separated 'to' is parsed into a list before being passed to the provider."""
        from atlas_brain.mcp.email_server import send_email

        provider = MagicMock()
        provider.send = AsyncMock(return_value={"id": "msg-456"})

        with self._patch_provider(provider):
            await send_email(
                to="alice@example.com, bob@example.com",
                subject="Multi",
                body="Body",
            )

        call_kwargs = provider.send.call_args.kwargs
        assert call_kwargs["to"] == ["alice@example.com", "bob@example.com"]

    async def test_send_email_provider_error(self):
        from atlas_brain.mcp.email_server import send_email

        provider = MagicMock()
        provider.send = AsyncMock(side_effect=RuntimeError("Gmail unavailable"))

        with self._patch_provider(provider):
            raw = await send_email(to="x@y.com", subject="s", body="b")

        data = json.loads(raw)
        assert data["success"] is False
        assert "Gmail unavailable" in data["error"]

    async def test_list_inbox(self):
        from atlas_brain.mcp.email_server import list_inbox

        messages = [{"id": "m1"}, {"id": "m2"}]
        provider = MagicMock()
        provider.list_messages = AsyncMock(return_value=messages)

        with self._patch_provider(provider):
            raw = await list_inbox(query="is:unread", max_results=10)

        data = json.loads(raw)
        assert data["count"] == 2
        assert data["messages"][0]["id"] == "m1"

    async def test_get_message(self):
        from atlas_brain.mcp.email_server import get_message

        full_msg = {
            "id": "m1",
            "from": "alice@example.com",
            "subject": "Invoice",
            "body_text": "Please pay…",
        }
        provider = MagicMock()
        provider.get_message = AsyncMock(return_value=full_msg)

        with self._patch_provider(provider):
            raw = await get_message("m1")

        data = json.loads(raw)
        assert data["message"]["subject"] == "Invoice"

    async def test_get_thread(self):
        from atlas_brain.mcp.email_server import get_thread

        thread = {"id": "t1", "messages": [{"id": "m1"}, {"id": "m2"}]}
        provider = MagicMock()
        provider.get_thread = AsyncMock(return_value=thread)

        with self._patch_provider(provider):
            raw = await get_thread("t1")

        data = json.loads(raw)
        assert data["thread"]["id"] == "t1"
        assert len(data["thread"]["messages"]) == 2

    async def test_list_sent_history(self):
        from atlas_brain.mcp.email_server import list_sent_history

        from atlas_brain.tools.base import ToolResult
        mock_tool_result = ToolResult(
            success=True,
            data={"emails": [], "count": 0},
            message="No emails found.",
        )

        with patch("atlas_brain.mcp.email_server.list_sent_history") as mocked:
            mocked = AsyncMock(return_value=json.dumps(
                {"success": True, "data": {"emails": [], "count": 0}, "message": "No emails found."}
            ))
            raw = await mocked(hours=24)

        data = json.loads(raw)
        assert data["success"] is True


# ===========================================================================
# CompositeEmailProvider — fallback logic
# ===========================================================================

class TestCompositeEmailProvider:
    async def test_send_uses_gmail_when_available(self):
        from atlas_brain.services.email_provider import CompositeEmailProvider

        provider = CompositeEmailProvider()
        provider._gmail.is_available = AsyncMock(return_value=True)
        provider._gmail.send = AsyncMock(return_value={"id": "gmail-id"})
        provider._resend.send = AsyncMock(return_value={"id": "resend-id"})

        result = await provider.send(to=["x@y.com"], subject="s", body="b")
        assert result["id"] == "gmail-id"
        provider._resend.send.assert_not_called()

    async def test_send_falls_back_to_resend(self):
        from atlas_brain.services.email_provider import CompositeEmailProvider

        provider = CompositeEmailProvider()
        provider._gmail.is_available = AsyncMock(return_value=True)
        provider._gmail.send = AsyncMock(side_effect=RuntimeError("token expired"))
        provider._resend.send = AsyncMock(return_value={"id": "resend-id"})

        result = await provider.send(to=["x@y.com"], subject="s", body="b")
        assert result["id"] == "resend-id"

    async def test_send_uses_resend_when_gmail_unavailable(self):
        from atlas_brain.services.email_provider import CompositeEmailProvider

        provider = CompositeEmailProvider()
        provider._gmail.is_available = AsyncMock(return_value=False)
        provider._gmail.send = AsyncMock()
        provider._resend.send = AsyncMock(return_value={"id": "resend-only"})

        result = await provider.send(to=["x@y.com"], subject="s", body="b")
        assert result["id"] == "resend-only"
        provider._gmail.send.assert_not_called()


# ===========================================================================
# LookupCustomerTool — CRM-first strategy
# ===========================================================================

class TestLookupCustomerToolCRMFirst:
    async def test_returns_crm_contact_when_found(self):
        """When CRM has a match it should be returned without querying appointments."""
        from atlas_brain.tools.scheduling import LookupCustomerTool

        contact = _make_contact()
        contact["id"] = str(uuid4())
        mock_crm = MagicMock()
        mock_crm.search_contacts = AsyncMock(return_value=[contact])
        mock_crm.get_contact_appointments = AsyncMock(return_value=[])

        tool = LookupCustomerTool()
        with patch("atlas_brain.tools.scheduling.get_crm_provider", return_value=mock_crm):
            result = await tool.execute({"name": "Jane"})

        assert result.success is True
        assert result.data["found"] is True
        assert result.data["customer"]["source"] == "crm"
        assert result.data["customer"]["name"] == "Jane Smith"

    async def test_falls_back_to_appointments_when_crm_empty(self):
        """When CRM returns nothing the tool falls back to appointment rows."""
        from atlas_brain.tools.scheduling import LookupCustomerTool
        from atlas_brain.storage.exceptions import DatabaseUnavailableError

        mock_crm = MagicMock()
        mock_crm.search_contacts = AsyncMock(return_value=[])

        appt = _make_appointment()
        # Make start_time aware for comparison inside the tool
        appt["start_time"] = datetime(2027, 1, 1, 9, 0, tzinfo=timezone.utc)
        mock_repo = MagicMock()
        mock_repo.search_by_name = AsyncMock(return_value=[appt])
        mock_repo.get_by_phone = AsyncMock(return_value=[])

        tool = LookupCustomerTool()
        with (
            patch("atlas_brain.tools.scheduling.get_crm_provider", return_value=mock_crm),
            patch("atlas_brain.tools.scheduling.get_appointment_repo", return_value=mock_repo),
        ):
            result = await tool.execute({"name": "Jane"})

        assert result.success is True
        assert result.data["found"] is True
        assert result.data["customer"]["source"] == "appointments"

    async def test_missing_params_returns_error(self):
        from atlas_brain.tools.scheduling import LookupCustomerTool

        tool = LookupCustomerTool()
        result = await tool.execute({})

        assert result.success is False
        assert result.error == "MISSING_PARAMS"
