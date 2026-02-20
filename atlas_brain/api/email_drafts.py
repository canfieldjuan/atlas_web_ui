"""
REST API for email draft management.

Provides list, view, approve, reject, and edit operations for
LLM-generated email reply drafts.
"""

import logging
from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

logger = logging.getLogger("atlas.api.email_drafts")

router = APIRouter(prefix="/email/drafts", tags=["email-drafts"])


def _affected_rows(result: str) -> int:
    """Parse affected row count from asyncpg execute result."""
    try:
        return int(result.split()[-1])
    except (ValueError, IndexError, AttributeError):
        return -1


def _row_to_dict(row) -> dict:
    """Convert a DB row to a response dict."""
    d = {}
    for key in row.keys():
        val = row[key]
        if isinstance(val, UUID):
            d[key] = str(val)
        elif isinstance(val, datetime):
            d[key] = val.isoformat()
        else:
            d[key] = val
    return d


class EditDraftBody(BaseModel):
    draft_body: str | None = None
    draft_subject: str | None = None


@router.get("/")
async def list_drafts(
    status: str = Query(default="pending", description="Filter by status: pending, approved, sent, rejected, expired, or all"),
    limit: int = Query(default=50, ge=1, le=500),
):
    """List email drafts with optional status filter."""
    from ..storage.database import get_db_pool

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"count": 0, "drafts": []}

    if status == "all":
        rows = await pool.fetch(
            """
            SELECT id, gmail_message_id, thread_id, original_from, original_subject,
                   draft_subject, draft_body, model_provider, model_name,
                   status, approved_at, sent_at, gmail_sent_id,
                   created_at, expires_at
            FROM email_drafts
            ORDER BY created_at DESC
            LIMIT $1
            """,
            limit,
        )
    else:
        rows = await pool.fetch(
            """
            SELECT id, gmail_message_id, thread_id, original_from, original_subject,
                   draft_subject, draft_body, model_provider, model_name,
                   status, approved_at, sent_at, gmail_sent_id,
                   created_at, expires_at
            FROM email_drafts
            WHERE status = $1
            ORDER BY created_at DESC
            LIMIT $2
            """,
            status, limit,
        )

    drafts = [_row_to_dict(r) for r in rows]
    return {"count": len(drafts), "drafts": drafts}


@router.get("/{draft_id}")
async def get_draft(draft_id: UUID):
    """Get full draft detail including original body."""
    from ..storage.database import get_db_pool

    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(503, "Database not available")

    row = await pool.fetchrow(
        "SELECT * FROM email_drafts WHERE id = $1", draft_id,
    )
    if not row:
        raise HTTPException(404, "Draft not found")

    return _row_to_dict(row)


@router.post("/{draft_id}/approve")
async def approve_draft(draft_id: UUID):
    """Approve a pending draft and send it via Gmail."""
    from ..storage.database import get_db_pool
    from ..tools.gmail import get_gmail_transport

    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(503, "Database not available")

    # Fetch and validate
    row = await pool.fetchrow(
        "SELECT * FROM email_drafts WHERE id = $1", draft_id,
    )
    if not row:
        raise HTTPException(404, "Draft not found")
    if row["status"] != "pending":
        raise HTTPException(409, f"Draft is already '{row['status']}', cannot approve")

    # Mark as approved
    now = datetime.now(timezone.utc)
    await pool.execute(
        "UPDATE email_drafts SET status = 'approved', approved_at = $1 WHERE id = $2",
        now, draft_id,
    )

    # Extract reply-to address from original_from
    original_from = row["original_from"]
    # Parse email from "Name <email@domain.com>" format
    if "<" in original_from and ">" in original_from:
        reply_to_addr = original_from.split("<")[1].split(">")[0].strip()
    else:
        reply_to_addr = original_from.strip()

    # Send via Gmail with threading
    # Use RFC 2822 Message-ID for In-Reply-To/References headers
    # (not the Gmail API hex ID which won't thread correctly cross-client)
    message_id = row.get("original_message_id") or row["gmail_message_id"]
    try:
        transport = get_gmail_transport()
        send_result = await transport.send(
            to=[reply_to_addr],
            subject=row["draft_subject"],
            body=row["draft_body"],
            thread_id=row.get("thread_id"),
            in_reply_to=message_id,
            references=message_id,
        )

        gmail_sent_id = send_result.get("id", "")
        sent_at = datetime.now(timezone.utc)

        await pool.execute(
            """
            UPDATE email_drafts
            SET status = 'sent', sent_at = $1, gmail_sent_id = $2
            WHERE id = $3
            """,
            sent_at, gmail_sent_id, draft_id,
        )

        logger.info("Draft %s approved and sent: gmail_id=%s", draft_id, gmail_sent_id)

        return {
            "draft_id": str(draft_id),
            "status": "sent",
            "gmail_sent_id": gmail_sent_id,
            "sent_to": reply_to_addr,
        }

    except Exception as e:
        # Revert to pending on send failure
        await pool.execute(
            "UPDATE email_drafts SET status = 'pending', approved_at = NULL WHERE id = $1",
            draft_id,
        )
        logger.error("Failed to send approved draft %s: %s", draft_id, e)
        raise HTTPException(502, f"Failed to send email: {e}")


@router.post("/{draft_id}/reject")
async def reject_draft(draft_id: UUID):
    """Reject a pending draft."""
    from ..storage.database import get_db_pool

    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(503, "Database not available")

    result = await pool.execute(
        """
        UPDATE email_drafts SET status = 'rejected'
        WHERE id = $1 AND status = 'pending'
        """,
        draft_id,
    )

    if _affected_rows(result) == 0:
        raise HTTPException(404, "Draft not found or not pending")

    logger.info("Draft %s rejected", draft_id)
    return {"draft_id": str(draft_id), "status": "rejected"}


@router.post("/{draft_id}/edit")
async def edit_draft(draft_id: UUID, body: EditDraftBody):
    """Edit a pending draft's body and/or subject."""
    from ..storage.database import get_db_pool

    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(503, "Database not available")

    row = await pool.fetchrow(
        "SELECT status FROM email_drafts WHERE id = $1", draft_id,
    )
    if not row:
        raise HTTPException(404, "Draft not found")
    if row["status"] != "pending":
        raise HTTPException(409, f"Draft is '{row['status']}', can only edit pending drafts")

    updates = []
    params = []
    idx = 1

    if body.draft_body is not None:
        updates.append(f"draft_body = ${idx}")
        params.append(body.draft_body)
        idx += 1

    if body.draft_subject is not None:
        updates.append(f"draft_subject = ${idx}")
        params.append(body.draft_subject)
        idx += 1

    if not updates:
        raise HTTPException(400, "No fields to update")

    params.append(draft_id)
    query = f"UPDATE email_drafts SET {', '.join(updates)} WHERE id = ${idx}"
    await pool.execute(query, *params)

    updated = await pool.fetchrow(
        "SELECT * FROM email_drafts WHERE id = $1", draft_id,
    )
    return _row_to_dict(updated)
