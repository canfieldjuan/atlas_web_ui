"""Pattern matchers for proactive reflection.

Each pattern detector queries recent data and returns findings
that the reflection node can act on.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("atlas.reasoning.patterns")


async def detect_stale_threads() -> list[dict[str, Any]]:
    """Find threads where Atlas sent a reply but received no response in 3+ days."""
    from ..storage.database import get_db_pool

    pool = get_db_pool()
    if not pool.is_initialized:
        return []

    rows = await pool.fetch(
        """
        SELECT ed.id AS draft_id, ed.original_from, ed.draft_subject,
               ed.sent_at, pe.intent, pe.contact_id
        FROM email_drafts ed
        LEFT JOIN processed_emails pe
            ON pe.gmail_message_id = ed.gmail_message_id
        WHERE ed.status = 'sent'
          AND ed.sent_at < NOW() - INTERVAL '3 days'
          AND NOT EXISTS (
              SELECT 1 FROM processed_emails pe2
              WHERE pe2.followup_of_draft_id = ed.id
          )
        ORDER BY ed.sent_at ASC
        LIMIT 20
        """
    )
    findings = []
    for r in rows:
        sent_at = r.get("sent_at")
        if not sent_at:
            continue
        findings.append({
            "pattern": "stale_thread",
            "description": (
                f"Reply to {r['original_from']} re: {r['draft_subject']} "
                f"sent {sent_at.strftime('%b %d')} with no response"
            ),
            "entity_type": "contact",
            "entity_id": r.get("contact_id"),
            "draft_id": str(r["draft_id"]),
            "original_intent": r.get("intent"),
            "days_since_reply": (datetime.now(timezone.utc) - sent_at).days,
        })
    return findings


async def detect_scheduling_gaps() -> list[dict[str, Any]]:
    """Find estimates sent but no booking made within 5+ days."""
    from ..storage.database import get_db_pool

    pool = get_db_pool()
    if not pool.is_initialized:
        return []

    rows = await pool.fetch(
        """
        SELECT ed.id AS draft_id, ed.original_from, ed.draft_subject,
               ed.sent_at, pe.contact_id
        FROM email_drafts ed
        LEFT JOIN processed_emails pe
            ON pe.gmail_message_id = ed.gmail_message_id
        WHERE ed.status = 'sent'
          AND pe.intent = 'estimate_request'
          AND ed.sent_at < NOW() - INTERVAL '5 days'
          AND pe.contact_id IS NOT NULL
          AND NOT EXISTS (
              SELECT 1 FROM appointments a
              WHERE a.contact_id = pe.contact_id
                AND a.created_at > ed.sent_at
          )
        ORDER BY ed.sent_at ASC
        LIMIT 20
        """
    )
    findings = []
    for r in rows:
        sent_at = r.get("sent_at")
        if not sent_at:
            continue
        findings.append({
            "pattern": "scheduling_gap",
            "description": (
                f"Estimate for {r['original_from']} sent {sent_at.strftime('%b %d')} "
                f"but no appointment booked"
            ),
            "entity_type": "contact",
            "entity_id": r.get("contact_id"),
            "draft_id": str(r["draft_id"]),
        })
    return findings


async def detect_missing_followups() -> list[dict[str, Any]]:
    """Find completed appointments with no invoice sent."""
    from ..storage.database import get_db_pool

    pool = get_db_pool()
    if not pool.is_initialized:
        return []

    try:
        rows = await pool.fetch(
            """
            SELECT a.id AS appointment_id, a.contact_id, a.service_type,
                   a.scheduled_at, c.full_name
            FROM appointments a
            LEFT JOIN contacts c ON c.id::text = a.contact_id
            WHERE a.status = 'completed'
              AND a.scheduled_at < NOW() - INTERVAL '1 day'
              AND NOT EXISTS (
                  SELECT 1 FROM invoices i
                  WHERE i.contact_id = a.contact_id
                    AND i.created_at > a.scheduled_at
              )
            ORDER BY a.scheduled_at DESC
            LIMIT 10
            """
        )
    except Exception:
        # invoices table may not exist
        return []

    findings = []
    for r in rows:
        scheduled_at = r.get("scheduled_at")
        if not scheduled_at:
            continue
        findings.append({
            "pattern": "missing_followup",
            "description": (
                f"Appointment with {r.get('full_name', 'unknown')} on "
                f"{scheduled_at.strftime('%b %d')} completed but no invoice"
            ),
            "entity_type": "contact",
            "entity_id": r.get("contact_id"),
            "appointment_id": str(r["appointment_id"]),
        })
    return findings


async def detect_news_market_correlation() -> list[dict[str, Any]]:
    """Find news events that correlate with significant price moves."""
    from ..config import settings
    from ..storage.database import get_db_pool

    pool = get_db_pool()
    if not pool.is_initialized:
        return []

    cfg = settings.external_data
    news_lookback = cfg.correlation_news_lookback_hours
    market_window = cfg.correlation_market_window_hours

    try:
        rows = await pool.fetch(
            """
            SELECT na.id AS news_article_id,
                   na.title AS news_title,
                   na.matched_keywords,
                   na.created_at AS news_at,
                   ms.symbol, ms.price, ms.change_pct, ms.snapshot_at,
                   dw.name AS asset_name
            FROM news_articles na
            JOIN data_watchlist dw ON dw.enabled = true
                AND dw.category IN ('stock','etf','commodity','crypto','forex')
            JOIN market_snapshots ms ON ms.symbol = dw.symbol
                AND ms.snapshot_at > na.created_at
                AND ms.snapshot_at < na.created_at + make_interval(hours => $1)
                AND ABS(ms.change_pct) >= COALESCE(dw.threshold_pct, 5.0)
            WHERE na.created_at > NOW() - make_interval(hours => $2)
            ORDER BY ABS(ms.change_pct) DESC
            LIMIT 10
            """,
            market_window,
            news_lookback,
        )
    except Exception:
        logger.debug("News-market correlation query failed", exc_info=True)
        return []

    seen = set()
    findings = []
    for r in rows:
        key = f"{r['news_article_id']}:{r['symbol']}"
        if key in seen:
            continue
        seen.add(key)
        findings.append({
            "pattern": "news_market_correlation",
            "description": (
                f"News: \"{r['news_title'][:100]}\" followed by "
                f"{r['symbol']} ({r['asset_name']}) moving {r['change_pct']:+.1f}% "
                f"within {(r['snapshot_at'] - r['news_at']).total_seconds() / 3600:.0f}h"
            ),
            "entity_type": None,
            "entity_id": None,
            "news_article_id": str(r["news_article_id"]),
            "symbol": r["symbol"],
            "change_pct": float(r["change_pct"]),
        })
    return findings


async def run_all_pattern_detectors() -> list[dict[str, Any]]:
    """Run all pattern detectors and aggregate findings."""
    import asyncio

    results = await asyncio.gather(
        detect_stale_threads(),
        detect_scheduling_gaps(),
        detect_missing_followups(),
        detect_news_market_correlation(),
        return_exceptions=True,
    )

    findings = []
    for result in results:
        if isinstance(result, list):
            findings.extend(result)
        elif isinstance(result, Exception):
            logger.warning("Pattern detector failed: %s", result)

    return findings
