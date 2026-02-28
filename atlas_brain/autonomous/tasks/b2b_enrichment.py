"""
B2B review enrichment: extract churn signals from pending reviews via LLM
using the b2b_churn_extraction skill.

Single-pass enrichment (one LLM call per review). Polls b2b_reviews WHERE
enrichment_status = 'pending', calls LLM, stores result in enrichment JSONB
column, sets status to 'enriched'.

Runs on an interval (default 5 min). Returns _skip_synthesis so the
runner does not double-synthesize.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any

from ...config import settings
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.autonomous.tasks.b2b_enrichment")


async def run(task: ScheduledTask) -> dict[str, Any]:
    """Autonomous task handler: enrich pending B2B reviews with churn signals."""
    cfg = settings.b2b_churn
    if not cfg.enabled:
        return {"_skip_synthesis": "B2B churn pipeline disabled"}

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": "DB not ready"}

    max_batch = cfg.enrichment_max_per_batch
    max_attempts = cfg.enrichment_max_attempts

    rows = await pool.fetch(
        """
        SELECT id, vendor_name, product_name, product_category,
               source, raw_metadata,
               rating, rating_max, summary, review_text, pros, cons,
               reviewer_title, reviewer_company, company_size_raw,
               reviewer_industry, enrichment_attempts
        FROM b2b_reviews
        WHERE enrichment_status = 'pending'
          AND enrichment_attempts < $1
        ORDER BY imported_at ASC
        LIMIT $2
        """,
        max_attempts,
        max_batch,
    )

    if not rows:
        return {"_skip_synthesis": "No B2B reviews to enrich"}

    enriched = 0
    failed = 0

    for row in rows:
        ok = await _enrich_single(pool, row, max_attempts, cfg.enrichment_local_only,
                                  cfg.enrichment_max_tokens)
        if ok:
            enriched += 1
        elif (row["enrichment_attempts"] + 1) >= max_attempts:
            failed += 1

    logger.info(
        "B2B enrichment: %d enriched, %d failed (of %d)",
        enriched, failed, len(rows),
    )

    return {
        "_skip_synthesis": "B2B enrichment complete",
        "total": len(rows),
        "enriched": enriched,
        "failed": failed,
    }


async def _enrich_single(pool, row, max_attempts: int, local_only: bool,
                         max_tokens: int) -> bool:
    """Enrich a single B2B review with churn signals. Returns True on success."""
    review_id = row["id"]

    try:
        result = _classify_review(row, local_only, max_tokens)

        if result and _validate_enrichment(result):
            await pool.execute(
                """
                UPDATE b2b_reviews
                SET enrichment = $1,
                    enrichment_status = 'enriched',
                    enrichment_attempts = enrichment_attempts + 1,
                    enriched_at = $2
                WHERE id = $3
                """,
                json.dumps(result),
                datetime.now(timezone.utc),
                review_id,
            )
            return True
        else:
            await _increment_attempts(pool, review_id, row["enrichment_attempts"], max_attempts)
            return False

    except Exception:
        logger.exception("Failed to enrich B2B review %s", review_id)
        try:
            await pool.execute(
                "UPDATE b2b_reviews SET enrichment_attempts = enrichment_attempts + 1 WHERE id = $1",
                review_id,
            )
        except Exception:
            pass
        return False


def _smart_truncate(text: str, max_len: int = 3000) -> str:
    """Truncate preserving both beginning and end of review text.

    Churn signals often appear at the end ("I'm switching to X next quarter"),
    so naive head-only truncation loses them.
    """
    if len(text) <= max_len:
        return text
    half = max_len // 2 - 15
    return text[:half] + "\n[...truncated...]\n" + text[-half:]


def _classify_review(row, local_only: bool, max_tokens: int) -> dict[str, Any] | None:
    """Call LLM with b2b_churn_extraction skill."""
    from ...skills import get_skill_registry
    from ...services.protocols import Message
    from ...pipelines.llm import get_pipeline_llm, clean_llm_output

    skill = get_skill_registry().get("digest/b2b_churn_extraction")
    if not skill:
        logger.warning("Skill 'digest/b2b_churn_extraction' not found")
        return None

    llm = get_pipeline_llm(
        prefer_cloud=not local_only,
        try_openrouter=False,
        auto_activate_ollama=True,
    )
    if llm is None:
        logger.warning("No LLM available for B2B churn extraction")
        return None

    review_text = _smart_truncate(row["review_text"] or "")

    # Extract source context from raw_metadata
    raw_meta = row.get("raw_metadata") or {}
    if isinstance(raw_meta, str):
        try:
            raw_meta = json.loads(raw_meta)
        except (json.JSONDecodeError, TypeError):
            raw_meta = {}

    payload = {
        "vendor_name": row["vendor_name"],
        "product_name": row["product_name"] or "",
        "product_category": row["product_category"] or "",
        "source_name": row.get("source") or "",
        "source_weight": raw_meta.get("source_weight", 0.7),
        "source_type": raw_meta.get("source_type", "unknown"),
        "rating": float(row["rating"]) if row["rating"] is not None else None,
        "rating_max": int(row["rating_max"]),
        "summary": row["summary"] or "",
        "review_text": review_text,
        "pros": row["pros"] or "",
        "cons": row["cons"] or "",
        "reviewer_title": row["reviewer_title"] or "",
        "reviewer_company": row["reviewer_company"] or "",
        "company_size_raw": row["company_size_raw"] or "",
        "reviewer_industry": row["reviewer_industry"] or "",
    }

    messages = [
        Message(role="system", content=skill.content),
        Message(role="user", content=json.dumps(payload)),
    ]

    try:
        result = llm.chat(messages=messages, max_tokens=max_tokens, temperature=0.1)
        text = result.get("response", "").strip()
        if not text:
            return None
        text = clean_llm_output(text)
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            return None
        return parsed
    except json.JSONDecodeError:
        logger.debug("Failed to parse B2B enrichment JSON")
        return None
    except Exception:
        logger.exception("B2B enrichment LLM call failed")
        return None


_KNOWN_PAIN_CATEGORIES = {
    "pricing", "features", "reliability", "support", "integration",
    "performance", "security", "ux", "onboarding", "other",
}


def _coerce_bool(value: Any) -> bool | None:
    """Coerce a value to bool. Returns None if unrecognizable."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        if value.lower() in ("true", "yes", "1"):
            return True
        if value.lower() in ("false", "no", "0"):
            return False
    return None


_CHURN_SIGNAL_BOOL_FIELDS = (
    "intent_to_leave",
    "actively_evaluating",
    "migration_in_progress",
    "support_escalation",
    "contract_renewal_mentioned",
)


def _validate_enrichment(result: dict) -> bool:
    """Validate enrichment output structure and data consistency."""
    if "churn_signals" not in result:
        return False
    if "urgency_score" not in result:
        return False
    if not isinstance(result.get("churn_signals"), dict):
        return False

    # Type check: urgency_score must be numeric
    urgency = result.get("urgency_score")
    if isinstance(urgency, str):
        try:
            urgency = float(urgency)
            result["urgency_score"] = urgency
        except (ValueError, TypeError):
            logger.warning("urgency_score is non-numeric string: %r", urgency)
            return False

    if not isinstance(urgency, (int, float)):
        logger.warning("urgency_score has unexpected type: %s", type(urgency).__name__)
        return False

    # Range check: 0-10
    if urgency < 0 or urgency > 10:
        logger.warning("urgency_score out of range [0,10]: %s", urgency)
        return False

    # Boolean coercion: churn_signals fields used in ::boolean casts
    signals = result["churn_signals"]
    for field in _CHURN_SIGNAL_BOOL_FIELDS:
        if field in signals:
            coerced = _coerce_bool(signals[field])
            if coerced is None:
                logger.warning("churn_signals.%s unrecognizable bool: %r -- rejecting", field, signals[field])
                return False
            signals[field] = coerced

    # Consistency warning: high urgency with no intent_to_leave
    intent = signals.get("intent_to_leave")
    if urgency >= 9 and intent is False:
        logger.warning(
            "Contradictory: urgency=%s but intent_to_leave=false -- accepting with warning",
            urgency,
        )

    # Boolean coercion: reviewer_context.decision_maker (used in ::boolean cast)
    reviewer_ctx = result.get("reviewer_context")
    if isinstance(reviewer_ctx, dict) and "decision_maker" in reviewer_ctx:
        coerced = _coerce_bool(reviewer_ctx["decision_maker"])
        if coerced is None:
            logger.warning("reviewer_context.decision_maker unrecognizable bool: %r -- rejecting", reviewer_ctx["decision_maker"])
            return False
        reviewer_ctx["decision_maker"] = coerced

    # Type check: competitors_mentioned must be list; items must be dicts with "name"
    competitors = result.get("competitors_mentioned")
    if competitors is not None and not isinstance(competitors, list):
        logger.warning("competitors_mentioned is not a list: %s", type(competitors).__name__)
        result["competitors_mentioned"] = []
    elif isinstance(competitors, list):
        result["competitors_mentioned"] = [
            c for c in competitors
            if isinstance(c, dict) and "name" in c
        ]

    # Type check: quotable_phrases must be list if present
    qp = result.get("quotable_phrases")
    if qp is not None and not isinstance(qp, list):
        logger.warning("quotable_phrases is not a list: %s", type(qp).__name__)
        result["quotable_phrases"] = []

    # Type check: feature_gaps must be list if present
    fg = result.get("feature_gaps")
    if fg is not None and not isinstance(fg, list):
        logger.warning("feature_gaps is not a list: %s", type(fg).__name__)
        result["feature_gaps"] = []

    # Coerce unknown pain_category to "other"
    pain = result.get("pain_category")
    if pain and pain not in _KNOWN_PAIN_CATEGORIES:
        logger.warning("Unknown pain_category: %r -- coercing to 'other'", pain)
        result["pain_category"] = "other"

    return True


async def _increment_attempts(pool, review_id, current_attempts: int, max_attempts: int) -> None:
    """Bump attempts; mark failed if exhausted."""
    new_attempts = current_attempts + 1
    await pool.execute(
        "UPDATE b2b_reviews SET enrichment_attempts = $1 WHERE id = $2",
        new_attempts, review_id,
    )
    if new_attempts >= max_attempts:
        await pool.execute(
            "UPDATE b2b_reviews SET enrichment_status = 'failed' WHERE id = $1",
            review_id,
        )
