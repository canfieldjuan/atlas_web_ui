"""
Complaint enrichment pipeline: classify pending product reviews via LLM
using the complaint_classification skill.

Single phase (review text already stored -- no HTTP fetch needed).
Polls product_reviews WHERE enrichment_status = 'pending', calls LLM,
updates enrichment columns, sets status to 'enriched'.

Supports two modes:
  - Single (reviews_per_call=1): one LLM call per review (default, most reliable)
  - Batch (reviews_per_call=2-10): N reviews per call (higher throughput, falls
    back to single on parse failure)

LLM routing:
  - local_only=False (default): tries triage LLM (Claude) first, falls back to local
  - local_only=True: skips triage LLM, uses only the active local model

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

logger = logging.getLogger("atlas.autonomous.tasks.complaint_enrichment")


async def run(task: ScheduledTask) -> dict[str, Any]:
    """Autonomous task handler: enrich pending product reviews."""
    cfg = settings.external_data
    if not cfg.complaint_mining_enabled:
        return {"_skip_synthesis": "Complaint mining disabled"}

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": "DB not ready"}

    max_batch = cfg.complaint_enrichment_max_per_batch
    max_attempts = cfg.complaint_enrichment_max_attempts
    reviews_per_call = min(cfg.complaint_enrichment_reviews_per_call, 10)

    rows = await pool.fetch(
        """
        SELECT id, asin, rating, summary, review_text,
               hardware_category, issue_types,
               enrichment_attempts
        FROM product_reviews
        WHERE enrichment_status = 'pending'
          AND enrichment_attempts < $1
        ORDER BY imported_at ASC
        LIMIT $2
        """,
        max_attempts,
        max_batch,
    )

    if not rows:
        return {"_skip_synthesis": "No reviews to enrich"}

    enriched = 0
    failed = 0

    if reviews_per_call > 1:
        # Batch mode: process N reviews per LLM call
        for i in range(0, len(rows), reviews_per_call):
            chunk = rows[i:i + reviews_per_call]
            batch_results = await _classify_batch(chunk, cfg.complaint_enrichment_local_only)

            if batch_results and len(batch_results) == len(chunk):
                # Batch succeeded -- apply all results
                for row, classification in zip(chunk, batch_results):
                    if classification and classification.get("root_cause"):
                        await _apply_enrichment(pool, row, classification)
                        enriched += 1
                    else:
                        await _increment_attempts(pool, row, max_attempts)
                        if (row["enrichment_attempts"] + 1) >= max_attempts:
                            failed += 1
            else:
                # Batch parse failed -- fall back to individual calls
                logger.debug(
                    "Batch parse failed for %d reviews, falling back to single mode",
                    len(chunk),
                )
                for row in chunk:
                    ok = await _enrich_single(pool, row, max_attempts, cfg.complaint_enrichment_local_only)
                    if ok:
                        enriched += 1
                    elif (row["enrichment_attempts"] + 1) >= max_attempts:
                        failed += 1
    else:
        # Single mode: one LLM call per review
        for row in rows:
            ok = await _enrich_single(pool, row, max_attempts, cfg.complaint_enrichment_local_only)
            if ok:
                enriched += 1
            elif (row["enrichment_attempts"] + 1) >= max_attempts:
                failed += 1

    logger.info(
        "Complaint enrichment: %d enriched, %d failed (of %d) [batch=%d]",
        enriched, failed, len(rows), reviews_per_call,
    )

    return {
        "_skip_synthesis": "Complaint enrichment complete",
        "total": len(rows),
        "enriched": enriched,
        "failed": failed,
        "batch_size": reviews_per_call,
    }


# ------------------------------------------------------------------
# Single-review enrichment
# ------------------------------------------------------------------


async def _enrich_single(pool, row, max_attempts: int, local_only: bool) -> bool:
    """Classify and store a single review. Returns True on success."""
    review_id = row["id"]
    attempts = row["enrichment_attempts"]

    try:
        classification = await _classify_review(
            asin=row["asin"],
            rating=float(row["rating"]),
            summary=row["summary"] or "",
            review_text=row["review_text"] or "",
            hardware_category=list(row["hardware_category"] or []),
            issue_types=list(row["issue_types"] or []),
            local_only=local_only,
        )

        if classification:
            await _apply_enrichment(pool, row, classification)
            return True
        else:
            await _increment_attempts(pool, row, max_attempts)
            return False

    except Exception:
        logger.exception("Failed to enrich review %s", review_id)
        try:
            await pool.execute(
                "UPDATE product_reviews SET enrichment_attempts = enrichment_attempts + 1 WHERE id = $1",
                review_id,
            )
        except Exception:
            pass
        return False


async def _apply_enrichment(pool, row, classification: dict[str, Any]) -> None:
    """Write classification results to the review row."""
    await pool.execute(
        """
        UPDATE product_reviews
        SET root_cause = $1,
            specific_complaint = $2,
            severity = $3,
            pain_score = $4,
            time_to_failure = $5,
            workaround_found = $6,
            workaround_text = $7,
            alternative_mentioned = $8,
            alternative_asin = $9,
            alternative_name = $10,
            actionable_for_manufacturing = $11,
            manufacturing_suggestion = $12,
            enrichment_status = 'enriched',
            enrichment_attempts = $13,
            enriched_at = $14
        WHERE id = $15
        """,
        classification.get("root_cause"),
        classification.get("specific_complaint"),
        classification.get("severity"),
        classification.get("pain_score"),
        classification.get("time_to_failure"),
        classification.get("workaround_found"),
        classification.get("workaround_text"),
        classification.get("alternative_mentioned"),
        classification.get("alternative_asin"),
        classification.get("alternative_name"),
        classification.get("actionable_for_manufacturing"),
        classification.get("manufacturing_suggestion"),
        row["enrichment_attempts"] + 1,
        datetime.now(timezone.utc),
        row["id"],
    )


async def _increment_attempts(pool, row, max_attempts: int) -> None:
    """Bump attempts; mark failed if exhausted."""
    review_id = row["id"]
    new_attempts = row["enrichment_attempts"] + 1
    await pool.execute(
        "UPDATE product_reviews SET enrichment_attempts = $1 WHERE id = $2",
        new_attempts, review_id,
    )
    if new_attempts >= max_attempts:
        await pool.execute(
            "UPDATE product_reviews SET enrichment_status = 'failed' WHERE id = $1",
            review_id,
        )


# ------------------------------------------------------------------
# LLM classification (single)
# ------------------------------------------------------------------


def _get_llm(local_only: bool):
    """Resolve the LLM to use for classification."""
    from ...pipelines.llm import get_pipeline_llm

    return get_pipeline_llm(
        prefer_cloud=not local_only,
        try_openrouter=False,  # enrichment uses simpler prompts, local is fine
        auto_activate_ollama=True,
    )


async def _classify_review(
    asin: str,
    rating: float,
    summary: str,
    review_text: str,
    hardware_category: list[str],
    issue_types: list[str],
    local_only: bool = False,
) -> dict[str, Any] | None:
    """Classify a single product review via LLM."""
    from ...skills import get_skill_registry
    from ...services.protocols import Message

    skill = get_skill_registry().get("digest/complaint_classification")
    if not skill:
        logger.warning("Skill 'digest/complaint_classification' not found")
        return None

    llm = _get_llm(local_only)
    if llm is None:
        logger.warning("No LLM available for complaint classification")
        return None

    truncated = review_text[:2000] if review_text else ""

    user_payload = json.dumps({
        "asin": asin,
        "rating": rating,
        "summary": summary,
        "review_text": truncated,
        "hardware_category": hardware_category,
        "issue_types": issue_types,
    })

    messages = [
        Message(role="system", content=skill.content),
        Message(role="user", content=user_payload),
    ]

    return _call_and_parse(llm, messages, max_tokens=256)


# ------------------------------------------------------------------
# LLM classification (batch)
# ------------------------------------------------------------------


_BATCH_SYSTEM_SUFFIX = """

## BATCH MODE

You will receive a JSON array of reviews. Classify EACH review independently.
Return a JSON array of classification objects in the SAME ORDER as the input.
Each object must follow the single-review output format above.
Return ONLY the JSON array -- no prose, no markdown fencing."""


async def _classify_batch(rows, local_only: bool) -> list[dict[str, Any]] | None:
    """Classify multiple reviews in a single LLM call. Returns list or None on failure."""
    from ...skills import get_skill_registry
    from ...services.protocols import Message

    skill = get_skill_registry().get("digest/complaint_classification")
    if not skill:
        return None

    llm = _get_llm(local_only)
    if llm is None:
        return None

    reviews = []
    for row in rows:
        reviews.append({
            "asin": row["asin"],
            "rating": float(row["rating"]),
            "summary": row["summary"] or "",
            "review_text": (row["review_text"] or "")[:2000],
            "hardware_category": list(row["hardware_category"] or []),
            "issue_types": list(row["issue_types"] or []),
        })

    messages = [
        Message(role="system", content=skill.content + _BATCH_SYSTEM_SUFFIX),
        Message(role="user", content=json.dumps(reviews)),
    ]

    # More tokens for batch (256 per review + overhead)
    max_tokens = 300 * len(rows)
    result = _call_and_parse_array(llm, messages, max_tokens=max_tokens)
    if result and len(result) == len(rows):
        return result
    return None


# ------------------------------------------------------------------
# LLM call helpers
# ------------------------------------------------------------------


def _call_and_parse(llm, messages, max_tokens: int) -> dict[str, Any] | None:
    """Call LLM and parse single JSON object response."""
    try:
        result = llm.chat(messages=messages, max_tokens=max_tokens, temperature=0.1)
        text = result.get("response", "").strip()
        if not text:
            return None
        text = _clean_llm_output(text)
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            return None
        if "root_cause" not in parsed or "specific_complaint" not in parsed:
            return None
        return parsed
    except json.JSONDecodeError:
        logger.debug("Failed to parse classification JSON: %.200s", text)
        return None
    except Exception:
        logger.exception("Classification LLM call failed")
        return None


def _call_and_parse_array(llm, messages, max_tokens: int) -> list[dict[str, Any]] | None:
    """Call LLM and parse JSON array response."""
    try:
        result = llm.chat(messages=messages, max_tokens=max_tokens, temperature=0.1)
        text = result.get("response", "").strip()
        if not text:
            return None
        text = _clean_llm_output(text)
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            return None
        # Validate each item has required fields
        for item in parsed:
            if not isinstance(item, dict) or "root_cause" not in item:
                return None
        return parsed
    except json.JSONDecodeError:
        logger.debug("Failed to parse batch classification JSON: %.200s", text)
        return None
    except Exception:
        logger.exception("Batch classification LLM call failed")
        return None


def _clean_llm_output(text: str) -> str:
    """Strip think tags and markdown fencing from LLM output."""
    from ...pipelines.llm import clean_llm_output

    return clean_llm_output(text)
