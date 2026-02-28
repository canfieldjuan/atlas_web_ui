"""
B2B churn intelligence: aggregate enriched review data, feed to LLM
for synthesis, persist intelligence products, and notify.

Runs weekly (default Sunday 9 PM). Produces 4 report types:
  - weekly_churn_feed: ranked companies showing churn intent
  - vendor_scorecard: per-vendor health metrics
  - displacement_report: competitive flow map
  - category_overview: cross-vendor trends

Handles its own LLM call, report persistence, churn_signals upserts,
and ntfy notification -- returns _skip_synthesis so the runner does
not double-synthesize.
"""

import asyncio
import json
import logging
from datetime import date, datetime, timezone
from typing import Any

from ...config import settings
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.autonomous.tasks.b2b_churn_intelligence")


def _safe_json(value: Any, default: Any = None) -> Any:
    """Safely deserialize a JSON value, returning *default* on failure."""
    if default is None:
        default = []
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return default
    return default


async def run(task: ScheduledTask) -> dict[str, Any]:
    """Autonomous task handler: weekly B2B churn intelligence."""
    cfg = settings.b2b_churn
    if not cfg.enabled or not cfg.intelligence_enabled:
        return {"_skip_synthesis": "B2B churn intelligence disabled"}

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": "DB not ready"}

    window_days = cfg.intelligence_window_days
    min_reviews = cfg.intelligence_min_reviews
    urgency_threshold = cfg.high_churn_urgency_threshold
    today = date.today()

    # Gather all 10 data sources in parallel
    (
        vendor_scores, high_intent, competitive_disp,
        pain_dist, feature_gaps,
        negative_counts, price_rates, dm_rates,
        churning_companies, quotable_evidence,
    ) = await asyncio.gather(
        _fetch_vendor_churn_scores(pool, window_days, min_reviews),
        _fetch_high_intent_companies(pool, urgency_threshold, window_days),
        _fetch_competitive_displacement(pool, window_days),
        _fetch_pain_distribution(pool, window_days),
        _fetch_feature_gaps(pool, window_days),
        _fetch_negative_review_counts(pool, window_days),
        _fetch_price_complaint_rates(pool, window_days),
        _fetch_dm_churn_rates(pool, window_days),
        _fetch_churning_companies(pool, window_days),
        _fetch_quotable_evidence(pool, window_days),
        return_exceptions=True,
    )

    # Convert exceptions to empty values
    def _safe(val: Any, name: str) -> list:
        if isinstance(val, Exception):
            logger.warning("%s fetch failed: %s", name, val)
            return []
        return val

    vendor_scores = _safe(vendor_scores, "vendor_scores")
    high_intent = _safe(high_intent, "high_intent")
    competitive_disp = _safe(competitive_disp, "competitive_disp")
    pain_dist = _safe(pain_dist, "pain_dist")
    feature_gaps = _safe(feature_gaps, "feature_gaps")
    negative_counts = _safe(negative_counts, "negative_counts")
    price_rates = _safe(price_rates, "price_rates")
    dm_rates = _safe(dm_rates, "dm_rates")
    churning_companies = _safe(churning_companies, "churning_companies")
    quotable_evidence = _safe(quotable_evidence, "quotable_evidence")

    # Check if there's enough data
    if not vendor_scores and not high_intent:
        return {"_skip_synthesis": "No enriched B2B reviews to analyze"}

    # Fetch prior reports for trend comparison
    prior_reports = await _fetch_prior_reports(pool)

    # Build payload
    payload = {
        "date": str(today),
        "analysis_window_days": window_days,
        "vendor_churn_scores": vendor_scores,
        "high_intent_companies": high_intent,
        "competitive_displacement": competitive_disp,
        "pain_distribution": pain_dist,
        "feature_gaps": feature_gaps,
        "negative_review_counts": negative_counts,
        "price_complaint_rates": price_rates,
        "decision_maker_churn_rates": dm_rates,
        "prior_reports": prior_reports,
    }

    # Load skill and call LLM
    from ...pipelines.llm import call_llm_with_skill, parse_json_response

    analysis = call_llm_with_skill(
        "digest/b2b_churn_intelligence", payload,
        max_tokens=cfg.intelligence_max_tokens, temperature=0.4,
    )
    if not analysis:
        return {"_skip_synthesis": "LLM analysis failed"}

    parsed = parse_json_response(analysis, recover_truncated=True)

    # Persist intelligence reports
    report_types = [
        ("weekly_churn_feed", parsed.get("weekly_churn_feed", [])),
        ("vendor_scorecard", parsed.get("vendor_scorecards", [])),
        ("displacement_report", parsed.get("displacement_map", [])),
        ("category_overview", parsed.get("category_insights", [])),
    ]

    for report_type, data in report_types:
        try:
            await pool.execute(
                """
                INSERT INTO b2b_intelligence (
                    report_date, report_type, intelligence_data,
                    executive_summary, data_density, status, llm_model
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                today,
                report_type,
                json.dumps(data, default=str),
                parsed.get("executive_summary", ""),
                json.dumps({
                    "vendors_analyzed": len(vendor_scores),
                    "high_intent_companies": len(high_intent),
                    "competitive_flows": len(competitive_disp),
                    "pain_categories": len(pain_dist),
                    "feature_gaps": len(feature_gaps),
                }),
                "published",
                "pipeline_default",
            )
        except Exception:
            logger.exception("Failed to store %s report", report_type)

    # Build lookups for upsert
    pain_lookup = _build_pain_lookup(pain_dist)
    competitor_lookup = _build_competitor_lookup(competitive_disp)
    feature_gap_lookup = _build_feature_gap_lookup(feature_gaps)
    neg_lookup = {r["vendor"]: r["negative_count"] for r in negative_counts}
    price_lookup = {r["vendor"]: r["price_complaint_rate"] for r in price_rates}
    dm_lookup = {r["vendor"]: r["dm_churn_rate"] for r in dm_rates}
    company_lookup = {r["vendor"]: r["companies"] for r in churning_companies}
    quote_lookup = {r["vendor"]: r["quotes"] for r in quotable_evidence}

    # Upsert per-vendor churn signals
    await _upsert_churn_signals(
        pool, vendor_scores,
        neg_lookup, pain_lookup, competitor_lookup, feature_gap_lookup,
        price_lookup, dm_lookup, company_lookup, quote_lookup,
    )

    # Send ntfy notification
    await _send_notification(task, parsed, high_intent)

    return {
        "_skip_synthesis": "B2B churn intelligence complete",
        "date": str(today),
        "vendors_analyzed": len(vendor_scores),
        "high_intent_companies": len(high_intent),
        "competitive_flows": len(competitive_disp),
        "report_types": len(report_types),
    }


# ------------------------------------------------------------------
# Data fetchers
# ------------------------------------------------------------------


async def _fetch_vendor_churn_scores(pool, window_days: int, min_reviews: int) -> list[dict[str, Any]]:
    """Per-vendor health metrics from enriched reviews."""
    rows = await pool.fetch(
        """
        SELECT vendor_name, product_category,
            count(*) AS total_reviews,
            count(*) FILTER (
                WHERE (enrichment->'churn_signals'->>'intent_to_leave')::boolean = true
            ) AS churn_intent,
            -- Source-weighted urgency: weighted avg preserving 0-10 scale
            -- Falls back to 0.7 for pre-existing reviews without source_weight
            avg(
                (enrichment->>'urgency_score')::numeric
                * COALESCE((raw_metadata->>'source_weight')::numeric, 0.7)
            ) / NULLIF(avg(COALESCE((raw_metadata->>'source_weight')::numeric, 0.7)), 0)
            AS avg_urgency,
            avg(rating / NULLIF(rating_max, 0)) AS avg_rating_normalized,
            count(*) FILTER (
                WHERE (enrichment->>'would_recommend')::boolean = true
            ) AS recommend_yes,
            count(*) FILTER (
                WHERE (enrichment->>'would_recommend')::boolean = false
            ) AS recommend_no
        FROM b2b_reviews
        WHERE enrichment_status = 'enriched'
          AND enriched_at > NOW() - make_interval(days => $1)
        GROUP BY vendor_name, product_category
        HAVING count(*) >= $2
        ORDER BY avg((enrichment->>'urgency_score')::numeric) DESC
        """,
        window_days,
        min_reviews,
    )
    return [
        {
            "vendor_name": r["vendor_name"],
            "product_category": r["product_category"],
            "total_reviews": r["total_reviews"],
            "churn_intent": r["churn_intent"],
            "avg_urgency": float(r["avg_urgency"]) if r["avg_urgency"] else 0,
            "avg_rating_normalized": float(r["avg_rating_normalized"]) if r["avg_rating_normalized"] else None,
            "recommend_yes": r["recommend_yes"],
            "recommend_no": r["recommend_no"],
        }
        for r in rows
    ]


async def _fetch_high_intent_companies(pool, urgency_threshold: int, window_days: int) -> list[dict[str, Any]]:
    """Companies showing high churn intent -- the money feed."""
    rows = await pool.fetch(
        """
        SELECT reviewer_company, vendor_name, product_category,
            enrichment->'reviewer_context'->>'role_level' AS role_level,
            (enrichment->'reviewer_context'->>'decision_maker')::boolean AS is_dm,
            (enrichment->>'urgency_score')::numeric AS urgency,
            enrichment->>'pain_category' AS pain,
            enrichment->'competitors_mentioned' AS alternatives,
            enrichment->'quotable_phrases' AS quotes,
            enrichment->'contract_context'->>'contract_value_signal' AS value_signal
        FROM b2b_reviews
        WHERE enrichment_status = 'enriched'
          AND (enrichment->>'urgency_score')::numeric >= $1
          AND reviewer_company IS NOT NULL AND reviewer_company != ''
          AND enriched_at > NOW() - make_interval(days => $2)
        ORDER BY (enrichment->>'urgency_score')::numeric DESC
        """,
        urgency_threshold,
        window_days,
    )
    results = []
    for r in rows:
        try:
            urgency = float(r["urgency"]) if r["urgency"] is not None else 0
        except (ValueError, TypeError):
            urgency = 0
        results.append({
            "company": r["reviewer_company"],
            "vendor": r["vendor_name"],
            "category": r["product_category"],
            "role_level": r["role_level"],
            "decision_maker": r["is_dm"],
            "urgency": urgency,
            "pain": r["pain"],
            "alternatives": _safe_json(r["alternatives"]),
            "quotes": _safe_json(r["quotes"]),
            "contract_signal": r["value_signal"],
        })
    return results


async def _fetch_competitive_displacement(pool, window_days: int) -> list[dict[str, Any]]:
    """Who's winning from whom -- competitive flows."""
    rows = await pool.fetch(
        """
        SELECT vendor_name,
            comp.value->>'name' AS competitor,
            comp.value->>'context' AS direction,
            count(*) AS mention_count
        FROM b2b_reviews
        CROSS JOIN LATERAL jsonb_array_elements(enrichment->'competitors_mentioned') AS comp(value)
        WHERE enrichment_status = 'enriched'
          AND enriched_at > NOW() - make_interval(days => $1)
        GROUP BY vendor_name, comp.value->>'name', comp.value->>'context'
        ORDER BY mention_count DESC
        """,
        window_days,
    )
    return [
        {
            "vendor": r["vendor_name"],
            "competitor": r["competitor"],
            "direction": r["direction"],
            "mention_count": r["mention_count"],
        }
        for r in rows
    ]


async def _fetch_pain_distribution(pool, window_days: int) -> list[dict[str, Any]]:
    """What's driving churn per vendor."""
    rows = await pool.fetch(
        """
        SELECT vendor_name,
            enrichment->>'pain_category' AS pain,
            count(*) AS complaint_count,
            avg((enrichment->>'urgency_score')::numeric) AS avg_urgency
        FROM b2b_reviews
        WHERE enrichment_status = 'enriched'
          AND enriched_at > NOW() - make_interval(days => $1)
        GROUP BY vendor_name, enrichment->>'pain_category'
        ORDER BY complaint_count DESC
        """,
        window_days,
    )
    return [
        {
            "vendor": r["vendor_name"],
            "pain": r["pain"],
            "complaint_count": r["complaint_count"],
            "avg_urgency": float(r["avg_urgency"]) if r["avg_urgency"] else 0,
        }
        for r in rows
    ]


async def _fetch_feature_gaps(pool, window_days: int) -> list[dict[str, Any]]:
    """Most-mentioned missing features per vendor."""
    rows = await pool.fetch(
        """
        SELECT vendor_name,
            gap.value #>> '{}' AS feature_gap,
            count(*) AS mentions
        FROM b2b_reviews
        CROSS JOIN LATERAL jsonb_array_elements(enrichment->'feature_gaps') AS gap(value)
        WHERE enrichment_status = 'enriched'
          AND enriched_at > NOW() - make_interval(days => $1)
        GROUP BY vendor_name, gap.value #>> '{}'
        HAVING count(*) >= 2
        ORDER BY mentions DESC
        """,
        window_days,
    )
    return [
        {
            "vendor": r["vendor_name"],
            "feature_gap": r["feature_gap"],
            "mentions": r["mentions"],
        }
        for r in rows
    ]


async def _fetch_negative_review_counts(pool, window_days: int) -> list[dict[str, Any]]:
    """Count reviews with below-50% ratings per vendor."""
    rows = await pool.fetch(
        """
        SELECT vendor_name, count(*) AS negative_count
        FROM b2b_reviews
        WHERE enrichment_status = 'enriched'
          AND enriched_at > NOW() - make_interval(days => $1)
          AND rating IS NOT NULL AND rating_max > 0
          AND (rating / rating_max) < 0.5
        GROUP BY vendor_name
        """,
        window_days,
    )
    return [{"vendor": r["vendor_name"], "negative_count": r["negative_count"]} for r in rows]


async def _fetch_price_complaint_rates(pool, window_days: int) -> list[dict[str, Any]]:
    """Fraction of reviews with pain_category='pricing' per vendor."""
    rows = await pool.fetch(
        """
        SELECT vendor_name,
            count(*) FILTER (WHERE enrichment->>'pain_category' = 'pricing') AS pricing_count,
            count(*) AS total
        FROM b2b_reviews
        WHERE enrichment_status = 'enriched'
          AND enriched_at > NOW() - make_interval(days => $1)
        GROUP BY vendor_name
        HAVING count(*) > 0
        """,
        window_days,
    )
    return [
        {
            "vendor": r["vendor_name"],
            "price_complaint_rate": r["pricing_count"] / r["total"] if r["total"] else 0,
        }
        for r in rows
    ]


async def _fetch_dm_churn_rates(pool, window_days: int) -> list[dict[str, Any]]:
    """Decision-maker churn rate: DMs with intent_to_leave / total DMs, per vendor."""
    rows = await pool.fetch(
        """
        SELECT vendor_name,
            count(*) FILTER (
                WHERE (enrichment->'reviewer_context'->>'decision_maker')::boolean = true
                  AND (enrichment->'churn_signals'->>'intent_to_leave')::boolean = true
            ) AS dm_churning,
            count(*) FILTER (
                WHERE (enrichment->'reviewer_context'->>'decision_maker')::boolean = true
            ) AS dm_total
        FROM b2b_reviews
        WHERE enrichment_status = 'enriched'
          AND enriched_at > NOW() - make_interval(days => $1)
        GROUP BY vendor_name
        HAVING count(*) FILTER (
            WHERE (enrichment->'reviewer_context'->>'decision_maker')::boolean = true
        ) > 0
        """,
        window_days,
    )
    return [
        {
            "vendor": r["vendor_name"],
            "dm_churn_rate": r["dm_churning"] / r["dm_total"] if r["dm_total"] else 0,
        }
        for r in rows
    ]


async def _fetch_churning_companies(pool, window_days: int) -> list[dict[str, Any]]:
    """Companies with high churn intent, aggregated per vendor."""
    rows = await pool.fetch(
        """
        SELECT vendor_name,
            jsonb_agg(jsonb_build_object(
                'company', reviewer_company,
                'urgency', (enrichment->>'urgency_score')::numeric,
                'role', enrichment->'reviewer_context'->>'role_level',
                'pain', enrichment->>'pain_category'
            ) ORDER BY (enrichment->>'urgency_score')::numeric DESC)
            AS companies
        FROM b2b_reviews
        WHERE enrichment_status = 'enriched'
          AND enriched_at > NOW() - make_interval(days => $1)
          AND (enrichment->'churn_signals'->>'intent_to_leave')::boolean = true
          AND reviewer_company IS NOT NULL AND reviewer_company != ''
        GROUP BY vendor_name
        """,
        window_days,
    )
    results = []
    for r in rows:
        companies = _safe_json(r["companies"])
        results.append({"vendor": r["vendor_name"], "companies": companies})
    return results


async def _fetch_quotable_evidence(pool, window_days: int) -> list[dict[str, Any]]:
    """High-urgency quotable phrases per vendor."""
    rows = await pool.fetch(
        """
        SELECT vendor_name,
            jsonb_agg(phrase.value ORDER BY (enrichment->>'urgency_score')::numeric DESC)
            AS quotes
        FROM b2b_reviews
        CROSS JOIN LATERAL jsonb_array_elements_text(
            COALESCE(enrichment->'quotable_phrases', '[]'::jsonb)
        ) AS phrase(value)
        WHERE enrichment_status = 'enriched'
          AND enriched_at > NOW() - make_interval(days => $1)
          AND (enrichment->>'urgency_score')::numeric >= 6
        GROUP BY vendor_name
        """,
        window_days,
    )
    results = []
    for r in rows:
        quotes = _safe_json(r["quotes"])
        results.append({"vendor": r["vendor_name"], "quotes": quotes})
    return results


async def _fetch_prior_reports(pool) -> list[dict[str, Any]]:
    """Fetch most recent prior intelligence reports for trend comparison.

    Includes both weekly_churn_feed and vendor_scorecard, with full
    intelligence_data so the LLM can compute trends from actual numbers
    instead of guessing from prose.
    """
    rows = await pool.fetch(
        """
        SELECT report_type, intelligence_data, executive_summary, report_date
        FROM b2b_intelligence
        WHERE report_type IN ('weekly_churn_feed', 'vendor_scorecard')
        ORDER BY report_date DESC
        LIMIT 4
        """,
    )
    results = []
    for r in rows:
        intel_data = r["intelligence_data"]
        # asyncpg auto-deserializes JSONB to dict/list, but handle string fallback
        if isinstance(intel_data, str):
            try:
                intel_data = json.loads(intel_data)
            except (json.JSONDecodeError, TypeError):
                intel_data = {}
        results.append({
            "report_type": r["report_type"],
            "report_date": str(r["report_date"]),
            "executive_summary": r["executive_summary"],
            "intelligence_data": intel_data,
        })
    return results


# ------------------------------------------------------------------
# Lookup builders (pure Python, no DB)
# ------------------------------------------------------------------


def _build_pain_lookup(pain_dist: list[dict]) -> dict[str, list[dict]]:
    """vendor -> sorted list of {category, count, avg_urgency}."""
    lookup: dict[str, list[dict]] = {}
    for row in pain_dist:
        vendor = row.get("vendor", "")
        lookup.setdefault(vendor, []).append({
            "category": row.get("pain", "other"),
            "count": row.get("complaint_count", 0),
            "avg_urgency": round(row.get("avg_urgency", 0), 1),
        })
    for v in lookup:
        lookup[v].sort(key=lambda x: x["count"], reverse=True)
    return lookup


def _build_competitor_lookup(competitive_disp: list[dict]) -> dict[str, list[dict]]:
    """vendor -> sorted list of {name, direction, mentions}."""
    lookup: dict[str, list[dict]] = {}
    for row in competitive_disp:
        vendor = row.get("vendor", "")
        lookup.setdefault(vendor, []).append({
            "name": row.get("competitor", ""),
            "direction": row.get("direction", ""),
            "mentions": row.get("mention_count", 0),
        })
    for v in lookup:
        lookup[v].sort(key=lambda x: x["mentions"], reverse=True)
    return lookup


def _build_feature_gap_lookup(feature_gaps: list[dict]) -> dict[str, list[dict]]:
    """vendor -> sorted list of {feature, mentions}."""
    lookup: dict[str, list[dict]] = {}
    for row in feature_gaps:
        vendor = row.get("vendor", "")
        lookup.setdefault(vendor, []).append({
            "feature": row.get("feature_gap", ""),
            "mentions": row.get("mentions", 0),
        })
    for v in lookup:
        lookup[v].sort(key=lambda x: x["mentions"], reverse=True)
    return lookup


# ------------------------------------------------------------------
# Persistence helpers
# ------------------------------------------------------------------


async def _upsert_churn_signals(
    pool,
    vendor_scores: list[dict],
    neg_lookup: dict[str, int],
    pain_lookup: dict[str, list[dict]],
    competitor_lookup: dict[str, list[dict]],
    feature_gap_lookup: dict[str, list[dict]],
    price_lookup: dict[str, float],
    dm_lookup: dict[str, float],
    company_lookup: dict[str, list[dict]],
    quote_lookup: dict[str, list[str]],
) -> None:
    """Upsert b2b_churn_signals with all 15 columns from SQL-computed data."""
    now = datetime.now(timezone.utc)

    for vs in vendor_scores:
        vendor = vs["vendor_name"]
        category = vs.get("product_category")

        total = vs["total_reviews"]
        recommend_yes = vs.get("recommend_yes", 0)
        recommend_no = vs.get("recommend_no", 0)
        nps = ((recommend_yes - recommend_no) / total * 100) if total > 0 else None

        try:
            await pool.execute(
                """
                INSERT INTO b2b_churn_signals (
                    vendor_name, product_category,
                    total_reviews, negative_reviews, churn_intent_count,
                    avg_urgency_score, avg_rating_normalized, nps_proxy,
                    top_pain_categories, top_competitors, top_feature_gaps,
                    price_complaint_rate, decision_maker_churn_rate,
                    company_churn_list, quotable_evidence,
                    last_computed_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11,
                          $12, $13, $14, $15, $16)
                ON CONFLICT (vendor_name, COALESCE(product_category, '')) DO UPDATE SET
                    total_reviews = EXCLUDED.total_reviews,
                    negative_reviews = EXCLUDED.negative_reviews,
                    churn_intent_count = EXCLUDED.churn_intent_count,
                    avg_urgency_score = EXCLUDED.avg_urgency_score,
                    avg_rating_normalized = EXCLUDED.avg_rating_normalized,
                    nps_proxy = EXCLUDED.nps_proxy,
                    top_pain_categories = EXCLUDED.top_pain_categories,
                    top_competitors = EXCLUDED.top_competitors,
                    top_feature_gaps = EXCLUDED.top_feature_gaps,
                    price_complaint_rate = EXCLUDED.price_complaint_rate,
                    decision_maker_churn_rate = EXCLUDED.decision_maker_churn_rate,
                    company_churn_list = EXCLUDED.company_churn_list,
                    quotable_evidence = EXCLUDED.quotable_evidence,
                    last_computed_at = EXCLUDED.last_computed_at
                """,
                vendor,
                category,
                total,
                neg_lookup.get(vendor, 0),
                vs.get("churn_intent", 0),
                vs.get("avg_urgency", 0),
                vs.get("avg_rating_normalized"),
                nps,
                json.dumps(pain_lookup.get(vendor, [])[:5]),
                json.dumps(competitor_lookup.get(vendor, [])[:5]),
                json.dumps(feature_gap_lookup.get(vendor, [])[:5]),
                price_lookup.get(vendor),
                dm_lookup.get(vendor),
                json.dumps(company_lookup.get(vendor, [])[:20]),
                json.dumps(quote_lookup.get(vendor, [])[:10]),
                now,
            )
        except Exception:
            logger.exception("Failed to upsert churn signal for %s", vendor)


# ------------------------------------------------------------------
# Notification
# ------------------------------------------------------------------


async def _send_notification(task: ScheduledTask, parsed: dict, high_intent: list) -> None:
    """Send ntfy push notification with executive summary."""
    from ...pipelines.notify import send_pipeline_notification

    # Build a custom notification body for churn intelligence
    parts: list[str] = []

    summary = parsed.get("executive_summary", "")
    if summary:
        parts.append(summary.strip())

    # Top high-intent companies
    feed = parsed.get("weekly_churn_feed", [])
    if feed and isinstance(feed, list):
        items = []
        for entry in feed[:5]:
            if isinstance(entry, dict):
                company = entry.get("company", "Unknown")
                vendor = entry.get("vendor", "")
                urgency = entry.get("urgency", "?")
                pain = entry.get("pain", "")
                role = entry.get("reviewer_role", "")
                quote = entry.get("key_quote", "")
                line = f"- **{company}** ({role}) -- {vendor}, urgency {urgency}/10"
                if pain:
                    line += f"\n  Pain: {pain}"
                if quote:
                    line += f'\n  "{quote}"'
                items.append(line)
        if items:
            parts.append("\n**High-Intent Companies**\n" + "\n".join(items))

    message = "\n\n".join(parts) if parts else "Weekly churn intelligence report generated."

    high_count = len(high_intent)
    title = f"Atlas: Weekly Churn Feed ({high_count} high-intent compan{'y' if high_count == 1 else 'ies'})"

    await send_pipeline_notification(
        message, task,
        title=title,
        default_tags="brain,chart_with_downwards_trend",
    )
