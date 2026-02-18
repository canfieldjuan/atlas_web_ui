"""
Temporal pattern context for LLM prompts.

Queries learned daily routines from temporal_patterns and returns a compact
natural-language summary suitable for injection into the system prompt.
Results are cached in-memory and auto-invalidate at midnight (patterns only
change via the nightly pattern_learning task).
"""

import logging
import time as _time
from datetime import date

from ..config import settings
from ..utils.time import format_minutes

logger = logging.getLogger("atlas.orchestration.temporal")

# Cache: (date, day_of_week, min_samples) -> formatted string
_cache: dict[tuple[date, int, int], str] = {}

# Negative-result cache: suppress DB retries after a failure
_failure_until: float = 0.0

_DAY_ABBR = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


async def get_temporal_context(*, min_samples: int | None = None) -> str:
    """Return a compact natural-language summary of today's learned routines.

    Args:
        min_samples: Minimum sample_count required per pattern row.
                     Defaults to settings.temporal.min_samples.

    Fail-open: returns empty string on any error or if no patterns exist.
    """
    global _failure_until

    cfg = settings.temporal

    if not cfg.enabled:
        return ""

    if min_samples is None:
        min_samples = cfg.min_samples

    try:
        # Short-circuit if we recently failed (avoid DB hammering)
        if _time.monotonic() < _failure_until:
            return ""

        today = date.today()
        dow = today.weekday()  # 0=Mon, 6=Sun

        cache_key = (today, dow, min_samples)
        if cache_key in _cache:
            return _cache[cache_key]

        # Evict stale entries from previous days
        for k in list(_cache):
            if k[0] != today:
                del _cache[k]

        from ..storage.database import get_db_pool

        pool = get_db_pool()
        if not pool.is_initialized:
            return ""

        rows = await pool.fetch(
            """
            SELECT person_name, pattern_type, median_minutes
            FROM temporal_patterns
            WHERE day_of_week = $1 AND sample_count >= $2
            ORDER BY person_name, pattern_type
            """,
            dow,
            min_samples,
        )

        if not rows:
            _cache[cache_key] = ""
            return ""

        # Group by person
        persons: dict[str, list[str]] = {}
        for row in rows:
            name = row["person_name"]
            ptype = row["pattern_type"]
            time_str = format_minutes(row["median_minutes"], round_to=5)
            persons.setdefault(name, []).append(f"{ptype} ~{time_str}")

        day_label = _DAY_ABBR[dow]
        lines = []
        for name, patterns in persons.items():
            lines.append(f"{name}'s routine ({day_label}): {', '.join(patterns)}")

        result = "\n".join(lines)
        _cache[cache_key] = result
        return result

    except Exception as e:
        _failure_until = _time.monotonic() + cfg.failure_cooldown
        logger.debug("Could not build temporal context: %s", e)
        return ""
