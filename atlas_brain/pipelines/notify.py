"""
Shared notification helper for pipeline tasks.

Replaces identical _send_notification() functions across
complaint_analysis, daily_intelligence, and complaint_content_generation.
"""

from __future__ import annotations

import logging
from typing import Any

from ..storage.models import ScheduledTask

logger = logging.getLogger("atlas.pipelines.notify")


async def send_pipeline_notification(
    message: str,
    task: ScheduledTask,
    *,
    title: str | None = None,
    default_tags: str = "brain",
    max_chars: int = 4000,
) -> None:
    """Send an ntfy push notification for a pipeline task result.

    Checks autonomous config (notify_results), alerts config (ntfy_enabled),
    and per-task opt-out (metadata.notify). Truncates message to ``max_chars``
    (ntfy has a ~4KB limit).
    """
    from ..config import settings

    if not settings.autonomous.notify_results:
        return
    if not settings.alerts.ntfy_enabled:
        return
    if (task.metadata or {}).get("notify") is False:
        return

    if title is None:
        title = f"Atlas: {task.name.replace('_', ' ').title()}"

    priority = (task.metadata or {}).get("notify_priority", "default")
    tags = (task.metadata or {}).get("notify_tags", default_tags)

    try:
        from ..tools.notify import notify_tool

        await notify_tool._send_notification(
            message=message[:max_chars],
            title=title,
            priority=priority,
            tags=tags,
        )
        logger.info("Sent notification for task '%s'", task.name)
    except Exception:
        logger.warning("Failed to send notification for task '%s'", task.name, exc_info=True)
