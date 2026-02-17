"""
Calendar reminder builtin task.

Checks upcoming calendar events every 5 minutes and sends ntfy
notifications for events starting in 15-30 minutes. Only fires
when the house is occupied.
"""

import logging
from datetime import datetime, timedelta
from typing import Any

from ...config import settings
from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.autonomous.tasks.calendar_reminder")

# In-memory dedup: tracks notified event keys to avoid duplicate alerts.
# Keyed by "{event_id}:{date}" so recurring events get one alert per day.
_notified_events: set[str] = set()
_MAX_DEDUP_ENTRIES = 500


async def run(task: ScheduledTask) -> dict[str, Any]:
    """
    Check upcoming calendar events and notify when someone is home.

    Configurable via task.metadata:
        lead_minutes (int): Max minutes before event to notify (default: 30)
        min_minutes (int): Min minutes before event to notify (default: 15)
    """
    global _notified_events

    # 1. Presence gate -- skip if house is empty
    from ...autonomous.presence import get_presence_tracker, OccupancyState

    presence = get_presence_tracker()
    if presence.state.state == OccupancyState.EMPTY:
        return {"skipped": True, "reason": "house_empty"}

    # 2. Calendar config gate
    if not settings.tools.calendar_enabled or not settings.tools.calendar_refresh_token:
        return {"skipped": True, "reason": "calendar_not_configured"}

    metadata = task.metadata or {}
    lead_minutes = metadata.get("lead_minutes", 30)
    min_minutes = metadata.get("min_minutes", 15)

    # 3. Fetch upcoming events (1-hour window)
    from ...tools.calendar import calendar_tool

    try:
        events = await calendar_tool._fetch_events(hours_ahead=1, max_results=10)
    except Exception as e:
        logger.warning("Calendar fetch failed: %s", e)
        return {"error": str(e)}

    now = datetime.now().astimezone()
    window_start = now + timedelta(minutes=min_minutes)
    window_end = now + timedelta(minutes=lead_minutes)

    # 4. Filter to events in the notification window
    candidates = [
        e for e in events
        if not e.all_day and window_start <= e.start <= window_end
    ]

    # 5. Dedup -- reset set if it grows too large
    if len(_notified_events) > _MAX_DEDUP_ENTRIES:
        _notified_events = set()

    notified = []
    for event in candidates:
        dedup_key = f"{event.id}:{event.start.date()}"
        if dedup_key in _notified_events:
            continue
        _notified_events.add(dedup_key)

        minutes_until = int((event.start - now).total_seconds() / 60)
        message = f"{event.summary} in {minutes_until} minutes"
        if event.location:
            message += f" at {event.location}"
        message += "."

        # 6. Send notification
        if settings.alerts.ntfy_enabled:
            try:
                from ...tools.notify import notify_tool
                await notify_tool._send_notification(
                    message=message,
                    title="Upcoming Event",
                    priority="default",
                )
            except Exception as e:
                logger.warning("Failed to send calendar reminder: %s", e)

        notified.append({
            "summary": event.summary,
            "start": event.start.isoformat(),
            "minutes_until": minutes_until,
        })

    return {
        "checked": len(events),
        "in_window": len(candidates),
        "notified": len(notified),
        "events": notified,
    }
