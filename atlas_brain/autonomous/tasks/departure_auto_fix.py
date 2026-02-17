"""
Departure auto-fix builtin task.

Runs the departure_check to find security issues (lights on, locks
unlocked, covers open), then calls Home Assistant services to fix them.

Triggered by the presence_departure hook when the house goes empty.
"""

import logging
from typing import Any

from ...config import settings
from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.autonomous.tasks.departure_auto_fix")

# Maps departure_check issue labels to HA service calls
_SERVICE_MAP: dict[str, str] = {
    "lights on": "light/turn_off",
    "locks unlocked": "lock/lock",
    "covers/garage open": "cover/close_cover",
    "switches on": "switch/turn_off",
}


async def run(task: ScheduledTask) -> dict[str, Any]:
    """
    Run departure_check, then fix each issue via HA service calls.

    Returns a summary dict with issues found and fixes attempted.
    """
    from .departure_check import run as departure_check_run

    check_result = await departure_check_run(task)
    issues = check_result.get("issues", [])

    if not issues:
        return {
            **check_result,
            "fixes_attempted": 0,
            "fixes_succeeded": 0,
        }

    # Connect to HA to apply fixes
    ha_url = settings.homeassistant.url
    ha_token = settings.homeassistant.token
    if not ha_token:
        return {
            **check_result,
            "fixes_attempted": 0,
            "fixes_succeeded": 0,
            "summary": "Issues found but HA token not configured -- cannot fix.",
        }

    from ...capabilities.backends.homeassistant import HomeAssistantBackend

    ha = HomeAssistantBackend(ha_url, ha_token)
    try:
        await ha.connect()
    except Exception as e:
        logger.error("Failed to connect to HA for auto-fix: %s", e)
        return {
            **check_result,
            "fixes_attempted": 0,
            "fixes_succeeded": 0,
            "summary": f"Issues found but failed to connect to HA: {e}",
        }

    fixes: list[dict[str, Any]] = []
    try:
        for issue in issues:
            service = _SERVICE_MAP.get(issue["issue"])
            if not service:
                logger.warning("No service mapped for issue: %s", issue["issue"])
                continue

            try:
                await ha.send_command(service, {"entity_id": issue["entity_id"]})
                fixes.append({**issue, "fixed": True})
            except Exception as e:
                logger.error(
                    "Failed to fix %s (%s): %s",
                    issue["entity_id"], service, e,
                )
                fixes.append({**issue, "fixed": False, "error": str(e)})
    finally:
        await ha.disconnect()

    succeeded = sum(1 for f in fixes if f["fixed"])
    summary = f"Auto-fixed {succeeded}/{len(fixes)} issues on departure."
    logger.info(summary)

    # Include original issues (synthesis skill expects this key) plus
    # fixes with per-entity success/failure status.
    result: dict[str, Any] = {
        "total_checked": check_result.get("total_checked", 0),
        "issues": issues,
        "issues_found": len(issues),
        "fixes_attempted": len(fixes),
        "fixes_succeeded": succeeded,
        "fixes": fixes,
        "summary": summary,
    }

    # Thread alert context through for synthesis if present
    alert_ctx = (task.metadata or {}).get("_alert_context")
    if alert_ctx:
        result["trigger_context"] = alert_ctx

    return result
