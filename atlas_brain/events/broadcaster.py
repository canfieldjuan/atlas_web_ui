"""
System event broadcaster.

Provides a lightweight publish channel for system events to the Atlas UI.
Call register_broadcast_fn() once at startup (from main.py) to wire up
the WebSocket broadcast channel.  Any module can then call
broadcast_system_event() without importing from the API layer.

Categories: ha, alert, reminder, task, llm, comms
Levels: info, warning, error
"""

import logging
import uuid
from datetime import datetime, timezone

logger = logging.getLogger("atlas.events.broadcaster")

# Set by main.py after the orchestrated WS module is loaded.
# Points to api.orchestrated.websocket._broadcast (an async callable).
_broadcast_fn = None


def register_broadcast_fn(fn) -> None:
    """
    Register the async broadcast function for system events.

    Should be called once at startup with the orchestrated WebSocket
    _broadcast coroutine.

    Args:
        fn: async callable with signature (state: str, **kwargs) -> None
    """
    global _broadcast_fn
    _broadcast_fn = fn
    logger.info("System event broadcaster registered")


async def broadcast_system_event(category: str, level: str, message: str) -> None:
    """
    Broadcast a system event to all connected UI clients via WebSocket.

    Silently no-ops if no broadcast function has been registered yet
    (e.g. during early startup before any WS clients connect).

    Args:
        category: Event source category. One of: ha, alert, reminder,
                  task, llm, comms.
        level:    Severity level. One of: info, warning, error.
        message:  Human-readable event description.
    """
    if _broadcast_fn is None:
        logger.debug(
            "Broadcaster not registered, dropping event: [%s/%s] %s",
            category, level, message,
        )
        return

    event_id = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()

    try:
        await _broadcast_fn(
            "system_event",
            id=event_id,
            ts=ts,
            category=category,
            level=level,
            message=message,
        )
    except Exception as exc:
        logger.debug("Failed to broadcast system event: %s", exc)
