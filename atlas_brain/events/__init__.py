"""System event broadcasting for real-time UI feed."""

from .broadcaster import broadcast_system_event, register_broadcast_fn

__all__ = ["broadcast_system_event", "register_broadcast_fn"]
