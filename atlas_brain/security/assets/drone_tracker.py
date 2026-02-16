"""Drone asset tracker."""

from datetime import datetime
from typing import Any, Optional

from .asset_tracker import AssetTracker


class DroneTracker(AssetTracker):
    """Track drones with optional telemetry fields."""

    def __init__(self, stale_after_seconds: int, max_assets: int):
        super().__init__(
            asset_type="drone",
            stale_after_seconds=stale_after_seconds,
            max_assets=max_assets,
        )

    def update_telemetry(
        self,
        identifier: str,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        battery_level: Optional[int] = None,
        signal_strength: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
        observed_at: Optional[datetime] = None,
    ) -> dict[str, Any]:
        """Update drone observation with telemetry values."""
        payload = dict(metadata or {})
        if latitude is not None:
            payload["latitude"] = latitude
        if longitude is not None:
            payload["longitude"] = longitude
        if battery_level is not None:
            payload["battery_level"] = battery_level
        if signal_strength is not None:
            payload["signal_strength"] = signal_strength
        return self.observe(identifier=identifier, metadata=payload, observed_at=observed_at)
