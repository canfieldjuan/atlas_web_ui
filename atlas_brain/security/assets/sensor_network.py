"""Sensor network tracker."""

from datetime import datetime
from typing import Any, Optional

from .asset_tracker import AssetTracker


class SensorNetworkTracker(AssetTracker):
    """Track sensor assets and their latest readings."""

    def __init__(self, stale_after_seconds: int, max_assets: int):
        super().__init__(
            asset_type="sensor",
            stale_after_seconds=stale_after_seconds,
            max_assets=max_assets,
        )

    def update_reading(
        self,
        identifier: str,
        reading: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
        observed_at: Optional[datetime] = None,
    ) -> dict[str, Any]:
        """Update sensor state with latest reading payload."""
        payload = dict(metadata or {})
        if reading:
            payload["reading"] = reading
        return self.observe(identifier=identifier, metadata=payload, observed_at=observed_at)
