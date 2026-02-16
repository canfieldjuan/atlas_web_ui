"""Vehicle asset tracker."""

from datetime import datetime
from typing import Any, Optional

from .asset_tracker import AssetTracker


class VehicleTracker(AssetTracker):
    """Track vehicles with optional position and speed metadata."""

    def __init__(self, stale_after_seconds: int, max_assets: int):
        super().__init__(
            asset_type="vehicle",
            stale_after_seconds=stale_after_seconds,
            max_assets=max_assets,
        )

    def update_state(
        self,
        identifier: str,
        speed_mps: Optional[float] = None,
        heading_deg: Optional[float] = None,
        metadata: Optional[dict[str, Any]] = None,
        observed_at: Optional[datetime] = None,
    ) -> dict[str, Any]:
        """Update vehicle state from observed telemetry."""
        payload = dict(metadata or {})
        if speed_mps is not None:
            payload["speed_mps"] = speed_mps
        if heading_deg is not None:
            payload["heading_deg"] = heading_deg
        return self.observe(identifier=identifier, metadata=payload, observed_at=observed_at)
