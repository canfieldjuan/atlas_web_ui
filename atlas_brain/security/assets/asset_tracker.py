"""Base tracker for security assets."""

import copy
from datetime import datetime, timezone
from typing import Any, Optional


class AssetTracker:
    """Track asset presence and health based on observations."""

    def __init__(self, asset_type: str, stale_after_seconds: int, max_assets: int):
        self._asset_type = asset_type
        self._stale_after_seconds = stale_after_seconds
        self._max_assets = max_assets
        self._assets: dict[str, dict[str, Any]] = {}

    def observe(
        self,
        identifier: str,
        metadata: Optional[dict[str, Any]] = None,
        observed_at: Optional[datetime] = None,
    ) -> dict[str, Any]:
        """Record a new observation and return current asset state."""
        if observed_at is None:
            observed_at = datetime.now(timezone.utc)

        existing = self._assets.get(identifier)
        if existing:
            existing["last_seen"] = observed_at
            existing["status"] = "active"
            if metadata:
                existing["metadata"].update(metadata)
            existing["observations"] += 1
            return copy.deepcopy(existing)

        asset = {
            "asset_type": self._asset_type,
            "identifier": identifier,
            "status": "active",
            "first_seen": observed_at,
            "last_seen": observed_at,
            "metadata": dict(metadata or {}),
            "observations": 1,
        }
        self._assets[identifier] = asset
        self._prune_assets_if_needed()
        return copy.deepcopy(asset)

    def list_assets(self, status: Optional[str] = None) -> list[dict[str, Any]]:
        """Return tracked assets, optionally filtered by status."""
        self.refresh_health()
        records = list(self._assets.values())
        if status:
            records = [record for record in records if record["status"] == status]
        records.sort(key=lambda record: record["last_seen"], reverse=True)
        return [self._to_output(record) for record in records]

    def get_asset(self, identifier: str) -> Optional[dict[str, Any]]:
        """Return a single asset by identifier."""
        self.refresh_health()
        record = self._assets.get(identifier)
        if not record:
            return None
        return self._to_output(record)

    def refresh_health(self, now: Optional[datetime] = None) -> None:
        """Mark assets stale based on last observed time."""
        if now is None:
            now = datetime.now(timezone.utc)
        for record in self._assets.values():
            seconds_since_seen = (now - record["last_seen"]).total_seconds()
            if seconds_since_seen > self._stale_after_seconds:
                record["status"] = "stale"
            else:
                record["status"] = "active"

    def get_summary(self) -> dict[str, int]:
        """Return aggregate counts for active and stale assets."""
        self.refresh_health()
        active = sum(1 for record in self._assets.values() if record["status"] == "active")
        stale = sum(1 for record in self._assets.values() if record["status"] == "stale")
        return {"total": len(self._assets), "active": active, "stale": stale}

    def _prune_assets_if_needed(self) -> None:
        """Prune oldest assets to stay under configured max."""
        if self._max_assets <= 0:
            return
        if len(self._assets) <= self._max_assets:
            return
        ordered = sorted(self._assets.values(), key=lambda record: record["last_seen"])
        to_remove = len(self._assets) - self._max_assets
        for record in ordered[:to_remove]:
            self._assets.pop(record["identifier"], None)

    def _to_output(self, record: dict[str, Any]) -> dict[str, Any]:
        """Serialize datetime fields for API output."""
        return {
            "asset_type": record["asset_type"],
            "identifier": record["identifier"],
            "status": record["status"],
            "first_seen": record["first_seen"].isoformat(),
            "last_seen": record["last_seen"].isoformat(),
            "metadata": record["metadata"],
            "observations": record["observations"],
        }
