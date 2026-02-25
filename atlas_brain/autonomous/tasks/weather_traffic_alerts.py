"""
Proactive weather and traffic alert monitoring.

Polls NWS for severe weather alerts and TomTom for traffic incidents.
Sends ntfy push notifications (and TTS for urgent alerts).
"""

import logging
import math
from typing import Any

import httpx

from ...config import settings
from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.autonomous.tasks.weather_traffic_alerts")

# Module-level dedup state (persists across polls within a process lifetime)
_seen_weather_ids: set[str] = set()
_seen_traffic_ids: set[str] = set()
_last_commute_alert_delay: int = 0  # Avoid re-alerting same delay level

# NWS events that warrant TTS voice announcement
_URGENT_EVENTS = {
    "Tornado Warning",
    "Flash Flood Warning",
    "Severe Thunderstorm Warning",
}

# TomTom category filter: Accident, Fog, DangerousConditions, Ice, Jam,
# LaneClosed, RoadClosed, RoadWorks
_TOMTOM_CATEGORIES = "1,2,3,6,7,9,11,14"

_TOMTOM_TYPE_NAMES = {
    1: "Accident",
    2: "Fog",
    3: "Dangerous Conditions",
    6: "Ice on Road",
    7: "Congestion",
    9: "Lane Closed",
    11: "Road Closed",
    14: "Road Works",
}


async def run(task: ScheduledTask) -> dict:
    """Main entry point called by scheduler."""
    cfg = settings.alert_monitor
    if not cfg.enabled:
        return {"_skip_synthesis": "Alert monitor disabled"}

    home_lat = cfg.home_lat or settings.tools.weather_default_lat
    home_lon = cfg.home_lon or settings.tools.weather_default_lon

    if home_lat is None or home_lon is None:
        return {"_skip_synthesis": "No home coordinates configured"}

    results: dict[str, Any] = {}

    # 1. NWS weather alerts
    weather_alerts = await _check_nws_alerts(home_lat, home_lon, cfg)
    results["weather_alerts"] = len(weather_alerts)
    for alert in weather_alerts:
        await _notify_weather(alert, cfg)

    # 2. Area traffic incidents (radius around home)
    if settings.tools.traffic_enabled and settings.tools.traffic_api_key:
        incidents = await _check_traffic_incidents(home_lat, home_lon, cfg)
        results["traffic_incidents"] = len(incidents)
        for incident in incidents:
            await _notify_traffic_incident(incident)

        # 3. Commute route delay (if work location configured)
        if cfg.work_lat and cfg.work_lon:
            delay = await _check_commute_delay(
                home_lat, home_lon, cfg.work_lat, cfg.work_lon, cfg
            )
            results["commute_delay"] = delay

    # Prune expired alert IDs periodically (keep set bounded)
    _prune_seen_ids()

    total = results.get("weather_alerts", 0) + results.get("traffic_incidents", 0)
    commute = results.get("commute_delay")
    commute_notified = isinstance(commute, dict) and commute.get("notified", False)
    if total == 0 and not commute_notified:
        results["_skip_synthesis"] = "No new weather or traffic alerts."

    return results


# ---------------------------------------------------------------------------
# NWS Weather Alerts
# ---------------------------------------------------------------------------

async def _check_nws_alerts(lat: float, lon: float, cfg) -> list[dict]:
    """Fetch active NWS alerts and return only new ones matching severity filter."""
    allowed = {s.strip() for s in cfg.nws_severities.split(",")}
    url = f"https://api.weather.gov/alerts/active?point={lat},{lon}"

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                url,
                headers={
                    "User-Agent": "Atlas/1.0 (weather-alerts)",
                    "Accept": "application/geo+json",
                },
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.warning("NWS alerts fetch failed: %s", e)
        return []

    new_alerts = []
    for feature in data.get("features", []):
        props = feature.get("properties", {})
        alert_id = props.get("id", "")
        severity = props.get("severity", "")

        if severity not in allowed:
            continue
        if alert_id in _seen_weather_ids:
            continue

        _seen_weather_ids.add(alert_id)
        new_alerts.append({
            "id": alert_id,
            "event": props.get("event", "Unknown"),
            "severity": severity,
            "urgency": props.get("urgency", ""),
            "headline": props.get("headline", ""),
            "description": props.get("description", ""),
            "instruction": props.get("instruction", ""),
            "expires": props.get("expires", ""),
        })

    if new_alerts:
        logger.info("NWS: %d new alert(s) for %.2f,%.2f", len(new_alerts), lat, lon)

    return new_alerts


# ---------------------------------------------------------------------------
# TomTom Traffic Incidents
# ---------------------------------------------------------------------------

async def _check_traffic_incidents(lat: float, lon: float, cfg) -> list[dict]:
    """Fetch TomTom incidents within radius and return new ones above severity threshold."""
    api_key = settings.tools.traffic_api_key
    if not api_key:
        return []

    # Convert radius to bounding box
    lat_offset = cfg.traffic_radius_miles / 69.0
    lon_offset = cfg.traffic_radius_miles / (69.0 * math.cos(math.radians(lat)))
    bbox = f"{lon - lon_offset},{lat - lat_offset},{lon + lon_offset},{lat + lat_offset}"

    url = "https://api.tomtom.com/traffic/services/5/incidentDetails"
    params = {
        "bbox": bbox,
        "key": api_key,
        "fields": "{incidents{type,geometry{coordinates},properties{id,magnitudeOfDelay,events{description,code},from,to}}}",
        "language": "en-US",
        "categoryFilter": _TOMTOM_CATEGORIES,
    }

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.warning("TomTom incidents fetch failed: %s", e)
        return []

    new_incidents = []
    for incident in data.get("incidents", []):
        props = incident.get("properties", {})
        incident_id = props.get("id", "")
        magnitude = props.get("magnitudeOfDelay", 0)
        inc_type = incident.get("type", "")

        if magnitude < cfg.traffic_min_severity:
            continue
        if incident_id in _seen_traffic_ids:
            continue

        _seen_traffic_ids.add(incident_id)

        # Build description from events list
        events = props.get("events", [])
        description = "; ".join(e.get("description", "") for e in events if e.get("description"))
        from_road = props.get("from", "")
        to_road = props.get("to", "")
        location = f"from {from_road} to {to_road}" if from_road and to_road else from_road or to_road

        type_name = _TOMTOM_TYPE_NAMES.get(inc_type, str(inc_type)) if isinstance(inc_type, int) else str(inc_type)

        new_incidents.append({
            "id": incident_id,
            "type": type_name,
            "magnitude": magnitude,
            "description": description,
            "location": location,
        })

    if new_incidents:
        logger.info("TomTom: %d new incident(s) near %.2f,%.2f", len(new_incidents), lat, lon)

    return new_incidents


# ---------------------------------------------------------------------------
# Commute Delay
# ---------------------------------------------------------------------------

async def _check_commute_delay(
    home_lat: float, home_lon: float,
    work_lat: float, work_lon: float,
    cfg,
) -> dict | None:
    """Check commute route delay and notify if above threshold."""
    global _last_commute_alert_delay

    api_key = settings.tools.traffic_api_key
    if not api_key:
        return None

    try:
        from ...tools.traffic import traffic_tool
        route_data = await traffic_tool._fetch_route_traffic(
            home_lat, home_lon, work_lat, work_lon, api_key
        )
    except Exception as e:
        logger.warning("Commute delay check failed: %s", e)
        return None

    delay_min = route_data.get("traffic_delay_minutes", 0)
    travel_min = route_data.get("travel_time_minutes", 0)
    distance = route_data.get("distance_miles", 0)

    if delay_min >= cfg.commute_delay_threshold_minutes:
        # Only alert if delay level changed meaningfully (5-min buckets)
        bucket = (delay_min // 5) * 5
        if bucket != _last_commute_alert_delay:
            _last_commute_alert_delay = bucket
            await _notify_commute_delay(delay_min, travel_min, distance)
            return {"delay_minutes": delay_min, "notified": True}
    else:
        # Delay dropped below threshold -- reset
        _last_commute_alert_delay = 0

    return {"delay_minutes": delay_min, "notified": False}


# ---------------------------------------------------------------------------
# Notification helpers
# ---------------------------------------------------------------------------

async def _push_ntfy(
    title: str,
    body: str,
    priority: str = "default",
    tags: str = "",
) -> None:
    """Send a push notification via ntfy."""
    if not settings.alerts.ntfy_enabled:
        return
    try:
        url = f"{settings.alerts.ntfy_url.rstrip('/')}/{settings.alerts.ntfy_topic}"
        headers: dict[str, str] = {
            "Title": title,
            "Priority": priority,
        }
        if tags:
            headers["Tags"] = tags
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(url, content=body.encode(), headers=headers)
        logger.info("ntfy push: %s", title)
    except Exception as e:
        logger.warning("ntfy push failed: %s", e)


async def _notify_weather(alert: dict, cfg) -> None:
    """Send ntfy (and optionally TTS) for a weather alert."""
    event = alert["event"]
    severity = alert["severity"]

    if severity == "Extreme":
        priority = "urgent"
        tags = "warning,rotating_light"
    else:
        priority = "high"
        tags = "warning,cloud"

    title = f"Weather Alert: {event}"
    body = alert.get("headline", "")
    instruction = alert.get("instruction", "")
    if instruction:
        body = f"{body}\n\n{instruction}" if body else instruction

    await _push_ntfy(title, body, priority=priority, tags=tags)

    # TTS for urgent alerts (best-effort via voice pipeline if available)
    if cfg.tts_on_urgent and event in _URGENT_EVENTS:
        await _tts_announce(f"Attention. {event} issued for your area.")


async def _notify_traffic_incident(incident: dict) -> None:
    """Send ntfy push for a traffic incident."""
    inc_type = incident["type"]
    magnitude = incident["magnitude"]

    # Higher priority for road closures and major accidents
    is_major = inc_type in ("Road Closed", "Accident") or magnitude >= 3
    priority = "high" if is_major else "default"
    tags = "car,warning"

    title = f"Traffic Alert: {inc_type}"
    body = incident.get("description", "")
    location = incident.get("location", "")
    if location:
        body = f"{body}\n{location}" if body else location

    await _push_ntfy(title, body, priority=priority, tags=tags)


async def _notify_commute_delay(delay_min: int, travel_min: int, distance: float) -> None:
    """Send ntfy push for commute delay."""
    normal_min = travel_min - delay_min
    title = f"Commute Alert: {delay_min} min delay"
    body = f"Your commute is currently {travel_min} min (normally ~{normal_min} min). Distance: {distance} miles."

    await _push_ntfy(title, body, priority="high", tags="car,clock")


async def _tts_announce(text: str) -> None:
    """Best-effort TTS announcement via the voice pipeline's playback controller.

    Only works when the voice pipeline is running (microphone active).
    Silently skips if the pipeline is not available.
    """
    try:
        import asyncio
        from ...voice.launcher import get_voice_pipeline

        pipeline = get_voice_pipeline()
        if pipeline is None:
            logger.debug("Voice pipeline not running, skipping TTS announcement")
            return

        # PlaybackController.speak() is synchronous (blocks in a thread);
        # run it in the default executor to avoid blocking the async loop.
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, pipeline.playback.speak, text)
        logger.info("TTS announced: %s", text[:60])
    except Exception as e:
        logger.debug("TTS weather announcement unavailable: %s", e)


# ---------------------------------------------------------------------------
# Dedup pruning
# ---------------------------------------------------------------------------

_MAX_SEEN_IDS = 500


def _prune_seen_ids() -> None:
    """Keep seen-ID sets bounded. If they exceed the limit, clear and re-learn on next poll."""
    global _seen_weather_ids, _seen_traffic_ids
    if len(_seen_weather_ids) > _MAX_SEEN_IDS:
        _seen_weather_ids = set()
        logger.debug("Pruned _seen_weather_ids")
    if len(_seen_traffic_ids) > _MAX_SEEN_IDS:
        _seen_traffic_ids = set()
        logger.debug("Pruned _seen_traffic_ids")
