"""
Rogue access point detector.

Detects evil twin and suspicious APs.
"""

import logging
import time
from typing import Any, Dict, Optional, Set

from ...config import settings

logger = logging.getLogger("atlas.security.wireless.rogue_ap")


class RogueAPDetector:
    """
    Detects rogue and evil twin access points.
    
    Monitors beacon frames for suspicious AP patterns.
    """
    
    def __init__(self):
        self._known_ssids: Set[str] = set(settings.security.known_ssids)
        self._known_bssids: Set[str] = set(
            mac.lower() for mac in settings.security.known_ap_bssids
        )
        self._seen_aps: Dict[str, Dict[str, Any]] = {}
        
    def process_beacon(
        self, bssid: str, ssid: str, channel: int, signal_strength: int
    ) -> Optional[Dict[str, Any]]:
        """
        Process beacon frame and check for rogue AP patterns.
        
        Returns alert details if rogue AP detected, None otherwise.
        """
        bssid_lower = bssid.lower()
        
        if not ssid or ssid.strip() == "":
            return None
            
        ap_key = f"{bssid_lower}:{ssid}"
        
        if ap_key not in self._seen_aps:
            # Memory leak protection
            if len(self._seen_aps) > 1000:
                self._seen_aps.clear()
                
            self._seen_aps[ap_key] = {
                "bssid": bssid_lower,
                "ssid": ssid,
                "channel": channel,
                "signal": signal_strength,
                "first_seen": time.time()
            }
            
        self._seen_aps[ap_key]["signal"] = max(
            self._seen_aps[ap_key]["signal"], signal_strength
        )
        
        if ssid in self._known_ssids and bssid_lower not in self._known_bssids:
            logger.warning(
                "Evil twin detected: SSID '%s' with unknown BSSID %s on channel %d",
                ssid, bssid, channel
            )
            return {
                "type": "evil_twin",
                "ssid": ssid,
                "bssid": bssid,
                "channel": channel,
                "signal": signal_strength
            }
            
        return None
        
    def get_seen_aps(self) -> Dict[str, Dict[str, Any]]:
        """Get all seen APs."""
        return self._seen_aps.copy()