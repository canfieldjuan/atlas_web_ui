"""
Deauthentication attack detector.

Monitors WiFi deauth frames and alerts on suspicious patterns.
"""

import logging
import time
from collections import defaultdict
from typing import Dict, List, Tuple

from ...config import settings

logger = logging.getLogger("atlas.security.wireless.deauth")

DEAUTH_WINDOW_SECONDS = 10

class DeauthDetector:
    """
    Detects WiFi deauthentication attacks.
    
    Tracks deauth frame counts and generates alerts when thresholds exceeded.
    """
    
    def __init__(self):
        self._threshold = settings.security.deauth_threshold
        self._window_seconds = DEAUTH_WINDOW_SECONDS
        self._deauth_counts: Dict[str, List[Tuple[float, str]]] = defaultdict(list)
        
    def process_deauth_frame(self, src_mac: str, dst_mac: str) -> bool:
        """
        Process deauth frame and check if attack threshold exceeded.
        
        Returns True if alert should be generated.
        """
        current_time = time.time()
        
        # Memory leak protection
        if len(self._deauth_counts) > 1000:
             # Remove empty entries to free up space
             stale_keys = [
                 k for k, v in self._deauth_counts.items() 
                 if not v or v[-1][0] < current_time - self._window_seconds
             ]
             for k in stale_keys:
                 del self._deauth_counts[k]
             # If still too big, brute force clear
             if len(self._deauth_counts) > 1000:
                 self._deauth_counts.clear()

        cutoff_time = current_time - self._window_seconds
        
        self._deauth_counts[src_mac] = [
            (ts, target) for ts, target in self._deauth_counts[src_mac]
            if ts > cutoff_time
        ]
        
        self._deauth_counts[src_mac].append((current_time, dst_mac))
        
        frame_count = len(self._deauth_counts[src_mac])
        if frame_count >= self._threshold:
            targets = set(target for _, target in self._deauth_counts[src_mac])
            logger.warning(
                "Deauth attack detected: %s sent %d frames to %d targets in %ds",
                src_mac, frame_count, len(targets), self._window_seconds
            )
            self._deauth_counts[src_mac].clear()
            return True
            
        return False
        
    def get_stats(self) -> Dict[str, int]:
        """Get current deauth frame counts by source MAC."""
        current_time = time.time()
        cutoff_time = current_time - self._window_seconds
        
        stats = {}
        for src_mac, frames in self._deauth_counts.items():
            recent_frames = [ts for ts, _ in frames if ts > cutoff_time]
            if recent_frames:
                stats[src_mac] = len(recent_frames)
                
        return stats
