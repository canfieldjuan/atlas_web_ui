"""
Traffic analyzer for detecting network anomalies.

Tracks bandwidth usage and connection patterns to identify threats.
"""

import logging
import time
from collections import defaultdict, deque
from typing import Any, Dict, Optional

from ...config import settings

logger = logging.getLogger("atlas.security.network.traffic")



class TrafficAnalyzer:
    """
    Analyzes network traffic for anomalies.
    
    Tracks metrics and compares against baseline to detect threats.
    """
    
    def __init__(self):
        self._enabled = settings.security.traffic_analysis_enabled
        self._baseline_hours = 24.0  # Default to 24h if not set correctly
        if hasattr(settings.security, 'baseline_period_hours'):
             self._baseline_hours = settings.security.baseline_period_hours
             
        self._spike_multiplier = settings.security.bandwidth_spike_multiplier
        
        # Traffic history: deque of (timestamp, bytes) per second
        # maxlen=3600 means we keep exactly 1 hour of second-by-second history
        self._history_in: deque = deque(maxlen=3600)
        self._history_out: deque = deque(maxlen=3600)
        
        # Current second accumulation
        self._current_second = int(time.time())
        self._current_bytes_in = 0
        self._current_bytes_out = 0
        
        # Connection tracking with cleanup
        self._connections: Dict[str, int] = defaultdict(int)
        self._last_cleanup = time.time()
        self._cleanup_interval = 300  # 5 minutes
        
        self._baseline_bps_in: float = 0.0
        self._baseline_bps_out: float = 0.0
        self._baseline_established = False
        self._baseline_start_time = time.time()
        
    def _rotate_bucket(self, current_time: int) -> None:
        """Rotate aggregation bucket if second has changed."""
        # Add completed bucket to history
        self._history_in.append((self._current_second, self._current_bytes_in))
        self._history_out.append((self._current_second, self._current_bytes_out))
        
        # Fill gaps with zeros if more than 1 second passed
        # (e.g. no traffic for 10 seconds)
        gap = min(current_time - self._current_second - 1, 60)
        for i in range(gap):
            ts = self._current_second + 1 + i
            self._history_in.append((ts, 0))
            self._history_out.append((ts, 0))

        # Reset bucket
        self._current_second = current_time
        self._current_bytes_in = 0
        self._current_bytes_out = 0

    def _cleanup_connections(self) -> None:
        """Remove old connection counters to prevent memory leak."""
        if len(self._connections) > 5000:
            self._connections.clear()
            
    def record_traffic(
        self, direction: str, bytes_count: int, src_ip: str
    ) -> Optional[Dict[str, Any]]:
        """
        Record traffic and check for anomalies.
        
        Args:
            direction: 'in' or 'out'
            bytes_count: Number of bytes
            src_ip: Source IP address
            
        Returns alert details if anomaly detected, None otherwise.
        """
        if not self._enabled:
            return None
            
        current_time_int = int(time.time())
        
        # Rotate bucket if needed
        if current_time_int > self._current_second:
            self._rotate_bucket(current_time_int)
            
            # Periodic cleanup
            if current_time_int - self._last_cleanup > self._cleanup_interval:
                self._cleanup_connections()
                self._last_cleanup = current_time_int
        
        if direction == "in":
            self._current_bytes_in += bytes_count
        elif direction == "out":
            self._current_bytes_out += bytes_count
        
        self._connections[src_ip] += 1
        
        if not self._baseline_established:
            self._check_baseline_ready(current_time_int)
            return None
            
        # Optimization: calculate rate only if we have baseline established
        current_bps_in = self._calculate_instant_rate(self._history_in, self._current_bytes_in)
        current_bps_out = self._calculate_instant_rate(self._history_out, self._current_bytes_out)
        
        # Use a minimum baseline to avoid zero division and noise
        min_baseline = 1000.0 
        
        if self._baseline_bps_in > min_baseline:
            trigger_in = self._baseline_bps_in * self._spike_multiplier
            if current_bps_in > trigger_in:
                logger.warning(
                    "Inbound bandwidth spike: %d bps (baseline: %d bps)",
                    int(current_bps_in), int(self._baseline_bps_in)
                )
                return {
                    "type": "bandwidth_spike",
                    "severity": "medium",
                    "direction": "inbound",
                    "current_bps": int(current_bps_in),
                    "baseline_bps": int(self._baseline_bps_in),
                    "multiplier": current_bps_in / self._baseline_bps_in
                }
            
        if self._baseline_bps_out > min_baseline:
            trigger_out = self._baseline_bps_out * self._spike_multiplier
            if current_bps_out > trigger_out:
                logger.warning(
                    "Outbound bandwidth spike: %d bps (baseline: %d bps)",
                    int(current_bps_out), int(self._baseline_bps_out)
                )
                return {
                    "type": "bandwidth_spike",
                    "severity": "medium",
                    "direction": "outbound",
                    "current_bps": int(current_bps_out),
                    "baseline_bps": int(self._baseline_bps_out),
                    "multiplier": current_bps_out / self._baseline_bps_out
                }
            
        return None
        
    def _check_baseline_ready(self, current_time: float) -> None:
        """Check if baseline period has elapsed."""
        elapsed_hours = (current_time - self._baseline_start_time) / 3600
        if elapsed_hours >= self._baseline_hours:
            self._establish_baseline()
            
    def _establish_baseline(self) -> None:
        """Calculate baseline metrics from collected data."""
        # Calculate average bps over the entire history available (last hour)
        val_in = self._calculate_average_bps(self._history_in)
        val_out = self._calculate_average_bps(self._history_out)
        
        self._baseline_bps_in = val_in
        self._baseline_bps_out = val_out
        
        self._baseline_established = True
        logger.info(
            "Traffic baseline established: in=%d bps, out=%d bps",
            int(self._baseline_bps_in), int(self._baseline_bps_out)
        )
        
    def _calculate_instant_rate(self, history: deque, current_bytes: int) -> float:
        """Calculate rate over last 60 seconds."""
        # Sum last 59 seconds from history + current partial
        total_bytes = current_bytes
        
        count = 0
        limit = 59
        
        # Iterate backwards
        for i in range(len(history) - 1, -1, -1):
            total_bytes += history[i][1]
            count += 1
            if count >= limit:
                break
                
        seconds = count + 1
        return total_bytes / float(seconds)

    def _calculate_average_bps(self, history: deque) -> float:
        """Calculate long-term average bps from history."""
        if not history:
            return 0.0
        # Average of all recorded seconds
        total = sum(x[1] for x in history)
        return total / float(len(history))
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get current traffic metrics."""
        current_bps_in = self._calculate_instant_rate(self._history_in, self._current_bytes_in)
        current_bps_out = self._calculate_instant_rate(self._history_out, self._current_bytes_out)
        
        return {
            "bytes_per_sec_in": current_bps_in,
            "bytes_per_sec_out": current_bps_out,
            "baseline_established": self._baseline_established,
            "baseline_bps_in": self._baseline_bps_in,
            "baseline_bps_out": self._baseline_bps_out,
            "total_connections": sum(self._connections.values())
        }
