"""
Port scan detector.

Detects port scanning attempts by tracking unique port access patterns.
"""

import logging
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from ...config import settings

logger = logging.getLogger("atlas.security.network.port_scan")


class PortScanDetector:
    """
    Detects port scanning activity.
    
    Tracks TCP/UDP connection attempts and alerts on suspicious patterns.
    """
    
    def __init__(self):
        self._threshold = settings.security.port_scan_threshold
        self._window_seconds = settings.security.port_scan_window
        self._whitelist = set(settings.security.whitelist_ips)
        # Store access attempts as list of (timestamp, port)
        self._port_access: Dict[str, List[Tuple[float, int]]] = defaultdict(list)
        self._last_cleanup = time.time()
        self._cleanup_interval = 60.0 # Clean up every minute
        
    def process_connection_attempt(
        self, src_ip: str, dst_ip: str, dst_port: int
    ) -> Optional[Dict[str, any]]:
        """
        Process connection attempt and check for port scan patterns.
        
        Returns alert details if port scan detected, None otherwise.
        """
        if src_ip in self._whitelist:
            return None
            
        current_time = time.time()
        
        # Periodic cleanup of stale IPs (memory leak fix)
        if current_time - self._last_cleanup > self._cleanup_interval:
            self._cleanup_stale_entries(current_time)
            self._last_cleanup = current_time
            
        cutoff_time = current_time - self._window_seconds
        
        # Filter old entries for this IP
        # Optimization: If list is too long, just slice it (DoS protection)
        # If an IP sends 10k packets, we don't need to store all of them to know it's a scanner
        if len(self._port_access[src_ip]) > 1000:
             # Keep only recent 1000, assuming they are somewhat ordered or at least relevant
             self._port_access[src_ip] = self._port_access[src_ip][-1000:]
             
        self._port_access[src_ip] = [
            (ts, port) for ts, port in self._port_access[src_ip]
            if ts > cutoff_time
        ]
        
        self._port_access[src_ip].append((current_time, dst_port))
        
        # Count unique ports
        unique_ports = set(port for _, port in self._port_access[src_ip])
        port_count = len(unique_ports)
        
        if port_count >= self._threshold:
            severity = self._calculate_severity(port_count)
            
            logger.warning(
                "Port scan detected: %s scanned %d ports on %s in %ds",
                src_ip, port_count, dst_ip, int(self._window_seconds)
            )
            
            # Reset after alert to avoid spamming
            self._port_access[src_ip].clear()
            
            return {
                "type": "port_scan",
                "severity": severity,
                "source_ip": src_ip,
                "target_ip": dst_ip,
                "ports_scanned": port_count,
                "ports": sorted(list(unique_ports)),
                "time_window": self._window_seconds
            }
            
        return None
        
    def _cleanup_stale_entries(self, current_time: float) -> None:
        """Remove IPs that haven't been seen recently."""
        cutoff_time = current_time - self._window_seconds
        # Create list of keys to remove to avoid runtime error during iteration
        to_remove = []
        
        for src_ip, attempts in self._port_access.items():
            # Filter attempts in place or check if all are old
            # Efficient check: if last attempt is old, all are old (entries are appended sorted by time)
            if not attempts:
                to_remove.append(src_ip)
                continue
                
            last_ts = attempts[-1][0]
            if last_ts < cutoff_time:
                to_remove.append(src_ip)
            else:
                # If some are valid, we might want to prune the old ones here too, 
                # but lazy pruning in process_connection_attempt is usually enough for active IPs.
                # However, for semi-active IPs, we should prune to keep memory low.
                # Only prune if list is getting long
                if len(attempts) > 100:
                    self._port_access[src_ip] = [
                        (ts, p) for ts, p in attempts if ts > cutoff_time
                    ]
                    if not self._port_access[src_ip]:
                         to_remove.append(src_ip)

        for ip in to_remove:
            del self._port_access[ip]
            
    def _calculate_severity(self, port_count: int) -> str:
        """Calculate severity based on number of ports scanned."""
        if port_count >= 100:
            return "high"
        elif port_count >= 50:
            return "medium"
        else:
            return "low"
            
    def get_stats(self) -> Dict[str, int]:
        """Get current port access counts by source IP."""
        current_time = time.time()
        cutoff_time = current_time - self._window_seconds
        
        stats = {}
        for src_ip, access_list in self._port_access.items():
            recent_ports = set(
                port for ts, port in access_list if ts > cutoff_time
            )
            if recent_ports:
                stats[src_ip] = len(recent_ports)
                
        return stats
