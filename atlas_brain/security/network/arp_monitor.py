"""
ARP monitor for detecting ARP spoofing and poisoning attacks.

Tracks ARP table changes and detects malicious patterns.
"""

import logging
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from ...config import settings

logger = logging.getLogger("atlas.security.network.arp")


class ARPMonitor:
    """
    Monitors ARP traffic for poisoning and spoofing attacks.
    
    Tracks IP to MAC address mappings and alerts on suspicious changes.
    """
    
    def __init__(self):
        self._enabled = settings.security.arp_monitor_enabled
        # Threshold assignment removed (unused)
        self._known_gateways = set(settings.security.known_gateways)
        self._static_entries = dict(settings.security.static_arp_entries)
        self._arp_table: Dict[str, str] = {}
        self._change_history: Dict[str, List[Tuple[float, str, str]]] = (
            defaultdict(list)
        )
        self._last_cleanup = time.time()
        self._cleanup_interval = 300  # 5 minutes
        
    def process_arp_packet(
        self, src_ip: str, src_mac: str, op_code: int
    ) -> Optional[Dict[str, Any]]:
        """
        Process ARP packet and check for suspicious patterns.
        
        Args:
            src_ip: Source IP address
            src_mac: Source MAC address
            op_code: ARP operation code (1=request, 2=reply)
            
        Returns alert details if threat detected, None otherwise.
        """
        if not self._enabled:
            return None
            
        current_time = time.time()
        
        # Periodic cleanup
        if current_time - self._last_cleanup > self._cleanup_interval:
             self._cleanup_storage()
             self._last_cleanup = current_time
             
        src_mac_lower = src_mac.lower()
        
        if src_ip in self._static_entries:
            expected_mac = self._static_entries[src_ip].lower()
            if src_mac_lower != expected_mac:
                logger.error(
                    "Static ARP violation: %s has MAC %s, expected %s",
                    src_ip, src_mac, expected_mac
                )
                return {
                    "type": "arp_spoofing",
                    "severity": "critical",
                    "source_ip": src_ip,
                    "detected_mac": src_mac,
                    "expected_mac": expected_mac,
                    "details": {"static_entry_violated": True}
                }
        
        if src_ip in self._arp_table:
            old_mac = self._arp_table[src_ip]
            if old_mac != src_mac_lower:
                self._change_history[src_ip].append(
                    (current_time, old_mac, src_mac_lower)
                )
                
                # Limit history size per IP
                if len(self._change_history[src_ip]) > 50:
                    self._change_history[src_ip] = self._change_history[src_ip][-50:]
                
                is_gateway = src_ip in self._known_gateways
                severity = "high" if is_gateway else "medium"
                
                logger.warning(
                    "ARP table change detected: %s changed from %s to %s%s",
                    src_ip, old_mac, src_mac,
                    " (GATEWAY)" if is_gateway else ""
                )
                
                self._arp_table[src_ip] = src_mac_lower
                
                return {
                    "type": "arp_change",
                    "severity": severity,
                    "source_ip": src_ip,
                    "old_mac": old_mac,
                    "new_mac": src_mac_lower,
                    "is_gateway": is_gateway,
                    "change_count": len(self._change_history[src_ip])
                }
        else:
            # Prevent ARP table explosion
            if len(self._arp_table) < 5000:
                self._arp_table[src_ip] = src_mac_lower
            
        return None
        
    def _cleanup_storage(self) -> None:
        """Clean up old history and limit table size."""
        # Clean up history for IPs that haven't changed in a long time? 
        # Actually history is only updated on change, so old history implies stable network or old attack.
        # We can just clear history if it gets too large globally.
        if len(self._change_history) > 1000:
             self._change_history.clear()
             
        # ARP table cleanup is harder without timestamps, 
        # but if we are at limit, clearing it allows relearning active hosts.
        if len(self._arp_table) >= 5000:
            self._arp_table.clear()
        
    def get_arp_table(self) -> Dict[str, str]:
        """Get current ARP table."""
        return self._arp_table.copy()
        
    def get_change_history(self, ip: Optional[str] = None) -> Dict:
        """Get ARP change history for IP or all IPs."""
        if ip:
            return {ip: self._change_history.get(ip, [])}
        return dict(self._change_history)
