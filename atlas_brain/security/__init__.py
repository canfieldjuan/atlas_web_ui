"""
Network security monitoring for Atlas.

Provides WiFi threat detection, network IDS, and security asset tracking.
"""

from .monitor import SecurityMonitor, get_security_monitor

__all__ = [
    "SecurityMonitor",
    "get_security_monitor",
]
