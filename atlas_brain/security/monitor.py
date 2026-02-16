"""
Main security monitor service for Atlas network security.

Orchestrates wireless monitoring, network IDS, and threat detection.
"""

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Optional

from scapy.all import ARP, IP, TCP, UDP, AsyncSniffer, get_if_addr
from scapy.utils import PcapWriter

from ..config import settings

logger = logging.getLogger("atlas.security.monitor")


class SecurityMonitor:
    """
    Main security monitoring service.
    
    Coordinates wireless monitoring and threat detection systems.
    """
    
    def __init__(self):
        self._wireless_monitor = None
        self._port_scan_detector = None
        self._arp_monitor = None
        self._traffic_analyzer = None
        self._drone_tracker = None
        self._vehicle_tracker = None
        self._sensor_tracker = None
        self._network_sniffer: Optional[AsyncSniffer] = None
        self._network_interface_ip: Optional[str] = None
        self._pcap_writer: Optional[PcapWriter] = None
        self._pcap_lock = Lock()
        self._pcap_check_counter = 0
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._packets_processed = 0
        self._alerts_emitted = 0
        self._packets_dropped = 0
        
    @property
    def is_running(self) -> bool:
        """Check if security monitor is running."""
        return self._running
        
    async def start(self) -> None:
        """Start the security monitor."""
        if self._running:
            logger.warning("Security monitor already running")
            return
            
        if (
            not settings.security.network_monitor_enabled
            and not settings.security.network_ids_enabled
            and not settings.security.asset_tracking_enabled
        ):
            logger.info("All security monitoring disabled in config")
            return
            
        logger.info("Starting security monitor")
        self._running = True
        self._loop = asyncio.get_running_loop()
        
        try:
            if settings.security.network_monitor_enabled:
                if settings.security.wireless_interface:
                    from .wireless.monitor import WirelessMonitor
                    self._wireless_monitor = WirelessMonitor()
                    await self._wireless_monitor.start()
                    logger.info("Wireless monitoring started")
            
            if settings.security.network_ids_enabled:
                from .network import (
                    PortScanDetector,
                    ARPMonitor,
                    TrafficAnalyzer
                )
                self._port_scan_detector = PortScanDetector()
                self._arp_monitor = ARPMonitor()
                self._traffic_analyzer = TrafficAnalyzer()
                self._network_interface_ip = self._resolve_interface_ip()
                self._start_network_capture()
                logger.info("Network IDS components initialized")

            if settings.security.asset_tracking_enabled:
                from .assets import (
                    DroneTracker,
                    SensorNetworkTracker,
                    VehicleTracker,
                )
                stale_after = settings.security.asset_stale_after_seconds
                max_assets = settings.security.asset_max_tracked
                if settings.security.drone_tracking_enabled:
                    self._drone_tracker = DroneTracker(stale_after, max_assets)
                if settings.security.vehicle_tracking_enabled:
                    self._vehicle_tracker = VehicleTracker(stale_after, max_assets)
                if settings.security.sensor_tracking_enabled:
                    self._sensor_tracker = SensorNetworkTracker(stale_after, max_assets)
                logger.info("Security asset tracking initialized")
        except Exception as e:
            logger.error("Failed to start security components: %s", e)
            self._running = False
            raise
            
        logger.info("Security monitor started successfully")
        
    async def stop(self) -> None:
        """Stop the security monitor."""
        if not self._running:
            return
            
        logger.info("Stopping security monitor")
        self._running = False
        
        if self._wireless_monitor:
            await self._wireless_monitor.stop()
            self._wireless_monitor = None

        self._stop_network_capture()
        
        self._port_scan_detector = None
        self._arp_monitor = None
        self._traffic_analyzer = None
        self._drone_tracker = None
        self._vehicle_tracker = None
        self._sensor_tracker = None
        self._network_interface_ip = None
        self._loop = None
            
        logger.info("Security monitor stopped")
    
    def get_port_scan_detector(self):
        """Get the port scan detector instance."""
        return self._port_scan_detector
    
    def get_arp_monitor(self):
        """Get the ARP monitor instance."""
        return self._arp_monitor
    
    def get_traffic_analyzer(self):
        """Get the traffic analyzer instance."""
        return self._traffic_analyzer

    def get_drone_tracker(self):
        """Get the drone tracker instance."""
        return self._drone_tracker

    def get_vehicle_tracker(self):
        """Get the vehicle tracker instance."""
        return self._vehicle_tracker

    def get_sensor_tracker(self):
        """Get the sensor network tracker instance."""
        return self._sensor_tracker

    def observe_asset(
        self,
        asset_type: str,
        identifier: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        """Record asset observation in the matching tracker."""
        normalized_type = asset_type.strip().lower()
        if normalized_type == "drone" and self._drone_tracker:
            return self._drone_tracker.observe(identifier, metadata)
        if normalized_type == "vehicle" and self._vehicle_tracker:
            return self._vehicle_tracker.observe(identifier, metadata)
        if normalized_type == "sensor" and self._sensor_tracker:
            return self._sensor_tracker.observe(identifier, metadata)
        return None

    def get_asset_summary(self) -> dict[str, dict[str, int]]:
        """Return summary counts for each enabled asset tracker."""
        summary = {}
        if self._drone_tracker:
            summary["drone"] = self._drone_tracker.get_summary()
        if self._vehicle_tracker:
            summary["vehicle"] = self._vehicle_tracker.get_summary()
        if self._sensor_tracker:
            summary["sensor"] = self._sensor_tracker.get_summary()
        return summary

    def list_assets(self, asset_type: Optional[str] = None) -> list[dict[str, Any]]:
        """List tracked assets across trackers or for a specific type."""
        trackers = {
            "drone": self._drone_tracker,
            "vehicle": self._vehicle_tracker,
            "sensor": self._sensor_tracker,
        }
        if asset_type:
            tracker = trackers.get(asset_type.lower())
            return tracker.list_assets() if tracker else []

        assets = []
        for tracker in trackers.values():
            if tracker:
                assets.extend(tracker.list_assets())
        assets.sort(key=lambda item: item["last_seen"], reverse=True)
        return assets

    def get_runtime_stats(self) -> dict[str, Any]:
        """Get runtime counters for the network IDS packet pipeline."""
        return {
            "packets_processed": self._packets_processed,
            "alerts_emitted": self._alerts_emitted,
            "packets_dropped": self._packets_dropped,
            "sniffer_running": bool(self._network_sniffer),
        }

    def _resolve_interface_ip(self) -> Optional[str]:
        """Resolve IPv4 address for configured network IDS interface."""
        try:
            return get_if_addr(settings.security.network_interface)
        except Exception as exc:
            logger.warning("Unable to resolve interface IP: %s", exc)
            return None

    def _start_network_capture(self) -> None:
        """Start packet capture for network IDS detectors."""
        if self._network_sniffer:
            return

        self._start_pcap_writer()
        capture_filter = self._build_capture_filter()

        self._network_sniffer = AsyncSniffer(
            iface=settings.security.network_interface,
            filter=capture_filter,
            prn=self._process_network_packet,
            store=False,
        )
        self._network_sniffer.start()
        logger.info(
            "Network IDS capture started on %s (filter=%s)",
            settings.security.network_interface,
            capture_filter if capture_filter else "none",
        )

    def _stop_network_capture(self) -> None:
        """Stop packet capture for network IDS detectors."""
        if not self._network_sniffer:
            self._close_pcap_writer()
            return
        try:
            self._network_sniffer.stop()
        except Exception as exc:
            logger.warning("Network sniffer stop warning: %s", exc)
        finally:
            self._network_sniffer = None
            self._close_pcap_writer()

    def _process_network_packet(self, packet: Any) -> None:
        """Route captured packets through Phase 2 detectors."""
        self._packets_processed += 1
        try:
            self._write_packet_capture(packet)
            if ARP in packet:
                self._handle_arp_packet(packet)
            if IP in packet:
                self._handle_ip_packet(packet)
        except Exception as exc:
            self._packets_dropped += 1
            logger.debug("Packet processing error: %s", exc)

    def _build_capture_filter(self) -> Optional[str]:
        """Build BPF capture filter from configured protocol list."""
        protocol_map = {
            "TCP": "tcp",
            "UDP": "udp",
            "ICMP": "icmp",
            "ARP": "arp",
        }
        tokens = []
        for name in settings.security.protocols_to_monitor:
            token = protocol_map.get(str(name).upper())
            if token:
                tokens.append(token)
        if not tokens:
            return None
        return " or ".join(sorted(set(tokens)))

    def _start_pcap_writer(self) -> None:
        """Initialize pcap writer when packet evidence capture is enabled."""
        if not settings.security.pcap_enabled:
            return
        if self._pcap_writer:
            return
        try:
            pcap_dir = Path(settings.security.pcap_directory)
            pcap_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            file_name = (
                f"network_ids_{settings.security.network_interface}_{stamp}.pcap"
            )
            self._pcap_writer = PcapWriter(str(pcap_dir / file_name), append=True)
            self._enforce_pcap_storage_limit()
        except Exception as exc:
            logger.warning("Unable to initialize pcap writer: %s", exc)
            self._pcap_writer = None

    def _write_packet_capture(self, packet: Any) -> None:
        """Write packet to pcap evidence file when enabled."""
        if not self._pcap_writer:
            return
        with self._pcap_lock:
            self._pcap_writer.write(packet)
            self._pcap_check_counter += 1
        if self._pcap_check_counter >= 200:
            self._pcap_check_counter = 0
            self._enforce_pcap_storage_limit()

    def _enforce_pcap_storage_limit(self) -> None:
        """Prune oldest pcap files when total size exceeds configured limit."""
        max_bytes = settings.security.pcap_max_size_mb * 1024 * 1024
        if max_bytes <= 0:
            return

        pcap_dir = Path(settings.security.pcap_directory)
        if not pcap_dir.exists():
            return

        pcap_files = [path for path in pcap_dir.glob("*.pcap") if path.is_file()]
        if not pcap_files:
            return

        pcap_files.sort(key=lambda path: path.stat().st_mtime)
        total_size = sum(path.stat().st_size for path in pcap_files)
        if total_size <= max_bytes:
            return

        for old_file in pcap_files:
            if total_size <= max_bytes:
                break
            try:
                old_size = old_file.stat().st_size
                old_file.unlink()
                total_size -= old_size
                logger.warning("Removed old pcap file to enforce size limit: %s", old_file)
            except Exception as exc:
                logger.warning("Failed removing old pcap file %s: %s", old_file, exc)

    def _close_pcap_writer(self) -> None:
        """Close pcap writer if it is open."""
        if not self._pcap_writer:
            return
        try:
            with self._pcap_lock:
                self._pcap_writer.close()
        except Exception as exc:
            logger.warning("Error closing pcap writer: %s", exc)
        finally:
            self._pcap_writer = None

    def _handle_arp_packet(self, packet: Any) -> None:
        """Process ARP packets through ARP monitor."""
        if not self._arp_monitor:
            return
        arp_layer = packet[ARP]
        src_ip = getattr(arp_layer, "psrc", None)
        src_mac = getattr(arp_layer, "hwsrc", None)
        op_code = int(getattr(arp_layer, "op", 0))
        if not src_ip or not src_mac:
            return
        alert = self._arp_monitor.process_arp_packet(src_ip, src_mac, op_code)
        self._emit_network_alert(alert)

    def _handle_ip_packet(self, packet: Any) -> None:
        """Process IP packets through scan and traffic detectors."""
        ip_layer = packet[IP]
        src_ip = getattr(ip_layer, "src", "")
        dst_ip = getattr(ip_layer, "dst", "")
        if not src_ip or not dst_ip:
            return

        if TCP in packet and self._port_scan_detector:
            dst_port = int(packet[TCP].dport)
            alert = self._port_scan_detector.process_connection_attempt(
                src_ip, dst_ip, dst_port
            )
            self._emit_network_alert(alert)

        if UDP in packet and self._port_scan_detector:
            dst_port = int(packet[UDP].dport)
            alert = self._port_scan_detector.process_connection_attempt(
                src_ip, dst_ip, dst_port
            )
            self._emit_network_alert(alert)

        if self._traffic_analyzer:
            direction = self._classify_direction(src_ip)
            byte_count = len(packet)
            alert = self._traffic_analyzer.record_traffic(direction, byte_count, src_ip)
            self._emit_network_alert(alert)

    def _classify_direction(self, src_ip: str) -> str:
        """Classify traffic direction relative to monitored interface."""
        if self._network_interface_ip and src_ip == self._network_interface_ip:
            return "out"
        return "in"

    def _emit_network_alert(self, alert: Optional[dict[str, Any]]) -> None:
        """Publish detector alert through Atlas alert manager."""
        if not alert or not self._loop:
            return
        if not self._loop.is_running():
            return
        asyncio.run_coroutine_threadsafe(
            self._publish_alert_event(alert),
            self._loop,
        )

    async def _publish_alert_event(self, alert: dict[str, Any]) -> None:
        """Transform detector output into SecurityAlertEvent."""
        try:
            from ..alerts import SecurityAlertEvent
            from ..alerts.manager import get_alert_manager

            event = SecurityAlertEvent(
                source_id=settings.security.network_interface,
                timestamp=datetime.now(timezone.utc),
                detection_type=alert.get("type", "network_alert"),
                label=alert.get("type", "network_alert").replace("_", " "),
                confidence=1.0,
                metadata=alert,
            )
            await get_alert_manager().process_event(event)
            self._alerts_emitted += 1
        except Exception as exc:
            logger.warning("Failed to publish network alert: %s", exc)


_security_monitor: Optional[SecurityMonitor] = None


def get_security_monitor() -> SecurityMonitor:
    """Get or create the global security monitor instance."""
    global _security_monitor
    if _security_monitor is None:
        _security_monitor = SecurityMonitor()
    return _security_monitor


