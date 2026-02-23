"""
System stats endpoint — real CPU and network metrics via psutil.
"""

import time

import psutil
from fastapi import APIRouter

router = APIRouter(prefix="/system", tags=["system"])

# Module-level state for computing network throughput between polls
_last_net_io = psutil.net_io_counters()
_last_net_time = time.monotonic()


@router.get("/stats")
async def get_system_stats():
    """
    Return real-time system metrics.

    Returns:
        cpu_percent:    0–100 float
        net_mbps:       combined send+recv throughput in Mb/s (megabits)
        mem_percent:    RAM usage 0–100
    """
    global _last_net_io, _last_net_time

    # CPU — non-blocking (uses kernel-cached value since last call)
    cpu = psutil.cpu_percent(interval=None)

    # Network throughput delta
    now = time.monotonic()
    cur_net = psutil.net_io_counters()
    elapsed = now - _last_net_time or 0.001  # guard div-by-zero on first call

    bytes_delta = (
        (cur_net.bytes_sent - _last_net_io.bytes_sent)
        + (cur_net.bytes_recv - _last_net_io.bytes_recv)
    )
    mbps = round((bytes_delta * 8) / elapsed / 1_000_000, 1)  # bytes → megabits

    _last_net_io = cur_net
    _last_net_time = now

    # Memory
    mem = psutil.virtual_memory().percent

    return {
        "cpu_percent": round(cpu, 1),
        "net_mbps": max(0.0, mbps),
        "mem_percent": round(mem, 1),
    }
