"""
Scheduled task handlers for model_swap_day and model_swap_night.
"""
import logging
from typing import Any

from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.autonomous.tasks.model_swap")


async def run_day(task: ScheduledTask) -> dict[str, Any]:
    """Swap to day model: unload night model, pre-warm day model (7:30 AM)."""
    from ...jobs.model_swap import run_model_swap_day
    return await run_model_swap_day()


async def run_night(task: ScheduledTask) -> dict[str, Any]:
    """Swap to night: unload day model to free VRAM for graphiti (midnight)."""
    from ...jobs.model_swap import run_model_swap_night
    return await run_model_swap_night()
