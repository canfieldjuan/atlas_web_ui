"""
Model swap jobs for the day/night Ollama model split.

Day swap  (7:30 AM): unload qwen3:32b -> pre-warm qwen3:14b for the day.
Night swap (midnight): unload qwen3:14b -> free VRAM for graphiti-wrapper.

The graphiti-wrapper (port 8001) loads qwen3:32b on demand when email_graph_sync
runs at 1 AM; Atlas does NOT pre-load it here to avoid keeping 32b in VRAM all
night unnecessarily.

Both functions are no-ops when settings.llm.model_swap_enabled is False.
"""
import logging
from typing import Any

logger = logging.getLogger("atlas.jobs.model_swap")


async def run_model_swap_day() -> dict[str, Any]:
    """Day swap: unload night model, pre-warm day model.

    Called at 7:30 AM by the model_swap_day scheduled task.
    """
    from ..config import settings
    from ..services.llm.model_manager import (
        _matches,
        list_running_models,
        load_model,
        unload_model,
    )

    if not settings.llm.model_swap_enabled:
        return {"skipped": True, "_skip_synthesis": "Model swap disabled (ATLAS_LLM__MODEL_SWAP_ENABLED)."}

    base_url = settings.llm.ollama_url
    day = settings.llm.day_model
    night = settings.llm.night_model

    logger.info("Model swap DAY: unload %s, load %s", night, day)

    running = await list_running_models(base_url)
    night_was_running = any(_matches(r, night) for r in running)

    unloaded_night = False
    if night_was_running:
        unloaded_night = await unload_model(night, base_url)
        if not unloaded_night:
            logger.warning("Day swap: failed to unload %s", night)
    else:
        logger.debug("Day swap: %s was not running, skipping unload", night)

    loaded_day = await load_model(day, base_url, keep_alive="2h")
    if not loaded_day:
        logger.warning("Day swap: failed to pre-warm %s", day)

    # Also pre-warm the intent fallback classifier (phi3:mini etc.)
    fallback = settings.intent_router.llm_fallback_model
    loaded_fallback = False
    if settings.intent_router.llm_fallback_enabled and fallback:
        loaded_fallback = await load_model(fallback, base_url, keep_alive="2h")
        if not loaded_fallback:
            logger.warning("Day swap: failed to pre-warm fallback classifier %s", fallback)

    result = {
        "direction": "day",
        "day_model": day,
        "night_model": night,
        "night_was_running": night_was_running,
        "unloaded_night": unloaded_night,
        "loaded_day": loaded_day,
        "loaded_fallback": loaded_fallback,
    }
    logger.info("Model swap DAY complete: %s", result)
    return result


async def run_model_swap_night() -> dict[str, Any]:
    """Night swap: unload day model to free VRAM for graphiti-wrapper.

    Called at midnight by the model_swap_night scheduled task.
    The graphiti-wrapper loads qwen3:32b itself when email_graph_sync runs at 1 AM.
    """
    from ..config import settings
    from ..services.llm.model_manager import _matches, list_running_models, unload_model

    if not settings.llm.model_swap_enabled:
        return {"skipped": True, "_skip_synthesis": "Model swap disabled (ATLAS_LLM__MODEL_SWAP_ENABLED)."}

    base_url = settings.llm.ollama_url
    day = settings.llm.day_model
    night = settings.llm.night_model

    logger.info("Model swap NIGHT: unload %s (freeing VRAM for graphiti)", day)

    running = await list_running_models(base_url)
    day_was_running = any(_matches(r, day) for r in running)

    unloaded_day = False
    if day_was_running:
        unloaded_day = await unload_model(day, base_url)
        if not unloaded_day:
            logger.warning("Night swap: failed to unload %s", day)
    else:
        logger.debug("Night swap: %s was not running, nothing to unload", day)

    result = {
        "direction": "night",
        "day_model": day,
        "night_model": night,
        "day_was_running": day_was_running,
        "unloaded_day": unloaded_day,
    }
    logger.info("Model swap NIGHT complete: %s", result)
    return result
