"""
Ollama model manager -- load, unload, and inspect models in VRAM.

Used by the model-swap scheduled tasks to enforce the day/night split:
  - Day   (7:30 AM): unload qwen3:32b, pre-warm qwen3:14b
  - Night (00:00 AM): unload qwen3:14b (graphiti-wrapper loads 32b on demand)

All calls go to the Ollama REST API; no direct GPU interaction.
"""
import logging
from typing import Optional

import httpx

logger = logging.getLogger("atlas.services.llm.model_manager")

# Ollama endpoint constants
_PATH_PS = "/api/ps"
_PATH_GENERATE = "/api/generate"


def _matches(running_name: str, target: str) -> bool:
    """Case-insensitive model name match, ignoring trailing tag variants."""
    rn = running_name.lower()
    tn = target.lower()
    return rn == tn or rn.startswith(tn + ":") or tn.startswith(rn + ":")


async def list_running_models(base_url: str) -> list[str]:
    """Return names of models currently loaded in Ollama VRAM.

    Calls GET /api/ps. Returns empty list on error (fail-open).
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(base_url.rstrip("/") + _PATH_PS)
            resp.raise_for_status()
            data = resp.json()
            names = [m["name"] for m in data.get("models", [])]
            logger.debug("Ollama running models: %s", names)
            return names
    except Exception as e:
        logger.warning("list_running_models failed: %s", e)
        return []


async def is_model_running(model_name: str, base_url: str) -> bool:
    """Return True if the named model is currently loaded in VRAM."""
    running = await list_running_models(base_url)
    return any(_matches(r, model_name) for r in running)


async def unload_model(model_name: str, base_url: str) -> bool:
    """Unload a model from Ollama VRAM by sending keep_alive=0.

    Returns True if the request succeeded, False otherwise.
    The model is not deleted from disk -- only evicted from VRAM.
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                base_url.rstrip("/") + _PATH_GENERATE,
                json={"model": model_name, "keep_alive": 0, "stream": False},
            )
            resp.raise_for_status()
            logger.info("Unloaded model from VRAM: %s", model_name)
            return True
    except Exception as e:
        logger.warning("unload_model(%s) failed: %s", model_name, e)
        return False


async def load_model(
    model_name: str,
    base_url: str,
    keep_alive: str = "2h",
) -> bool:
    """Pre-warm a model into Ollama VRAM without generating output.

    Sends an empty generate request so Ollama loads the weights.
    keep_alive controls how long the model stays loaded after this call.

    Returns True on success, False otherwise.
    """
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                base_url.rstrip("/") + _PATH_GENERATE,
                json={
                    "model": model_name,
                    "keep_alive": keep_alive,
                    "stream": False,
                },
            )
            resp.raise_for_status()
            logger.info("Pre-warmed model into VRAM: %s (keep_alive=%s)", model_name, keep_alive)
            return True
    except Exception as e:
        logger.warning("load_model(%s) failed: %s", model_name, e)
        return False
