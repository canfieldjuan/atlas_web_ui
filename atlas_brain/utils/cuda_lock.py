"""Shared CUDA lock for serializing GPU access across agent graphs."""

import asyncio
from typing import Optional

_cuda_lock: Optional[asyncio.Lock] = None


def get_cuda_lock() -> asyncio.Lock:
    """Get or create the global CUDA lock.

    All agent graphs share this single lock to serialize LLM calls
    that require exclusive GPU access.
    """
    global _cuda_lock
    if _cuda_lock is None:
        _cuda_lock = asyncio.Lock()
    return _cuda_lock
