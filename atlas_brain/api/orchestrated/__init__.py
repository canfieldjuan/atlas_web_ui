"""
Orchestrated voice WebSocket API.

Full voice pipeline over WebSocket: audio in → ASR → agent → TTS → audio out.
"""

from .websocket import router

__all__ = ["router"]
