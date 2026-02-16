"""
Async ASR bridge for orchestrated WebSocket endpoint.

Connects to the Nemotron ASR streaming server via WebSocket and
provides async methods for sending audio and receiving transcripts.
"""

import asyncio
import json
import logging
from typing import Callable, Optional

import websockets

from ...config import settings

logger = logging.getLogger("atlas.api.orchestrated.asr_bridge")


class AsrBridge:
    """Async WebSocket client to the ASR streaming server.

    Usage:
        bridge = AsrBridge(on_partial=..., on_final=...)
        await bridge.connect()
        await bridge.send_audio(pcm_bytes)
        transcript = await bridge.finalize()
        await bridge.reset()
        await bridge.close()
    """

    def __init__(
        self,
        url: str | None = None,
        on_partial: Callable[[str], None] | None = None,
        on_final: Callable[[str], None] | None = None,
    ):
        self._url = url or settings.voice.asr_ws_url or "ws://localhost:8081/v1/asr/stream"
        self._on_partial = on_partial
        self._on_final = on_final
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._recv_task: Optional[asyncio.Task] = None
        self._final_event = asyncio.Event()
        self._final_text: str = ""
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    async def connect(self, timeout: float | None = None) -> None:
        """Open WebSocket connection to ASR server and start recv loop."""
        if self._connected:
            return

        timeout = timeout or settings.orchestrated.asr_connect_timeout

        try:
            self._ws = await asyncio.wait_for(
                websockets.connect(
                    self._url,
                    ping_interval=30,
                    ping_timeout=10,
                    max_size=2**20,  # 1MB
                ),
                timeout=timeout,
            )
            self._connected = True
            self._recv_task = asyncio.create_task(
                self._recv_loop(), name="asr-recv-loop"
            )
            logger.info("ASR bridge connected to %s", self._url)
        except asyncio.TimeoutError:
            raise ConnectionError(
                f"ASR connection timed out after {timeout}s to {self._url}"
            )
        except Exception as e:
            raise ConnectionError(f"ASR connection failed: {e}")

    async def send_audio(self, pcm_bytes: bytes) -> None:
        """Forward raw PCM binary frame to ASR server."""
        if not self._connected or not self._ws:
            return
        try:
            await self._ws.send(pcm_bytes)
        except websockets.ConnectionClosed:
            logger.warning("ASR connection closed while sending audio")
            self._connected = False

    async def finalize(self, timeout: float | None = None) -> str:
        """Send finalize command and wait for final transcript.

        Returns:
            Final transcript text (may be empty string)
        """
        if not self._connected or not self._ws:
            return ""

        timeout = timeout or settings.orchestrated.asr_finalize_timeout
        self._final_event.clear()
        self._final_text = ""

        try:
            await self._ws.send(json.dumps({"type": "finalize"}))
            await asyncio.wait_for(self._final_event.wait(), timeout=timeout)
            return self._final_text
        except asyncio.TimeoutError:
            logger.warning("ASR finalize timed out after %.1fs", timeout)
            return self._final_text or ""

    async def reset(self) -> None:
        """Reset ASR buffer for the next utterance."""
        if not self._connected or not self._ws:
            return
        try:
            await self._ws.send(json.dumps({"type": "reset"}))
        except websockets.ConnectionClosed:
            logger.warning("ASR connection closed while resetting")
            self._connected = False

    async def close(self) -> None:
        """Close the ASR connection and cancel recv loop."""
        self._connected = False
        if self._recv_task and not self._recv_task.done():
            self._recv_task.cancel()
            try:
                await self._recv_task
            except (asyncio.CancelledError, Exception):
                pass
            self._recv_task = None
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

    async def _recv_loop(self) -> None:
        """Background task receiving messages from ASR server."""
        try:
            async for message in self._ws:
                try:
                    data = json.loads(message)
                except (json.JSONDecodeError, TypeError):
                    continue

                msg_type = data.get("type", "")

                if msg_type == "partial":
                    text = data.get("text", "")
                    if self._on_partial and text:
                        try:
                            self._on_partial(text)
                        except Exception as e:
                            logger.debug("on_partial callback error: %s", e)

                elif msg_type == "final":
                    text = data.get("text", "")
                    self._final_text = text
                    self._final_event.set()
                    if self._on_final and text:
                        try:
                            self._on_final(text)
                        except Exception as e:
                            logger.debug("on_final callback error: %s", e)

                elif msg_type == "error":
                    logger.error("ASR error: %s", data.get("message", "unknown"))
                    # Unblock any pending finalize
                    self._final_event.set()

        except websockets.ConnectionClosed:
            logger.info("ASR connection closed")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.exception("ASR recv loop error: %s", e)
        finally:
            self._connected = False
            # Unblock any pending finalize
            self._final_event.set()
