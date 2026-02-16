"""
TTS bridge for orchestrated WebSocket endpoint.

Synthesizes text to WAV bytes using Kokoro KPipeline directly,
without playing through speakers. Returns base64-encodable WAV data
for sending over WebSocket to remote clients.
"""

import asyncio
import io
import logging
import wave

import numpy as np

from ...config import settings

logger = logging.getLogger("atlas.api.orchestrated.tts_bridge")

# Language code mapping: human-readable -> kokoro KPipeline codes
_LANG_MAP = {
    "en-us": "a",
    "en-gb": "b",
    "es": "e",
    "fr": "f",
    "hi": "h",
    "it": "i",
    "ja": "j",
    "pt-br": "p",
    "zh": "z",
}

# Module-level lazy singleton
_pipeline = None
_pipeline_lock = asyncio.Lock()


async def _ensure_pipeline():
    """Lazy-load Kokoro pipeline on first call (thread-safe)."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    async with _pipeline_lock:
        # Double-check after acquiring lock
        if _pipeline is not None:
            return _pipeline

        from kokoro import KPipeline

        tts_cfg = settings.tts
        lang_code = _LANG_MAP.get(tts_cfg.kokoro_lang, tts_cfg.kokoro_lang)

        logger.info("Loading Kokoro KPipeline (lang_code=%s)", lang_code)
        _pipeline = KPipeline(lang_code=lang_code)
        logger.info("Kokoro KPipeline loaded for orchestrated TTS")
        return _pipeline


def _synthesize_blocking(text: str, voice: str, speed: float) -> tuple[np.ndarray, int]:
    """Run Kokoro synthesis (blocking, meant for asyncio.to_thread)."""
    sr = 24000
    chunks = []
    for _graphemes, _phonemes, audio in _pipeline(text, voice=voice, speed=speed):
        if audio is not None and len(audio) > 0:
            chunks.append(np.asarray(audio, dtype=np.float32))
    if not chunks:
        return np.array([], dtype=np.float32), sr
    return np.concatenate(chunks), sr


def _float32_to_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    """Convert float32 samples to int16 WAV bytes."""
    # Clip and convert float32 [-1.0, 1.0] to int16
    clipped = np.clip(samples, -1.0, 1.0)
    int16_samples = (clipped * 32767).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(int16_samples.tobytes())
    return buffer.getvalue()


async def synthesize_to_wav_bytes(
    text: str,
    voice: str | None = None,
    speed: float | None = None,
    lang: str | None = None,
) -> bytes:
    """
    Synthesize text to WAV bytes using Kokoro KPipeline.

    Args:
        text: Text to synthesize
        voice: Voice name (default from config)
        speed: Speech speed (default from config)
        lang: Language code (default from config)

    Returns:
        WAV file bytes (16-bit PCM, mono, 24kHz)
    """
    await _ensure_pipeline()

    tts_cfg = settings.tts
    voice = voice or tts_cfg.voice
    speed = speed if speed is not None else tts_cfg.speed

    # Run synthesis in thread pool (it's CPU-bound)
    samples, sr = await asyncio.to_thread(
        _synthesize_blocking, text, voice, speed,
    )

    logger.info(
        "Synthesized %d samples (%.1fs) at %dHz for: %s",
        len(samples), len(samples) / sr if sr else 0, sr, text[:60],
    )

    # Convert to WAV bytes
    return _float32_to_wav_bytes(samples, sr)
