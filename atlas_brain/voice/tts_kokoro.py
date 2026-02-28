"""
Kokoro TTS engine for high-quality speech synthesis.

Uses kokoro KPipeline (82M params, 24kHz, 54 voices) as a drop-in
replacement for Piper TTS. Implements the SpeechEngine protocol.
"""

import logging
import threading
import time

import numpy as np
import sounddevice as sd

logger = logging.getLogger("atlas.voice.tts_kokoro")

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


class KokoroTTS:
    """Kokoro TTS engine using KPipeline for natural speech synthesis."""

    def __init__(
        self,
        voice: str = "af_heart",
        speed: float = 1.0,
        lang: str = "en-us",
    ):
        self.voice = voice
        self.speed = speed
        self.lang = lang
        self.stop_event = threading.Event()
        self.current_stream = None
        self._pipeline = None  # Lazy-loaded

    def _ensure_loaded(self):
        """Lazy-load Kokoro pipeline on first speak()."""
        if self._pipeline is None:
            from kokoro import KPipeline

            lang_code = _LANG_MAP.get(self.lang, self.lang)
            logger.info("Loading Kokoro KPipeline (lang_code=%s)", lang_code)
            self._pipeline = KPipeline(lang_code=lang_code)
            # Warmup inference — compile CUDA kernels so first real speak()
            # doesn't pay the 600ms+ cold-start penalty.
            for _g, _p, _a in self._pipeline("warmup", voice=self.voice, speed=self.speed):
                break  # One chunk is enough to warm the kernels
            logger.info("Kokoro KPipeline loaded and warmed")

    def speak(self, text: str):
        """Synthesize and stream audio from Kokoro to sounddevice."""
        self.stop_event.clear()
        try:
            self._ensure_loaded()
        except Exception as e:
            logger.error("Kokoro model load failed: %s", e)
            return

        start_time = time.perf_counter()
        sr = 24000  # KPipeline outputs 24kHz
        first_chunk = True

        try:
            stream = self._open_output_stream(sr)
            with stream:
                self.current_stream = stream
                for graphemes, phonemes, audio in self._pipeline(
                    text, voice=self.voice, speed=self.speed
                ):
                    if self.stop_event.is_set():
                        break
                    if audio is None or len(audio) == 0:
                        continue
                    if first_chunk:
                        latency_ms = (time.perf_counter() - start_time) * 1000
                        logger.info("TTS first chunk latency: %.0fms", latency_ms)
                        first_chunk = False

                    # Convert to float32 numpy if needed and stream in sub-chunks
                    samples = np.asarray(audio, dtype=np.float32)
                    chunk_size = 4800  # 200ms at 24kHz
                    for i in range(0, len(samples), chunk_size):
                        if self.stop_event.is_set():
                            break
                        stream.write(samples[i:i + chunk_size])

                try:
                    stream.stop()
                except Exception as e:
                    logger.debug("Stream stop failed: %s", e)
        except Exception as e:
            logger.error("Kokoro playback failed: %s", e)
        finally:
            self.current_stream = None

    def _open_output_stream(self, samplerate: int, retries: int = 2) -> sd.OutputStream:
        """Open an output stream with retry on ALSA errors.

        After stream.abort(), ALSA may need a moment to release resources.
        Retry with a short delay handles this race condition.
        """
        last_err = None
        for attempt in range(retries + 1):
            try:
                return sd.OutputStream(
                    samplerate=samplerate, channels=1, dtype="float32",
                )
            except Exception as e:
                last_err = e
                if attempt < retries:
                    logger.debug(
                        "OutputStream open failed (attempt %d/%d): %s, retrying...",
                        attempt + 1, retries + 1, e,
                    )
                    time.sleep(0.2)
        raise last_err

    def stop(self):
        """Stop current playback."""
        self.stop_event.set()
        try:
            if self.current_stream is not None:
                try:
                    self.current_stream.abort()
                except Exception as e:
                    logger.debug("Stream abort failed: %s", e)
            sd.stop()
            # Brief pause lets ALSA release the device so the next
            # OutputStream open doesn't hit PaErrorCode -9999.
            time.sleep(0.1)
        except Exception as e:
            logger.debug("sd.stop() failed: %s", e)
