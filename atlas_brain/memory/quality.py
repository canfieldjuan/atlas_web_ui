"""
Memory quality signal detection.

Detects two signals per user turn:
- Correction: regex patterns indicating the user is correcting Atlas
- Repetition: cosine similarity against cached session turn embeddings

Signals are stored in the existing metadata JSONB column on conversation_turns.
Fail-open: if the embedding model isn't loaded, only correction detection runs;
if anything throws, the turn stores normally without signals.
"""

import logging
import re
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger("atlas.memory.quality")

# ---------------------------------------------------------------------------
# Correction patterns -- compiled at module level (~0.05ms to match)
# ---------------------------------------------------------------------------

_CORRECTION_PATTERNS: dict[str, re.Pattern] = {
    "no_i_meant": re.compile(
        r"\bno[,.]?\s+I\s+meant\b", re.IGNORECASE,
    ),
    "thats_wrong": re.compile(
        r"\bthat'?s\s+(?:wrong|incorrect|not\s+(?:right|correct))\b", re.IGNORECASE,
    ),
    "already_told": re.compile(
        r"\bI\s+(?:already|just)\s+(?:told|said|mentioned)\b", re.IGNORECASE,
    ),
    "not_what_i_asked": re.compile(
        r"\b(?:that'?s\s+)?not\s+what\s+I\s+(?:asked|said|meant)\b", re.IGNORECASE,
    ),
    "youre_wrong": re.compile(
        r"\byou(?:'re| are)\s+(?:wrong|mistaken|confused)\b", re.IGNORECASE,
    ),
    "didnt_say": re.compile(
        r"\bI\s+didn'?t\s+say\b", re.IGNORECASE,
    ),
    "let_me_rephrase": re.compile(
        r"\blet\s+me\s+(?:rephrase|clarify|repeat)\b", re.IGNORECASE,
    ),
    "no_not_that": re.compile(
        r"\bno[,.]?\s+not\s+that\b", re.IGNORECASE,
    ),
    "misunderstood": re.compile(
        r"\b(?:you\s+)?misunderstood\b", re.IGNORECASE,
    ),
    "keep_telling": re.compile(
        r"\bI\s+keep\s+(?:telling|saying|asking)\b", re.IGNORECASE,
    ),
}

# Default values (used when not overridden via constructor)
DEFAULT_REPETITION_THRESHOLD = 0.85
DEFAULT_MAX_SESSION_TURNS = 20
DEFAULT_MAX_SESSIONS = 500


# ---------------------------------------------------------------------------
# QualitySignal dataclass
# ---------------------------------------------------------------------------

@dataclass
class QualitySignal:
    """Quality signal detected for a user turn."""

    correction: bool = False
    repetition: bool = False
    repetition_of_turn_id: Optional[str] = None
    repetition_similarity: Optional[float] = None
    correction_pattern: Optional[str] = None
    detection_ms: float = 0.0

    def to_metadata(self) -> dict:
        """Convert to metadata dict for JSONB storage.

        Returns dict with "memory_quality" key, or empty dict if no signals.
        """
        if not self.correction and not self.repetition:
            return {}

        inner: dict = {"detection_ms": round(self.detection_ms, 2)}

        if self.correction:
            inner["correction"] = True
            if self.correction_pattern:
                inner["correction_pattern"] = self.correction_pattern

        if self.repetition:
            inner["repetition"] = True
            if self.repetition_of_turn_id:
                inner["repetition_of"] = self.repetition_of_turn_id
            if self.repetition_similarity is not None:
                inner["repetition_similarity"] = round(self.repetition_similarity, 3)

        return {"memory_quality": inner}


# ---------------------------------------------------------------------------
# Internal session cache record
# ---------------------------------------------------------------------------

@dataclass
class _SessionTurnRecord:
    turn_id: str
    embedding: np.ndarray


# ---------------------------------------------------------------------------
# MemoryQualityDetector
# ---------------------------------------------------------------------------

class MemoryQualityDetector:
    """Detects correction and repetition signals in user turns."""

    def __init__(
        self,
        repetition_threshold: float = DEFAULT_REPETITION_THRESHOLD,
        max_session_turns: int = DEFAULT_MAX_SESSION_TURNS,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
    ) -> None:
        self.repetition_threshold = repetition_threshold
        self.max_session_turns = max_session_turns
        self.max_sessions = max_sessions
        self._session_turns: dict[str, list[_SessionTurnRecord]] = {}
        # Track access order for LRU eviction
        self._session_order: list[str] = []

    def _get_embedder(self):
        """Lazy-load the shared embedding service (already used by intent router)."""
        try:
            from ..services.embedding.sentence_transformer import get_embedding_service
            svc = get_embedding_service()
            if svc.is_loaded:
                return svc
        except Exception:
            pass
        return None

    def detect(
        self,
        session_id: str,
        user_content: str,
        turn_type: str = "conversation",
        turn_id: Optional[str] = None,
    ) -> QualitySignal:
        """Detect quality signals for a user turn.

        Args:
            session_id: Current session ID
            user_content: The user's message text
            turn_type: "conversation" or "command" -- commands are skipped
            turn_id: Optional turn UUID string

        Returns:
            QualitySignal with detected signals
        """
        start = time.perf_counter()
        signal = QualitySignal()

        # Skip command turns
        if turn_type == "command" or not user_content:
            signal.detection_ms = (time.perf_counter() - start) * 1000
            return signal

        # 1. Correction detection (regex, ~0.05ms)
        for pattern_name, pattern in _CORRECTION_PATTERNS.items():
            if pattern.search(user_content):
                signal.correction = True
                signal.correction_pattern = pattern_name
                break

        # 2. Repetition detection (embedding similarity, ~1-3ms)
        embedder = self._get_embedder()
        if embedder is not None:
            try:
                embedding = embedder.embed(user_content)

                # Check against cached session turns
                session_cache = self._session_turns.get(session_id, [])
                for record in session_cache:
                    sim = float(np.dot(embedding, record.embedding))
                    if sim >= self.repetition_threshold:
                        signal.repetition = True
                        signal.repetition_of_turn_id = record.turn_id
                        signal.repetition_similarity = sim
                        break

                # Cache this turn's embedding
                self._record_turn(session_id, turn_id or "", embedding)

            except Exception as e:
                logger.debug("Repetition detection failed: %s", e)

        signal.detection_ms = (time.perf_counter() - start) * 1000
        return signal

    def _record_turn(
        self,
        session_id: str,
        turn_id: str,
        embedding: np.ndarray,
    ) -> None:
        """Cache a turn embedding for future repetition checks."""
        if session_id not in self._session_turns:
            self._session_turns[session_id] = []
            self._session_order.append(session_id)
            self._evict_if_needed()

        # Move session to end of access order
        elif session_id in self._session_order:
            self._session_order.remove(session_id)
            self._session_order.append(session_id)

        self._session_turns[session_id].append(
            _SessionTurnRecord(turn_id=turn_id, embedding=embedding)
        )

        # Trim turns within session
        if len(self._session_turns[session_id]) > self.max_session_turns:
            self._session_turns[session_id] = (
                self._session_turns[session_id][-self.max_session_turns:]
            )

    def _evict_if_needed(self) -> None:
        """Evict oldest sessions when cache exceeds max_sessions."""
        while len(self._session_turns) > self.max_sessions and self._session_order:
            oldest = self._session_order.pop(0)
            self._session_turns.pop(oldest, None)

    def clear_session(self, session_id: str) -> None:
        """Evict a session from the turn cache."""
        self._session_turns.pop(session_id, None)
        try:
            self._session_order.remove(session_id)
        except ValueError:
            pass


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_quality_detector: Optional[MemoryQualityDetector] = None


def get_quality_detector() -> MemoryQualityDetector:
    """Get the global MemoryQualityDetector singleton."""
    global _quality_detector
    if _quality_detector is None:
        _quality_detector = MemoryQualityDetector()
    return _quality_detector
