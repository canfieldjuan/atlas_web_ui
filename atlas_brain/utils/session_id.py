"""Session ID normalization utilities.

Converts arbitrary session ID strings (hex hashes, telephony SIDs,
etc.) into valid UUID strings that are safe for the PostgreSQL UUID
columns in sessions / conversation_turns. Also ensures a matching
row exists in the sessions table so FK constraints are satisfied.
"""

import logging
from uuid import UUID, uuid4, uuid5

logger = logging.getLogger("atlas.utils.session_id")

# Fixed namespace for deterministic UUID5 derivation.
# Any non-UUID string is hashed with this namespace so the same
# input always produces the same session UUID.
_ATLAS_SESSION_NS = UUID("f47ac10b-58cc-4372-a567-0d02b2c3d479")


def normalize_session_id(raw: str | None) -> str:
    """Convert an arbitrary session ID string into a valid UUID string.

    - None / empty  -> new random UUID4
    - Already valid UUID string -> returned as-is (lower-case canonical)
    - Anything else -> deterministic UUID5 so the same input always
      maps to the same session UUID.
    """
    if not raw:
        return str(uuid4())

    # Fast path: already a well-formed UUID
    try:
        return str(UUID(raw))
    except (ValueError, AttributeError):
        pass

    # Deterministic mapping for non-UUID strings (sha256 hashes, SIDs, etc.)
    return str(uuid5(_ATLAS_SESSION_NS, raw))


async def ensure_session_row(session_id: str) -> None:
    """Create a sessions row if one does not already exist.

    Uses ON CONFLICT DO NOTHING so it is safe to call repeatedly.
    All columns except ``id`` have defaults or accept NULL, so a
    bare INSERT with just the id is sufficient.
    """
    from ..storage.database import get_db_pool

    try:
        pool = get_db_pool()
        if not pool.is_initialized:
            return
        await pool.execute(
            "INSERT INTO sessions (id) VALUES ($1) ON CONFLICT (id) DO NOTHING",
            UUID(session_id),
        )
    except Exception as exc:
        logger.warning("Could not ensure session row for %s: %s", session_id, exc)
