"""
Manual pipeline test -- bypasses download and ASR, feeds a transcript directly.

Usage:
    python scripts/test_call_pipeline.py

Tests: LLM extraction, DB storage, and ntfy notification with a fake call.
Set SKIP_NOTIFY=1 to suppress the ntfy push during testing.
Set SKIP_DB=1 to skip database storage.
"""

import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.WARNING)  # suppress noise; show only errors

# Fake call metadata
FAKE_CALL_SID = "TEST-CALL-DRY-RUN-001"
FROM_NUMBER = "+16185559876"
TO_NUMBER = "+16183683696"
CONTEXT_ID = "effingham-maids"
DURATION_SECONDS = 142

FAKE_TRANSCRIPT = """
Hi, thank you for calling Effingham Maids, how can I help you today?
Yeah hi, my name is Sarah Johnson. I'm looking to get a deep cleaning done for my house.
Sure, I can help you with that. What's your address?
It's 412 Lakewood Drive in Effingham.
Great. And when were you thinking? Do you have a preferred date or time?
Would next Thursday morning work? Like around 9 or 10 AM?
Let me check -- yes, Thursday morning is available. We have a 9 AM slot open.
That works perfectly. And is it just a standard deep clean or do you need anything specific?
Just a deep clean. Maybe extra attention to the kitchen and bathrooms.
Got it. And the best number to reach you at for confirmation?
This number is fine, 618-555-9876.
Perfect. I'll get you booked in for Thursday at 9 AM. You'll get a confirmation text shortly.
Thank you so much, I really appreciate it.
Of course, see you Thursday!
""".strip()


def _init_llm():
    """Activate the Ollama LLM the same way main.py does at startup."""
    from atlas_brain.config import settings
    from atlas_brain.services import llm_registry

    backend = settings.llm.default_model
    if backend == "ollama":
        llm_registry.activate(
            "ollama",
            model=settings.llm.ollama_model,
            base_url=settings.llm.ollama_url,
        )
        print(f"  LLM: {settings.llm.ollama_model} via {settings.llm.ollama_url}")
    else:
        # Fallback: try ollama directly with defaults
        llm_registry.activate(
            "ollama",
            model=settings.llm.ollama_model,
            base_url=settings.llm.ollama_url,
        )
        print(f"  LLM: {settings.llm.ollama_model} (backend={backend}, forced ollama)")

    llm = llm_registry.get_active()
    if not llm:
        print("  ERROR: LLM failed to activate. Is Ollama running?")
        sys.exit(1)


async def _init_db():
    """Initialize DB pool the same way main.py does at startup."""
    from atlas_brain.storage.database import get_db_pool
    pool = get_db_pool()
    if not pool.is_initialized:
        try:
            await pool.initialize()
            print("  DB: connected")
        except Exception as e:
            print(f"  DB: unavailable ({e}) -- will skip DB step")
    else:
        print("  DB: already initialized")
    return pool


def _get_business_context():
    """Resolve the business context from the context router."""
    from atlas_brain.comms.context import get_context_router
    ctx_router = get_context_router()
    ctx = ctx_router.get_context(CONTEXT_ID)
    if ctx:
        print(f"  Context: {ctx.name}")
    else:
        print(f"  Context: '{CONTEXT_ID}' not found -- continuing without business context")
    return ctx


async def main():
    print("=" * 60)
    print("CALL INTELLIGENCE PIPELINE -- DRY RUN")
    print("=" * 60)
    print(f"Call SID  : {FAKE_CALL_SID}")
    print(f"From      : {FROM_NUMBER}")
    print(f"Duration  : {DURATION_SECONDS}s ({DURATION_SECONDS // 60}m {DURATION_SECONDS % 60}s)")
    print()

    # --- Init ---
    print("Initializing...")
    _init_llm()
    pool = await _init_db()
    biz_ctx = _get_business_context()
    print()

    print("TRANSCRIPT:")
    print("-" * 40)
    print(FAKE_TRANSCRIPT)
    print("-" * 40)
    print()

    from atlas_brain.comms.call_intelligence import _extract_call_data, _notify_call_summary
    from atlas_brain.storage.repositories.call_transcript import get_call_transcript_repo

    # --- Step 1: LLM extraction ---
    print("Step 1: LLM extraction  (this takes ~10-20s)...")
    try:
        summary, extracted_data, proposed_actions = await _extract_call_data(
            FAKE_TRANSCRIPT, biz_ctx
        )
        print()
        print("SUMMARY:")
        print(f"  {summary}")
        print()
        print("EXTRACTED DATA:")
        for k, v in extracted_data.items():
            if v not in (None, "", [], False):
                print(f"  {k}: {v}")
        if not extracted_data:
            print("  (empty -- LLM returned no structured data)")
        print()
        print("PROPOSED ACTIONS:")
        for action in proposed_actions:
            label = action.get("label", action.get("type", "?"))
            reason = action.get("reason", "")
            print(f"  [{action.get('type')}] {label} -- {reason}")
        if not proposed_actions:
            print("  (none)")
        print()
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- Step 2: DB storage ---
    skip_db = os.environ.get("SKIP_DB", "").lower() in ("1", "true", "yes")
    tid = None

    if skip_db or not pool.is_initialized:
        print("Step 2: Skipping DB (SKIP_DB=1 or DB unavailable)")
    else:
        print("Step 2: Storing in database...")
        try:
            repo = get_call_transcript_repo()
            record = await repo.create(
                call_sid=FAKE_CALL_SID,
                from_number=FROM_NUMBER,
                to_number=TO_NUMBER,
                context_id=CONTEXT_ID,
                duration=DURATION_SECONDS,
            )
            tid = record["id"]
            await repo.update_transcript(tid, FAKE_TRANSCRIPT)
            await repo.update_extraction(tid, summary, extracted_data, proposed_actions)
            await repo.update_status(tid, "ready")
            print(f"  Stored. Record ID: {tid}")
        except Exception as e:
            print(f"  DB ERROR (non-fatal): {e}")
    print()

    # --- Step 3: ntfy notification ---
    skip_notify = os.environ.get("SKIP_NOTIFY", "").lower() in ("1", "true", "yes")

    if skip_notify:
        print("Step 3: Skipping ntfy (SKIP_NOTIFY=1)")
    else:
        print("Step 3: Sending ntfy notification...")
        try:
            from unittest.mock import AsyncMock
            fake_repo = AsyncMock()
            await _notify_call_summary(
                fake_repo,
                tid,
                FAKE_CALL_SID,
                FROM_NUMBER,
                DURATION_SECONDS,
                summary,
                extracted_data,
                proposed_actions,
                biz_ctx,
            )
            print("  Sent -- check your ntfy app.")
        except Exception as e:
            print(f"  NOTIFY ERROR: {e}")
            import traceback
            traceback.print_exc()

    print()
    print("=" * 60)
    print("DRY RUN COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
