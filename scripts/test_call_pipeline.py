"""
Manual pipeline test — bypasses download and ASR, feeds a transcript directly.

Usage:
    python scripts/test_call_pipeline.py

Tests: LLM extraction, DB storage, and ntfy notification with a fake call.
Set SKIP_NOTIFY=1 to suppress the ntfy push during testing.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Fake call metadata
FAKE_CALL_SID = "TEST-CALL-DRY-RUN-001"
FROM_NUMBER = "+16185559876"
TO_NUMBER = "+16183683696"
CONTEXT_ID = "effingham-maids"
DURATION_SECONDS = 142

# Simulated transcript of a two-party conversation
FAKE_TRANSCRIPT = """
Hi, thank you for calling Effingham Maids, how can I help you today?
Yeah hi, my name is Sarah Johnson. I'm looking to get a deep cleaning done for my house.
Sure, I can help you with that. What's your address?
It's 412 Lakewood Drive in Effingham.
Great. And when were you thinking? Do you have a preferred date or time?
Would next Thursday morning work? Like around 9 or 10 AM?
Let me check — yes, Thursday morning is available. We have a 9 AM slot open.
That works perfectly. And is it just a standard deep clean or do you need anything specific?
Just a deep clean. Maybe extra attention to the kitchen and bathrooms.
Got it. And the best number to reach you at for confirmation?
This number is fine, 618-555-9876.
Perfect. I'll get you booked in for Thursday at 9 AM. You'll get a confirmation text shortly.
Thank you so much, I really appreciate it.
Of course, see you Thursday!
""".strip()


async def main():
    print("=" * 60)
    print("CALL INTELLIGENCE PIPELINE — DRY RUN")
    print("=" * 60)
    print(f"Call SID  : {FAKE_CALL_SID}")
    print(f"From      : {FROM_NUMBER}")
    print(f"Duration  : {DURATION_SECONDS}s")
    print()
    print("TRANSCRIPT:")
    print("-" * 40)
    print(FAKE_TRANSCRIPT)
    print("-" * 40)
    print()

    from atlas_brain.config import settings
    from atlas_brain.comms.call_intelligence import _extract_call_data, _notify_call_summary
    from atlas_brain.storage.repositories.call_transcript import get_call_transcript_repo
    from atlas_brain.comms.context import get_context_router

    # Resolve business context
    ctx_router = get_context_router()
    biz_ctx = ctx_router.get_context(CONTEXT_ID)
    if biz_ctx:
        print(f"Business  : {biz_ctx.name}")
    else:
        print(f"Business  : (context '{CONTEXT_ID}' not found, using None)")
    print()

    # Step 1: LLM extraction
    print("Step 1: Running LLM extraction...")
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
            if v:
                print(f"  {k}: {v}")
        print()
        print("PROPOSED ACTIONS:")
        for action in proposed_actions:
            print(f"  [{action.get('type')}] {action.get('label')} — {action.get('reason')}")
        if not proposed_actions:
            print("  (none)")
        print()
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 2: Store in DB
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
        print(f"  Stored with ID: {tid}")
        print()
    except Exception as e:
        print(f"  DB ERROR (non-fatal): {e}")
        print("  (continuing without DB storage)")
        tid = None
        print()

    # Step 3: ntfy notification
    skip_notify = os.environ.get("SKIP_NOTIFY", "").lower() in ("1", "true", "yes")
    if skip_notify:
        print("Step 3: Skipping ntfy (SKIP_NOTIFY=1)")
    else:
        print("Step 3: Sending ntfy notification...")
        try:
            from unittest.mock import AsyncMock
            fake_repo = AsyncMock()
            if tid:
                fake_repo.mark_notified = AsyncMock()

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
            print("  Notification sent — check your ntfy app.")
        except Exception as e:
            print(f"  NOTIFY ERROR: {e}")

    print()
    print("=" * 60)
    print("DRY RUN COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
