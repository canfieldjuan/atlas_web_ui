"""
Post-call transcription and data extraction pipeline.

After a call ends and SignalWire produces a recording:
1. Download recording WAV from SignalWire
2. Transcribe via local ASR server (Nemotron)
3. Extract structured data via LLM (qwen3:14b)
4. Store results in call_transcripts table
5. Look up or create CRM contact, link call transcript
6. Push ntfy notification with summary
"""

import asyncio
import json
import logging
import re
from typing import Optional
from uuid import UUID

import httpx

from ..config import settings
from ..skills import get_skill_registry
from ..storage.repositories.call_transcript import get_call_transcript_repo

logger = logging.getLogger("atlas.comms.call_intelligence")


async def process_call_recording(
    call_sid: str,
    recording_url: str,
    from_number: str,
    to_number: str,
    context_id: str,
    duration_seconds: int,
    business_context=None,
) -> None:
    """
    Process a completed call recording through the intelligence pipeline.

    Triggered by SignalWire's recording-status webhook after a call ends.
    Downloads the recording, transcribes it, extracts structured data,
    stores everything in the DB, and sends a notification.

    Each step is wrapped in try/except to fail-open -- partial results
    are better than no results.
    """
    cfg = settings.call_intelligence

    logger.info(
        "Call intelligence pipeline started: call=%s url=%s from=%s to=%s duration=%ds enabled=%s",
        call_sid, recording_url, from_number, to_number, duration_seconds, cfg.enabled,
    )

    if not cfg.enabled:
        logger.info("Call intelligence disabled, skipping %s", call_sid)
        return

    if duration_seconds < cfg.min_duration_seconds:
        logger.info(
            "Call %s too short (%ds < %ds), skipping",
            call_sid, duration_seconds, cfg.min_duration_seconds,
        )
        return

    repo = get_call_transcript_repo()

    # Step 1: Create DB row
    try:
        record = await repo.create(
            call_sid=call_sid,
            from_number=from_number,
            to_number=to_number,
            context_id=context_id,
            duration=duration_seconds,
        )
        transcript_id = record["id"]
        logger.info("Step 1/6 OK: DB record created id=%s for call %s", transcript_id, call_sid)
    except Exception as e:
        logger.error("Step 1/6 FAIL: DB record creation for %s: %s", call_sid, e)
        return

    # Step 2: Download recording from SignalWire
    try:
        await repo.update_status(transcript_id, "transcribing")
        audio_bytes = await _download_recording(recording_url)
        logger.info("Step 2/6 OK: Downloaded recording %d bytes for %s", len(audio_bytes), call_sid)
    except Exception as e:
        logger.error("Step 2/6 FAIL: Download for %s: %s", call_sid, e)
        await _safe_update_status(repo, transcript_id, "error", f"Download: {e}")
        return

    # Step 3: Transcribe
    try:
        transcript = await _transcribe_audio(audio_bytes)
        if not transcript:
            logger.info("Step 3/6: Empty transcript for %s, marking ready", call_sid)
            await repo.update_transcript(transcript_id, "")
            await repo.update_extraction(
                transcript_id,
                summary="No speech detected",
                extracted_data={},
                proposed_actions=[],
            )
            await repo.update_status(transcript_id, "ready")
            return
        await repo.update_transcript(transcript_id, transcript)
        logger.info("Step 3/6 OK: Transcribed %d chars for %s", len(transcript), call_sid)
    except Exception as e:
        logger.error("Step 3/6 FAIL: Transcription for %s: %s", call_sid, e)
        await _safe_update_status(repo, transcript_id, "error", f"Transcription: {e}")
        return

    # Step 4: LLM extraction
    try:
        await repo.update_status(transcript_id, "extracting")
        summary, extracted_data, proposed_actions = await _extract_call_data(
            transcript, business_context,
        )
        await repo.update_extraction(transcript_id, summary, extracted_data, proposed_actions)
        await repo.update_status(transcript_id, "ready")
        logger.info(
            "Step 4/6 OK: Extracted data for %s: summary=%s actions=%d",
            call_sid, summary[:80], len(proposed_actions),
        )
    except Exception as e:
        logger.error("Step 4/6 FAIL: Extraction for %s: %s", call_sid, e)
        await _safe_update_status(repo, transcript_id, "error", f"Extraction: {e}")
        return

    # Step 5: Link to CRM contact
    contact_id = None
    try:
        contact_id = await _link_to_crm(
            repo, transcript_id, call_sid,
            from_number, context_id, extracted_data, summary,
        )
        if contact_id:
            logger.info("Step 5/6 OK: Linked call %s to contact %s", call_sid, contact_id)
        else:
            logger.info("Step 5/6 OK: No CRM link created for %s (insufficient data)", call_sid)
    except Exception as e:
        logger.error("Step 5/6 FAIL: CRM link for %s: %s", call_sid, e)

    # Step 6: Notify
    try:
        await _notify_call_summary(
            repo, transcript_id, call_sid,
            from_number, duration_seconds,
            summary, extracted_data, proposed_actions,
            business_context,
        )
        logger.info("Step 6/6 OK: Notification sent for %s", call_sid)
    except Exception as e:
        logger.error("Step 6/6 FAIL: Notification for %s: %s", call_sid, e)


async def _download_recording(recording_url: str) -> bytes:
    """Download a call recording from SignalWire.

    SignalWire recording URLs require HTTP Basic auth with
    project_id and api_token. Appends .wav to get WAV format.
    """
    from ..comms import comms_settings

    # Ensure we request WAV format
    url = recording_url
    if not url.endswith(".wav"):
        url = url.rstrip("/") + ".wav"

    auth = None
    # REST-initiated recordings live under the Twilio-compatible API, which
    # authenticates with project_id + api_token (the same creds the SDK uses).
    # Fabric API credentials (account_sid + recording_token) are for a
    # different API surface and will 401 on REST recording URLs.
    account_sid = comms_settings.signalwire_project_id or comms_settings.signalwire_account_sid
    recording_token = comms_settings.signalwire_api_token or comms_settings.signalwire_recording_token
    if account_sid and recording_token:
        auth = httpx.BasicAuth(account_sid, recording_token)
    logger.info("Recording download: url=%s auth_user=%s", url, account_sid[:12] + "..." if account_sid else "none")

    cfg = settings.call_intelligence
    async with httpx.AsyncClient(timeout=cfg.asr_timeout, follow_redirects=True) as client:
        resp = await client.get(url, auth=auth)
        if resp.status_code == 401 and auth:
            # First credential set failed; try the other one.
            alt_sid = comms_settings.signalwire_account_sid or ""
            alt_tok = comms_settings.signalwire_recording_token or ""
            if alt_sid and alt_tok and alt_sid != account_sid:
                logger.info("Recording download 401, retrying with account_sid auth")
                alt_auth = httpx.BasicAuth(alt_sid, alt_tok)
                resp = await client.get(url, auth=alt_auth)
        logger.info("Recording download response: status=%d size=%d", resp.status_code, len(resp.content))
        resp.raise_for_status()
        return resp.content


async def _transcribe_audio(audio_bytes: bytes) -> Optional[str]:
    """Send audio to the local ASR server and return transcript text.

    The ASR server (Nemotron) accepts WAV files directly.
    """
    cfg = settings.call_intelligence

    async with httpx.AsyncClient(timeout=cfg.asr_timeout) as client:
        resp = await client.post(
            cfg.asr_url,
            files={"file": ("call.wav", audio_bytes, "audio/wav")},
        )
        resp.raise_for_status()
        result = resp.json()

    text = result.get("text", "").strip()
    return text if text else None


async def _extract_call_data(
    transcript: str,
    business_context=None,
) -> tuple[str, dict, list]:
    """Use LLM to extract structured data from the transcript."""
    from ..services.protocols import Message
    from ..services.llm_router import get_triage_llm
    from ..services import llm_registry

    # Prefer Haiku (no local VRAM, always available) over local qwen3
    llm = get_triage_llm() or llm_registry.get_active()
    if not llm:
        logger.warning("No active LLM for call extraction")
        return transcript[:200], {}, []

    # Build business context string
    ctx_parts = []
    if business_context:
        if hasattr(business_context, "name"):
            ctx_parts.append(f"Business: {business_context.name}")
        if hasattr(business_context, "business_type") and business_context.business_type:
            ctx_parts.append(f"Type: {business_context.business_type}")
        if hasattr(business_context, "services") and business_context.services:
            ctx_parts.append(f"Services: {', '.join(business_context.services)}")
    business_context_str = "\n".join(ctx_parts) if ctx_parts else "General business"

    # Load skill prompt
    skill = get_skill_registry().get("call/call_extraction")
    if skill:
        system_prompt = skill.content.replace("{business_context}", business_context_str)
    else:
        system_prompt = (
            "Extract customer info, intent, and proposed actions from this call transcript. "
            f"Business context: {business_context_str}\n"
            "Return a JSON object with: customer_name, customer_phone, customer_email, "
            "intent, services_mentioned, address, preferred_date, preferred_time, "
            "urgency, follow_up_needed, notes. "
            "Then a blank line and a JSON array of proposed actions."
        )

    cfg = settings.call_intelligence
    messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=transcript),
    ]

    loop = asyncio.get_running_loop()
    result = await asyncio.wait_for(
        loop.run_in_executor(
            None,
            lambda: llm.chat(
                messages=messages,
                max_tokens=cfg.llm_max_tokens,
                temperature=cfg.llm_temperature,
            ),
        ),
        timeout=30.0,
    )

    text = result.get("response", "").strip()
    if not text:
        return transcript[:200], {}, []

    # Strip <think> tags (Qwen3 models)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    return _parse_extraction(text, transcript)


def _parse_extraction(text: str, transcript: str) -> tuple[str, dict, list]:
    """Parse LLM output into (summary, extracted_data, proposed_actions)."""
    extracted_data = {}
    proposed_actions = []

    # Find the first JSON object
    obj_start = text.find("{")
    obj_end = _find_matching_brace(text, obj_start) if obj_start >= 0 else -1

    if obj_start >= 0 and obj_end > obj_start:
        try:
            extracted_data = json.loads(text[obj_start : obj_end + 1])
        except json.JSONDecodeError:
            pass

        # Find the actions array after the object
        remainder = text[obj_end + 1 :].strip()
        arr_start = remainder.find("[")
        arr_end = remainder.rfind("]")

        if arr_start >= 0 and arr_end > arr_start:
            try:
                proposed_actions = json.loads(remainder[arr_start : arr_end + 1])
            except json.JSONDecodeError:
                pass

    # Build summary from extracted data
    parts = []
    name = extracted_data.get("customer_name")
    if name:
        parts.append(f"Customer: {name}")
    intent = extracted_data.get("intent")
    if intent:
        parts.append(f"Intent: {intent.replace('_', ' ')}")
    services = extracted_data.get("services_mentioned")
    if services:
        parts.append(f"Services: {', '.join(services)}")
    notes = extracted_data.get("notes")
    if notes:
        parts.append(notes)

    summary = ". ".join(parts) if parts else transcript[:200]

    return summary, extracted_data, proposed_actions


def _find_matching_brace(text: str, start: int) -> int:
    """Find the closing brace matching the opening brace at `start`."""
    if start < 0 or start >= len(text) or text[start] != "{":
        return -1
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1


async def _link_to_crm(
    repo,
    transcript_id: UUID,
    call_sid: str,
    from_number: str,
    context_id: str,
    extracted_data: dict,
    summary: str,
) -> Optional[str]:
    """Look up or create a CRM contact from call extraction data.

    Returns the contact_id if linked, or None if there wasn't enough data.
    Fail-open: errors are logged but never block the pipeline.
    """
    from ..services.crm_provider import get_crm_provider
    from ..storage.database import get_db_pool

    pool = get_db_pool()
    if not pool.is_initialized:
        logger.debug("CRM link skipped: DB pool not initialized")
        return None

    phone = extracted_data.get("customer_phone") or from_number
    email = extracted_data.get("customer_email")
    name = extracted_data.get("customer_name")

    if not phone and not email:
        return None

    crm = get_crm_provider()

    # Try to find an existing contact by phone first, then email
    existing = None
    if phone:
        results = await crm.search_contacts(phone=phone)
        if results:
            existing = results[0]
    if not existing and email:
        results = await crm.search_contacts(email=email)
        if results:
            existing = results[0]

    if existing:
        contact_id = str(existing["id"])
        # Update name if we learned it from this call and it was missing
        if name and not existing.get("full_name"):
            parts = name.split(None, 1)
            await crm.update_contact(contact_id, {
                "full_name": name,
                "first_name": parts[0],
                "last_name": parts[1] if len(parts) > 1 else None,
            })
    else:
        # Create a new contact from call data
        parts = name.split(None, 1) if name else []
        contact = await crm.create_contact({
            "full_name": name or phone or "Unknown Caller",
            "first_name": parts[0] if parts else None,
            "last_name": parts[1] if len(parts) > 1 else None,
            "phone": phone,
            "email": email,
            "address": extracted_data.get("address"),
            "business_context_id": context_id,
            "contact_type": "customer",
            "source": "phone_call",
            "source_ref": str(transcript_id),
        })
        contact_id = str(contact["id"])

    # Link the call transcript to the contact
    await repo.link_contact(transcript_id, contact_id)

    # Log the interaction
    await crm.log_interaction(
        contact_id=contact_id,
        interaction_type="call",
        summary=summary or f"Inbound call from {from_number}",
    )

    return contact_id


async def _notify_call_summary(
    repo,
    transcript_id: UUID,
    call_sid: str,
    from_number: str,
    duration_seconds: int,
    summary: str,
    extracted_data: dict,
    proposed_actions: list,
    business_context=None,
) -> None:
    """Send ntfy notification with the call summary."""
    if not settings.call_intelligence.notify_enabled:
        return
    if not settings.alerts.ntfy_enabled:
        return
    if not settings.alerts.ntfy_url or not settings.alerts.ntfy_topic:
        return

    from ..comms import comms_settings as _comms_cfg

    ntfy_url = f"{settings.alerts.ntfy_url.rstrip('/')}/{settings.alerts.ntfy_topic}"
    api_url = _comms_cfg.webhook_base_url.rstrip("/")

    # Format duration
    mins, secs = divmod(duration_seconds, 60)
    dur_str = f"{mins}m {secs}s" if mins else f"{secs}s"

    # Build message
    biz_name = business_context.name if business_context and hasattr(business_context, "name") else "Business"
    lines = [f"From: {from_number} ({dur_str})"]

    name = extracted_data.get("customer_name")
    if name:
        lines.append(f"Customer: {name}")

    phone = extracted_data.get("customer_phone")
    if phone:
        lines.append(f"Phone: {phone}")

    intent = extracted_data.get("intent")
    if intent:
        lines.append(f"Intent: {intent.replace('_', ' ').title()}")

    services = extracted_data.get("services_mentioned")
    if services:
        lines.append(f"Services: {', '.join(services)}")

    address = extracted_data.get("address")
    if address:
        lines.append(f"Address: {address}")

    date = extracted_data.get("preferred_date")
    time = extracted_data.get("preferred_time")
    if date and time:
        lines.append(f"Requested: {date} {time}")
    elif date:
        lines.append(f"Requested: {date}")
    elif time:
        lines.append(f"Requested: {time}")

    notes = extracted_data.get("notes")
    if notes:
        lines.append(f"Notes: {notes}")

    if proposed_actions:
        actions_str = "; ".join(
            a.get("label", a.get("type", "")) for a in proposed_actions
            if a.get("type") != "none"
        )
        if actions_str:
            lines.append(f"Follow-up: {actions_str}")

    message = "\n".join(lines)

    # Build ntfy action buttons from proposed_actions
    action_parts = []
    for action in proposed_actions:
        atype = action.get("type", "none")
        if atype in ("book_estimate", "create_appointment", "book_appointment"):
            action_parts.append(
                f"http, Book Appointment, {api_url}/api/v1/comms/call-actions/{transcript_id}/book, "
                f"method=POST, clear=true"
            )
        elif atype in ("send_sms", "send_followup", "schedule_callback"):
            action_parts.append(
                f"http, Send SMS, {api_url}/api/v1/comms/call-actions/{transcript_id}/sms, "
                f"method=POST, clear=true"
            )
    action_parts.append(
        f"view, View Transcript, {api_url}/api/v1/comms/call-actions/{transcript_id}/view"
    )
    actions_header = "; ".join(action_parts)

    headers = {
        "Title": f"{biz_name}: Call Summary",
        "Priority": "default",
        "Tags": "phone,call",
        "Actions": actions_header,
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(ntfy_url, content=message, headers=headers)
            resp.raise_for_status()
        await repo.mark_notified(transcript_id)
        await repo.update_status(transcript_id, "notified")
        logger.info("Call summary notification sent for %s", call_sid)
    except Exception as e:
        logger.warning("Failed to send call summary notification: %s", e)


async def _safe_update_status(repo, transcript_id, status, error_message=None):
    """Update status without raising on failure."""
    try:
        await repo.update_status(transcript_id, status, error_message)
    except Exception as e:
        logger.warning("Failed to update status for %s: %s", transcript_id, e)
