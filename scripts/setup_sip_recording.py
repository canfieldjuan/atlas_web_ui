"""
Configure SignalWire SIP endpoint outbound call recording.

Creates a SWML script resource that records every outbound call from Zoiper
and connects it to the destination. Assigns that SWML script as the outbound
call handler on the 'atlas' SIP endpoint.

Usage:
    python scripts/setup_sip_recording.py [--dry-run]

Reads from .env:
    ATLAS_COMMS_SIGNALWIRE_PROJECT_ID   - PT-format project ID (auth username)
    ATLAS_COMMS_SIGNALWIRE_API_TOKEN    - API token (auth password)
    ATLAS_COMMS_SIGNALWIRE_SPACE        - space name (e.g. finetunelab)
    ATLAS_COMMS_WEBHOOK_BASE_URL        - base URL for recording-status callback
    ATLAS_COMMS_SIP_USERNAME            - SIP endpoint username to update (default: atlas)
"""

import argparse
import json
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

import os

SPACE = os.environ.get("ATLAS_COMMS_SIGNALWIRE_SPACE", "")
# Fabric API uses account SID (UUID) + recording token for Basic auth
ACCOUNT_SID = os.environ.get("ATLAS_COMMS_SIGNALWIRE_ACCOUNT_SID", "")
RECORDING_TOKEN = os.environ.get("ATLAS_COMMS_SIGNALWIRE_RECORDING_TOKEN", "")
WEBHOOK_BASE_URL = os.environ.get("ATLAS_COMMS_WEBHOOK_BASE_URL", "").rstrip("/")
SIP_USERNAME = os.environ.get("ATLAS_COMMS_SIP_USERNAME", "atlas")

FABRIC_BASE = f"https://{SPACE}.signalwire.com/api/fabric"
RECORDING_STATUS_URL = f"{WEBHOOK_BASE_URL}/api/v1/comms/voice/recording-status"
SWML_SCRIPT_NAME = "Atlas Outbound Call Recorder"


def _check_config():
    missing = [k for k, v in {
        "ATLAS_COMMS_SIGNALWIRE_SPACE": SPACE,
        "ATLAS_COMMS_SIGNALWIRE_ACCOUNT_SID": ACCOUNT_SID,
        "ATLAS_COMMS_SIGNALWIRE_RECORDING_TOKEN": RECORDING_TOKEN,
        "ATLAS_COMMS_WEBHOOK_BASE_URL": WEBHOOK_BASE_URL,
    }.items() if not v]
    if missing:
        print(f"ERROR: Missing required env vars: {', '.join(missing)}")
        sys.exit(1)


def _build_swml_contents() -> str:
    """Build the SWML script JSON string for outbound call recording.

    record_call runs in the background (non-blocking), then connect
    passes the call through to the originally dialed destination.
    stereo=true captures both legs in separate channels.
    status_url (not status_callback) is the correct SWML field name.
    """
    swml = {
        "version": "1.0.0",
        "sections": {
            "main": [
                {
                    "record_call": {
                        "format": "wav",
                        "stereo": True,
                        "status_url": RECORDING_STATUS_URL,
                    }
                },
                {
                    "connect": {
                        "to": "%{to}",
                    }
                },
            ]
        },
    }
    return json.dumps(swml)


def _find_existing_script(client: httpx.Client) -> str | None:
    """Return the resource ID of an existing script with our name, or None."""
    resp = client.get(f"{FABRIC_BASE}/resources/swml_scripts")
    resp.raise_for_status()
    for item in resp.json().get("data", []):
        if item.get("display_name") == SWML_SCRIPT_NAME:
            return item["id"]
    return None


def _create_swml_script(client: httpx.Client, contents: str) -> str:
    """Create the SWML script resource and return its resource ID."""
    payload = {"name": SWML_SCRIPT_NAME, "contents": contents}
    resp = client.post(f"{FABRIC_BASE}/resources/swml_scripts", json=payload)
    resp.raise_for_status()
    resource_id = resp.json()["id"]
    print(f"  Created SWML script resource: {resource_id}")
    return resource_id


def _update_swml_script(client: httpx.Client, resource_id: str, contents: str) -> None:
    """Update an existing SWML script resource contents."""
    resp = client.put(
        f"{FABRIC_BASE}/resources/swml_scripts/{resource_id}",
        json={"name": SWML_SCRIPT_NAME, "contents": contents},
    )
    resp.raise_for_status()
    print(f"  Updated SWML script resource: {resource_id}")


def _find_sip_endpoint(client: httpx.Client) -> tuple[str, dict]:
    """Return (resource_id, sip_endpoint_dict) for the target SIP username."""
    resp = client.get(f"{FABRIC_BASE}/resources/sip_endpoints")
    resp.raise_for_status()
    for item in resp.json().get("data", []):
        sip = item.get("sip_endpoint", {})
        if sip.get("username") == SIP_USERNAME:
            return item["id"], sip
    raise RuntimeError(
        f"SIP endpoint with username '{SIP_USERNAME}' not found. "
        f"Check ATLAS_COMMS_SIP_USERNAME or create the endpoint first."
    )


def _update_sip_endpoint(client: httpx.Client, endpoint_id: str, script_id: str) -> None:
    """Set the SIP endpoint's outbound call handler to the SWML script."""
    payload = {
        "call_handler": "resource",
        "calling_handler_resource_id": script_id,
    }
    resp = client.put(f"{FABRIC_BASE}/resources/sip_endpoints/{endpoint_id}", json=payload)
    resp.raise_for_status()
    print(f"  Updated SIP endpoint {endpoint_id} -> call_handler=resource")


def main():
    parser = argparse.ArgumentParser(description="Wire SIP outbound call recording")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without executing")
    args = parser.parse_args()

    _check_config()

    print(f"Space:            {SPACE}")
    print(f"SIP username:     {SIP_USERNAME}")
    print(f"Recording URL:    {RECORDING_STATUS_URL}")
    print(f"SWML name:        {SWML_SCRIPT_NAME}")
    print()

    swml_contents = _build_swml_contents()
    print("SWML script contents:")
    print(json.dumps(json.loads(swml_contents), indent=2))
    print()

    if args.dry_run:
        print("DRY RUN - no changes made.")
        return

    with httpx.Client(auth=(ACCOUNT_SID, RECORDING_TOKEN), timeout=30.0) as client:
        print("Step 1: Check for existing SWML script...")
        existing_id = _find_existing_script(client)
        if existing_id:
            print(f"  Found existing script: {existing_id} (updating)")
            _update_swml_script(client, existing_id, swml_contents)
            script_id = existing_id
        else:
            print("  Creating new SWML script resource...")
            script_id = _create_swml_script(client, swml_contents)

        print(f"\nStep 2: Find SIP endpoint '{SIP_USERNAME}'...")
        endpoint_id, sip_info = _find_sip_endpoint(client)
        current_handler = sip_info.get("call_handler", "unknown")
        print(f"  Found: {endpoint_id} (current handler: {current_handler})")

        print(f"\nStep 3: Assign SWML script as outbound call handler...")
        _update_sip_endpoint(client, endpoint_id, script_id)

        print("\nDone. Outbound calls from Zoiper will now be recorded.")
        print("Recordings fire the same webhook as inbound calls.")


if __name__ == "__main__":
    main()
