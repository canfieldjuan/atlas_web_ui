"""
Invoice number detector for text pipelines.

Extracts invoice numbers (INV-YYYY-NNNN) from call transcripts,
emails, SMS, and other text sources, then resolves them to invoice records.
"""

import logging
import re
from typing import Optional

logger = logging.getLogger("atlas.comms.invoice_detector")

# Matches INV-2026-0001 style invoice numbers (case-insensitive)
_INVOICE_PATTERN = re.compile(r"\bINV-\d{4}-\d{4,}\b", re.IGNORECASE)


def extract_invoice_numbers(text: str) -> list[str]:
    """Extract all invoice numbers from text. Returns uppercase normalized list."""
    if not text:
        return []
    matches = _INVOICE_PATTERN.findall(text)
    # Deduplicate while preserving order, normalize to uppercase
    seen = set()
    result = []
    for m in matches:
        upper = m.upper()
        if upper not in seen:
            seen.add(upper)
            result.append(upper)
    return result


async def resolve_invoices(numbers: list[str]) -> list[dict]:
    """Look up invoice records by their invoice numbers.

    Returns a list of invoice dicts for numbers that exist in the DB.
    Missing numbers are silently skipped.
    """
    if not numbers:
        return []

    try:
        from ..storage.repositories.invoice import get_invoice_repo
        repo = get_invoice_repo()

        results = []
        for num in numbers:
            try:
                inv = await repo.get_by_number(num)
                if inv:
                    results.append(inv)
            except Exception as e:
                logger.debug("Failed to resolve invoice %s: %s", num, e)
        return results
    except Exception as e:
        logger.warning("Invoice resolution failed: %s", e)
        return []
