"""
Invoice overdue check -- daily autonomous task.

Scans for invoices past their due date, marks them overdue, logs CRM
interactions, and returns results for LLM synthesis and ntfy notification.
"""

import logging
from datetime import date

from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.autonomous.tasks.invoice_overdue_check")


async def run(task: ScheduledTask) -> dict:
    """Check for overdue invoices and mark them accordingly.

    Returns dict for synthesis, or _skip_synthesis when nothing is overdue.
    """
    from ...config import settings

    if not settings.invoicing.enabled:
        return {"_skip_synthesis": "Invoicing disabled"}

    from ...storage.repositories.invoice import get_invoice_repo

    repo = get_invoice_repo()

    try:
        overdue = await repo.get_overdue(as_of_date=date.today())
    except Exception as e:
        logger.error("Failed to query overdue invoices: %s", e)
        return {"_skip_synthesis": f"Error checking overdue invoices: {e}"}

    if not overdue:
        return {"_skip_synthesis": "No overdue invoices"}

    # Mark newly overdue invoices (those still in 'sent' or 'partial' status)
    newly_overdue = []
    for inv in overdue:
        if inv["status"] != "overdue":
            try:
                await repo.mark_overdue(inv["id"])
                newly_overdue.append(inv)

                # Log CRM interaction
                contact_id = inv.get("contact_id")
                if contact_id:
                    try:
                        from ...services.crm_provider import get_crm_provider
                        crm = get_crm_provider()
                        await crm.log_interaction(
                            contact_id=str(contact_id),
                            interaction_type="invoice",
                            summary=f"Invoice {inv['invoice_number']} overdue (${inv['amount_due']:.2f}, due {inv['due_date']})",
                        )
                    except Exception as e:
                        logger.warning("CRM log failed for %s: %s", inv["invoice_number"], e)

            except Exception as e:
                logger.error("Failed to mark invoice %s overdue: %s", inv["invoice_number"], e)

    total_outstanding = sum(float(i.get("amount_due", 0)) for i in overdue)

    result = {
        "total_overdue": len(overdue),
        "newly_marked": len(newly_overdue),
        "total_outstanding": round(total_outstanding, 2),
        "invoices": [
            {
                "invoice_number": inv["invoice_number"],
                "customer_name": inv["customer_name"],
                "amount_due": float(inv.get("amount_due", 0)),
                "due_date": str(inv["due_date"]),
                "days_overdue": (date.today() - inv["due_date"]).days if isinstance(inv["due_date"], date) else 0,
            }
            for inv in overdue
        ],
    }

    logger.info(
        "Overdue check: %d total overdue, %d newly marked, $%.2f outstanding",
        len(overdue), len(newly_overdue), total_outstanding,
    )
    return result
