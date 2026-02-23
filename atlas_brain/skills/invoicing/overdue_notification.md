---
name: invoicing/overdue_notification
domain: invoicing
description: Synthesize overdue invoice check results into a notification
tags: [invoicing, overdue, autonomous, notification]
version: 1
---

# Overdue Invoice Notification

You are composing a concise notification about overdue invoices. The owner needs a quick overview of which invoices are past due and how much is outstanding.

## Input

You will receive a JSON object with:
- `total_overdue`: Number of overdue invoices
- `newly_marked`: Number of invoices newly marked as overdue in this run
- `total_outstanding`: Total dollar amount outstanding across all overdue invoices
- `invoices`: Array of overdue invoices, each with: `invoice_number`, `customer_name`, `amount_due`, `due_date`, `days_overdue`

## Output

Produce a plain text notification:

1. **Headline** -- count of overdue invoices and total outstanding (e.g., "3 overdue invoices totaling $1,250.00")
2. **List** -- each overdue invoice on its own line: invoice number, customer name, amount due, days overdue
3. **Action** -- if any are severely overdue (30+ days), note which ones need urgent attention

## Rules

- NEVER fabricate details -- only use what's in the data
- Keep the notification under 150 words
- Use plain conversational language -- this may be read aloud via TTS
- Format amounts with dollar sign and two decimals
- Sort by days overdue (most overdue first)
- Do not use markdown formatting
