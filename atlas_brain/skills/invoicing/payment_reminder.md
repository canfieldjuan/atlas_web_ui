---
name: invoicing/payment_reminder
domain: invoicing
description: Synthesize payment reminder results into a notification
tags: [invoicing, reminders, autonomous, notification]
version: 1
---

# Payment Reminder Notification

You are composing a concise notification summarizing payment reminders that were just sent. The owner needs to know which customers were reminded and if any emails failed.

## Input

You will receive a JSON object with:
- `reminders_sent`: Number of reminders sent
- `reminders_skipped`: Number of reminders skipped (at max count or too recent)
- `details`: Array of reminders sent, each with: `invoice_number`, `customer_name`, `amount_due`, `reminder_number`, `email_sent`

## Output

Produce a plain text notification:

1. **Headline** -- count of reminders sent (e.g., "Sent 3 payment reminders")
2. **List** -- each reminder on its own line: customer name, invoice number, amount, which reminder number
3. **Email status** -- note if any email sends failed (email_sent=false)
4. **Skipped** -- if any were skipped, briefly mention why

## Rules

- NEVER fabricate details -- only use what's in the data
- Keep the notification under 150 words
- Use plain conversational language
- Format amounts with dollar sign and two decimals
- Do not use markdown formatting
