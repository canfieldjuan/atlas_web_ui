---
name: invoicing/invoice_email
domain: invoicing
description: Professional invoice email template for sending to customers
tags: [invoicing, email, template]
version: 1
---

# Invoice Email

You are composing a professional invoice email to send to a customer. The email should be clear, concise, and include all necessary payment details.

## Input

You will receive a JSON object with:
- `invoice_number`: The invoice number (e.g., INV-2026-0042)
- `customer_name`: Customer's name
- `line_items`: Array of items with description, quantity, unit_price, amount
- `subtotal`: Subtotal before tax/discount
- `tax_amount`: Tax amount
- `discount_amount`: Discount amount
- `total_amount`: Total due
- `due_date`: Payment due date
- `notes`: Optional notes

## Output

Compose a professional email body (plain text):

1. **Greeting** -- "Dear [customer_name],"
2. **Introduction** -- one sentence mentioning the invoice number
3. **Line items** -- formatted list of services/items with amounts
4. **Totals** -- subtotal, tax (if >0), discount (if >0), total due
5. **Due date** -- when payment is expected
6. **Payment instructions** -- generic (we'll customize later)
7. **Closing** -- thank them for their business

## Rules

- Professional but warm tone
- Keep under 200 words
- Format all amounts with dollar sign and two decimals
- Do not include any subject line -- just the body
- Plain text only, no markdown
