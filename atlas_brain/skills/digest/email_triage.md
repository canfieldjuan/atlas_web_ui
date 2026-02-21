---
name: digest/email_triage
domain: digest
description: Summarize pre-classified Gmail emails into a prioritized natural language digest
tags: [digest, email, triage, autonomous]
version: 4
---

# Email Triage Digest

/no_think

Your output MUST follow the exact format shown in the example below. No markdown. No bullet points. No bold. No emojis. No headers with #. No commentary. No closing remarks. Stop after the last email line.

## Example Output (copy this format exactly)

13 emails -- 2 need action, 5 FYI, 6 low priority

ACTION REQUIRED
[financial] Cash App -- Borrow payment of $65.62 due tomorrow
[personal] Tia Jackson (Red Cross) -- Requesting ACH enrollment form, needs contact info updated

FYI
[shopping] Amazon -- ADHD Cleaning Planner delivered to front door
[travel] Google Calendar -- Train to Effingham IL, Thu Feb 19 9:23pm
[financial] Cash App (3) -- Spent $42 at Amtrak, $14.56 at Casey's, $6.72 at Starbucks
[financial] Cash App -- Green status renewed through Mar 31

LOW PRIORITY
[promotion] January -- Blue Spruce Toolworks payment plan offer
[automated] Republic Services -- Solid waste service delayed 2 hours
[newsletter] GitLab -- Ultimate trial ended

## Input Fields

- `emails`: list of classified emails with `category`, `priority`, `from`, `subject`, `body_text`
- `graph_context`: list of historical facts from the knowledge graph (may be empty)

## Rules

- Line 1: overview count (e.g. "13 emails -- 2 need action, 5 FYI, 6 low priority")
- Section headers: ACTION REQUIRED, FYI, LOW PRIORITY (plain text, no # or **)
- Each email line: [category] Sender -- key info
- Collapse multiple emails from same sender: "Cash App (4) -- ..." with all key details on one line
- Use the `category` field as the [tag] — do not invent new ones
- group_id priority: action_required → ACTION REQUIRED, fyi → FYI, low_priority → LOW PRIORITY
- Skip empty sections entirely
- Extract key details: amounts, dates, confirmation numbers
- Keep under 300 words
- If graph_context is non-empty, append "(note: <fact>)" after relevant ACTION REQUIRED lines only
- If no emails: output only "No new emails to report."
- DO NOT add suggestions, questions, closing remarks, or any text after the last email line
- DO NOT use markdown formatting of any kind
