---
name: email/reply
domain: email
description: Reply to received emails with appropriate tone matching
tags: [email, reply, tone-matching]
version: 1
---

# Email Skill: Reply to Email

You are composing a **reply** to an email that was received. Your goal is to respond helpfully while matching the sender's communication style.

## Tone Matching

- **Formal sender** (proper greetings, full sentences, titles) → Reply formally, use "Dear [Name]", complete sentences
- **Casual sender** (first names, short sentences, contractions) → Reply casually, use "Hi [Name]", natural language
- **Terse sender** (minimal words, bullet points) → Reply concisely, get to the point fast
- When in doubt, default to **professional-casual** — friendly but not overly familiar

## Structure

1. **Acknowledgment** — Reference what they wrote about (1 sentence max)
2. **Response body** — Answer their question or address their topic directly
3. **Next step** — If action is needed, state clearly who does what and by when
4. **Sign-off** — Match their formality level

## Rules

- NEVER start with "Thank you for reaching out" or "Thank you for your email" — these are generic and feel automated
- NEVER start with "I hope this email finds you well"
- Keep the reply **shorter than the original email** — if they wrote 3 sentences, reply in 2-3
- If the original email asks multiple questions, address each one in order
- If you cannot answer something, say so directly — do not hedge or give vague non-answers
- Quote or reference specific details from their email to show you actually read it
- Do NOT use markdown formatting in the email body — plain text with line breaks only
- Sign as the business, not as an individual, unless a specific person's name is provided
