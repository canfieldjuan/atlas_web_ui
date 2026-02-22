-- Link call transcripts to CRM contacts
-- Migration: 036_call_contact_link.sql
--
-- Adds contact_id FK to call_transcripts so each call can be tied to a
-- CRM contact.  ON DELETE SET NULL keeps the transcript if a contact is
-- removed.  Partial index covers only linked rows.

ALTER TABLE call_transcripts
    ADD COLUMN IF NOT EXISTS contact_id UUID REFERENCES contacts(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_call_transcripts_contact_id
    ON call_transcripts(contact_id)
    WHERE contact_id IS NOT NULL;

COMMENT ON COLUMN call_transcripts.contact_id IS
    'CRM contact this call belongs to; NULL for calls not yet linked';
