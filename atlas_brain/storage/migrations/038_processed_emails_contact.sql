-- Link processed emails to CRM contacts
-- Migration: 038_processed_emails_contact.sql
--
-- Adds contact_id FK to processed_emails so lead emails can be tied to a
-- CRM contact.  ON DELETE SET NULL keeps the email record if a contact is
-- removed.  Partial index covers only linked rows (most emails won't have one).

ALTER TABLE processed_emails
    ADD COLUMN IF NOT EXISTS contact_id UUID REFERENCES contacts(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_processed_emails_contact_id
    ON processed_emails(contact_id)
    WHERE contact_id IS NOT NULL;

COMMENT ON COLUMN processed_emails.contact_id IS
    'CRM contact this email belongs to; NULL for non-lead emails';
