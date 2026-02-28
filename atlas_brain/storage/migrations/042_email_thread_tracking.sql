-- Migration 042: Email thread tracking
--
-- Store RFC 2822 threading headers on processed_emails so the intake pipeline
-- can detect follow-ups to threads Atlas previously participated in.
-- Also store the RFC Message-ID of emails we send (email_drafts.sent_message_id)
-- so incoming In-Reply-To headers can be matched back to our sent replies.

-- processed_emails: store RFC threading headers + follow-up linkage
ALTER TABLE processed_emails ADD COLUMN IF NOT EXISTS message_id TEXT;
ALTER TABLE processed_emails ADD COLUMN IF NOT EXISTS in_reply_to TEXT;
ALTER TABLE processed_emails ADD COLUMN IF NOT EXISTS references_header TEXT;
ALTER TABLE processed_emails ADD COLUMN IF NOT EXISTS followup_of_draft_id UUID
    REFERENCES email_drafts(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_processed_emails_in_reply_to
    ON processed_emails(in_reply_to) WHERE in_reply_to IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_processed_emails_followup_draft
    ON processed_emails(followup_of_draft_id) WHERE followup_of_draft_id IS NOT NULL;

-- email_drafts: store RFC Message-ID of the email we sent
ALTER TABLE email_drafts ADD COLUMN IF NOT EXISTS sent_message_id TEXT;
