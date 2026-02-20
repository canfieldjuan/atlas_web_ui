-- 029_email_drafts.sql
-- Email draft storage for draft-approve-send workflow.
-- Stores LLM-generated reply drafts for action_required emails.

CREATE TABLE IF NOT EXISTS email_drafts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    gmail_message_id TEXT NOT NULL,
    thread_id TEXT,
    original_message_id TEXT,
    original_from TEXT NOT NULL,
    original_subject TEXT NOT NULL,
    original_body_text TEXT,
    draft_subject TEXT NOT NULL,
    draft_body TEXT NOT NULL,
    model_provider VARCHAR(32) NOT NULL,
    model_name VARCHAR(64) NOT NULL,
    status VARCHAR(16) NOT NULL DEFAULT 'pending',
    approved_at TIMESTAMPTZ,
    sent_at TIMESTAMPTZ,
    gmail_sent_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMPTZ NOT NULL DEFAULT (CURRENT_TIMESTAMP + INTERVAL '24 hours'),
    metadata JSONB DEFAULT '{}'::jsonb,
    CONSTRAINT email_drafts_status_check
        CHECK (status IN ('pending', 'approved', 'sent', 'rejected', 'expired'))
);

CREATE INDEX IF NOT EXISTS idx_email_drafts_status
    ON email_drafts (status);
CREATE INDEX IF NOT EXISTS idx_email_drafts_message_id
    ON email_drafts (gmail_message_id);
CREATE INDEX IF NOT EXISTS idx_email_drafts_created_at
    ON email_drafts (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_email_drafts_expires_pending
    ON email_drafts (expires_at) WHERE status = 'pending';
