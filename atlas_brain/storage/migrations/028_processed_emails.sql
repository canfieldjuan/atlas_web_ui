-- 028_processed_emails.sql
-- Dedup table for Gmail digest: tracks which messages have been processed

CREATE TABLE IF NOT EXISTS processed_emails (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    gmail_message_id TEXT NOT NULL UNIQUE,
    sender TEXT,
    subject TEXT,
    category VARCHAR(32),
    priority VARCHAR(16),
    processed_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_processed_emails_message_id
    ON processed_emails (gmail_message_id);
CREATE INDEX IF NOT EXISTS idx_processed_emails_processed_at
    ON processed_emails (processed_at DESC);
CREATE INDEX IF NOT EXISTS idx_processed_emails_category
    ON processed_emails (category);
