-- Add synced_at to conversation_turns for nightly sync idempotency.
-- Only un-synced turns are sent to GraphRAG; only synced turns are purged.

ALTER TABLE conversation_turns
ADD COLUMN IF NOT EXISTS synced_at TIMESTAMP WITH TIME ZONE DEFAULT NULL;

CREATE INDEX IF NOT EXISTS idx_turns_synced_at
    ON conversation_turns(synced_at)
    WHERE synced_at IS NULL;
