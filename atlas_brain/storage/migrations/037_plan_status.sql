-- Add plan lifecycle tracking to call_transcripts
ALTER TABLE call_transcripts
    ADD COLUMN IF NOT EXISTS plan_status VARCHAR(32) DEFAULT 'pending',
    ADD COLUMN IF NOT EXISTS plan_decided_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS plan_results JSONB DEFAULT '[]'::jsonb;

-- Index for querying pending plans
CREATE INDEX IF NOT EXISTS idx_call_transcripts_plan_status
    ON call_transcripts (plan_status) WHERE plan_status IS NOT NULL;
