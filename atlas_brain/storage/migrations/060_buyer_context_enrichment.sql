-- Migration 060: Buyer context columns (Pass 3 + Pass 4)
--
-- These columns exist for backward compatibility and future backfill.
-- Deep extraction (Pass 2) now extracts all 32 fields in a single call,
-- so no autonomous task queues rows into these columns.  Defaults are
-- 'not_applicable' to prevent any stale pending-queue queries.

-- --- Pass 3: Buyer Psychology ---

ALTER TABLE product_reviews
    ADD COLUMN IF NOT EXISTS buyer_psychology              JSONB,
    ADD COLUMN IF NOT EXISTS buyer_psychology_status       TEXT NOT NULL DEFAULT 'not_applicable',
    ADD COLUMN IF NOT EXISTS buyer_psychology_attempts     INT  NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS buyer_psychology_enriched_at  TIMESTAMPTZ;

-- --- Pass 4: Extended Context ---

ALTER TABLE product_reviews
    ADD COLUMN IF NOT EXISTS extended_context              JSONB,
    ADD COLUMN IF NOT EXISTS extended_context_status       TEXT NOT NULL DEFAULT 'not_applicable',
    ADD COLUMN IF NOT EXISTS extended_context_attempts     INT  NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS extended_context_enriched_at  TIMESTAMPTZ;

-- Mark ALL existing rows as not_applicable (no separate Pass 3/4 processing).
UPDATE product_reviews
SET
    buyer_psychology_status = 'not_applicable',
    extended_context_status = 'not_applicable'
WHERE buyer_psychology_status != 'not_applicable'
   OR extended_context_status != 'not_applicable';

-- --- Indexes ---

-- GIN indexes for querying inside the JSONB blobs (useful if data is backfilled)
CREATE INDEX IF NOT EXISTS idx_product_reviews_buyer_psychology
    ON product_reviews USING GIN (buyer_psychology)
    WHERE buyer_psychology IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_product_reviews_extended_context
    ON product_reviews USING GIN (extended_context)
    WHERE extended_context IS NOT NULL;
