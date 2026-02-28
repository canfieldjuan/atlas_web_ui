-- B2B review scraping: watchlist targets and execution log
-- Migration 056

-- Scrape targets: each row = one product to scrape from one source
CREATE TABLE IF NOT EXISTS b2b_scrape_targets (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source                TEXT NOT NULL,           -- g2, capterra, trustradius, reddit
    vendor_name           TEXT NOT NULL,
    product_name          TEXT,
    product_slug          TEXT NOT NULL,           -- URL slug (varies by source)
    product_category      TEXT,
    max_pages             INT NOT NULL DEFAULT 5,
    enabled               BOOLEAN NOT NULL DEFAULT true,
    priority              INT NOT NULL DEFAULT 0,  -- higher = scraped first
    last_scraped_at       TIMESTAMPTZ,
    last_scrape_status    TEXT,                    -- success | partial | failed | blocked
    last_scrape_reviews   INT DEFAULT 0,
    scrape_interval_hours INT NOT NULL DEFAULT 168, -- default weekly
    metadata              JSONB NOT NULL DEFAULT '{}',
    created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_b2b_scrape_targets_enabled
    ON b2b_scrape_targets(enabled) WHERE enabled = true;
CREATE UNIQUE INDEX IF NOT EXISTS idx_b2b_scrape_targets_dedup
    ON b2b_scrape_targets(source, product_slug);

-- Scrape execution log for observability
CREATE TABLE IF NOT EXISTS b2b_scrape_log (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    target_id         UUID NOT NULL REFERENCES b2b_scrape_targets(id),
    source            TEXT NOT NULL,
    status            TEXT NOT NULL,            -- success | partial | failed | blocked
    reviews_found     INT NOT NULL DEFAULT 0,
    reviews_inserted  INT NOT NULL DEFAULT 0,
    pages_scraped     INT NOT NULL DEFAULT 0,
    errors            JSONB NOT NULL DEFAULT '[]',
    duration_ms       INT,
    proxy_type        TEXT,                     -- datacenter | residential | none
    started_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_b2b_scrape_log_target
    ON b2b_scrape_log(target_id, started_at DESC);
