-- Persist news articles for daily intelligence (news_intake currently discards them)
CREATE TABLE IF NOT EXISTS news_articles (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dedup_key             TEXT NOT NULL,
    title                 TEXT NOT NULL,
    source_name           TEXT NOT NULL DEFAULT 'unknown',
    url                   TEXT NOT NULL DEFAULT '',
    published_at          TEXT,
    summary               TEXT NOT NULL DEFAULT '',
    matched_keywords      TEXT[] NOT NULL DEFAULT '{}',
    matched_watchlist_ids UUID[] NOT NULL DEFAULT '{}',
    is_market_related     BOOLEAN NOT NULL DEFAULT false,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_news_articles_created ON news_articles(created_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS idx_news_articles_dedup ON news_articles(dedup_key);

-- Persistent reasoning memory -- each daily session stores its conclusions
CREATE TABLE IF NOT EXISTS reasoning_journal (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_date          DATE NOT NULL,
    analysis_type         TEXT NOT NULL DEFAULT 'daily',
    analysis_window_days  INT NOT NULL DEFAULT 7,
    raw_data_summary      JSONB NOT NULL DEFAULT '{}',
    reasoning_output      TEXT NOT NULL DEFAULT '',
    key_insights          JSONB NOT NULL DEFAULT '[]',
    connections_found     JSONB NOT NULL DEFAULT '[]',
    recommendations       JSONB NOT NULL DEFAULT '[]',
    market_summary        JSONB NOT NULL DEFAULT '{}',
    news_summary          JSONB NOT NULL DEFAULT '{}',
    business_implications JSONB NOT NULL DEFAULT '[]',
    created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_reasoning_journal_date ON reasoning_journal(session_date DESC);
