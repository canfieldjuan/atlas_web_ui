-- Add pressure_direction column to news_articles (building/steady/releasing/unclear)
ALTER TABLE news_articles
    ADD COLUMN IF NOT EXISTS pressure_direction TEXT;

-- Partial index for 'fetched' status (complements existing 'pending' index)
CREATE INDEX IF NOT EXISTS idx_news_articles_enrichment_fetched
    ON news_articles(enrichment_status) WHERE enrichment_status = 'fetched';
