ALTER TABLE presence_events ADD COLUMN IF NOT EXISTS unknown_count INT DEFAULT 0;
