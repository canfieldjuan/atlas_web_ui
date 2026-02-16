ALTER TABLE presence_events ADD COLUMN IF NOT EXISTS arrival_times JSONB DEFAULT '{}';
