-- Temporal patterns: learned arrival/departure/wake/sleep norms per person per day-of-week.
-- Populated nightly by the pattern_learning task; read by anomaly_detection.

CREATE TABLE IF NOT EXISTS temporal_patterns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    person_name VARCHAR(128) NOT NULL,
    pattern_type VARCHAR(32) NOT NULL,   -- 'arrival', 'departure', 'wake', 'sleep'
    day_of_week SMALLINT NOT NULL,       -- 0=Mon, 6=Sun
    median_minutes SMALLINT NOT NULL,    -- Minutes since midnight
    stddev_minutes SMALLINT NOT NULL DEFAULT 60,
    sample_count SMALLINT NOT NULL DEFAULT 0,
    earliest_minutes SMALLINT,
    latest_minutes SMALLINT,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_temporal_pattern_key
    ON temporal_patterns (person_name, pattern_type, day_of_week);

CREATE INDEX IF NOT EXISTS idx_temporal_patterns_person
    ON temporal_patterns (person_name);
