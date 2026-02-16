-- 022_network_security_threats.sql
-- Network security monitoring: WiFi threats and network anomalies

CREATE TABLE IF NOT EXISTS security_threats (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    threat_type VARCHAR(64) NOT NULL,
    severity VARCHAR(16) NOT NULL DEFAULT 'medium',
    source_mac VARCHAR(17),
    target_mac VARCHAR(17),
    source_id VARCHAR(128) NOT NULL DEFAULT 'unknown',
    detection_type VARCHAR(64) NOT NULL,
    label VARCHAR(256),
    confidence FLOAT DEFAULT 1.0,
    details JSONB NOT NULL DEFAULT '{}',
    pcap_file VARCHAR(512),
    resolved BOOLEAN NOT NULL DEFAULT FALSE,
    resolved_at TIMESTAMP,
    notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_security_threats_timestamp
    ON security_threats (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_security_threats_type
    ON security_threats (threat_type, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_security_threats_resolved
    ON security_threats (resolved, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_security_threats_source_mac
    ON security_threats (source_mac) WHERE source_mac IS NOT NULL;
