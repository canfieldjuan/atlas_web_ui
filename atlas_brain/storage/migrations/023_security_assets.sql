-- 023_security_assets.sql
-- Security asset registry and telemetry history

CREATE TABLE IF NOT EXISTS security_assets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    asset_type VARCHAR(32) NOT NULL,
    identifier VARCHAR(128) NOT NULL,
    name VARCHAR(128),
    status VARCHAR(16) NOT NULL DEFAULT 'active',
    first_seen TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_seen TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB NOT NULL DEFAULT '{}',
    UNIQUE (asset_type, identifier)
);

CREATE INDEX IF NOT EXISTS idx_security_assets_type
    ON security_assets (asset_type, last_seen DESC);

CREATE INDEX IF NOT EXISTS idx_security_assets_status
    ON security_assets (status, last_seen DESC);

CREATE TABLE IF NOT EXISTS security_asset_telemetry (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    asset_type VARCHAR(32) NOT NULL,
    identifier VARCHAR(128) NOT NULL,
    observed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_security_asset_telemetry_lookup
    ON security_asset_telemetry (asset_type, identifier, observed_at DESC);
