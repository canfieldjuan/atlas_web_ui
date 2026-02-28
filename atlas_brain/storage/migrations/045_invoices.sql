-- Invoicing: invoices + payments for manual billing/payment tracking
-- Links to CRM contacts, supports partial payments and payment behavior analytics

-- Auto-incrementing invoice number sequence
CREATE SEQUENCE IF NOT EXISTS invoice_number_seq START WITH 1 INCREMENT BY 1;

CREATE TABLE IF NOT EXISTS invoices (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    invoice_number      VARCHAR(32) NOT NULL UNIQUE,

    -- Customer (denormalized for PDF/email rendering; FK for CRM link)
    contact_id          UUID REFERENCES contacts(id) ON DELETE SET NULL,
    customer_name       VARCHAR(256) NOT NULL,
    customer_email      VARCHAR(256),
    customer_phone      VARCHAR(32),
    customer_address    TEXT,

    -- Line items: [{description, quantity, unit_price, amount}]
    line_items          JSONB NOT NULL DEFAULT '[]'::jsonb,

    -- Amounts
    subtotal            NUMERIC(12,2) NOT NULL DEFAULT 0,
    tax_rate            NUMERIC(5,4) NOT NULL DEFAULT 0,
    tax_amount          NUMERIC(12,2) NOT NULL DEFAULT 0,
    discount_amount     NUMERIC(12,2) NOT NULL DEFAULT 0,
    total_amount        NUMERIC(12,2) NOT NULL DEFAULT 0,
    amount_paid         NUMERIC(12,2) NOT NULL DEFAULT 0,
    amount_due          NUMERIC(12,2) GENERATED ALWAYS AS (total_amount - amount_paid) STORED,

    -- Dates
    issue_date          DATE NOT NULL DEFAULT CURRENT_DATE,
    due_date            DATE NOT NULL,

    -- Status: draft -> sent -> partial -> paid / overdue -> void
    status              VARCHAR(16) NOT NULL DEFAULT 'draft',

    -- Tracking
    sent_at             TIMESTAMPTZ,
    sent_via            VARCHAR(32),
    paid_at             TIMESTAMPTZ,
    voided_at           TIMESTAMPTZ,
    void_reason         TEXT,

    -- Reminders
    reminder_count      INT NOT NULL DEFAULT 0,
    last_reminder_at    TIMESTAMPTZ,

    -- Source
    source              VARCHAR(32) DEFAULT 'manual',
    source_ref          VARCHAR(256),
    appointment_id      UUID,

    -- Standard
    business_context_id VARCHAR(64),
    notes               TEXT,
    metadata            JSONB DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS invoice_payments (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    invoice_id          UUID NOT NULL REFERENCES invoices(id) ON DELETE CASCADE,
    amount              NUMERIC(12,2) NOT NULL,
    payment_date        DATE NOT NULL DEFAULT CURRENT_DATE,
    payment_method      VARCHAR(32) NOT NULL DEFAULT 'other',
    reference           VARCHAR(256),
    notes               TEXT,
    recorded_by         VARCHAR(128),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata            JSONB DEFAULT '{}'::jsonb
);

-- Invoice number lookup (already UNIQUE via column constraint, but explicit for clarity)
CREATE INDEX IF NOT EXISTS idx_invoices_contact_id ON invoices(contact_id);
CREATE INDEX IF NOT EXISTS idx_invoices_status_due ON invoices(status, due_date);
CREATE INDEX IF NOT EXISTS idx_invoices_context_status ON invoices(business_context_id, status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_invoices_overdue ON invoices(due_date) WHERE status IN ('sent', 'partial');
CREATE INDEX IF NOT EXISTS idx_invoices_customer_name ON invoices USING gin(to_tsvector('english', customer_name));
CREATE INDEX IF NOT EXISTS idx_invoices_appointment ON invoices(appointment_id) WHERE appointment_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_invoices_created ON invoices(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_invoice_payments_invoice_date ON invoice_payments(invoice_id, payment_date DESC);
