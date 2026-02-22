# Session Coordination

## Session Roles
- **Senior (this session)**: Architecture decisions, CRM wiring, agency workflow, cross-reference intelligence
- **Junior (other session)**: MCP server refinement, provider hardening, tool migration tasks assigned by senior

## Branch Discipline
- **Both sessions work on `main`** with pull-before-edit discipline
- Before editing ANY file: `git pull origin main`
- Commit frequently (small atomic commits) to minimize conflict windows
- If a conflict occurs: the session that caused it resolves it

## File Ownership (Conflict Avoidance)

### Senior Session OWNS (do not edit without asking):
- `atlas_brain/agents/` (all graphs, workflows, routing)
- `atlas_brain/autonomous/` (task handlers, runner, scheduler)
- `atlas_brain/comms/call_intelligence.py`
- `atlas_brain/voice/` (pipeline, launcher)
- `atlas_brain/memory/` (RAG, feedback, quality)
- `atlas_brain/storage/migrations/` (new migrations)

### Junior Session OWNS (do not edit without asking):
- `atlas_brain/mcp/` (all 4 MCP servers)
- `atlas_brain/services/calendar_provider.py`
- `atlas_brain/services/email_provider.py`
- `atlas_brain/services/crm_provider.py`
- `atlas_brain/templates/` (email templates)
- `atlas_comms/` (telephony providers)
- `tests/test_mcp_servers.py`

### SHARED FILES (coordinate before editing):
- `atlas_brain/config.py` — Senior adds new config sections, junior adds provider config
- `atlas_brain/main.py` — Senior manages lifespan, junior adds MCP startup if needed
- `atlas_brain/services/__init__.py` — Coordinate exports
- `atlas_brain/services/protocols.py` — Senior defines new protocols, junior implements
- `atlas_brain/tools/scheduling.py` — Senior owns CRM wiring, junior owns tool interface
- `requirements.txt` — Either can add deps, pull first
- `CLAUDE.md` — Senior updates architecture sections
- `docker-compose.yml` — Coordinate

### Protocol for shared files:
1. Announce intent: "I need to add X to config.py"
2. Pull latest
3. Make minimal, isolated edit
4. Commit + push immediately
5. Tell the other session: "pushed config.py change, pull before editing"

## Current Task Assignments — Round 1

### Senior Session - Active Tasks:

**S1. Migration 036: Add `contact_id` FK to `call_transcripts`**
- File: `atlas_brain/storage/migrations/036_call_contact_link.sql`
- Add `contact_id UUID REFERENCES contacts(id) ON DELETE SET NULL` to `call_transcripts`
- Add index on `contact_id` for join queries
- This enables linking calls to CRM contacts

**S2. Wire `call_intelligence.py` → CRM auto-population**
- File: `atlas_brain/comms/call_intelligence.py`
- After LLM extraction produces `extracted_data` (customer_name, phone, email, intent, services):
  - Lookup existing contact by phone/email via CRM provider
  - If not found → create new contact
  - Link call transcript to contact via new `contact_id` column
  - Store interaction record in `contact_interactions`
- Depends on: S1 (migration), J1 (duplicate protection)

**S3. Design `CustomerContextService` (cross-reference layer)**
- New file: `atlas_brain/services/customer_context.py`
- Unified view of a customer: recent calls, emails, appointments, CRM notes
- Query by contact_id, phone, or email
- Powers the agency workflow: "what do we know about this customer?"
- Depends on: S2 (call→CRM link exists)

### Junior Session - Active Tasks:

**J1. CRM duplicate protection**
- File: `atlas_brain/services/crm_provider.py`
- Problem: `contacts` table has NO unique constraints on phone/email — duplicates will accumulate
- Options (pick one):
  - Add UNIQUE constraint on `phone` and `email` columns (migration 037)
  - OR implement upsert logic in `create_contact()` — check existing first, update if found
- Recommendation: upsert in provider + partial unique index (allows NULLs) in migration
- Test: verify `create_contact()` with same phone twice doesn't create duplicates

**J2. MCP server smoke test**
- Files: `atlas_brain/mcp/` (all 4 servers)
- Verify each server starts without import errors: `python -c "from atlas_brain.mcp.crm_server import mcp; print('OK')"`
- Verify tool registration counts match expectations (CRM=9, Email=8, Calendar=8, Twilio=10)
- Fix any import or config issues found

**J3. Wire `BookAppointmentTool` → CRM contact linkage**
- File: `atlas_brain/tools/scheduling.py`
- `BookAppointmentTool.execute()` creates appointments but never sets `contact_id`
- After booking: lookup/create contact via CRM provider, set `contact_id` on appointment record
- `LookupCustomerTool` CRM-first path already works — reuse that pattern
- Depends on: J1 (duplicate protection must be in place first)

**J4. Email provider IMAP validation**
- File: `atlas_brain/services/email_provider.py`
- Verify IMAP connection settings are validated on init (host, port, SSL)
- Add graceful error if IMAP credentials are missing/invalid (don't crash MCP server)
- Test: `CompositeEmailProvider` falls back cleanly if IMAP is misconfigured

### Task Dependencies:
```
S1 (migration) ──→ S2 (call→CRM wire)  ──→ S3 (context service)
J1 (dedup)     ──→ J3 (booking→CRM)
J2 (smoke test) — independent, do first
J4 (IMAP)      — independent, do anytime
```

### Completed Tasks:
(none yet)
