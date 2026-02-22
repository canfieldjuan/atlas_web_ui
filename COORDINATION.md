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

### Completed — Round 1:
- **S1** Migration 036: `contact_id` FK on `call_transcripts` ✓ (fb17380)
- **S2** Wire `call_intelligence.py` → CRM auto-population ✓ (fb17380)
- **S3** `CustomerContextService` cross-reference layer ✓ (fb17380)
- **J1** CRM duplicate protection (`find_or_create_contact` + partial unique indexes) ✓ (0a29b39)
- **J2** MCP server smoke test + fixes ✓ (0a29b39)
- **J3** `BookAppointmentTool` → CRM contact linkage ✓ (06a5d3a)
- **J4** Email provider IMAP graceful fallback ✓ (06a5d3a)

---

## Round 2 — Agency Workflow

### Senior Session - Active Tasks:

**S4. Action Planner — LLM + CustomerContext → structured action plan**
- After CRM linkage in call_intelligence pipeline, build full CustomerContext
- Feed context + call data to LLM with a skill prompt
- LLM outputs structured JSON action plan: `[{action, params, rationale}]`
- Store plan on transcript record (enrich `proposed_actions`)
- New skill: `skills/call/action_planning.md`

**S5. Plan approval + execution endpoint**
- Enhanced ntfy notification showing full plan summary
- `POST /call-actions/{id}/approve-plan` → execute all planned actions
- Reuse existing action logic (calendar, email, SMS)
- Log each executed action to `contact_interactions`
- `POST /call-actions/{id}/reject-plan` → mark plan rejected

### Junior Session - Active Tasks:

**J5. Wire email inbox into CustomerContext**
- File: `atlas_brain/services/customer_context.py` (SHARED — coordinate with senior)
- Add `_get_inbox_emails()` method: search IMAP/Gmail for recent emails from/to the customer's email
- Uses `email_provider.list_messages(query="from:{email}")` for inbound
- Merge into `CustomerContext` as `inbox_emails: list[dict]`
- Fail-open: if IMAP unavailable, return empty list

**J6. MCP tool for CustomerContext**
- New tool in `atlas_brain/mcp/crm_server.py`: `get_customer_context`
- Wraps `CustomerContextService.get_context()` / `get_context_by_phone()`
- Exposes full customer view to MCP clients (NocoDB, external agents)

### Task Dependencies:
```
S4 (action planner) ──→ S5 (approval + execution)
J5 (inbox context)  — independent, enhances S4 output
J6 (MCP tool)       — independent, can start anytime
```
