# Simple user selection: owner-approved browsers + "who is entering?" (migration 091)

Task marker FL-SIMPLE-USER-20261103. Branch `worktree-simple-user-20261103`, base `2e25d17`.
Status: **built and validated locally only. NOT deployed. Staging proof held for Codex target verification.**

Evidence labels used below:
- **[V]** verified in this worktree's source or by local tests (local `factory_ledger_test` only).
- **[C]** checked live by Codex, read-only, and reported to this session.
- **[R]** reported in docs/memory; not independently verified.

## 1. What changes for people

- Michael approves each browser once. The browser shows an 8-character code. Michael, signed in with his **own** key or phone PIN, enters it on **Staff & phones → Browsers for name selection**.
- On an approved browser every page shows **Who is entering?** at once, with four large buttons in this order: **Michael (Owner), Luz (Office), Arturo (Floor Lead), Miriam (Office)**.
- After a tap the header shows **Entering as: Arturo (Floor Lead)** with **Change user**. The choice survives navigation, new tabs and a browser restart. It never expires.
- Nobody types a PIN, password or key for routine work.
- A tapped name **identifies** the person. It does not prove who they are, so Michael selected by name can do routine work but no owner decision. Approvals, kosher attestation, backdating over 14 days, reopening orders, shipped-not-recorded closes, voiding other people's or earlier posts, and staff/browser admin all still need his own sign-in.

## 2. Three separate things

| Concern | Mechanism | Stored | Never |
|---|---|---|---|
| (A) Human identification | `X-FL-Person: <actor id>`, checked on every request against `person_choices` + `actors` (no cache) | `localStorage fl-person` bound to the browser token | never proof, never owner authorization, never substituted |
| (B) Machine authentication | browser token `flb_…` (32 random bytes) in `X-API-Key`; owner approves its pairing code once; no expiry; owner revokes | sha256 only (`trusted_browsers.secret_hash`); pairing code sha256 until approval | never publicly issued as approved; Origin is never trusted |
| (C) Transaction attribution | unchanged columns: `operator_id = actors.name` snapshot, `entered_by_actor_id = actors.id` | plus `write_tickets.identity_via='selection'`, `trusted_browser_id`; `actor_write_audit.trusted_browser_id`; `shift_confirmations.identity_via/trusted_browser_id` | no rename of historical snapshots; id 1 stays `Blubber` [C] |

## 3. Owner decisions and review resolutions

- **Owner decision (Michael):** approve each browser once, then staff tap a name with no credential and no expiry. Network-only trust was not chosen. Owner admin proof stays independent. Custom GPT work was skipped by owner request ("can we skip the custom gpt stuff").
- **Ordinary defaults (Codex-approved D1–D7):**
  - D1 pairing code (10 minutes, D7);
  - D2 selected Michael voids only his own posts from the same plant day;
  - D3 credentialed API/session/PIN compatibility kept, with no automatic UI fallback;
  - D4 display "Michael", with id 1 `Blubber` unchanged;
  - D5 a self shift-confirm is identification, recorded with selection assurance;
  - D6 existing privileged sessions keep their timeouts.
- **Codex review fixes:**
  - R1: explicit positive allowlist for the selected owner, so new owner actions are denied by default;
  - R2: the selection marker lives in the actor dict itself;
  - R3: exact four-person configuration;
  - R5: one inactive person is disabled while the other three continue; a malformed configuration fails everyone closed;
  - R6: storage failure is sticky-unavailable and never restores a prior name;
  - R7: stale refresh answers are ignored;
  - R8: a missing script fails closed in every transport;
  - R9: selected-identity facade for page `ui.session`;
  - R10: hydration is not a user change; the Assistant has no reload loops and stays blocked on a storage failure;
  - R11: one authoritative config per page. An unreadable config is UNKNOWN (blocked, no key/PIN prompt, no old session restored), never "selection off", and every key-session client shares `FLDashboard.config`;
  - R12: normal pages in selector mode (Dashboard tabs and runs through `installDashboard`, Shift Summary, Record entry/Office/Receipt through `ledger-ui`, FL Assistant) never initialise the legacy key/PIN client, so no old token is sent, no identity restored and no PIN offer shown. Owner-proof pages (Staff & phones, My PIN, `?owner=1`) keep Michael's own sign-in;
  - R13: until the page's one config answers, the mode is unknown. Shift Summary starts with no credential panel and neutral "Checking settings…" copy; the legacy panel appears only when config is known and selection is off. A sign-in or read-only action submitted while settings are pending is dropped (Shift Summary, FL Assistant, `ledger-ui`), never queued;
  - R14: a selected credential (browser token + name) is sent only to the API that issued it (`FLDashboard.base`). Shift Summary's `?api=` override to another origin is refused before any request is made ("different Factory Ledger address"), with no redirect. Normal hosted/staging targets and machine API auth are unchanged.

## 4. Route and integration matrix

### 4.1 Integrations

| Integration | Credential | Live state | Effect of this change |
|---|---|---|---|
| Office GPT "Factory-Ledger 2.0" (`openapi-gpt-v3.yaml`, 30 ops) | master key [R] | live [R]; UI read smoke failed before any task deploy ("This response couldn't load") [C] | **none**: schema untouched (still 30 ops), master-key behaviour unchanged [V]. Live GPT UI **not tested**; owner skipped GPT work |
| Floor GPT (`gpt-configs/schemas/openapi-floor.yaml`) | master key [R] | listed as "Factory Ledger — Floor"; editor shows API-key auth and pre-existing "Updates pending" [C] | none [V]. Live GPT UI not tested |
| Read-only GPT key (`READONLY_API_KEY`) | read-only key | code live, not set in production [R] | unchanged; read-only sweep covers the new routes (403) [V] |
| Personal actor keys / key sessions / phone PIN | actor key, `fls_` session | production key sessions on, PIN on, idle 180 min, PIN 12 h [C] | unchanged; remain the **owner-proof** path [V] |
| Dashboard (Netlify) | public dashboard key + sessions today | live | name selection when `PERSON_SELECT_ENABLED=1`. With the flag off, tested credentialed API compatibility is preserved [V]; availability errors (for example an unreadable `/auth/session/config`) now fail closed on every page instead of showing key/PIN prompts (R11), an intended UI change |
| FL Assistant (`/assistant/*`) | session today; selection after | staging `ASSISTANT_ENABLED=1` [C]; production believed off [R] | accepts selection identities; internal relays keep the selection scope and cap [V] |
| Packing-slip signed links | HMAC (`PACKING_SLIP_SIGNING_KEY`) | live | unchanged [V] |
| Workers `expire-tickets`, `exceptions-sweep` | database | live | unchanged |
| `factory-ledger-mcp-staging` | none configured; `MCP_AUTH_MODE=locked`, `MCP_WRITES_ENABLED=false`; backend = staging API [C] | SUCCESS [C] | unchanged; preserve as is |
| `mcp-spike` | `SPIKE_PATH_SECRET` only, no ledger credentials [C] | SUCCESS [C] | unchanged |
| `origin/feature/mcp-writes` | would use named-actor keys for writes (source) [C] | production MCP use not established [C] | actor-key behaviour unchanged [V] |

The machine credentials are not changed. The master key's writes stay attributed to the shared placeholder `legacy-shared-key` exactly as before (the existing shared-attribution limitation). A selected human never falls back to it.

### 4.2 Who reaches what (selector mode on)

| Credential | Reads | Routine writes (tickets, dashboard metadata, notes, runs, supplies, ER, orders, attachments, Assistant) | Legacy direct writes (`POST /receive`, `/make`, `/pack`, `/adjust`, `/ship`, `/void/{id}`, `/customers`, `/sales/orders`, …) and old shortcuts (`/receive/commit`, `/receive/preview`, `/sales/orders/{id}/ship/commit`) | Owner-proof actions | Admin `/admin/*`, `/auth/devices*`, `/auth/actors/*`, `/auth/browsers` list/approve/revoke |
|---|---|---|---|---|---|
| Master key (GPTs) | ✓ | ✓ (shared attribution) | ✓ | ✓ (shared) | ✓ |
| Personal actor key / key session / PIN session | ✓ | ✓ (role matrix) | ✓ (role matrix) | owner role only | owner session only |
| Public dashboard key | GETs + `POST /resolve` (SELECT-only, proven no writes) | **✗ 403 PERSON_REQUIRED** | ✗ | ✗ | ✗ |
| Approved browser, no name | dashboard GETs | ✗ 403 PERSON_REQUIRED | ✗ | ✗ | ✗ |
| Approved browser + name (Arturo/Luz/Miriam) | ✓ | ✓ (their role matrix) | ✗ | ✗ | ✗ |
| Approved browser + **Michael** | ✓ | ✓ (`owner_selected` allowlist) | ✗ | **✗ 403 OWNER_PROOF_REQUIRED** | ✗ |
| Pending/expired/revoked browser token | ✗ 401 | ✗ | ✗ | ✗ | ✗ |

With the flag off, the dashboard key keeps its full old allowlist (compatibility, [V]).

Human route callers were checked in source [V]:
- the dashboard calls only `GET /customers`, and creates customers through `/customers/prepare`;
- the Assistant relays only to prepares, `/tickets/{t}/commit`, `/attachments`, `/resolve` and GETs.

So no legacy direct write had to be opened to people.

### 4.3 Privileged actions that reject a selected identity, even when Michael is chosen

`permissions.OWNER_PROOF_ACTIONS` (computed as every owner action not on the reviewed allowlist):
- `approve_exception`: approve or reject held corrections, plus owner-kind resolutions and waivers;
- `kosher_attestation`: also `kosher_tiers.owner()`/`attest()`, which refuse `via='selection'` directly, including the make-commit fallback;
- `backdate_over_14d`;
- `close_order_shipped_not_recorded`, `reopen_order`, `reopen_order_shipped_not_recorded`.

Also refused:
- voids of other people's posts or other plant days (floor rule);
- `/auth/devices*`, `/auth/actors/*` and `/auth/browsers` approve/list/revoke (owner key session only);
- every master-only route.

A selected Michael's late entries (48 h to 14 d) open a LATE_ENTRY exception like staff entries, and his corrections over 500 lb are held for owner approval.

### 4.4 Assurance never crosses

- Tickets bind to the person **and** the assurance level. A selected Michael cannot commit a draft prepared with his own credential, and the reverse is also refused (`TICKET_WRONG_USER`).
- A held ticket approved by the owner posts as the preparer and keeps the selection assurance.
- Change user cannot commit the previous person's ticket.

## 5. Schema, flags, files

- **Migration `migrations/091_trusted_browsers.sql`:** additive and re-runnable.
  - New tables `trusted_browsers`, `trusted_browser_attempts`, `trusted_browser_audit` (append-only trigger) and `person_choices`. All have RLS on and PUBLIC/anon/authenticated revoked.
  - Nullable assurance columns on `write_tickets`, `actor_write_audit` and `shift_confirmations`.
  - Down script `migrations/down/091_trusted_browsers_down.sql` refuses while selection attribution exists, unless the rows were exported and `factory_ledger.confirm_selection_export='yes'` is set.
- **Flag `PERSON_SELECT_ENABLED`** (default `0`).
  - `/auth/session/config` reports MODE (`person_select_enabled`, the flag) separately from AVAILABILITY (`person_select_available`, `person_select_problem`).
  - Mode on with availability off means blocked with "not available". It never shows key or PIN prompts.
  - `KEY_SESSIONS_ENABLED` and `PIN_SIGNIN_ENABLED` stay as they are; they are the owner-proof path.
- **Seed:** `scripts/seed_person_choices.py` prints a self-checking transaction. It never connects to a database. It checks exact actor id, name, routine role and active status, and aborts on the first mismatch. It never renames.
  - Production mapping (Codex verified [C]): `--michael 1:Blubber --luz 3:Luz --arturo 2:Arturo --miriam 4:Miriam`.
  - Staging choices [C]: Michael 1000000028, Arturo 1000000029, Luz 1000000030, Miriam 1000000031. 1000000116 "Michael (staging)" is an extra test owner and is not a choice. Read the exact staging actor names before generating the seed.
- **Backend:** `trusted_browsers.py` (new), `main.py`, `key_sessions.py`, `permissions.py`, `write_tickets.py`, `exceptions_enforcement.py`, `shift_summary.py`, `kosher_tiers.py`, `resolution_receipts.py`, `ship_tickets.py`, `attachments.py`.
- **Dashboard:**
  - new: `person-select.js`, `person-select.css`;
  - changed: `dashboard-runtime.js`, `ledger-ui.js`, `key-session.js`, `fl-assistant.js`, `shift-summary.js`, `staff-access.js`, and 14 HTML pages (script tags, page flags, cache-busters).
  - Owner-proof pages: `staff-access.html`, `pin.html`, and any page with `?owner=1` (for example `exceptions.html?owner=1` for approvals).
- **New routes** (none in any GPT schema): `POST /auth/browsers/request`, `GET /auth/browsers/status`, `GET /auth/browsers`, `POST /auth/browsers/approve`, `POST /auth/browsers/{browser_id}/revoke`.

## 6. Regression Guard inventory (completed before app edits)

- **Read:** Known Root Causes and Permanent Rules, unchanged (no GPT instruction edits, no 6543 readonly, ET time, migrations idempotent).
- **Full rows read, untruncated:** 62, 142, 157, 159, 163–166, 169–172, 176–180, 183, 185, 189–200, 202–204 (both 204s), 207, 210, 234, 237–240. Breaks-If-Reverted text was read for 117, 121, 131, 133, 143, 162, 173 (both), 175, 182, 191, 192, 205, 219, 221 and 72/68. These were widened by every touched helper, route and table (`_authorize_api_key`, `_actor_identity`, `_VerifiedApiIdentity`, `role_of`, `require_route`, `kosher_tiers.owner/attest`, `require_void`, `write_tickets`, `actor_write_audit`, `shift_confirmations`, `DASHBOARD_KEY_ALLOWLIST`, `ACTOR_WRITE_ALLOWLIST`, `key_kind`, `caller_source_tag`, `_operator_id`, `data-fl-write-ui`).
- **Preserved:**
  - legacy key comparisons first, and the `flb_` branch only after them (142);
  - actor-key scope unchanged, and the 60 s cache unchanged (142);
  - browser tokens are checked for exact format BEFORE any trusted-browser lookup: only a well-formed `flb_` token (flag on) reaches `trusted_browsers.authenticate`, which reads the live `trusted_browsers` row on every request (no cache, so revocation is immediate); a well-formed but unknown token therefore costs one indexed lookup, like a key session. A malformed `flb_` value never reaches that lookup; it falls through to the unchanged actor-key path (60 s cache, unknown keys never force a refresh) and is refused as an unknown key;
  - the read-only key gets no write (189);
  - `/auth/dashboard-key` still publishes only the scoped key (204/207);
  - PIN salt/limits/089 check, session-only owner routes (238);
  - exact-true write-UI gate and `X-FL-Background` no-slide (190/195/200);
  - person-bound tickets and their expiry (180);
  - `entered_by_actor_id` on every write (169);
  - attribution audit fails closed in the same transaction (157/240);
  - Notes read needs a key (240);
  - owner-only reopen/shipped-not-recorded (202–204);
  - no kosher bypass (173/178);
  - void eligibility comes from FL, not role guesses (193);
  - shift hash contract (177);
  - ship retry guard (234);
  - supplier-lot, pooled-lot, append-only, posted-only and duplicate guards (not touched);
  - dashboard-key writes were removed only in selector mode, and the dashboard tabs now write through selection instead (68/72 note).

## 7. Tests (local only)

| Suite | Command | Result |
|---|---|---|
| New backend tests | `pytest tests/test_trusted_browsers.py` (selector **on**) | 43 passed [V] |
| Selected-person flows | `pytest tests/test_trusted_browsers_flows.py`: Receive by all four people; FL Assistant from a selected session (mocked model only) prepare_receive → Record → DB attribution and browser id, duplicate replay, Change-user refusal; Assistant Make/Pack/adjust correction; Ship order by Luz with a BOL photo | 10 passed [V]; disabling the ticket-assurance write makes 12 tests fail (mutation check) |
| Route/OpenAPI contract | `pytest tests/test_route_contract.py` under exact `requirements.txt` pins (pydantic 2.14.1, task-local venv) | passed; exactly the five new browser routes added [V]. The shared `.venv-test` (pydantic 2.13.5) fails `POST /resolve` at HEAD too: a pre-existing environment drift; the HEAD line was kept |
| Full Python suite, both key-session modes | exact pins, task-isolated clean DB `fl_simple_user_a9d200a3` built from `tests/schema/schema.sql` | **7373 passed, exit 0, in BOTH `KEY_SESSIONS_ENABLED=0` and `=1`**, each on its own fresh clean DB. On the shared `factory_ledger_test` (176 leftover transactions, 135 products) six count-assuming tests fail **identically on untouched HEAD**; the worktree passes them on the clean DB. The shared DB was not reset |
| Pre-existing test defects fixed (test only) | `tests/test_device_pins.py` near-miss secret equalled the real one 1 time in 16; `tests/test_seed_staging.py` counted products committed by the state-model concurrency tests | both reproduced on untouched HEAD (forced trailing `A`: HEAD fails at line 79; state-model tests then the seed test on a fresh DB: HEAD fails 15 == 3) and pass after the narrow fixes. Assertions were not weakened; no backend change |
| JS | `node --test tests/*.js` | 676/676 [V], including 20 in `tests/test_person_select.js` |
| Real browser | `PLAYWRIGHT_MODULE=… node tests/visual/run-person-select.mjs` | 39/39 [V]: real dashboard files + synthetic in-memory FL, no network. Covers selector-off parity, pairing, auto-open names, reload/new tab, a persistent-profile browser close and relaunch without re-seeding, Office grants, Receipt void, Assistant hydration and Change user, composed storage-clear failure, cross-tab, missing script, unavailable mode, R11 transient/permanent config outages on 5 pages with leftover key/PIN identities, R12 on 7 normal pages (leftover key session, enrolled phone and PIN session whose legacy whoami fails: zero legacy requests, the selected request works, zero PIN offers), the owner-proof Staff page keeping its owner sign-in, R13 delayed config (1 s) on Shift Summary, Assistant and Record entry in both known modes, and R14 `?api=` override with no credentialed foreign request |
| Classic visual audit (92 screens × 6 variants, 388 captures) | run by Codex on current worktree assets AND untouched HEAD, task-local output, with the new explicit selector-OFF `/auth/session/config` stub fixture (`tests/visual/lib/stub.mjs`) | 429 FAIL / 0 ERROR on **both**; the only differences are 44 LAYOUT-020 refresh cells. Existing visual debt deferred (scope freeze) |

**Contract tests vs live GPT UI:**
- Compatibility for the Floor/Office schemas' API auth is covered by source/contract tests: the read-only sweep, machine-key-unchanged tests and the route contract.
- **No live Custom GPT UI test passed or was run by this work.** Codex's pre-deploy GPT UI smoke attempts failed in ChatGPT before any task deploy, and the owner skipped further GPT work.

## 8. Rollout (each step needs Michael's explicit approval, a verified backup first and verification after)

**Status 2026-11-03: staging and production are both HELD on Michael's explicit no-deploy instruction.** Codex has verified the staging targets. Nothing in this section has been run.

1. **Staging (held for Codex's exact target verification; do not overwrite another staging effort):**
   - apply 091 on the staging database (port 5432, one transaction, `lock_timeout`);
   - generate and apply the staging seed with exact actor ids and names;
   - deploy the branch to `FastAPI-staging`, then set `PERSON_SELECT_ENABLED=1` there only;
   - pair one browser with the staging owner;
   - record a receive/make/pack/adjust/ship and an order edit as each person;
   - confirm owner-only actions are refused for selected Michael and work from his session;
   - confirm the public dashboard key cannot write;
   - revoke the browser;
   - read the rows back.
2. **Production, in this order:**
   - backup (`BACKUP OK`);
   - apply 091;
   - deploy the backend with the flag still **off**;
   - publish Netlify (Railway before Netlify, per row 204);
   - seed `person_choices` (production mapping above);
   - verify `/auth/session/config` reports availability;
   - set `PERSON_SELECT_ENABLED=1`;
   - Michael pairs each staff browser;
   - check one write per person and read back `write_tickets.identity_via`.
3. **Communicate:** Michael approves browsers on Staff & phones. Owner decisions use his own PIN or key (for example `exceptions.html?owner=1`).

## 9. Rollback

- **Fast:** set `PERSON_SELECT_ENABLED=0`. Browser tokens become unknown keys, the dashboard key regains its old reach, and personal keys/PIN work as before. Nothing is deleted.
- **Code:** revert the merge. 091 objects are harmless while the flag is off.
- **Schema:** run the 091 down script only after exporting selection attribution rows; it refuses otherwise.

## 10. Acceptance checklist

- [ ] An approved browser shows the four names on opening. A tap needs no credential, survives navigation, tabs and a browser restart, and has no expiry. [V locally]
- [ ] The active name is prominent, and Change user works. A change in another tab is adopted. Drafts and the Assistant conversation from the previous name are dropped. [V locally]
- [ ] No write without a chosen name, and no silent substitution. An inactive person is disabled while the others continue. A bad configuration blocks everyone. [V]
- [ ] Selected Michael cannot approve, attest, backdate more than 14 days, reopen, or void others' or old posts. His own session can. [V]
- [ ] The public dashboard key and unapproved browsers cannot write in selector mode. Origin is not trusted. [V]
- [ ] Master/GPT, read-only, actor key and session/PIN behaviour unchanged; packing-slip links unchanged. [V contract; live GPT UI not tested]
- [ ] Ledger snapshots unchanged (`Blubber`); selection assurance and the browser are recorded on tickets, metadata audit and shift confirmations. [V]
- [ ] Staging proof complete. [pending Codex targets]
