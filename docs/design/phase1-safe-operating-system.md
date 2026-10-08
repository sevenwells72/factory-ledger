# Phase 1 — Factory Ledger "minimum safe operating system"

**Status:** design document only. Nothing in this document has been built, migrated, deployed or tested. No app code, no migrations, no DB writes.
**Revision 2 (2026-10-07 13:18):** Michael's decisions on PR #73 recorded in §11; the R2 correction-reason list is now fixed (§5 R2); interface neutrality added (§0.7, §1.8).
**Revision 3 (2026-10-07 15:00):** FL Assistant proof of concept = **GO** — F1 is the floor front end (§0.2, §0.7, §1.8, §10 track F); ChatGPT is never taken offline: GPTs go read-only at cutover, the MCP plugin ships read-only before Dec 11, MCP writes move after cutover (§0.6, §10, §10.1 cutover checklist, §11 items 11–16); F1 gets dictation + photo-to-draft (§1.8, §8.1); A4 short aliases exact-match only (§3.2, §3.3); floor identity recommendation recorded as OPEN (§4.4, §11) — **decided later the same day: PIN-only on any device (rev 3.1, §4.4, §11 item 16)**. Part A verification of the GPT auth model, the MCP branch and the PoC architecture is in §11.2.
**Revision 3.1 (2026-10-07 15:11):** floor identity decided — PIN-only login on any device, FL-issued sessions as the only browser credential (§4.4, §11 item 16, A11 2½ d).
**Revision 3.3 (2026-10-07 15:48):** kosher answers — 4,000 CT standard (284 formula → 72 via catalog cleanup), Aug 12 mis-pack left as-is, paper log of extra-kosher batches until cutover + checklist step 1a (§5.3, §10.1, §11 item 19).
**Revision 3.2 (2026-10-07 15:42):** supplier tracking approved — every receipt stores a real `supplier_id`, the lot-code prefix is a label only (§5.2, A5/D2 +1 d, gate 10(e)); Classic #9 split into Regular and Extra-Kosher tiers with owner PIN attestation and pack-source rules (§5.3, A12 3–3½ d, §3.2 alias rule, §11 items 17–18).
**Revision 3.4 (2026-10-07 16:00):** schedule only — no rule changes. Michael approved **3 Codex lanes** and four re-sequencing changes (§10.2, §11 item 20): A3 split into A3a (tables + seeds, 1 d) and A3b so A5/A6 start a day after A1; A9 runs in parallel with A12; D3 split into D3-lite (shift-summary page + Confirm, before the pilot) and the rest after cutover; G1's engineering is built early with only the two editor pastes left for Nov 20, and A10 moves to cutover+1 in two stages. Critical path ≈ 17–18 d; staging pilot soft-starts Mon Nov 2; gate (d) window Nov 9–13. The "simplify for cutover" set (§10.2 item 9) is **held** as the Nov 6 checkpoint lever together with A8. Lane plan with dates: `~/Documents/fl-audits/lane-schedule.md` (outside the repo).
**Revision 3.5 (2026-10-08 12:30):** decisions only — Michael approved the A9 shift-summary contract's three open points (legacy keys get a read-only all-actors view; confirm must echo `summary_hash`; re-confirming appends, history kept) and the A3a backfill rule (blank/missing legacy adjust reasons → `unknown`) on PRs #84/#85 (§11 items 21–24; §7.3 now points at `docs/contracts/shift-summary.md`; §5.1 backfill note).
**Date:** 2026-10-07 · **Against:** `origin/main` @ `ebf153e` (PR #72; rev 3 re-verified against `cb2705c`, PR #73), MCP branch `integration/mcp-with-66` @ `2c6d625` / `feature/mcp-server` @ `d74f747`
**Inputs read:** `FOLLOWUPS.md` (incl. P1.1 and §7), `~/Documents/fl-audits/mcp-readiness-audit.md` (2026-10-07), the GPT retirement map with the Oct 6 inventory rules R1–R10 (`~/Documents/Claude/2026-10-06/gpt-retirement-map/docs-draft/gpt-retirement-map.md`), `docs/mcp-auth-contract.md`, `docs/mcp-migration-plan.md`, `docs/named-actor-writes.md`, `docs/order-create-contract.md`, `docs/staging.md`, `audits/fresh-start/v3/*` (count packet, build backlog, Sunshine reconciliation), `IDEMPOTENCY_KEY_PLAN.md`, `main.py` and `tests/schema/schema.sql` on main, `mcp_server/` on the MCP branches.
**Line numbers** are `main.py` on `ebf153e` and `tests/schema/schema.sql` ("schema:") unless stated. They drift; grep by function name before editing.

---

## 0. Decisions already made (not reopened here)

1. **FL is the authority; AI is only an interface.** Every rule in this document is enforced in `main.py` + Postgres. The ChatGPT plugin, the dashboard and any future client go through the same endpoints and get the same refusals.
2. **The FL Assistant page in the dashboard is the primary chat interface for writes** (decided 2026-10-07 on the proof of concept, §11 item 11 — superseding rev 1's "ChatGPT + private MCP plugin"). ChatGPT stays as a **read** interface and is never taken offline: the two custom GPTs until cutover unchanged, then read-only; the MCP plugin read-only before Dec 11. The MCP adapter on `feature/mcp-server` (Google sign-in, per-user actor keys, two-call confirm) is still reused — for reads first, for writes after cutover (M1–M3); its SQLite ticket moves into FL (§1).
3. **No critical rule may depend on AI instructions.** If a rule only exists in an MCP/GPT instruction file, it is not a rule. Instruction files may *explain* FL's behaviour; they may not *be* the behaviour.
4. **Chat writes use prepare → commit.** One prepare call that returns an exact draft and a ticket; one commit call that posts only that draft.
5. **Ambiguity is resolved against FL records with the user choosing, never guessed.** Products, lots, orders, customers and units (§3).
6. **Cutover = physical count + reset, readiness-gated, target Nov 20** (count packet v3 is ready; reset applies only after owner approval — `audits/fresh-start/`). At cutover the two custom GPTs become **read-only** (write operations removed from their action schemas; a new read-only key issued in the same rotation — §10.1). The read-only MCP plugin is deployed before **Dec 11**; the GPTs are retired on Dec 11 once it is live. ChatGPT is never offline for reads.
7. **Interface-neutral core (added 2026-10-07; decided 2026-10-07 15:00).** Floor transaction entry moves from ChatGPT to the **"FL Assistant" page in the dashboard** (F1), which is also available to office users. Prepare/commit, receipts and resolution are plain FL HTTP endpoints with no client-specific fields: the FL Assistant backend and the MCP adapter are two thin front ends over the same `POST /{action}/prepare`, `POST /tickets/{ticket}/commit`, `GET /receipts/*` and `POST /resolve` (§1.8). Arturo uses F1 on Nov 20; the MCP adapter gains writes after cutover.

Vocabulary used below: **ledger post** = a `transactions` row (receive / make / pack / adjust / ship / found). **Metadata write** = customers, lots, orders, lines, expected receipts. **Actor** = a row in `actors` (migration 052). **Owner** = Michael (`actors.role = 'owner'`; the brief says "admin" — FL's existing role value is `owner`, keep it).

---

## 1. Prepare → commit with an FL-side ticket

### 1.1 What exists today (facts)

- The adapter already implements prepare/commit correctly *except for location*: `ConfirmationStore` is SQLite at `work/mcp-confirmations.sqlite3` (adapter.py:768–855), TTL 120 s, token bound to `digest([env, base_url, group, tool, payload])` + `digest([email, role, actor_key])` + `revision = digest(state)`. No Railway volume is declared, so it resets on every deploy. After the mutating request is sent, any failure becomes `write_outcome_uncertain` and the row *blocks* re-issuing the same proposal — the operator must reconcile by hand.
- The backend has request-level idempotency on exactly one route: `POST /sales/orders` (`sales_order_create_receipts`, PR #67: same `request_hash` → original response; different → 409 `EXTERNAL_ORDER_REFERENCE_CONFLICT`).
- `/receive`, `/make`, `/pack`, `/adjust`, `/ship`, `/sales/orders/{id}/ship`, `/sales/orders/{id}/lines`, `/inventory/found`, `/production/runs` have **no** retry protection: a dropped response + retry posts twice. They do have `mode=preview|commit`, `FOR UPDATE` locks and `occurred_at`/`backfill`.
- Every commit already returns `confirmation_code = "TXN-" + sha256("txn-{id}-cns")[:6]` (main:3276). It is derived from the id after the fact; it is not stored, not searchable, and not a receipt.
- `IDEMPOTENCY_KEY_PLAN.md` proposes an `idempotency_keys` table keyed on a client header. This design supersedes it: the key is FL-issued (a ticket), not client-invented, so a client cannot forget to send one.

### 1.2 Tables (migration 058 `write_tickets`)

```sql
CREATE TABLE write_tickets (
  id              bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  ticket_hash     text NOT NULL UNIQUE,          -- sha256 of the opaque ticket string; the string itself is never stored
  action          text NOT NULL CHECK (action IN (
                    'receive','make','pack','adjust','found','void',
                    'ship_order','ship_standalone','ship_bulk',
                    'rename_lot','update_supplier_lot',
                    'create_order','add_order_lines','update_order_line','cancel_order_line',
                    'update_order_header','update_order_status','close_order','cancel_order',
                    'create_expected_receipt','resolve_exception')),
  actor_id        integer NOT NULL REFERENCES actors(id),
  client_source   text NOT NULL CHECK (client_source IN ('mcp','dashboard','fl_assistant','api')),
  payload         jsonb NOT NULL,                -- the NORMALISED draft: ids only (product_id, lot_id, customer_id, order_id, line_id), numeric qty, unit, occurred_at, reason codes, confirmations
  payload_hash    text NOT NULL,                 -- sha256(canonical JSON of payload); returned to the client, echoed back on commit
  state_hash      text NOT NULL,                 -- sha256 of the validation snapshot (lot balances, order status, lot status, expected-receipt status) at prepare
  draft           jsonb NOT NULL,                -- what was SHOWN to the user: labels, lot codes, lb, warnings, suggested lots (for the receipt page)
  warnings        jsonb NOT NULL DEFAULT '[]',   -- [{code, message, message_es, refs}] e.g. POSSIBLE_DUPLICATE
  status          text NOT NULL DEFAULT 'prepared' CHECK (status IN ('prepared','committed','expired','rejected','superseded')),
  prepared_at     timestamptz NOT NULL DEFAULT clock_timestamp(),
  expires_at      timestamptz NOT NULL,          -- prepared_at + 10 minutes (chat) / 30 minutes (dashboard forms)
  committed_at    timestamptz,
  receipt_number  text UNIQUE,                   -- §2; set in the same transaction as the post
  result_ref      jsonb,                         -- {transaction_ids:[], lot_ids:[], shipment_id, order_id, correction_id}
  response        jsonb,                         -- the exact commit response, replayed verbatim on retry
  acknowledged    jsonb NOT NULL DEFAULT '[]',   -- warning codes the committer explicitly acknowledged
  reject_reason   text
);
CREATE INDEX ON write_tickets (actor_id, prepared_at DESC);
CREATE INDEX ON write_tickets (status) WHERE status = 'prepared';
-- append-only on committed rows: trigger refuses UPDATE when OLD.status = 'committed' (same pattern as actor_write_audit_append_only, schema:4266)
```

`transactions` gains two nullable columns, set at INSERT only (the table is append-only via the 039 triggers, so they must be present at insert): `receipt_number text` (unique, partial index) and `ticket_id bigint REFERENCES write_tickets(id)`. `shipments`, `ledger_corrections`, `sales_orders`, `sales_order_lines`, `lots` get `ticket_id` the same way (metadata writes are UPDATEs, so a nullable column + the `actor_write_audit` row carries it).

### 1.3 Endpoints

**Prepare** — one per action, all `POST`, all return the same envelope. Each reuses the existing preview code path (`mode='preview'` already runs the full validators: receive main:5451, ship 7890, make 8226, pack 8734, adjust 9065, ship_order 15340) and then *adds* the ticket:

| Endpoint | Replaces (becomes internal-only, §1.7) | Input identity rule |
|---|---|---|
| `POST /receive/prepare` | `POST /receive` mode=preview, `/receive/preview` | `product_id` required (no `product_name`); `supplier_id` or `supplier_name` → resolved, ambiguity 409 |
| `POST /make/prepare` | `POST /make` preview, `/make/preview` | `product_id` (batch); `lot_confirmations[]` optional at prepare, **required at commit** (R4, §5) |
| `POST /pack/prepare` | `POST /pack` preview, `/pack/preview` | `source_product_id`, `target_product_id`, `lot_confirmations[]` |
| `POST /adjust/prepare` | `POST /adjust` preview, `/adjust/preview` | `lot_id`, `delta_lb` (signed), `reason_code` from the fixed list (R2) |
| `POST /inventory/found/prepare` | `POST /inventory/found` | `product_id`, qty + uom, `reason_code` |
| `POST /void/{transaction_id}/prepare` | `POST /void/{id}` | returns the full transaction in `draft` so the user sees what is being voided |
| `POST /sales/orders/{id}/ship/prepare` | `POST /sales/orders/{id}/ship` mode=preview, `/ship/preview` | `line_ids` + quantities, `bol_reference` (§8), `lot_confirmations[]` |
| `POST /ship/prepare` (standalone) | `POST /ship` | **`customer_id` required**; no `customer_name`, no `force_create_customer` (audit: REPLACE) |
| `POST /ship/bulk/prepare` | new (§8.3 Sunshine bins) | `customer_id`, `bins[]` |
| `POST /lots/{lot_id}/rename/prepare`, `POST /lots/{lot_id}/supplier-lot/prepare` | `PATCH …/rename`, `PATCH …/supplier-lot` | `reason_code` (R2/R8) |
| `POST /sales/orders/prepare` | `POST /sales/orders` direct | PR #67 contract: `customer_id`, line `product_id`, `unit`, `customer_po`, `external_order_reference` (the ticket *is* the external reference for chat — the adapter passes `receipt_number` as `external_order_reference`, so the existing `sales_order_create_receipts` replay keeps working unchanged) |
| `POST /sales/orders/{id}/lines/prepare`, `…/lines/{line_id}/update/prepare`, `…/lines/{line_id}/cancel/prepare`, `POST /sales/orders/{id}/header/prepare`, `…/status/prepare`, `…/close/prepare`, `…/cancel/prepare` | the matching direct routes | ids only |
| `POST /expected-receipts/prepare` | `POST /expected-receipts` | `product_id`, `supplier_id` |

Prepare response envelope:

```json
{
  "ticket": "wt_7f3c…(43 chars, opaque)",     "ticket_id": 1842,
  "action": "make",       "expires_at": "2026-10-07T17:12:00Z",
  "payload_hash": "sha256…",
  "draft": { "product": {"id":107,"name":"Granola Classic Batch #9","odoo_code":"90001"},
             "output_lb": 323, "lot_code": "26-10-07-CLS9-001 (new)",
             "ingredients": [ {"product_id":12,"name":"Coconut Flake Desiccated","need_lb":40,
                               "suggested_lot": {"lot_id":9123,"lot_code":"24-09-30-…","on_hand_lb":260,"suggested":true,"confirmed":false} } ],
             "occurred_at": "2026-10-07T12:55:00-04:00", "happened_vs_now_minutes": 3,
             "production_warning": {…kosher…} },
  "warnings": [ {"code":"POSSIBLE_DUPLICATE","message":"A make of Granola Classic Batch #9, 323 lb, was posted 14 minutes ago as receipt MK-261007-006 by Arturo.","refs":{"receipt_number":"MK-261007-006","transaction_id":2451}} ],
  "blockers": [ {"code":"LOT_NOT_CONFIRMED","message":"Confirm the coconut lot (type its last 4 characters) before committing."} ],
  "can_commit": false,
  "permissions": {"actor":"Arturo","role":"floor","allowed":true}
}
```

`blockers` non-empty ⇒ the ticket is still issued (so the conversation can continue — e.g. the user types the last-4 and the client calls prepare again, which **supersedes** the first ticket), but commit of this ticket will fail with the same blocker codes.

**Commit** — one endpoint for every action:

```
POST /tickets/{ticket}/commit
{ "payload_hash": "sha256…", "acknowledged_warnings": ["POSSIBLE_DUPLICATE"], "lot_confirmations": [...optional late additions...] }
```

Inside one DB transaction:

1. `SELECT … FROM write_tickets WHERE ticket_hash = sha256($ticket) FOR UPDATE`. No row → 404 `TICKET_NOT_FOUND`.
2. `actor_id` must equal the authenticated actor → else 403 `TICKET_WRONG_USER` (bound to user).
3. `payload_hash` in body must equal stored → else 409 `TICKET_PAYLOAD_MISMATCH` (bound to exact payload — the client is proving it is committing what it showed).
4. `status = 'committed'` → **return the stored `response` with HTTP 200 and `"replayed": true`** (retry returns the original receipt). This is the whole point of the FL-side ticket: a client that lost the response calls commit again and gets the same receipt number, never a second post.
5. `status IN ('expired','rejected','superseded')` → 409 `TICKET_NOT_COMMITTABLE` with the status. `expires_at < now()` → flip to `expired`, 409 `TICKET_EXPIRED` ("prepare again").
6. **Re-validate** by running the action's validator against *current* state: posted-only lot balances (`lot_on_hand()`), lot `status <> 'merged'`, order `state='open'`/status rules, expected receipt still `open`, actor still `active`, role still allowed for this action (§4), `occurred_at` still inside the back-dating limit for this role (§6), every warning in `warnings` that is `requires_ack` is present in `acknowledged_warnings` → else 409 `WARNING_NOT_ACKNOWLEDGED`. If re-validation fails → ticket `rejected` with `reject_reason`, 409 `TICKET_STALE` carrying the fresh validation errors. (The stored `state_hash` is compared and reported as `state_changed: true/false` in the receipt, but it is **not** by itself a reason to refuse — a lot balance moving from 260 → 250 lb while the draft needs 40 lb is fine; the validator decides.)
7. Post exactly the stored `payload` (never the request body) through the existing commit code (receive main:5445 …), passing `ticket_id` and the pre-allocated `receipt_number` into every INSERT.
8. `UPDATE write_tickets SET status='committed', committed_at=now(), receipt_number=…, result_ref=…, response=…`.
9. COMMIT. The response is the normal commit response plus `{receipt_number, ticket_id, replayed:false, state_changed}`.

Because step 1's lock, the post and step 8 are one transaction, a crash anywhere rolls the ticket back to `prepared` and the retry proceeds normally. There is no `inflight`/`uncertain` state and nothing for a human to reconcile. The adapter's `write_outcome_uncertain` path goes away: on timeout the adapter simply calls commit again.

Concurrency: two commits of the same ticket serialize on the `FOR UPDATE`; the second sees `committed` and replays. Two *different* tickets for the same physical event are what the possible-duplicate warning (§1.5) is for.

### 1.4 Expiry, supersession, cleanup

- Expiry: 10 minutes for `client_source='mcp'` (a floor conversation: "here is the draft — yes"), 30 minutes for `dashboard` forms. Decision D2 (§11).
- A new prepare by the same actor for the same `(action, payload_hash)` marks older `prepared` tickets `superseded` (adapter behaviour kept). Different payload → independent tickets; both may commit (that is two different events).
- Nightly job (`scripts/expire_tickets.py` or a startup sweep — **gated by a migration marker like 051's sweep is not; make it a plain idempotent UPDATE**) flips `prepared` past `expires_at` to `expired`. Rows are never deleted; the ticket log is part of the audit trail.

### 1.5 "Possible duplicate" warning

Computed at prepare, stored in `warnings`, `requires_ack: true`:

| Action | Match rule (posted, `effective_status='posted'`, any actor) | Window |
|---|---|---|
| receive | same `product_id` and (same `supplier_lot_code` **or** same `total_lb` ± 0) | 24 h |
| make | same batch `product_id` and same `output_lb` | 45 min |
| pack | same `target_product_id` and same `cases` | 45 min |
| adjust / found | same `lot_id` (found: same product) and same `delta_lb` | 2 h |
| ship_order | same `sales_order_line_id` with any shipped qty since the prepare's own preview | 24 h |
| ship_standalone / bulk | same `customer_id` and same total lb | 24 h |
| create_order | handled by PR #67 `DUPLICATE_CUSTOMER_PO` — surfaced as a warning in the draft with `allow_duplicate_po` as the acknowledgement |

The warning names the earlier receipt number, who posted it and how long ago. It never blocks; the commit must carry the acknowledgement. Make/pack windows are short on purpose: two Classic batches an hour apart are normal; two in 20 minutes are usually one batch typed twice.

### 1.6 What the MCP adapter changes to (after cutover — M1; the read-only deploy before Dec 11 ships the adapter's 17 read tools unchanged, §10 M0)

- `ConfirmationStore` is deleted. `_write()` preview phase calls `/{action}/prepare`, returns FL's `ticket`, `draft`, `warnings`, `blockers`, `expires_at` to the model verbatim. Commit phase calls `POST /tickets/{ticket}/commit` with `payload_hash` and the acknowledgements the user gave. Token binding/principal/revision logic moves into FL (§1.3 steps 2–6).
- `verify_order_readback` stays (cheap, catches adapter bugs) but a readback failure is reported as `verification: pending`, never as uncertain-do-not-retry.
- Catalog: write tools keep `phase` + `confirmation_token` (renamed `ticket`) control fields so the ChatGPT-side contract is unchanged; `receive/make/pack/adjust/ship` input schemas switch from `product_name`/`customer_name` to `product_id`/`customer_id` and the model obtains ids via `POST /resolve` (§3). `createOrder` blocker and the 200-order scan are removed (stale after PR #67; audit §2.3 row 29). Standalone `ship` loses `force_create_customer`.

### 1.7 Endpoints that become internal-only

"Internal-only" = reachable with the master `API_KEY` only (removed from `DASHBOARD_KEY_ALLOWLIST` and `ACTOR_WRITE_ALLOWLIST`), kept for migration tooling and `audits/fresh-start/apply_reset.py`, and deleted after GPT retirement (Dec 11).

- `POST /receive`, `/make`, `/pack`, `/adjust`, `/ship` with `mode=commit`, and the `/{action}/preview|commit` siblings (main:15746–15791).
- `POST /sales/orders/{id}/ship` (mode=commit), `/ship/commit`.
- `POST /inventory/found`, `/inventory/found-with-new-product` (direct).
- `POST /void/{id}` direct; `PATCH /lots/{id}/rename`, `PATCH /lots/{code}/supplier-lot` direct.
- `POST /sales/orders` direct (dashboard intake `/sales/orders/extract/approve` keeps calling `_create_sales_order_core` internally — same core, so PR #67 idempotency is identical through both doors), `POST …/lines`, `PATCH …/lines/{id}/update|cancel`, `PATCH /sales/orders/{id}`, `PATCH …/status`, `/close|/cancel|/reopen`.
- `POST /records/transactions/{id}/corrections`, `/records/certifications`, `/lots/{id}/reassign`, `/admin/*` — already master-only; unchanged.

Reads are untouched. The dashboard moves its existing ship preview to `/ship/prepare` and gains the commit button it never had (retirement map F5).

### 1.8 Interface neutrality — the FL Assistant (F1) and the MCP adapter are both thin clients over the same FL endpoints

**Decision (2026-10-07, §11 item 11): F1 is the floor front end and is available to office users.** Evidence from the proof of concept (`spikes/fl-assistant/reports/reliability.md`, 2026-10-07): 30/30 turns produced a forced tool call, 0 false "recorded" claims (raw model notes audited, not just filtered), mean 4.2 s/turn, ≈ $3/month at 50 entries × 30 days plus audio at $0.0045/min, double-Record returned the same receipt across sequential and concurrent requests, desktop and phone push-to-talk passed.

**What the PoC actually is (verified from `spikes/fl-assistant/backend.mjs`, `server.mjs`):** a browser page plus a small Node server. The server holds the OpenAI key, runs the model with tool calls *forced* (no free-text escape tool), keeps the prepare ticket in server memory and hands the browser only an unrelated draft id, and on "Record" calls the ledger's commit **itself** — the model never commits. The PoC did **not** call Factory Ledger: it drove a throw-away `spike_*` MCP server through OpenAI's hosted-MCP tool. Rev 2 said F1 "calls the same endpoints from the browser" — that was wrong on two counts and is corrected here: F1 needs a server-side piece (key custody, ticket custody, commit), and that piece talks to FL directly, not through the MCP service.

**F1 architecture (binding):**

- **Page:** `dash/fl-assistant` in the dashboard — a ChatGPT-style chat: type, **push-to-talk dictation** (server-side transcription; the transcript is shown and editable *before* send — dictation never sends a turn by itself), **photo upload** (§8.1). Bilingual.
- **Backend:** `POST /assistant/*` routes **inside the FastAPI service on Railway** — the only place it runs (never a laptop, never a local tunnel; the PoC's cloudflared tunnel is PoC-only). `OPENAI_API_KEY` is a Railway secret on that service; it never reaches the browser, a URL, a log or the repo. Before go-live: raise the OpenAI hard spend cap above the current $20 and turn on e-mail alerts.
- **Tools are FL endpoints and nothing else.** The model's tools are thin function-tool wrappers whose implementations are the same `POST /resolve`, `POST /{action}/prepare`, `GET /receipts/*`, `GET /reports/shift-summary`, `GET /inventory/lookup` handlers every other client uses. **No hosted-MCP tool** in F1 (the MCP service is not on F1's path and F1 does not depend on M0–M4). **No business logic in F1's backend** — it formats FL's `draft`/`warnings`/`blockers`/`ask` and FL's refusals; it never computes balances, picks a product, or decides a permission. Likewise the MCP adapter (§1.6) only relays. Both front ends therefore get byte-identical refusals from FL.
- **Commit is never a model action.** `POST /tickets/{ticket}/commit` is called by the F1 backend when the user presses **Record** on a draft card; the model has no commit tool. F1 shows a draft card only for a real prepare result. Pressing Record twice or retrying after a dropped response re-sends the same ticket and gets FL's replayed receipt (§1.3 step 4).
- **Real read tools from day one:** "what did I enter today" (`GET /receipts?date=today&actor=me`), inventory lookup, the shift summary — so the model answers questions from FL data, not from memory. No free-text escape tool: every turn must end in a tool call or in a refusal the UI shows as such.
- **Lot numbers** are first-class: the user can say or type a lot code / last-4; F1 passes it as `lot_confirmations[]` / to `POST /resolve kind=lot` — it never invents one.
- **Clear "NOT recorded" state.** Network failure, timeout, 4xx/5xx or a missing receipt number renders a red *NOT recorded — try Record again or use the dashboard form* card. The only green state is a receipt number returned by FL. Model prose is a grey note and is never allowed to read as a confirmation.
- **Photo and voice only fill a DRAFT.** Anything read from a photo (BOL number, carrier, lot tag) or from dictation lands in the draft the user sees and confirms; nothing read from media is ever posted directly (§8.1).
- **Identity:** every F1 entry is recorded against the **person** (`write_tickets.actor_id`, `entered_by_actor_id`), never a device. Sign-in is a personal 4-digit PIN on any device → FL-issued 10-min session (§4.4, decided).
- **`client_source='fl_assistant'`**, 10-minute ticket life (same as `mcp`; 30 min for `dashboard` forms). Binding to actor and `payload_hash`, re-validation, replay and acknowledgements are identical for every client.
- **Resolution is a plain endpoint** (`POST /resolve`, §3.2) returning candidates and the `ask` string; F1 shows the choices, the user picks, F1 sends the chosen id back. The decision rules (never auto-pick when ≥ 2 plausible; short aliases exact-match only) live in FL, so neither client's prompt can loosen them.
- **Receipts and the shift summary are endpoints first, screens second**; F1, the dashboard page and the MCP tool render the same JSON.
- **Build-plan effect** (§10): F1 depends on A1 + A4 (A5/A6 for lot confirmations, BOL and photo evidence). F1 is on the critical path; M1–M4 (MCP writes) are off it and after cutover; M0 (read-only MCP deploy) is off it and before Dec 11.

---

## 2. Receipt numbers

**Rule the floor can apply:** *no FL receipt number = not recorded.* Anyone can type a receipt number into the dashboard and see what it posted; if the number does not exist, the event was not recorded, whatever the chat said.

**Format:** `<PREFIX>-<YYMMDD>-<NNN>` — `RCV` receive, `MK` make, `PK` pack, `ADJ` adjust, `FND` found, `SHP` ship (order, standalone, bulk), `VD` void, `LOT` rename/supplier-lot, `SO` order create/edit (chat-created orders also get their `SO-YYMMDD-NNN` order number as today; the receipt is the ticket), `XR` exception resolution. `NNN` is per prefix per plant day from a `receipt_counters (prefix, business_date, next)` row locked `FOR UPDATE` inside the commit transaction (same pattern as the SO number). Short enough to say over the phone and write on paper; the prefix tells Arturo what it was without looking it up.

**Storage:** `write_tickets.receipt_number` (unique) and `transactions.receipt_number` (set at INSERT). A ship that posts three transactions carries one receipt number on all three. The existing `confirmation_code` stays in responses for one release for GPT compatibility, then is dropped.

**Endpoints (dashboard allowlist + actor):**

- `GET /receipts/{receipt_number}` → `{receipt_number, action, status, actor:{name,role}, client_source, happened_at, entered_at, draft, response, transactions:[{id, type, lines:[{product, lot_code, quantity_lb}], effective_status}], lots, shipment, order, warnings, acknowledged, exceptions_opened:[…]}`. 404 if unknown — that 404 is the "not recorded" answer.
- `GET /receipts?date=&actor=&action=&status=` → list for the day (feeds the end-of-shift summary §7.3).
- `GET /receipts/by-transaction/{transaction_id}` → reverse lookup for Ledger History rows.

**Dashboard:** a "Receipt #" box in the shell header (every page) → receipt page. Ledger History and Activity rows show the receipt number as the primary id instead of the raw transaction id. Voided posts show the `VD-` receipt that voided them.

**Chat (F1 and, after cutover, the MCP adapter):** the commit result is "Recorded — receipt **MK-261007-007**." If the client cannot show a receipt number returned by FL, nothing was recorded — F1 renders that as its red *NOT recorded* card (§1.8); the MCP instruction file says so; the *enforcement* is that the dashboard lookup is the truth.

---

## 3. Resolution: product, lot, customer, order, unit

### 3.1 Today (facts) and the staging test cases

`_tiered_product_search` (main:4139): tier 1 exact (`odoo_code` if all digits, else `LOWER(name)`), tier 2 keyword (`name ILIKE ALL(%word%)`, noise words stripped), tier 3 trigram > 0.25. No alias table is consulted. `POST /products/resolve` (main:4578) **always answers 200 with a `match`** and only lists `alternatives` for keyword/trigram; `resolve_product_id` (main:3980) auto-picks on exact, on keyword-with-one-hit, or trigram > 0.4 with a 0.15 gap, else 409 `PRODUCT_AMBIGUOUS` / `PRODUCT_UNCERTAIN`.

Staging (`FOLLOWUPS.md` P1.1, catalog copy of production, 2026-10-07):

| Query | Today | Required outcome |
|---|---|---|
| `Classic` | auto-resolved to **Granola Classic 25 LB (136)** over four other Classic SKUs, tier `keyword`, confidence `medium` | `ambiguous` — list all 5 Classic products ranked; no `match` |
| `chocolate chip` | auto-resolved to **White Chocolate Chips (53)** over Chocolate Chips Sugar Free / Real 1,000 CT / Real 4,000 CT / Granola Chocolate Chip 25 LB, `keyword`/`medium` | `ambiguous` — 5 candidates; a floor context (`group: floor`, action `make`) ranks the ingredient chips above the finished-goods granola, an order context ranks the granola first, but neither auto-picks |
| `Sunshine 9` | **nothing** (no alias; "Sunshine" is not in any product name — the family is "SS Classic #9", batches 90025/90026 ids 283/284, finished goods 70013–70018 ids 285–290) | after alias `SS ⇄ Sunshine`: `ambiguous` with the 8 SS #9 products grouped (2 batch, 6 finished); with action `make` → the 2 batch products; with action `pack` + a source batch → the 6 finished goods |

These three are the acceptance tests for PR-R (§10), run against staging, plus: `90001` → exact match (odoo code); `Granola Classic 25 LB` → exact; `BS cacao` → Blue Stripes Whole Cacao Beans (alias expansion then keyword); `coconut` → ambiguous (product 12 and 190 are the known duplicate — never pick).

### 3.2 One resolution endpoint

`POST /resolve` (read-only POST; add to `SAFE_POSTS` in the adapter and to both allowlists):

```json
{ "kind": "product" | "lot" | "customer" | "order" | "unit",
  "query": "Sunshine 9",
  "context": { "action": "make" | "pack" | "receive" | "ship" | "order" | null,
               "group": "floor" | "office",
               "product_id": 283, "customer_id": 5, "order_id": 1201 },
  "limit": 8, "offset": 0 }
```

Response:

```json
{ "outcome": "match" | "ambiguous" | "none",
  "query_normalized": "ss 9",   "expansions_applied": [{"from":"sunshine","to":"ss","alias_id":4}],
  "match": { "id": 283, "label": "SS Classic #9 Batch (90025)" }  // only when outcome = match
  "candidates": [ {"id":283,"label":"…","score":0.92,"tier":"alias","why":"alias 'SS' + token '#9'", "context_boost":"batch product for make"},
                  {"id":284,"label":"…","score":0.92,"tier":"alias","why":"…"} ],
  "confidence": "high" | "medium" | "low" | "none",
  "ask": "Which one? 1) SS Classic #9 Batch 90025 (323 lb)  2) SS Classic #9 Batch 90026 (348 lb)" }
```

**Decision rules (the critical part — in FL, not in the prompt):**

- `match` is returned **only** when exactly one candidate is at tier `exact` (odoo code, exact full name, exact alias, exact lot code, exact SO number, exact customer name/alias) **or** exactly one candidate scores ≥ 0.5 at all. "Several plausibly match" = two or more candidates ≥ 0.5 → `ambiguous`, *regardless* of the gap. The old "keyword with one hit auto-picks" survives (one hit is one candidate); "trigram 0.4 with 0.15 gap" is removed.
- **Short codes and aliases match EXACTLY, never as substrings (Michael, 2026-10-07, §11 item 15).** A `search_aliases` row is applied only when the whole query, or a whole whitespace-delimited token of it, equals `alias_norm` — `SS` expands in `SS 9` and `ss classic`, never inside `glass`, `mass` or `SSX`; `BS` never inside `herbs`. Aliases never enter the keyword (`ILIKE %word%`) or trigram tiers; after expansion the *expanded* text goes through the normal tiers. Odoo codes and lot codes are already exact-only; this makes aliases the same. Acceptance tests (added to the §3.1 set, run on staging): `glass jar` → no `SS` expansion; `SSX` → none/no expansion; `SS 9` → the SS #9 family (`ambiguous`); `bs cacao` → Blue Stripes Whole Cacao Beans.
- **Kosher-tier alias rule (Michael, 2026-10-07, §5.3):** `SS` / `Sunshine` together with `#9` / `9` / `Classic 9` → only the **Extra-Kosher** Classic #9 products (283, 284 for `make`; 285–290 for `pack`/`order`). `#9` / `Classic 9` / `9` **alone** → both tiers as candidates (regular 107/108 and their finished goods, extra-kosher 283–290), never auto-picked. `SS` means Sunshine's product, which is the extra-kosher version.
- `none` when no candidate ≥ 0.25. The response still returns the best 3 below threshold under `near_misses` so the user can say "no, I meant…", but `candidates` is empty and the model has nothing to pick.
- **Pagination and ranking (corrected 2026-10-08):** `limit` defaults to 8 (1–25); `offset` defaults to 0 (nonnegative). Counts and ambiguity decisions use the complete deduplicated candidate set before pagination. Rank by **score → context rank → recency → name → id** (first three descending, name/id ascending). Michael's context decision: make → batch/ingredient products (ingredients first for floor make); order, ship, and pack without a product ID → finished goods first (order also includes services). Pack with `context.product_id` restricts candidates to that batch's finished children; receive favors ingredients/packaging. Product recency is the latest posted transaction occurrence or open, unfulfilled order line created within 180 days. Context and recency only reorder equal scores; neither changes scores or promotes an ambiguous result to `match`. The prior recency-before-context attribution to Michael was incorrect.
- Inactive products/customers/merged lots are excluded, except that an exact lot-code hit on a merged lot returns `outcome: none` with `note: "lot merged into …"`.
- `confidence` is derived: `high` = tier exact/alias; `medium` = keyword; `low` = trigram. It is informational — the model may *not* use it to pick; FL has already decided.

Per kind:

| kind | Sources searched, in order | Context use |
|---|---|---|
| product | `odoo_code` exact → `search_aliases` (global) exact → `customer_product_aliases` (when `customer_id`) / `supplier_product_aliases` (when `supplier_id`) → name exact → keyword after token expansion → trigram | action filters type (`batch`/`finished`/ingredient) as boost; `confirmed_sku` logic in make (sibling SKUs) becomes an `ambiguous` outcome here |
| lot | exact `lot_code` (upcased, `LOT` suffix normalised per `lots_product_code_norm_uniq`) within `product_id` if given → `supplier_lot_code` / `lot_supplier_codes` → **last-4 suffix match** (R4) → trigram on lot_code | `product_id` required for suffix matching; suffix hitting 2+ lots → `ambiguous` ("type the full code or scan"); lots with `on_hand_lb ≤ 0` excluded unless action is `adjust`/`void` |
| customer | exact name → `customer_aliases` → `LIKE` on name/alias → trigram | existing `_pick_by_address` becomes a boost only when `customer_address` is supplied (keeps FOLLOWUPS §3 tuning path) |
| order | exact `order_number` → `customer_po` (+customer) → "open orders for customer X" (candidates = open orders, newest first) | `context.customer_id`; `status`/`state` filters; "the Setton order" with 2 open orders → `ambiguous` always |
| unit | strict enum `lb`, `cases`, `bags`, `boxes`, `each`, `oz`. **Michael's approved rule (FOLLOWUPS P1.2, 2026-10-07; confirmed 2026-10-08):** for a pouch product, convert a whole pouch quantity to cases only when its verified pouches-per-case divides evenly; show the conversion in the draft. Remainders or missing/conflicting factors → clarify, never round; non-pouch products never silently change units | `pack_format`, catalog UOM, verified bag/case weight and `is_service` constrain allowed units. Bulk ingredient bags require a bag UOM and a consistent positive catalog weight. A bare number without a unit → `ambiguous`; prepare refuses a draft without a unit (422 `UNIT_REQUIRED`) |

`POST /products/resolve` (bulk, used by PO intake) keeps its shape but its per-name result follows the same rule: `match` only when unambiguous, else `match: null` + `candidates`. `resolve_product_id` / `resolve_customer_id` inside write handlers are retired for ticketed actions (ids only), kept for the internal-only direct routes until they are deleted.

### 3.3 Aliases (migration 060)

**A4 split approved by Michael, 2026-10-08:** PR #81 is **A4 part 1 (resolution + seeds)**. Migration 060 ships only the alias table and the five approved token rows below. `resolution_log` and `/aliases` write/prepare/deactivate operations are **A4 part 2, scheduled after A1 merges**; the read endpoint itself does not learn aliases or write business records.

```sql
CREATE TABLE search_aliases (
  id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  kind text NOT NULL CHECK (kind IN ('token','product','customer')),
  alias text NOT NULL,                       -- 'SS', 'BS', 'classic 9'
  alias_norm text GENERATED ALWAYS AS (lower(regexp_replace(alias,'\s+',' ','g'))) STORED,
  expansion text,                            -- kind=token: 'Sunshine' (bidirectional: both directions are tried)
  product_id integer REFERENCES products(id),  -- kind=product
  customer_id integer REFERENCES customers(id),-- kind=customer (supersedes customer_aliases over time; migrate its rows)
  language text NOT NULL DEFAULT 'any' CHECK (language IN ('any','en','es')),
  active boolean NOT NULL DEFAULT true,
  created_by integer REFERENCES actors(id), created_at timestamptz DEFAULT clock_timestamp(),
  deactivated_by integer REFERENCES actors(id), deactivated_at timestamptz,
  UNIQUE (kind, alias_norm, COALESCE(product_id,0), COALESCE(customer_id,0))
);
```

Seed rows: token `SS ⇄ Sunshine`, `BS ⇄ Blue Stripes`, `CLS ⇄ Classic`, `choc ⇄ chocolate`, `#9 ⇄ 9`; the Spanish forms Arturo uses (`coco ⇄ coconut`, `avena ⇄ oats`, `chispas ⇄ chips`) — to be collected from Arturo in week 1 (Decision D5). Matching is **whole-token exact** on `alias_norm` (§3.2) — an alias is never found inside a longer word, so short codes like `SS`/`BS` are safe to seed. `customer_product_aliases` / `supplier_product_aliases` stay (they carry case sizes and lb-per-unit) and are consulted when the context has a customer/supplier.

**Alias maintenance on the dashboard:** Settings → *Aliases* tab: list (kind, alias, points to, who added, last matched), add, deactivate (never delete; deactivation keeps the resolution log honest). Endpoints `GET /aliases`, `POST /aliases/prepare` + commit (owner/office), `PATCH /aliases/{id}/deactivate`. Also an inline "Add alias" on the receipt page and on an `ambiguous` resolution shown in the dashboard ("Classic → always mean Granola Classic 25 LB for **Setton**" writes a `customer_product_aliases` row). Chat **cannot** create aliases in Phase 1 — aliases change what every future resolution does, so they are an office/owner screen action.

`resolution_log (id, actor_id, kind, query, context, outcome, candidates, chosen_id, ticket_id, created_at)` records every call; it is how the thresholds get tuned from data (FOLLOWUPS §3) and how "the model picked wrong" is investigated.

---

## 4. Identity and roles

### 4.1 People → FL actors

| Person | FL `actors.role` (existing CHECK: `owner`,`floor`,`office`) | MCP allowlist role today | Phase 1 |
|---|---|---|---|
| Michael | `owner` ("admin" in the brief) | `admin_office_floor` | owner |
| Arturo | `floor` — **inventory owner** (R1) | `floor` | floor; default owner of shortage / unidentified-lot exceptions |
| Luz | `office` | `admin_office_floor` | office |
| Miriam | `office` | `admin_office_floor` | office |

Today `actors.role` is **informational only** (main:2802 "permission enforcement stays in MCP"); the MCP has its own two-role vocabulary (`admin_office_floor`, `floor`) in `identity.py`. Phase 1 makes FL's role the only one that matters: the adapter stops deciding permissions (its `can_write_*` helpers reduce to "is this user allowlisted at all" — M2, after cutover; the read-only M0 deploy has no writes to permission) and FL answers 403 per action. F1 relies on the same FL 403s and its own `/auth/whoami` `permissions` map for greying out.

### 4.2 How Google sign-in maps to an FL user

Unchanged mechanics (reuse): Google OAuth (`auth.py`) → verified `email` → `MCP_ALLOWED_USERS` entry → `actor_key_env` → `MCP_ACTOR_KEY_<NAME>` → sent as `X-API-Key` on every call → `_authorize_api_key` (main:3123) sha256-lookup in `actors.key_hash` → `request.state.actor`. The adapter already refuses to write unless `/auth/whoami` says `key_kind == 'actor'`.

Additions (migration 059):

- `actors.email text UNIQUE` — the Google email, lowercased. `GET /auth/whoami` returns `{actor:{id,name,role,email}, key_kind, permissions:{<action>: true|false}}` so the dashboard and the adapter can grey out what the user cannot do (display only; FL still enforces).
- Startup check in the MCP: for every allowlisted user, `/auth/whoami` with that user's key must return an actor whose `email` equals the allowlist email, else refuse to start (catches a swapped `MCP_ACTOR_KEY_*`). Today a swapped key would silently post Luz's writes as Arturo.
- `MCP_ALLOWED_USERS.role` becomes optional and ignored for permissions; keep it only to decide which tool groups to *list* (floor users see the floor server). Long-term the allowlist is just emails.
- No shared key is ever sent by the adapter (audit confirms). The master `API_KEY` rotation (audit §7: the live value is committed in 7 public files) is **Phase 1 PR-0** — before any of this is deployed.

Dashboard identity (Decision D3): today every dashboard write is `operator = 'dashboard'` via the shared `DASHBOARD_API_KEY` shipped in `dashboard.js`. Phase 1 minimum: dashboard write screens (ship commit, exceptions, aliases, corrections) require a *personal* actor key entered once per browser (stored in `localStorage`, sent as `X-API-Key` only on write calls); the shared dashboard key stays for reads. Google sign-in on the dashboard is Phase 2. **Rev 3 (decided, §11 item 16):** replaced by FL-issued short-lived sessions obtained with a personal 4-digit PIN (or, for office/owner, the personal actor key) for both F1 and the D2 forms; no actor key is stored in a browser (§4.4, A11).

### 4.3 Permission matrix (enforced in `_authorize_api_key` → `ROLE_PERMISSIONS[action][role]`, checked again at ticket commit)

| Action (ticket `action`) | owner (Michael) | floor (Arturo) | office (Luz, Miriam) |
|---|---|---|---|
| All reads, `POST /resolve`, receipts, exceptions list, shift summary | ✓ | ✓ | ✓ |
| `receive` | ✓ | ✓ | ✓ (office books paperwork; `lot_confirmations` still required) |
| `make`, `pack` | ✓ | ✓ | ✗ |
| `adjust` ≤ 500 lb and ≤ 10 % of lot | ✓ | ✓ | ✗ |
| `adjust` > 500 lb or > 10 % | ✓ | prepare ✓ / commit → opens exception `LARGE_CORRECTION`, posts only after owner approval (§7) | ✗ |
| `found` | ✓ | ✓ | ✗ |
| `void` | any | own posts, same plant day | own order-side posts? — ✗ (void is ledger-only) |
| `rename_lot`, `update_supplier_lot` (resolve unidentified lot) | ✓ | ✓ | ✓ |
| `ship_order` prepare | ✓ | ✓ | ✓ |
| `ship_order` commit (needs BOL, §8) | ✓ | ✓ | ✓ |
| `ship_standalone` (customer_id, no SO) | ✓ | ✗ | ✓ |
| `ship_bulk` (Sunshine bins) | ✓ | ✓ | ✓ |
| `create_order`, `add_order_lines`, `update_order_line`, `cancel_order_line`, `update_order_header` | ✓ | ✗ | ✓ |
| `update_order_status` | ✓ | only `ready` (existing `/sales-orders/{so}/ready`) | ✓ (never `shipped`/`partial_ship` — already 400) |
| `close_order` with `shipped_not_recorded`, `cancel_order`, `reopen` | ✓ | ✗ | `cancel_order` ✓; `shipped_not_recorded` ✗ (R6) |
| `create_expected_receipt` | ✓ | ✗ | ✓ |
| customers create/update | ✓ | ✗ | ✓ |
| aliases add/deactivate | ✓ | ✗ | ✓ |
| `resolve_exception`: shortage, unidentified lot | ✓ | ✓ (owner of the queue) | ✗ |
| `resolve_exception`: large correction approval, late-entry acceptance, shipment-proof waiver | ✓ | ✗ | ✗ |
| kosher attestation on an extra-kosher `make` ticket (§5.3) | ✓ (PIN, per batch) | ✗ | ✗ |
| Back-dating > 48 h (§6) | ✓ | ✗ | ✗ |
| `/admin/*`, lot merge, BOM, product create, `/records/*` | ✓ (master key today → owner actor in Phase 2) | ✗ | ✗ |

Every ✗ is a 403 `ROLE_NOT_ALLOWED {action, role}` from FL. The MCP lists tools by group as today, but a floor user calling an office tool gets FL's 403, not the adapter's.

### 4.4 Floor identity on the FL Assistant — DECIDED by Michael (§11 item 16)

**Decision of record (2026-10-07):** **PIN-only login, on any device.** No name picker: each person has a unique personal **4-digit PIN** that *is* their identity; typing it on a shared floor tablet or on their own phone gives an FL-issued short-lived session for that person. FL rejects obvious PINs (all-same digits, ascending/descending runs such as `1234`/`4321`, and any PIN already held by another actor), locks a PIN temporarily after several wrong attempts, and ends the session after **10 min idle**. Every entry records the **person**, never the device. Office/owner may also sign in with their personal actor key (Google sign-in in Phase 2). FL-issued sessions are the **only** credential a browser holds — this replaces D3's actor-key-in-`localStorage` for the D2 forms as well. **Owner note:** whether phones are allowed in production areas is CNS's food-safety call; the design supports a shared tablet and personal phones equally.

**Mechanics (A11, §10):**

- `actors.pin_hash text UNIQUE` = HMAC-SHA256(`PIN_PEPPER`, pin) with one **server-side pepper** held as a Railway secret, looked up by equality. (With no name picker the server must find the actor from the PIN alone; a per-row salt would force a scan of every actor per attempt, and the uniqueness rule below is what makes a single lookup unambiguous.) `actors.pin_set_at`, `pin_locked_until timestamptz`, `pin_failed_attempts int`. PINs are set/reset only from the owner screen (`POST /actors/{id}/pin`, owner or the person themself after a session); the setter enforces: exactly 4 digits, not all-same, not a strict ascending/descending run, not `0000`-style, **not equal to any other active actor's PIN** (`409 PIN_IN_USE` — the person picks another; the message never reveals whose it is).
- `actor_sessions (id, token_hash UNIQUE, actor_id, device_label text, created_at, last_seen_at, expires_at, ended_at, ended_reason: 'idle'|'logout'|'revoked'|'pin_reset')`. `POST /auth/session {pin}` → 200 `{session_token, actor:{name, role}, expires_at}` or 401 `PIN_INVALID` (same response for unknown and locked PINs, with `Retry-After` when locked). The token is an opaque 32-byte value sent as `X-API-Key`; `_authorize_api_key` gains `key_kind='session'` (checked **after** the two legacy keys and actor keys, resolving the actor and reaching the same routes as that actor's key). Every authenticated call slides `last_seen_at`; `expires_at = last_seen_at + 10 min`; a call after expiry → 401 `SESSION_EXPIRED` and the client re-asks the PIN. Setting a new PIN ends every open session of that actor.
- **Brute-force limits, because there is no username:** a 4-digit space is ≤ 10,000 values and with ~4–6 actors a blind guess succeeds with probability ≈ 5 × 10⁻⁴ per try. So FL rate-limits *per source* (device cookie `fl_device` set on first visit + client IP): 5 failures → 15-min lock of that source; additionally the PIN *being guessed* is not tracked (unknown), but any **actor** whose PIN is hit after the source is locked is not signed in — the lock applies first. A global ceiling (e.g. 50 failures/hour across all sources → owner e-mail + 1-hour lock on `POST /auth/session`) catches a distributed guess. All attempts are logged (`pin_attempts (source, ok, actor_id?, at)`) and the weekly view (§7.2) shows failed-attempt bursts. The readiness gate (§11 item 10) adds: a scripted 10,000-PIN sweep on staging must sign nobody in.
- **UI:** one screen — a 4-digit keypad, nothing else to pick. After success the header shows "**Arturo**" and every draft card says "Recording as **Arturo**". Any commit after idle expiry re-asks the PIN and then replays the same ticket (the ticket is bound to the actor, so a different person's PIN gets `TICKET_WRONG_USER`). "Sign out" ends the session explicitly for hand-over on a shared device.
- Personal actor keys stay valid as a server-held credential (MCP adapter, scripts) and as the office/owner sign-in; they are never stored in a browser after A11.

**Original analysis (kept for the record):**

**Requirement (Michael, 2026-10-07):** every entry records the actual **person** who entered and confirmed it — "Arturo", never "the production iPad". FL already binds the ticket to the actor (`write_tickets.actor_id`, `TICKET_WRONG_USER` on commit, `entered_by_actor_id` on the row), so the only question is how a person on the floor proves who they are to the F1 backend.

**Facts today:** `actors` (migration 052) holds `name, role, key_hash, active` — one long-lived key per person, minted by `scripts/mint_actor_keys.py`; no PIN, no e-mail (059 adds it), no device or session concept. The dashboard sends the shared `dashboard-key-2026` from `dashboard.js`; no screen asks who you are. D3 (§11 item 3) plans "personal actor key entered once per browser, kept in `localStorage`" for dashboard write screens.

| Option | How it works | For | Against |
|---|---|---|---|
| (a) Per-person login on personal phones | Each person opens F1 on their own phone; personal actor key entered once (Google sign-in in Phase 2). | Strongest identity; nothing to build beyond D3; phone mic already tested. | Puts a work credential on personal devices; phones near open product / with gloves and sticky hands are unreliable; a new or temporary worker needs a phone + a minted key; no floor-wide screen. |
| (b) Shared floor device + per-person PIN session with auto-timeout | One shared iPad/phone at the station; the page shows a **person picker + 4–6-digit PIN**; FL issues a short-lived **session** bound to the actor; idle timeout = ticket life (10 min); every draft card says "Recording as **Arturo**"; a commit after timeout re-asks the PIN. PIN per *entry* was considered and rejected: it adds a tap to every post without adding identity (the ticket is already bound). | Matches how the floor already identifies itself (time clock PIN); no personal device; device holds only the read-scoped dashboard key, never a person's long-lived key; bounded exposure (10 min) if someone walks off; `actors` grows `pin_hash` + lockout, nothing else. | PINs can be watched or shared — mitigated by lockout after 5 tries, PIN change from the owner screen, and the weekly view listing entries per person so a borrowed PIN is visible; ~2 d API + ~1 d UI to build (`actor_sessions`, `POST /auth/session`, `key_kind='session'` in `_authorize_api_key`). |
| (c) Both doors, one session mechanism **(recommended)** | FL-issued short-lived sessions are the *only* credential a browser holds. On the shared device the session comes from picker + PIN (b); on a personal phone it comes from the personal actor key or, in Phase 2, Google sign-in (a). Same `actor_sessions` table, same timeout, same "Recording as …" card, same `TICKET_WRONG_USER` binding. This also replaces D3's "actor key in `localStorage`" for dashboard write screens, so a long-lived key never sits in any browser. | One mechanism for F1, D2 forms and the shared device; floor gets (b), office/owner get (a); identity is the person in every case. | Same build as (b) plus ~½ d to accept a session where D2 planned a raw actor key. |

**Recommendation was (c)** (shared floor device + picker + PIN as Arturo's default); Michael took (c) and removed the picker — PIN only, any device. Reasons given: the person is recorded either way because the ticket is bound to the actor at prepare and checked at commit; a shared device fits the floor (gloves, wet hands, no personal phone near product, visible to whoever is covering); the 10-minute idle timeout equals the ticket life, so a session cannot outlive the draft it could commit; and no long-lived credential ever lives in a browser — the device key stays read-only, the person's key stays on the server. Cost ≈ 2½–3 d total (API `actor_sessions` + PIN + lockout, F1/D2 sign-in UI), folded into A2 (roles) and F1. Resolved by the decision above: PIN is 4 digits; the phones-in-production question is an owner food-safety call, not a design input.

---

## 5. Oct 6 inventory rules → where FL enforces them

| Rule | Where enforced (table / endpoint / check) | Phase |
|---|---|---|
| **R1** Arturo owns inventory accuracy and traceability | `actors.role='floor'` is the default `owner_actor_id` on `exceptions` of kind `SHORTAGE`, `UNIDENTIFIED_LOT`, `NEGATIVE_BALANCE`; owner weekly view (§7.2) shows them under his name; every ledger post carries `entered_by_actor_id` | 1 |
| **R2** corrections need a fixed reason; all on owner's weekly view; > 500 lb or > 10 % highlighted; photo > 500 lb | `correction_reasons (code PK, label_en, label_es, applies_to[], note_required bool, active)` seeded with **Michael's fixed list of 8 (decided 2026-10-07, D4)** — see table below. `adjust/prepare`, `void/prepare`, `rename/prepare`, `supplier-lot/prepare`, `found/prepare` → 422 `REASON_CODE_INVALID` unless `reason_code` ∈ active list; `unknown` → 422 `NOTE_REQUIRED` without a note. Threshold: `abs(delta_lb) > 500 OR abs(delta_lb) > 0.10 * lot_on_hand_before` → `exceptions(LARGE_CORRECTION)` + ticket blocker `PHOTO_REQUIRED` unless an `attachments` row (§8.1) is linked at commit. Weekly view = `GET /reports/weekly` §7.2 | 1 |
| **R3** make/pack never blocked by "not enough stock"; post + flag; Arturo resolves in 2 days; never add stock to get around it | `make`/`pack` commit: replace the 400 "Insufficient inventory" (main ~8524, `validate_lot_deduction` 1767, pack ~9029) with: post the consumption against the confirmed lot (balance goes negative — `lot_on_hand()` already allows it for adjust), insert `shortage_flags (id, transaction_id, product_id, lot_id, short_lb, opened_at, due_at = opened_at + 2 business days, owner_actor_id, status, resolution_kind, resolution_ticket_id)` and `exceptions(SHORTAGE)`. "Never add stock to get around it": `adjust/prepare` with `delta_lb > 0` on a lot/product with an **open** shortage → 409 `SHORTAGE_OPEN_RESOLVE_INSTEAD` (resolution goes through `resolve_exception` with kind `count` → a counted adjust that records the shortage id, or `missing_movement` → the receive/make that was never entered, or `void`); `found/prepare` for a product with an open shortage → same 409. Also: any positive adjust within 30 min *before* a make/pack on the same ingredient by the same actor is tagged `pre_make_adjust=true` and listed on the weekly view | 1 |
| **R4** actual ingredient lot confirmed; FL may suggest oldest, no click-through default; pallet lot recorded on move to production; case items need last-4 typed | `make/pack/ship` prepare returns `suggested_lot` with `confirmed:false`; commit → 422 `LOT_NOT_CONFIRMED` unless every consumed lot has a `lot_confirmations[]` entry `{lot_id, method: 'last4'|'full_code'|'scan'|'pallet', value}` where `value` matches the lot (`last4` = last 4 chars of `lot_code`, unique within the product's active lots else `AMBIGUOUS_SUFFIX`; `pallet` = a `lot_moves` row for that lot with `to_location='production'` within the last 24 h). The adapter cannot fabricate this: it must pass what the user typed. Stored in `transaction_lot_confirmations (transaction_id, lot_id, method, value, actor_id, created_at)`. Pallet moves: `POST /lots/{id}/move/prepare` (`to_location` enum `storage|staging|production`) — a small `lot_moves` table, not full locations (build backlog "Storage locations" 8–12 d stays Phase 2) | 1 (last-4 / full / pallet move); scan = client feature later |
| **R5** substitutions recorded on the batch with a reason | `make/prepare` body `substitutions[] {ingredient_product_id, substitute_product_id, lot_id, reason_code, note}`; stored in `transaction_substitutions`; `excluded_ingredients` requires a `reason_code` too; shown on the trace page and the day summary | 1 |
| **R6** order can't close as shipped without a recorded shipment (proof = loaded truck + BOL, same day) | Already: status `shipped` only via `ship_order` (main:14736 400 on manual). Add: `ship_order` commit → 422 `BOL_REQUIRED` unless `bol_reference` non-blank; stored on `shipments.bol_reference` (new column — today it only exists on `transactions`); `shipments.proof_status` (`complete`|`photo_pending`) — photo attachment required by end of the plant day else `exceptions(SHIPMENT_PROOF_MISSING)` auto-opened at 23:00 local; same-day check: `occurred_at::date = current plant date` else blocker `SHIP_NOT_SAME_DAY` (owner may acknowledge); `close_order` with `shipped_not_recorded` → owner only (§4.3) | 1 |
| **R7** Sunshine bulk: weighed bins (bin ID + tare) leave as a sale, invoiced immediately; pouches invoiced at packing, tracked as Sunshine-owned stock at CNS | §8.3–8.4: `bins`, `ship_bulk` action, `invoice_triggers`; pouch packs set `lots.ownership='sunshine'` + invoice trigger; on-hand reports split by ownership. Full custody/ownership model (backlog 7–12 d) is Phase 2 | 1-lite |
| **R8** unidentified lots allowed but flagged; resolved within 7 days | `lots.identity_status` (`identified`|`unidentified`) + `identify_by date`; `receive/prepare` with blank/`N/A`/`UNKNOWN` `supplier_lot_code` → draft shows "UNIDENTIFIED — resolve by <date>", commit sets status + `exceptions(UNIDENTIFIED_LOT, due_at = received_at + 7 d, owner = floor)`; `supplier-lot/prepare` with a real code and `reason_code` resolves it; the count packet's "Unidentified lot" rows land here too | 1 |
| **R9** weekly/monthly recounts and a monthly mock recall | `recount_schedule` from `audits/fresh-start/v3/recount-groups.csv` (77 weekly / 131 monthly, pending owner approval) → dashboard To-Do reminders + `recounts (product_id, counted_lb, fl_lb_at_count, actor_id, counted_at)`; mock recall = a saved `traceSupplierLot` run with `recall_drills (supplier_lot, run_by, run_at, lots_found, orders_affected)` | 2 (reminders in 1) |
| **R10** full physical count — separate workstream; connection points only | `adjust` reason `physical_count` + R2 thresholds; `FND` receipts with reason `physical_count` or `missing_receipt`; unidentified-lot queue; `apply_reset.py` keeps using the master key on the internal-only direct routes (§1.7) and must write `receipt_number`s so the reset is auditable like everything else | — |

### 5.1 The fixed correction-reason list (R2) — decided by Michael 2026-10-07

Seed of `correction_reasons`. These 8 are the only values accepted on `adjust`, `found`, `void`, `rename_lot`, `update_supplier_lot` and `resolve_exception`; the list is edited by migration only, never from a screen.

| `code` | `label_en` | `label_es` | `note_required` | `applies_to` | Typical use |
|---|---|---|---|---|---|
| `physical_count` | Physical count | Conteo físico | no | adjust, found, resolve_exception(SHORTAGE→counted) | weekly/monthly recounts, the Nov cutover reset, resolving a shortage by counting |
| `missing_receipt` | Missing receipt | Recepción no registrada | no | found, adjust(+), resolve_exception | stock on the floor that was never received in FL; the fix is a `RCV` (preferred) or a found with this reason |
| `missing_production` | Missing production | Producción no registrada | no | adjust, found, resolve_exception | a make/pack that happened but was never posted; the fix is a late `MK`/`PK` with the paper time (preferred) or an adjust with this reason |
| `wrong_lot` | Wrong lot used | Lote equivocado | no | adjust (paired −/+), void, rename_lot, update_supplier_lot | consumption posted against the wrong lot; corrected as a paired adjust or a void + re-post |
| `damage_disposal` | Damage/disposal | Daño o desecho | no | adjust(−) | damaged, spoiled, swept, discarded |
| `unrecorded_usage` | Unrecorded usage | Uso no registrado | no | adjust(−) | samples, R&D, giveaways, hydration/yield loss, anything consumed without a posted movement |
| `data_entry_error` | Data-entry error | Error de captura | no | void, adjust, rename_lot, update_supplier_lot | typed wrong quantity/lot/product; the preferred fix is `void` of the bad receipt + a new post |
| `unknown` | Unknown | Desconocido | **yes** | adjust, found | nothing above fits; the note is mandatory and these rows are always highlighted on the weekly view regardless of size |

**Mapping of today's codes** (hard-coded in `/reason-codes`, main:11256, and free text in `transactions.adjust_reason`). Applied by migration 061 to `transactions.adjust_reason` → new column `transactions.reason_code` for history, and used by the dashboard/MCP to translate any legacy value a client still sends during the overlap (422 after A10):

| Today (adjust) | → Phase 1 | Today (found, `inventory_adjustments.reason_code`) | → Phase 1 |
|---|---|---|---|
| `count_correction` | `physical_count` | `found_during_count` | `physical_count` |
| `damage` | `damage_disposal` | `found_back_stock` | `missing_receipt` |
| `spoilage` | `damage_disposal` | `predates_system` | `missing_receipt` (note auto-filled "predates system") |
| `sample` | `unrecorded_usage` | `unreceived_delivery` | `missing_receipt` |
| `hydration_yield` | `unrecorded_usage` (note auto-filled "hydration/yield") | reassignment codes (`incorrect_receive`, `product_merge`, `supplier_relabel`) | stay on the admin-only `/lots/{id}/reassign` route; not exposed to tickets |
| `other` / any free text | `unknown` with the original text as the note | `data_entry_error` (reassignment) | `data_entry_error` |

Free-text `adjust_reason` rows that cannot be mapped keep their text and get `reason_code = 'unknown'`; the weekly view's first run will show the backlog of `unknown` so it can be cleaned up or left as history. **Decided 2026-10-08 (item 24, PR #85): adjust rows with a blank or missing `adjust_reason` are backfilled to `unknown` too** — every `type='adjust'` row carries a code; nothing is left NULL on history.

### 5.2 Supplier tracking on receipts — APPROVED by Michael 2026-10-07

**Today (verified on production, read-only):** `generate_lot_code` (main:5429–5445) builds the lot code from the plant date plus the **first four letters of whatever supplier name was typed** (`DUTC`, `ABAK`, `QUAL`, `FOUND` …); nothing on `transactions` or `lots` stores a `supplier_id`. The `suppliers` table (52 rows, with duplicates such as Dutch Valley ×4 and the typo row `DUTC Valley`) is referenced by only two expected receipts. "Which vendor did this lot come from" is therefore answered today by guessing from four letters — `DUTC` is both Dutch Valley and Dutch Gold.

**Rule from cutover:** every receipt stores a real `supplier_id` chosen through resolution; the 4-letter lot-code prefix stays as a **label only** and is never used to identify a supplier.

- `POST /receive/prepare` requires `supplier_id` (or `supplier_name` → `POST /resolve kind=supplier`, new sixth kind: exact name → `supplier_aliases` → keyword → trigram, same decision rules as §3.2, ambiguity shown, never auto-picked). No `supplier_id` → 422 `SUPPLIER_REQUIRED`. The pseudo-suppliers (`FOUND`, `INITIAL INVENTORY`, `PHYSICAL COUNT`, `UNKNOWN`, …) are excluded from `kind=supplier` — found stock goes through `found/prepare` with a reason code (R2), not through a fake vendor.
- Storage: `transactions.supplier_id integer REFERENCES suppliers(id)` and `lots.supplier_id` set at INSERT (both tables are append-only for these columns); `expected_receipts.supplier_id` already exists and is carried onto the receive when the receipt is matched. The lot code keeps its prefix from the resolved supplier's **display short code** (`suppliers.short_code`, 4 letters, unique, seeded from today's tokens) so existing labels and the trace pages keep working — but trace, reports and resolution read `supplier_id`, never the prefix.
- Dashboard D2 receive form: supplier picker backed by `kind=supplier`; F1: same resolution in the draft ("from **Dutch Valley Foods** — confirm").
- Catalog prerequisite: the supplier duplicates in `~/Documents/fl-audits/catalog-cleanup.csv` (reviewed separately; not applied by this document) should be merged before cutover so the picker shows one row per vendor.
- **Readiness gate 10(e):** during the five confirmed shift summaries on staging, 100 % of `RCV` receipts carry a `supplier_id`; a staging report lists any receive without one.

Effort: **+1 d** split across A5 (API: columns, `kind=supplier`, `SUPPLIER_REQUIRED`, `short_code`) and D2/F1 (picker + draft line).

### 5.3 Classic #9 — Regular vs Extra-Kosher tiers — decided by Michael 2026-10-07

**What exists (production, read-only, 2026-10-07):**

| id | odoo | name | type | formula / parent | activity |
|---|---|---|---|---|---|
| 107 | 90002 | Batch Classic Granola #9 | batch, 323 lb | 7 ingredients, 322.6 lb | 291 txns (226 in 180 d), 86 lots — the working batch |
| 108 | 90001 | Batch Classic Chocolate Chip Granola #9 | batch, 348 lb | 8 ingredients, 347.6 lb, chips = Real **4,000 CT** (72) | 46 txns, 22 lots |
| 283 | 90025 | Batch SS Classic Granola #9 (Kosher Ignition) | batch, 323 lb | formula **byte-identical to 107** (same md5) | 0 txns, 0 lots since insert 2026-08-12 |
| 284 | 90026 | Batch SS Classic Chocolate Chip Granola #9 (Kosher Ignition) | batch, 348 lb | identical to 108 **except** chips = Real **1,000 CT** (73) | 0 txns |
| 285 / 286 / 287 | 70013 / 70014 / 70015 | Granola SS Classic #9 Bulk per/lb / 25 LB / 10 LB | finished, parent 283 | no `product_bom` | 285: 2 txns (INV-RECON pseudo pack+ship, "QB Original (#9) Bulk"); 286: 2 txns (packed 63 cases **from a regular 107 lot** on 2026-08-12, then adjusted out); 287: 0 |
| 288 / 289 / 290 | 70016 / 70017 / 70018 | Granola SS Classic Chocolate Chip #9 Bulk / 25 LB / 10 LB | finished, parent 284 | no `product_bom` | 288: 0 txns, 1 open order line; 289/290: 0 |
| Regular finished goods packed from 107 today (`product_bom`) | | `Granola Classic 25 LB` (136), `CQ Granola 10 LB` (144), `Granola Crunchy CNS 10 LB Case` (137), `Granola Wheat Free 25 LB` (138), `Granola Setton Good Ol 25 LB` (129), `Granola Fruit Nut Batch` (179); from 108: `Granola Chocolate Chip 25 LB` (143) | | | 136: 93 txns / 43 order lines; 144: 102 txns / 16 order lines (all Restaurant Depot) |
| Live demand | | Sunshine order **SO-260817-001** (298, `open`/`confirmed`): 4,000 lb of 285 + 6,000 lb of 288; an adjust reason notes bulk "likely shipped to Sunshine, order 298 (not enterable in system)" | | | |

**Michael's decision:** two versions, same formula — **Regular Classic #9** and **Extra-Kosher Classic #9**. The difference is the kosher procedure actually followed (Michael personally turns on the oven), not the recipe. FL must distinguish them in production records, product identification and labeling.

**Proposed mapping (for Michael to confirm):**

| Tier | Batch (make) | Finished (pack) | Notes |
|---|---|---|---|
| **Regular Classic #9** | **107** `Batch Classic Granola #9` | 136, 144, 137, 138, 129, 179 (unchanged) | keep name; `kosher_tier='regular'` |
| **Regular Classic Chocolate Chip #9** | **108** | 143 | `regular` |
| **Extra-Kosher Classic #9** | **283** → rename `Batch Classic Granola #9 — EXTRA-KOSHER (SS)` | 285, 286, 287 (rename `… SS Classic #9 …` → keep "SS", add "Extra-Kosher" to the label text) | `extra_kosher`; formula stays identical to 107 — the tier, not the recipe, is the difference |
| **Extra-Kosher Classic Chocolate Chip #9** | **284** → rename likewise | 288, 289, 290 | `extra_kosher`. **Resolved (Michael, 2026-10-07):** 1,000 CT and 4,000 CT real chips are interchangeable; the standard going forward is **4,000 CT** (72). 284's formula is changed from 73 to 72 — pre-approved, applied with the other catalog-cleanup changes (`catalog-cleanup.csv`), not by this document. Using 1,000 CT on any batch remains allowed as a recorded substitution (R5, `substitutions[]`) |

Both tiers of batch stay separate products (not one product with a flag) because FL identifies lots by product and the two must never share a lot; the tier is additionally stored so rules can be enforced across products.

**Rules (enforced in FL, A12):**

1. `products.kosher_tier text NOT NULL DEFAULT 'regular' CHECK (kosher_tier IN ('regular','extra_kosher'))` on batch and finished products; `lots.kosher_tier` copied from the product at lot creation (make output, pack output, receive) and immutable.
2. **Recording an extra-kosher batch requires Michael's PIN attestation.** `make/prepare` for a product with `kosher_tier='extra_kosher'` returns blocker `KOSHER_ATTESTATION_REQUIRED`; the attestation is `POST /tickets/{ticket}/attest {pin}` — the PIN must resolve (§4.4) to an actor with `role='owner'`; FL stores `ticket_attestations (ticket_id, kind='kosher_procedure', attested_by_actor_id, attested_at)` and the make transaction/lot carry `kosher_attested_by_actor_id`. The attestation is per ticket (per batch), expires with the ticket, and can be given from any device — Arturo's tablet shows "Michael: confirm the kosher procedure was followed — enter your PIN". No attestation → commit refused. The attestation is not a session: Michael's PIN here signs one batch and does not sign the floor device in as Michael.
3. **Packing: extra-kosher finished products can ONLY be packed from extra-kosher batch lots.** `pack/prepare` with a target `kosher_tier='extra_kosher'` → every source lot must have `lots.kosher_tier='extra_kosher'` else blocker `KOSHER_SOURCE_REQUIRED` (this is exactly what happened on 2026-08-12 when 63 cases of 286 were packed from a regular 107 lot and then adjusted out — **Michael, 2026-10-07: treat as a probable recording error; do not correct the historical records; the priority is prevention via this blocker**). The `suggested_lot` for such a pack is drawn only from extra-kosher lots.
4. **Extra-kosher batch lots MAY be packed as regular Classic #9** (e.g. 283 lot → 136). Allowed; recorded as `transaction_kosher_downgrade=true` on the pack transaction and `lot_events(event='kosher_downgrade')` on the output lot, which is `regular`. The weekly view (§7.2) lists downgrades so Michael sees extra-kosher material consumed as regular.
5. **Identification and labeling:** `/resolve` returns `kosher_tier` on every candidate and the `ask` string spells it out ("1) Classic #9 — Regular  2) Classic #9 — EXTRA-KOSHER (SS)"); prepare drafts, receipts (`GET /receipts/{n}`), the shift summary, lot labels and the trace page print **EXTRA-KOSHER** on tier rows and show the attesting owner on the batch receipt. The §3.2 alias rule: `SS`/`Sunshine` + `#9` → extra-kosher only; `#9`/`Classic 9` alone → both tiers shown.
6. Ship/orders: a Sunshine order line for 285–290 can only be allocated/shipped from `extra_kosher` lots (follows from rule 3 — the finished lot is already tiered); `QTY` and BOL rules unchanged.

**Interim procedure until cutover (Michael, 2026-10-07):** extra-kosher batches are logged **on paper** — date + lot number — by whoever runs the batch after Michael turns on the oven. On cutover day those lots are tagged `kosher_tier='extra_kosher'` in FL from the paper log (§10.1 step 1a); until then FL has no tier and the 283/284 products are not used for posting.

**Effort and build order:** new row **A12 `feat/kosher-tier`, 3–3½ d** — migration (`products.kosher_tier`, `lots.kosher_tier`, `ticket_attestations`, `lot_events` kind, data update setting tiers on 107/108/136/137/138/129/143/144/179 = regular and 283–290 = extra-kosher, renames per the mapping) ~1 d; make/pack validators + `POST /tickets/{t}/attest` + downgrade recording ~1½ d; `/resolve` tier field + alias rule + receipt/label/trace text ~½ d; F1/D2 attest step UI ~½ d. Depends on A1 (tickets), A5 (lot confirmation, `lot_events`), A11 (PIN → actor). Sits after A5 and before A9 in the critical set; it **cannot slip** past cutover because Sunshine order 298 (10,000 lb of #9 bulk) is open and the Aug 12 mis-pack shows the rule is needed now.

---

## 6. `happened_at` vs `entered_at` vs `entered_by`

### 6.1 Today

`transactions` has all three: `occurred_at` (happened; trigger fills from `timestamp` if missing), `created_at` (entered; `clock_timestamp()`, immutable), `operator_id` (text = actor name, default `'legacy-shared-key'`), plus `business_date` (NY date of `occurred_at`), `created_at_source`, `entry_backfilled`. `validate_inventory_occurred_at` (main:3235): ≤ 5 min future, ≤ 14 days back without `backfill=true`. Daily Entries already computes `days_late`.

**Not separable today:** order create (`order_date = CURRENT_DATE`), order status/close/cancel (`state_changed_at` server), lot rename/supplier-lot (no timestamp of the real event at all), void (`ledger_corrections.created_at` only), production-run complete (`clock_timestamp()`), `lots.received_at` PATCH (no attribution).

### 6.2 Phase 1 contract — every write

Three fields on every row that records something that happened, with one meaning each:

| Field | Meaning | Who sets it |
|---|---|---|
| `happened_at timestamptz` (existing `occurred_at` on `transactions`; new on `ledger_corrections`, `shipments`, `lots` (rename/supplier-lot events go to `lot_events`), `sales_orders.state_changed_at` → keep name, add `state_happened_at`, `production_runs.completed_happened_at`) | when it physically happened, plant time | the user, via the draft; defaults to now; shown back in the draft as "happened 3 min ago" / "happened **yesterday 16:40**" |
| `entered_at timestamptz` (existing `created_at`) | when FL recorded it | DB clock only; never settable |
| `entered_by_actor_id integer REFERENCES actors(id)` (new; `operator_id` text stays as a snapshot for the legacy readers) | the authenticated person | `_authorize_api_key`; never from the body (already true for `operator_id` since PR #66) |

Plus `ticket_id` (which draft produced it) and `client_source`. `lot_events (lot_id, event: 'rename'|'supplier_lot'|'identified'|'move'|'ownership', old, new, reason_code, happened_at, entered_at, entered_by_actor_id, ticket_id)` replaces "no audit of why" on rename (retirement map F3).

### 6.3 Back-dating limits (checked at prepare and again at commit)

| `now − happened_at` | floor / office | owner |
|---|---|---|
| future > 5 min | 400 `OCCURRED_AT_IN_FUTURE` (unchanged) | same |
| 0 – 48 h | allowed | allowed |
| 48 h – 14 d | allowed; receipt flagged `late_entry`; `exceptions(LATE_ENTRY)` opened for the owner to acknowledge; the draft says "This will be recorded as a late entry (2 days)" | allowed, no exception |
| > 14 d | 403 `BACKFILL_OWNER_ONLY` | allowed with `backfill:true` + `reason_code` (existing `api_backfill` stamping) |

Why 48 h: the count packet already treats > 15 min after count start as late; day-to-day, "I forgot yesterday's make" is normal and must not be blocked, but it must be visible. Decision D6.

---

## 7. Exceptions queue, owner weekly view, end-of-shift summary

### 7.1 `exceptions` (migration 061)

```sql
CREATE TABLE exceptions (
  id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  kind text NOT NULL CHECK (kind IN ('SHORTAGE','UNIDENTIFIED_LOT','LARGE_CORRECTION','LATE_ENTRY',
                                     'SHIPMENT_PROOF_MISSING','NEGATIVE_BALANCE','POSSIBLE_DUPLICATE_ACK',
                                     'UNSHIPPED_PAST_DUE','SUNSHINE_INVOICE_PENDING','SHIFT_DISCREPANCY')),
  status text NOT NULL DEFAULT 'open' CHECK (status IN ('open','resolved','waived','escalated')),
  severity text NOT NULL CHECK (severity IN ('info','warn','block')),
  product_id int, lot_id int, transaction_id bigint, sales_order_id int, shipment_id int,
  receipt_number text, ticket_id bigint REFERENCES write_tickets(id),
  detail jsonb NOT NULL,                       -- {short_lb, delta_lb, pct, days_late, …}
  owner_actor_id int REFERENCES actors(id),    -- who must resolve (floor by default for SHORTAGE/UNIDENTIFIED_LOT/NEGATIVE_BALANCE; owner for the rest)
  opened_at timestamptz DEFAULT clock_timestamp(),
  due_at timestamptz,                          -- SHORTAGE +2 business days; UNIDENTIFIED_LOT +7 days; SHIPMENT_PROOF_MISSING end of day; others NULL
  escalated_at timestamptz,                    -- set by the nightly job when now() > due_at; owner is told on the weekly view and the shift summary
  resolved_at timestamptz, resolved_by_actor_id int, resolution_kind text, resolution_note text,
  resolution_ticket_id bigint REFERENCES write_tickets(id)   -- the posting that fixed it (counted adjust, late receive, void, supplier-lot update, photo attach)
);
```

Endpoints: `GET /exceptions?status=open&kind=&owner=&overdue=true` (dashboard + MCP read tool `listExceptions`); `GET /exceptions/{id}`; `POST /exceptions/{id}/resolve/prepare` → `{resolution_kind, note, linked_ticket?}` → commit (ticketed, so it has a receipt `XR-…`). `resolution_kind` per kind: SHORTAGE → `counted` (requires a linked `ADJ` receipt with reason `physical_count`), `missing_movement` (linked `RCV`/`MK` receipt, or an `ADJ` with `missing_receipt`/`missing_production`), `voided` (linked `VD`); UNIDENTIFIED_LOT → `identified` (linked `LOT` receipt) or `written_off` (owner); LARGE_CORRECTION → `approved`/`declined` (owner; approval posts the held adjust — the held ticket's `expires_at` is extended to 7 days for this case); LATE_ENTRY → `acknowledged`; SHIPMENT_PROOF_MISSING → `photo_attached` / `waived` (owner).

Nightly job (`scripts/exceptions_sweep.py`, run by Railway cron or GitHub Action — **not** an app-startup sweep): escalate overdue; open `SHIPMENT_PROOF_MISSING` for the day's shipments without a photo; open `NEGATIVE_BALANCE` for any lot `lot_on_hand() < 0` without an open shortage; open `UNSHIPPED_PAST_DUE` for open orders with `requested_ship_date < today`.

**Dashboard:** *Exceptions* tab — open items grouped by kind, overdue first, owner column, "Resolve" opens the matching form (counted adjust, attach photo, supplier lot, approve). Each row links to its receipt.

### 7.2 Michael's weekly view

`GET /reports/weekly?week=2026-W41` → one page, printable, Monday–Sunday:

1. Open exceptions by kind with age; overdue in red (escalated).
2. All corrections this week (`ADJ`, `FND`, `VD`, `LOT`) with reason code, Δ lb, % of lot, who, happened/entered, photo ✓/✗; rows over threshold highlighted (R2).
3. Shortages opened / resolved / still open > 2 days (R3) and any `pre_make_adjust` tags.
4. Late entries (> 48 h) and back-fills (R-§6).
5. Unidentified lots opened / resolved / overdue (R8).
6. Shipments without proof; orders closed `shipped_not_recorded` (R6).
7. Sunshine: bulk dispatches awaiting invoice; pouch packs awaiting invoice (R7).
8. Shift summaries: days confirmed / confirmed with discrepancies / not confirmed (§7.3).
9. Recount reminders due (R9) — Phase 2 data, Phase 1 shows the schedule.

Dashboard *Weekly* tab (owner/office visible), plus MCP read tool `weeklyReview` so Michael can ask for it in chat.

### 7.3 End-of-shift summary — what Arturo checks against reality

`GET /reports/shift-summary?date=2026-10-07&actor=Arturo` (default: today, the caller) — MCP tool `endOfShift` (floor + office) and a dashboard page on the phone:

```
SHIFT SUMMARY — Tue Oct 7 — Arturo                         FL receipts today: 11
RECEIVED   RCV-261007-001  Coconut Flake Desiccated   1,100 lb  lot 24-09-30-… (supplier lot 8812-A)   08:41
           RCV-261007-002  Oats Rolled                2,000 lb  lot …  ⚠ UNIDENTIFIED — resolve by Oct 14
MADE       MK-261007-001   SS Classic #9 Batch 90025    323 lb  lot 26-10-07-CLS9-001   ingredients: coconut lot …(last-4 typed), …
           MK-261007-002   …                                                      ⚠ SHORT 12 lb oats lot … — resolve by Oct 9
PACKED     PK-261007-001   SS Original 12x10 OZ       120 cases (900 lb) from lot 26-10-07-CLS9-001   ownership: Sunshine → invoice pending
SHIPPED    SHP-261007-001  SO-261003-004 Setton Farms  40 cases  BOL 55821  photo ✓
ADJUSTED   ADJ-261007-001  Almonds lot …  −35 lb  reason: damage
VOIDED     (none)
NOT ENTERED YET?  expected receipt ER-… (Graham crumbs, due today) — no RCV receipt
                  production run #88 SS Original pack, planned today — status planned, no PK receipt
LOT BALANCES TOUCHED TODAY   coconut lot … 260 → 220 lb · oats lot … 0 → −12 lb (short) · …
OPEN EXCEPTIONS (yours)   2 shortages (1 due Oct 9), 1 unidentified lot (due Oct 14)
LATE ENTRIES TODAY        none
Does this match what happened on the floor?   [Confirm]   [Confirm with notes]   [Something is missing]
```

Sections are driven by `write_tickets` (today's committed receipts for the plant day, `happened_at` *or* `entered_at` today — both are listed, late ones marked) + `expected_receipts` due today without a linked receive + `production_runs` planned today without a linked pack/make + `lot_on_hand()` for every lot touched + open `exceptions` owned by the actor.

`POST /reports/shift-summary/confirm` → `shift_confirmations (business_date, actor_id, confirmed_at, outcome: 'match'|'discrepancy', notes, receipts_seen int, summary_snapshot jsonb)`. **The frozen JSON contract is `docs/contracts/shift-summary.md` (PR #84, approved 2026-10-08 — §11 items 21–23); it changes only by revision of this document.** "Something is missing" opens `exceptions(SHIFT_DISCREPANCY)` with the note; the weekly view (§7.2 item 8) shows which days were never confirmed. Not confirming is not blocked — it is visible. The summary is bilingual (`label_es` on every section; reason codes carry `label_es`).

---

## 8. Shipment + BOL gate; Sunshine flows

### 8.1 Attachments (migration 063)

`attachments (id, kind: 'bol'|'truck_photo'|'correction_photo'|'lot_tag'|'count_sheet'|'other', storage_path, sha256, content_type, bytes, uploaded_by_actor_id, uploaded_at, ticket_id, transaction_id, shipment_id, exception_id, extracted jsonb)` in the existing Supabase storage bucket pattern (`purchase-documents` is already wired for ER/SO intake; add bucket `evidence`). `POST /attachments` (multipart, dashboard + actor keys) returns `attachment_id`; a prepare/commit payload references `attachment_ids[]`. Photos come from the **FL Assistant page (F1) or the dashboard forms (D2)**; the MCP cannot reliably relay binaries, so a ChatGPT/MCP flow (after cutover) says "ship it, photo pending" and FL opens the proof exception if no photo arrives by end of day.

**Photo → draft in F1 (Michael, 2026-10-07, §11 item 13; +1–2 d on F1).** Three uses, one rule — *a photo only fills a draft the user confirms; nothing read from a photo is ever posted directly*:

1. **BOL / carrier into a shipment draft:** the user photographs the bill of lading; the F1 backend sends the image to the model (vision) and gets `{bol_reference, carrier, pieces?}`; F1 uploads the photo as `attachments.kind='bol'` with `extracted` = what was read, and calls `/sales/orders/{id}/ship/prepare` with `bol_reference` **pre-filled in the draft** and the attachment id. The draft card shows the read value next to a crop of the photo; the user corrects or confirms; the photo satisfies R6's proof requirement (`proof_status='complete'`).
2. **Lot tag into lot confirmation:** a photo of the ingredient bag's lot tag is read to a lot code candidate; F1 runs it through `POST /resolve kind=lot` (exact or last-4, within the product) and shows the match as a *suggested* confirmation. The user still confirms it (tap "this one" or type last-4); FL stores `lot_confirmations[].method='photo'` with the attachment id (`kind='lot_tag'`). An unreadable or ambiguous tag → `ambiguous`, never a guess.
3. **Evidence for corrections > 500 lb / > 10 %:** the R2 photo requirement is met by uploading from F1 (`kind='correction_photo'`); the attachment id goes on the `adjust/prepare` payload and clears the `PHOTO_REQUIRED` blocker. Nothing is read from this photo.

Storage reuses A6; `extracted` is informational (what the model read) and is never treated as the value of record — the value of record is what the user confirmed in the draft.

### 8.2 Ship gate (all three doors: office `shipOrder`, floor `commitShipOrder`, dashboard button — all hit `ship_order` main:15333)

Prepare draft shows: order, customer, lines with lots (confirmed per R4), total lb/cases, `bol_reference`, `happened_at`, `photo: attached|pending`. Commit refuses: `BOL_REQUIRED` (blank BOL), `LOT_NOT_CONFIRMED`, `SHIP_NOT_SAME_DAY` unless acknowledged by owner, `QTY_EXCEEDS_REMAINING`/`LINE_CANCELLED` etc. (existing). On success: `shipments` row gains `bol_reference`, `proof_status`, `ticket_id`, `entered_by_actor_id`; order status → `shipped`/`partial_ship` (existing); receipt `SHP-…`; packing slip link becomes `GET /sales/orders/{id}/packing-slip?receipt=SHP-…&sig=<hmac>` (short-lived signed link, replaces the static `?key=` — retirement map I1).

Standalone ship (`/ship/prepare`): `customer_id` only, `OPEN_SALES_ORDER_EXISTS` stays as a blocker that only office/owner may acknowledge, no customer auto-create.

### 8.3 Sunshine bulk (weighed bins → sale on departure → invoice)

Tables: `bins (id, bin_code UNIQUE, tare_lb numeric, tare_verified_at, tare_verified_by_actor_id, active)`; `bulk_dispatch_lines (shipment_id, bin_id, lot_id, product_id, gross_lb, tare_lb, net_lb, weighed_at, scale_note, estimated bool)`; `invoice_triggers (id, kind: 'bulk_dispatch'|'pouch_pack', shipment_id, transaction_id, customer_id, product_id, qty, unit, price_basis, status: 'pending'|'invoiced'|'void', invoice_reference, invoiced_by_actor_id, invoiced_at)`.

Flow: `POST /ship/bulk/prepare {customer_id (Sunshine), bins:[{bin_code, lot_id, gross_lb}], happened_at}` → draft computes `net_lb = gross − tare` per bin (tare missing or unverified → blocker `BIN_TARE_UNVERIFIED`; owner can verify tare on the dashboard), total net lb, warns if a bin was dispatched in the last 24 h (possible duplicate). Commit posts one `SHP` ship transaction per lot (balances leave CNS — it *is* a sale), `bulk_dispatch_lines`, and an `invoice_triggers(bulk_dispatch, pending)` row. Office works `GET /invoice-triggers?status=pending` on the dashboard and marks each `invoiced` with the QuickBooks invoice number (ticketed); pending triggers older than 1 business day appear as `SUNSHINE_INVOICE_PENDING` exceptions. Price basis and the later yield-credit question are **not** modelled (build backlog flags it as an owner decision — D8).

### 8.4 Sunshine pouches (invoice at pack, Sunshine-owned stock at CNS)

Products 145–149 (`70003, 70002, 70011, 70070, 70010`, 12×10 oz) are flagged `products.ownership_on_pack = 'sunshine'` (new column, default NULL). `pack/prepare` of such a product: draft says "This pack becomes Sunshine-owned stock held at CNS and triggers an invoice." Commit: output lot gets `lots.ownership='sunshine'` (new column, default `'cns'`), an `invoice_triggers(pouch_pack, pending)` row for the cases packed, and a `lot_events(ownership)` row. Inventory reports (`/inventory/lookup`, FG tab, count sheet) show ownership as a column and totals split `CNS-owned / Sunshine-owned`. A later ship of that lot to Sunshine posts normally but does **not** create a second invoice trigger (already invoiced at pack); a ship of a Sunshine-owned lot to anyone else → blocker `OWNED_BY_OTHER_PARTY` (owner acknowledge only). The historical reconciliation of today's pouch stock (`audits/fresh-start/v3/sunshine-ownership-reconciliation.md`) is an office review after the count, not a Phase 1 code path.

---

## 9. Offline / paper fallback

Three failure modes, one rule: **the paper carries exactly the draft fields, and the later FL entry carries the paper's time and reference.**

1. **FL Assistant unavailable (OpenAI outage, spend cap hit, model refusing), FL up** → use the dashboard forms (ship commit, receive, adjust, found, exceptions) which exist after PR-D1/D2; same tickets, same rules. The dashboard is the designated fallback (migration plan line 320). ChatGPT is read-only after cutover and is **not** a write fallback.
2. **FL down (Railway/Supabase)** → paper. A one-page bilingual *FL Late Entry Sheet* (`docs/manual-tests/late-entry-sheet.pdf`, generated like the count packet): one row per event with the ticket fields — action, product (name + odoo code), lot code **and last-4 confirmation initialled**, supplier lot, qty + unit, happened time (HH:MM), who, reason code, BOL #, bin codes/gross for bulk, a box for the receipt number to be written back. Sheets are numbered (`LE-261007-A`), photographed, and entered when FL returns.
3. **Phone dead / hands full** → same sheet, entered later the same shift.

Entering paper later: `*/prepare` with `happened_at` = the paper time, `entry_source: 'paper'`, `paper_ref: 'LE-261007-A/3'` (both stored on the ticket and the transaction; `transactions.created_at_source` stays `database`/`api_backfill` — the DB clock is still the entered time). FL's response is the receipt number, which is written back onto the sheet. The sheet is then photographed and attached (`attachments.kind='count_sheet'`) to the first receipt of the batch. Late-entry flags (§6.3) apply automatically, so the weekly view shows how much went through paper. Traceability survives because the lot confirmation, the happened time and the actor are captured on paper in the same form FL requires and FL refuses the entry without them; the one thing paper cannot provide is the possible-duplicate check at the time — FL still runs it at entry against what *was* posted, and the end-of-shift summary (§7.3) lists paper-sourced receipts separately for Arturo to check against the sheet.

Expected receipts and production runs due that day stay in the "NOT ENTERED YET?" section until the paper is keyed in, so a lost sheet is noticed the same day.

---

## 10. Build plan

Prerequisite **PR-0 — rotate the master `API_KEY`** (audit §7; live value is in 7 files of a public repo). Railway variable change + scrub the literals; ½ day; owner approval per deploy step. Nothing below should deploy before it.

Four tracks that can run in parallel once PR-A1 defines the contract: API, dashboard, the FL Assistant page (F — the floor front end, §1.8), and MCP (M — reads before Dec 11, writes after cutover; **off the Nov 20 critical path**). Effort = engineering days incl. tests against the local harness (`ledger_harness.py`) and staging, before owner acceptance. Migrations are numbered from 058 and must be renumbered against then-current main.

| # | PR | Track | Contents | Depends on | Effort |
|---|---|---|---|---|---|
| A1 | `feat/write-tickets` | API | migration 058 `write_tickets`, `receipt_counters`, `transactions.receipt_number/ticket_id`; `/receive\|make\|pack\|adjust\|found/prepare`, `POST /tickets/{t}/commit` with replay; possible-duplicate warnings; `GET /receipts/*`; allowlist changes; direct routes still open (not yet internal-only) | PR-0 | 5–6 d |
| A2 | `feat/roles-enforced` | API | migration 059 `actors.email`, `entered_by_actor_id` on transactions/shipments/corrections/sales_orders/lines + `lot_events`; `ROLE_PERMISSIONS`; `/auth/whoami` permissions; back-dating limits (§6.3) with LATE_ENTRY (needs 061 — order A2 after A3 or ship the enum in A2) | A1 | 3 d |
| A3a | `feat/exceptions-tables` | API | **rev 3.4 split.** migration 061 `exceptions`, `shortage_flags`, `correction_reasons` (seeded with the §5.1 list of 8 + legacy-code mapping onto `transactions.reason_code`) and the `exceptions.kind` enum — tables and seeds only, no enforcement; it is the contract A3b, A5 and A6 build against | A1 | 1 d |
| A3b | `feat/exceptions-core` | API | R2 reason enforcement + thresholds, R3 post-and-flag in make/pack + `SHORTAGE_OPEN_RESOLVE_INSTEAD`, `/exceptions/*`, nightly sweep script. **File ownership (rev 3.4):** A3b owns the "insufficient inventory" branch of make/pack commit; A5 owns the prepare validators — both rebase daily | A3a | 4–5 d |
| A4 part 1 | `feat/resolution` — **PR #81** | API | migration 060 `search_aliases` + five approved token seeds; read-only `POST /resolve` (6 kinds, including supplier), pagination/recency and §3.2 safety rules; `/products/resolve` no-auto-pick; `GET /aliases`; staging acceptance | Resolution reads can start day 1 | Part of A4's 4 d |
| A4 part 2 | alias ticket integration | API | `resolution_log`; `/aliases` prepare/commit and deactivate writes with owner/office authorization | **Scheduled after A1 merges** (Michael, 2026-10-08); builds on A4 part 1 | Remainder of A4's 4 d |
| A5 | `feat/lot-confirmation` | API | R4 `lot_confirmations` + `transaction_lot_confirmations` + `lot_moves`; R5 `substitutions`; R8 `lots.identity_status`/`identify_by` + receive flagging + UNIDENTIFIED_LOT exceptions; **§5.2 supplier tracking:** `transactions.supplier_id`/`lots.supplier_id`, `POST /resolve kind=supplier`, 422 `SUPPLIER_REQUIRED`, `suppliers.short_code` (lot prefix = label only) | A1, **A3a** (rev 3.4: the tables, not A3b) | **4½ d** (+½ supplier) |
| A6 | `feat/ship-gate` | API | migration 063 `attachments`, `shipments.bol_reference/proof_status`; `/ship/prepare`, `/sales/orders/{id}/ship/prepare`, BOL/photo/same-day gate, signed packing-slip link; standalone ship by `customer_id` only | A1, **A3a** | 3 d |
| A7 | `feat/order-tickets` | API | prepare/commit wrappers for create/lines/header/status/close/cancel/expected-receipt (thin: reuse `_create_sales_order_core`; ticket receipt = `external_order_reference`) | A1 | 2–3 d |
| A8 | `feat/sunshine-lite` | API | migration 064 `bins`, `bulk_dispatch_lines`, `invoice_triggers`, `products.ownership_on_pack`, `lots.ownership`; `/ship/bulk/prepare`; pouch pack ownership + trigger; `/invoice-triggers` | A1, A6 | 4–5 d |
| A9 | `feat/reports` | API | `/reports/shift-summary` + confirm (**pilot-critical, 1½ d, JSON frozen day 1 — §10.2**), `/reports/weekly` (1½ d, may land after cutover; first weekly review is Nov 27) | A2, A3b, A5, A6 — **not A12** (rev 3.4: runs in parallel with A12; A12 adds its EXTRA-KOSHER line as a ½-d follow-up) | 3 d |
| A10 | `chore/internal-only-routes` | API | remove direct write routes from both allowlists (§1.7); delete `confirmation_code`. **Rev 3.4: lands at cutover+1 as a pre-approved PR, in two stages** — stage 1 (Mon Nov 23) the ledger routes (`/receive\|make\|pack\|adjust\|found\|ship\|void`, lot PATCHes); stage 2 (after A7 is verified on the dashboard) the order routes, otherwise the office loses order editing | all A, D2 live, GPTs read-only (G1); stage 2 also A7 | 1 d (prepared Nov 3) |
| A12 | `feat/kosher-tier` | API | **§5.3:** `products.kosher_tier`, `lots.kosher_tier`, `ticket_attestations`, `POST /tickets/{t}/attest` (owner PIN), `KOSHER_ATTESTATION_REQUIRED` / `KOSHER_SOURCE_REQUIRED` blockers, downgrade recording + `lot_events`, `/resolve` tier field + SS/#9 alias rule, EXTRA-KOSHER on receipts/labels/trace/shift summary, data update + renames per the §5.3 mapping | A1, A5, A11 | **3–3½ d** (incl. ~½ d F1/D2 attest UI) |
| A11 | `feat/pin-sessions` | API | **Decided (§4.4):** `actors.pin_hash` (peppered, UNIQUE) + `pin_locked_until`/`pin_failed_attempts`, PIN set/reset endpoint with the obvious-PIN and uniqueness rules, `actor_sessions`, `pin_attempts`, `POST /auth/session {pin}` (PIN-only, no name) and the actor-key variant for office/owner, `key_kind='session'` in `_authorize_api_key`, 10-min sliding idle expiry, per-source (device cookie + IP) lockout + global ceiling + owner alert, 10,000-PIN staging sweep in the acceptance scripts | A2 | **2½ d** (was 2: +½ d for the no-username brute-force limits and the uniqueness/obvious-PIN rules; UI is simpler — a keypad, no picker — so F1/D2 sign-in drops to ~½ d) |
| **G1** | `gpt/read-only-at-cutover` | GPT | **new, part of the cutover key rotation (§10.1).** `READONLY_API_KEY` env + `key_kind='readonly'` accepted only on `GET` entries of `DASHBOARD_KEY_ALLOWLIST` + `POST /products/resolve` (~½ d main.py + tests); strip the 17 write ops from `openapi-gpt-v3.yaml` (→ 13 GET + `resolveProducts`) and the 11 from `gpt-configs/schemas/openapi-floor.yaml` (→ 11 GET); rewrite both instruction files ("read-only; writes go through the FL Assistant"), ≤ 8,000 chars; paste schema + instructions + the new key into **both** GPT editors (manual, owner/office); smoke each read action. Verified: today both GPTs send the **master `API_KEY`** and FL has no read-only key kind (§11.2). **Rev 3.4:** the engineering (key kind, trimmed YAMLs, instruction rewrites, tests) is built **early, Fri Oct 30, in a free lane** and merged behind the unset `READONLY_API_KEY`; only issuing the key and the two editor pastes remain on cutover day | PR-0 (same rotation) | **≈ 1 d + 2 editor pastes** (not "10 minutes") |
| **M0** | `mcp/read-only-deploy` | MCP | **before Dec 11, off the critical path, when a Codex lane is free.** Rebase `feature/mcp-server` onto main (branch is purely additive: 32 files/+10,135 lines, collisions only in the two changelogs); add `MCP_WRITES_ENABLED=false` (hide `WRITE_CATALOG` in `list_tools`, refuse write calls, adjust the server instruction string; ~½ d + tests); ship the **17 read tools + Google sign-in** unchanged (all 17 routes exist on main and are reachable by actor keys — verified §11.2); M4's infra: Google OAuth client, Railway `factory-ledger-mcp` service via the IaC partial, `MCP_ACTOR_KEY_*` for the four users, staging first, then ChatGPT connector setup | A2 actors only (uses existing actor keys); **not** A1 | 2–3 d + owner steps |
| M1 | `mcp/rebase-and-tickets` | MCP | **after cutover.** delete `ConfirmationStore`; `_write()` → prepare/commit; id-only input schemas; drop createOrder blocker/scan, `_order()` workaround, `external_order_ref` rename; standalone ship without auto-create; FOUND tool; `MCP_WRITES_ENABLED=true` | A1, M0 | 4 d |
| M2 | `mcp/roles-from-fl` | MCP | **after cutover.** permissions from `/auth/whoami`; startup email↔actor check; allowlist role optional | A2, M0 | 1 d |
| M3 | `mcp/phase1-tools` | MCP | **after cutover.** `resolve`, `listExceptions`, `resolveException`, `endOfShift`, `weeklyReview`, `shipBulk`, `moveLot`; bilingual summaries; instruction file rewrite (explain, don't enforce) | A3–A9, M1 | 3 d |
| ~~M4~~ | ~~`infra/mcp-deploy`~~ | MCP | folded into **M0** (the infra ships with the read-only deploy) | — | — |
| D1 | `dash/receipts-and-exceptions` | Dashboard | receipt lookup box + page; Exceptions tab + resolve forms; receipt numbers on History/Activity | A1, A3 | 3 d |
| D2 | `dash/floor-forms` | Dashboard | ship commit button (BOL, photo upload), receive form **with supplier picker (§5.2)**, adjust/found forms with reason dropdown + photo, lot-confirmation inputs; sign-in = PIN keypad → FL session (§4.4, A11); no actor key in the browser | A1, A5, A6 | **4½–5½ d** (+½ supplier picker) |
| D3-lite | `dash/shift-summary` | Dashboard | **rev 3.4 split.** shift-summary page + Confirm / Confirm with notes / Something is missing (renders the frozen A9 JSON) — needed for gate (d), before the pilot | A9 shift summary | 1 d |
| D3 | `dash/aliases-weekly` | Dashboard | Aliases settings tab; Weekly view; ownership column on inventory tabs; invoice-triggers list. **Rev 3.4: after cutover, Nov 23–27** (the first weekly review is Fri Nov 27); week-1 Spanish aliases are seeded by migration, which §3.3 already allows | A4, A8, A9 | 2–3 d |
| **F1** | `dash/fl-assistant` | FL Assistant | **GO (§11 item 11). The floor front end; also for office users.** Productionise the PoC per §1.8: dashboard page (type / push-to-talk with editable transcript / photo upload, bilingual) + `POST /assistant/*` routes in the FastAPI service on Railway (`OPENAI_API_KEY` as Railway secret; forced tool calls; no free-text escape tool; model tools = thin wrappers over `POST /resolve`, `/{action}/prepare`, `GET /receipts/*`, `GET /inventory/lookup`, `GET /reports/shift-summary`; commit called by the server on **Record**, never by the model; red *NOT recorded* state; lot numbers via `/resolve kind=lot` + `lot_confirmations`); photo → draft (BOL/carrier, lot tag, correction evidence — §8.1, +1–2 d); sign-in = PIN keypad → FL session, "Recording as Arturo" on every card, re-PIN after 10 min idle (§4.4, A11); `client_source='fl_assistant'`; OpenAI spend cap raised above $20 + e-mail alerts before go-live. No business rules in F1 — FL decides | A1, A4 (A5/A6 for lot confirmations, BOL and photo evidence; A11 for sign-in) | **6–7 d** (was 4–5; +1–2 photo/voice) |
| P1 | `docs/paper-fallback` | Docs | late-entry sheet PDF + instructions (EN/ES), `docs/manual-tests/` acceptance scripts for staging | A1 | 1 d |

**Critical path (rev 3.2):** PR-0 (½) → A1 (5–6) → A3 (5–6) ∥ A4 (4) ∥ F1 (6–7) → A5 (4½) / A6 (3) → A12 (3–3½, needs A11) → A9 (3) → D3 (3–4) → G1 (1, at cutover) → A10 (1). Longest chain = PR-0 ½ + A1 6 + A3 6 + A5 4½ + A12 3½ + A9 3 + D3 4 + G1 1 + A10 1 ≈ **29–30 engineering days end to end** (rev 3 was 25–26; +A12 and +½ supplier tracking). Totals: **≈ 52–65 d API** (A1–A12 incl. G1's ½ d, A11's 2½ d, A12's 3–3½ d), **10–13 d dashboard**, **6–7 d FL Assistant**, **MCP 2–3 d before Dec 11 (M0) + 8 d after cutover (M1–M3)**. With three parallel lanes (API / dashboard+F1 / Codex) that is about 4–5 weeks for the Nov 20 set — same tightness as rev 2, with MCP work removed from it. What can slip past cutover without breaking the safety rules: A8 Sunshine-lite (bulk can stay on paper + standalone ship with `customer_id` until it lands — D8), R9 recounts, Google sign-in for the dashboard, **all of M0–M3**. What cannot: PR-0, A1, A2, A3, A4, A5 (incl. supplier tracking), A6, A11, **A12** (Sunshine order 298 is open), **F1**, D1, D2, G1 (the GPTs must be read-only on cutover day).

### 10.2 Schedule (rev 3.4, approved by Michael 2026-10-07) — 3 Codex lanes, re-sequenced critical path

**Why the rev 3.2 chain did not fit.** The binding constraint is gate 10(d), not the 29–30 d chain: five consecutive confirmed shift summaries by Nov 17 means the pilot stack (A1, A4, A5, A11, A9 shift summary, F1 core) must be on staging by **Nov 10** — 24 working days from Oct 8 — and the rev 3.2 chain reached A9 on day ≈ 23½ at 100 % efficiency. Three of its links were sequencing, not dependencies: A3 → A5/A6 (A5 and A6 need the `exceptions` table, not R2/R3 enforcement), A12 → A9 (A9 does not depend on A12), and D3/G1/A10 on the tail although nothing before cutover needs them.

**Approved changes (no rule changes):**

5. **Split A3** into A3a (migration 061 tables + the 8-reason seed, 1 d) and A3b (enforcement, `/exceptions/*`, sweep). A5 and A6 start the day after A3a. Saves 4–5 d. Risk: A3b and A5 both touch make/pack commit in `main.py` — ownership fixed in the A3b row; rebase daily.
6. **A9 in parallel with A12** (A9 depends on A2/A3b/A5/A6). Saves 3 d. Risk: one small merge.
7. **D3 off the chain**: only D3-lite (shift-summary page + Confirm, 1 d) ships before the pilot; Aliases tab, Weekly view, ownership column and invoice-trigger list land Nov 23–27. Saves 2–3 d. Risk: Michael's weekly view arrives one week after cutover; receipt lookup + shift summaries cover that week.
8. **G1 early, A10 at cutover+1**: G1's engineering merged Oct 30 behind an unset key; A10 as a pre-approved PR the Monday after cutover, staged (ledger routes first; order routes after A7 — otherwise the office loses order editing). Saves ≈ 2 d off the tail. Risk: one extra working day where direct routes remain reachable by the dashboard key — the status quo today, GPTs already read-only, so no new exposure to the new baseline.
9. **HELD — the Nov 6 lever (with A8):** F1 photo = upload-as-attachment only (vision BOL/lot-tag reading later; BOL still typed and still mandatory); `/reports/weekly` after cutover; defer A5 `lot_moves`/pallet method (every make types last-4 — stricter than R4, not weaker); defer `/aliases/*` write endpoints (seed by migration). Saves 3–4 d of lane capacity. Applied only if the Nov 6 checkpoint is red; any item of this set not merged by Nov 6 then moves after cutover.

**Critical path (rev 3.4):** A1 part 2 (3, PR #78 is part 1) → A3a (1) → A5 (4½) → A12 (3½) ∥ A9 shift summary (1½) → F1 tail (1) ≈ **17–18 engineering days**; pilot stack on staging target Fri Oct 30, latest Fri Nov 6. **Dates:** staging pilot soft-start **Mon Nov 2** (Arturo + Michael on staging through F1; shift summary confirmed daily); **Nov 6 checkpoint**; gate (d) window **Mon Nov 9 – Fri Nov 13** with Nov 16–17 as restart reserve (the latest clean five-day run that still meets "by Nov 17" starts Nov 11); gate (e) exercised Tue Nov 10; gate decision **Tue Nov 17**; cutover **Fri Nov 20**; A10 stage 1 Mon Nov 23. **Estimates:** 2 lanes → cutover Nov 24–25 best case, realistically Dec 1 (Thanksgiving week); 3 lanes → Nov 20 holds with ≈ 3–4 working days of slack. With three lanes the binding resource becomes Michael's review/acceptance time: one owner-acceptance slot per working day is part of the plan. **Day-1 contract freezes (Thu Oct 8):** the A1 ticket envelope (prepare response, commit body, receipt JSON — as implemented in PR #78 and §1.3) and the A9 shift-summary JSON (`GET /reports/shift-summary`, `POST …/confirm`, §7.3) are written down first and change only by doc revision, so the F1, D2 and D3-lite lanes build against contracts instead of waiting. Lane-by-lane plan with dates and the daily owner slot: `~/Documents/fl-audits/lane-schedule.md`.

*Recorded from PR #78 (A1 part 1), for Michael to confirm in the next revision:* the A1 session reports an owner override that the PR-0 master-key rotation is performed at cutover (§10.1 step 2) rather than before A1 deploys, and that direct write routes stay open until A10. The schedule above follows that; §10's "PR-0 prerequisite" wording is not yet updated.

### 10.1 Cutover checklist (Nov 20) — key rotation and ChatGPT

Run in this order on cutover day, each step owner-approved, after the readiness gate (§11 item 10) is green:

1. **Physical count reviewed, reset approved and applied** (`audits/fresh-start/apply_reset.py`, master key, writes `receipt_number`s — §5 R10).
1a. **Tag the extra-kosher lots from the paper log** (§5.3 interim procedure): for each (date, lot number) on the paper log, set `lots.kosher_tier='extra_kosher'` on the matching 107/108 lot via the A12 admin route (owner, master key — a one-time data step, receipted like the reset); photograph the paper log and attach it (`attachments.kind='count_sheet'`) to the first tagged lot; any paper row that matches no FL lot becomes an `UNIDENTIFIED_LOT`-style exception for Michael to resolve. Only after this may extra-kosher packs (rule 3) run on cutover day.
2. **Rotate the master `API_KEY`** if PR-0 did not already (it should have — PR-0 is a prerequisite). Rotate `DASHBOARD_API_KEY`; redeploy the dashboard with the new value.
3. **Issue the `READONLY_API_KEY`** (G1) on Railway; verify with the test suite that it 403s every non-GET route and `POST /products/resolve` is the only POST it reaches.
4. **GPTs → read-only** (G1): paste the trimmed `openapi-gpt-v3.yaml` (13 GET + `resolveProducts`) into the office GPT and the trimmed `openapi-floor.yaml` (11 GET) into the floor GPT; set each GPT's action auth to the `READONLY_API_KEY`; paste the rewritten instructions; smoke one read per GPT; confirm a write attempt in either GPT shows no write action. The old master key is now dead for both GPTs regardless.
5. **Scrub key literals** from the 7 tracked files the audit lists (`CHANGE_LOG.md`, `DEPLOYMENT.md`, `gpt-instructions-v3.md`, `gpt-configs/dist/GPT_FLOOR_INSTRUCTIONS.md`, `gpt-configs/sources/floor-specific.md`, `archive/superseded-instructions/GPT_INSTRUCTIONS.md`, `audits/reports/DASHBOARD_ACTIONABILITY_AUDIT.md`) if PR-0 left any; the public repo's *history* still holds the old values, which is why rotation — not scrubbing — is the control.
6. **F1 live** on Railway with the production `OPENAI_API_KEY`, spend cap raised above $20, e-mail alerts on; Arturo signed in as Arturo (§4.4); first entry of the day confirmed by receipt lookup.
7. **A10** — **rev 3.4: cutover+1.** Stage 1 on Mon Nov 23 (ledger direct routes drop from both allowlists, master key only) once the first day's F1 receipts are confirmed by lookup; stage 2 (order routes) after A7 is verified on the dashboard. Pre-approved PR prepared Nov 3.
8. Announce: "ChatGPT answers questions; entries go through the FL Assistant or the dashboard forms." Luz is the fallback operator on the dashboard forms.
9. Before **Dec 11**: M0 read-only MCP plugin live in ChatGPT for the four users; then retire both GPTs.

Every PR: branch off `origin/main`, hunk-level staging on the shared checkout, suite green against the local DB (`TEST_DATABASE_URL` set), staging deploy and manual acceptance before any prod step, per-action approval for push/merge/deploy, `FACTORY_LEDGER_CHANGELOG.md` row on deploy. Migrations applied to staging first, prod only with approval.

---

## 11. DECISIONS — recorded 2026-10-07 (Michael, on PR #73)

All ten were answered. Items 1–3, 5–7, 9–10 are **approved as recommended**; item 4 is **fixed by Michael**; item 8 is **approved with two open questions**. The text below is the decision of record; the original recommendation is kept for context.

| # | Decision | Status | Effect in this document |
|---|---|---|---|
| 1 | Direct write routes become master-key-only from PR-A1, deleted after Dec 11 (one write path for staff). | **Approved** | §1.7, A10 |
| 2 | Ticket life 10 min for conversational clients (MCP, FL Assistant), 30 min for dashboard forms. | **Approved** | §1.4, §1.8 |
| 3 | Dashboard identity in Phase 1 = personal actor key entered once per browser for write screens; Google sign-in on the dashboard is Phase 2. | **Approved** | §4.2, D2 |
| 4 | **Correction reasons are fixed at these 8 (EN / ES):** Physical count / Conteo físico; Missing receipt / Recepción no registrada; Missing production / Producción no registrada; Wrong lot used / Lote equivocado; Damage/disposal / Daño o desecho; Unrecorded usage / Uso no registrado; Data-entry error / Error de captura; Unknown / Desconocido (**note required**). Today's adjust and found codes are mapped onto these. | **Decided by Michael** | §5.1 (codes, labels, `note_required`, legacy mapping), A3 |
| 5 | Aliases are added/deactivated by owner + office on the dashboard only; chat never. Arturo's Spanish shorthand collected and seeded in week 1. | **Approved** | §3.3 |
| 6 | Back-dating: 0–48 h free; 48 h–14 d allowed and flagged `LATE_ENTRY` for owner acknowledgement; > 14 d owner-only with `backfill`. Revisit 48 h after a month of weekly views. | **Approved** | §6.3 |
| 7 | Large corrections (> 500 lb or > 10 %) post immediately with a photo required; held only when the photo is missing. | **Approved** | §4.3, §5 R2, §7.1 |
| 8 | Sunshine-lite (A8: bins + invoice trigger + pouch ownership) ships if the critical tracks are green at the **Nov 6 checkpoint**; otherwise deferred, bulk stays on standalone ship to Sunshine (`customer_id`) + paper bin log. **PENDING Michael:** (a) pouch invoice-at-pack price basis (per case? fixed?); (b) how later Sunshine yield reports / credits alter the invoice trigger. Both are left as open questions; `invoice_triggers.price_basis` is nullable and no credit model is designed. | **Approved; two open questions** | §8.3–8.4, A8 |
| 9 | Shipment photo required by end of plant day (exception if missing); BOL number at commit is the hard gate. | **Approved** | §8.2, §7.1 |
| 10 | Fallback operator = Luz on the dashboard forms. Readiness gate for Nov 20: (a) PR-0 master-key rotation done; (b) A1–A6 + A11 + **F1** + D1 + D2 + G1 ready on staging with the §3.1, §4.4 (10,000-PIN sweep signs nobody in) and §7.3 acceptance scripts passed by Michael, Arturo and one office user (rev 3: MCP writes M1–M4 are **removed** from the gate); (c) physical count reviewed and reset approved; (d) five consecutive confirmed shift summaries on staging with zero unexplained discrepancies; **(e) rev 3.2:** every `RCV` receipt in those five days carries a resolved `supplier_id` (§5.2) and at least one extra-kosher make + pack has run on staging with the owner attestation and the `KOSHER_SOURCE_REQUIRED` refusal exercised (§5.3). If (d) is not met by Nov 17, cutover moves — the date does not override the gate. | **Approved; (b) amended rev 3, (e) added rev 3.2** | §9, §10, §10.1 |

**Added by Michael (rev 2, not numbered; resolved in rev 3 by item 11):** floor transaction entry may move from ChatGPT to an *FL Assistant* page in the dashboard (proof of concept running). Prepare/commit, receipts and resolution are kept interface-neutral so either front end works — §0.7, §1.8, and track F in §10.

### Decisions recorded 2026-10-07 15:00 — after the FL Assistant proof of concept

| # | Decision | Status | Effect in this document |
|---|---|---|---|
| 11 | **FL Assistant proof of concept = GO.** F1 (FL Assistant page in the dashboard) is the floor front end and is available to office users. Evidence: 30/30 forced tool calls, 0 false "recorded", ≈ 4 s/turn, ≈ $3/month, desktop + phone mic passed. The Nov 6 "which front end" question is closed. | **Decided by Michael** | §0.2, §0.7, §1.8, §10 F1, gate 10(b) |
| 12 | **ChatGPT is never taken offline.** Until cutover the custom GPTs are unchanged. At cutover they become **read-only** — write operations removed from both action schemas + a new read-only key, as part of the cutover key rotation (§10.1 steps 3–4). Before Dec 11 the existing MCP plugin is deployed **read-only** (17 read tools + Google sign-in), off the critical path, when a Codex lane is free (M0). MCP writes (M1–M3) come after cutover. M1–M4 writes are removed from the Nov 20 critical path and from the readiness gate. *Verification note (§11.2):* "read-only GPTs" is ≈ 1 day of work plus two editor pastes, not 10 minutes, because FL has no read-only key kind today and both GPTs currently hold the master key. | **Decided by Michael; effort corrected** | §0.2, §0.6, §1.6, §10 G1/M0–M3, §10.1, gate 10(b) |
| 13 | **F1 is a ChatGPT-style chat:** type, push-to-talk dictation (transcript shown and editable before send), photo upload. Photos: read BOL number/carrier into shipment drafts, lot tags into lot confirmation, evidence for corrections > 500 lb (reuse A6 attachment storage). Anything read from a photo or from voice only fills a **draft** the user confirms — never posts directly. +1–2 d on F1. | **Decided by Michael** | §1.8, §8.1, §10 F1 |
| 14 | **F1 requirements from the PoC:** real read tools (today's entries, inventory lookup); no free-text escape tool; lot-number support; clear "NOT recorded" state on network failure; hosted inside the FL dashboard on Railway only (never local tunnels); OpenAI key as a Railway secret; raise the $20 hard spend cap and turn on e-mail alerts before go-live. *Architecture correction (§11.2 item 3):* F1 has a server-side piece in the FastAPI service — the browser never holds the OpenAI key or the ticket, and F1 calls FL's endpoints directly, not the MCP service. | **Decided by Michael; wording fixed** | §1.8, §10 F1, §10.1 step 6 |
| 15 | **A4 resolution: short codes/aliases (SS, BS) match EXACTLY only — never as substrings.** Whole-token match on `alias_norm`; aliases never enter the keyword/trigram tiers. | **Decided by Michael** | §3.2, §3.3, A4 acceptance tests |
| 16 | **Floor identity — PIN only, any device.** Every entry records the person, never the device. Each person has a unique personal 4-digit PIN that identifies them (no name picker), usable on a shared floor tablet or their own phone; FL rejects obvious PINs (repeated digits, sequences like 1234) and PINs already in use, locks temporarily after several wrong attempts, ends the session after 10 min idle. Office/owner may also use their own sign-in (personal actor key; Google in Phase 2). FL-issued short-lived sessions are the only browser credential (replaces D3's `localStorage` actor key). **Owner note:** whether phones are allowed in production areas is CNS's food-safety call; the design supports both. ≈ 2½ d API (A11) + ~½ d UI. | **Decided by Michael** | §4.4, §10 A11, D2, F1, gate 10(b) |

### Decisions recorded 2026-10-07 15:42 — supplier tracking and Classic #9 tiers

| # | Decision | Status | Effect in this document |
|---|---|---|---|
| 17 | **Supplier tracking APPROVED.** From cutover every receipt stores a real `supplier_id` chosen via resolution; the 4-letter lot-code prefix is a label only, never used for supplier identification. | **Decided by Michael** | §5.2, A5 +½ d, D2 +½ d, gate 10(e) |
| 18 | **Classic #9 = two versions, same formula: Regular and Extra-Kosher.** Difference = the kosher procedure actually followed (Michael turns on the oven), not the recipe. FL distinguishes them in production records, identification and labeling. Recording an extra-kosher batch requires Michael's PIN attestation; extra-kosher finished goods pack only from extra-kosher lots; extra-kosher lots may be packed as regular (recorded). `SS`/`Sunshine` + `#9` → extra-kosher; `#9` alone → both shown. Proposed id mapping in §5.3 (**Michael to confirm**). | **Decided by Michael; mapping proposed** | §5.3, §3.2 alias rule, §4.3, A12 3–3½ d, gate 10(e) |

| 19 | **Kosher follow-ups:** (a) 1,000 CT and 4,000 CT real chips are interchangeable; standard going forward is 4,000 CT — 284's formula changes 73 → 72, pre-approved, applied with the catalog cleanup, 1,000 CT stays allowed as a recorded substitution; (b) the 2026-08-12 mis-pack of 286 is a probable recording error — **no historical correction**, prevention via `KOSHER_SOURCE_REQUIRED`; (c) interim: extra-kosher batches logged on paper (date + lot number) until cutover; cutover checklist step 1a tags those lots in FL from the paper log. | **Decided by Michael** | §5.3, §10.1 step 1a, `catalog-cleanup.csv` |

### Decisions recorded 2026-10-07 16:00 — schedule (rev 3.4)

| # | Decision | Status | Effect in this document |
|---|---|---|---|
| 20 | **3 Codex lanes; re-sequencing changes 5–8 approved** (A3 split A3a/A3b; A9 ∥ A12; D3 → D3-lite before the pilot, rest after cutover; G1 engineering early, A10 at cutover+1 in two stages). **Change 9 (simplify for cutover) is held** as the Nov 6 checkpoint lever together with A8. No safety rule changes. Pilot soft-start Nov 2; gate (d) window Nov 9–13; gate decision Nov 17; cutover Nov 20. | **Decided by Michael** | §10 rows A3a/A3b/A5/A6/A9/A10/G1/D3-lite/D3, §10.1 step 7, §10.2, `~/Documents/fl-audits/lane-schedule.md` |

**Still open (owner):** 8(a) pouch price basis; 8(b) yield credits; **18 — confirm the §5.3 id mapping** (chip size is resolved); **§10.2 note — confirm the PR-0 "rotate at cutover" override recorded in PR #78.**

### Decisions recorded 2026-10-08 12:30 — A9 contract and A3a backfill (Michael, on PRs #84 / #85)

| # | Decision | Status | Effect in this document |
|---|---|---|---|
| 21 | **Legacy keys (master / dashboard, no actor identity) get a read-only all-actors shift summary** — `GET /reports/shift-summary` without `actor` returns `actor: null`, `can_confirm: false`; no 422. | **YES** (PR #84 a) | `docs/contracts/shift-summary.md` §1.1 / §1.2; §7.3 unchanged |
| 22 | **Confirm must echo the `summary_hash` of the summary that was shown.** A changed summary → `409 SHIFT_SUMMARY_STALE` carrying the fresh document; the page asks again. | **YES** (PR #84 b) | contract §1.2, §3.1, §4; `shift_confirmations.summary_hash` + `summary_snapshot` (§3.4) |
| 23 | **Re-confirming the same `(date, actor)` appends a new row; history is kept, the latest outcome is the day's status** on the weekly view (§7.2 item 8). No 409. | **YES** (PR #84 c) | contract §3.1; `shift_confirmations` append-only |
| 24 | **Migration 061 backfill: adjust rows whose legacy `adjust_reason` is blank or missing become `reason_code='unknown'`** (not left NULL); original text/notes untouched. | **YES** (PR #85 d) | §5.1 backfill note; `migrations/061_exceptions_tables.sql` as written |

### 11.2 Part A verification (2026-10-07 15:00, against `origin/main` @ `cb2705c` and `origin/feature/mcp-server` @ `d74f747`)

1. **GPT authentication today.** Both custom GPTs send `X-API-Key` (`securitySchemes.ApiKeyAuth` in `openapi-gpt-v3.yaml` and `gpt-configs/schemas/openapi-floor.yaml`) carrying the **master `API_KEY`** — the live literal appears in `gpt-instructions-v3.md` and `gpt-configs/dist/GPT_FLOOR_INSTRUCTIONS.md` (audit §7a). Office schema: 30 ops = 13 GET + 17 POST/PATCH (of which `POST /products/resolve` is a read). Floor schema: 22 ops = 11 GET + 11 POST/PATCH (incl. `ship/preview` and `ship/commit`). Schemas and instruction sources live in the repo (`gpt-configs/README.md` workflow) but take effect only when pasted into each GPT's Actions/Instructions panel by hand. **FL has no read-only key kind:** `_authorize_api_key` (main:3123) knows master (everything), `DASHBOARD_API_KEY` (`DASHBOARD_KEY_ALLOWLIST` — includes writes: ship commit, receive commit, order PATCHes, production runs, close/cancel) and actor keys (that list + the 14 `ACTOR_WRITE_ALLOWLIST` writes). A read-only key must be built (G1, ~½ d: env var + `key_kind='readonly'` accepted on GET allowlist entries + `POST /products/resolve`, tests). With YAML trimming, two instruction rewrites under 8,000 chars and two editor pastes, "read-only GPTs" is **≈ 1 day**, not 10 minutes — and it is a real safety gain, because today the only alternative keys both reach writes.
2. **Read-only MCP before Dec 11.** `feature/mcp-server` is 33 commits behind / 9 ahead of main and **purely additive** (32 files, +10,135 lines under `mcp_server/`, `.railway/railway.ts`, two docs; only `CHANGE_LOG.md` / `FACTORY_LEDGER_CHANGELOG.md` collide), so a rebase is mechanical. The read catalog (17 tools) was generated from YAMLs byte-identical to main (audit §1); all 17 routes exist on main and are reachable by actor keys (`GET` entries of `DASHBOARD_KEY_ALLOWLIST` + `POST /products/resolve`; `SAFE_POSTS` in `adapter.py` covers `resolveProducts` and `shipOrder` preview). Google sign-in (`auth.py`, `MCP_AUTH_MODE=google`) is independent of main. **What is missing:** there is no writes-off switch — `server.py list_tools()` appends `WRITE_CATALOG` for any allowlisted role; M0 adds `MCP_WRITES_ENABLED=false` (hide + refuse + instruction string, ~½ d with tests). M4's infra has never been applied (Google OAuth client, Railway service via the `factory-ledger-mcp` IaC partial, four `MCP_ACTOR_KEY_*` variables, ChatGPT connector setup — owner steps). Write-side tests (`test_writes.py`, `test_named_actor_pending.py`) are stale after PR #66/#67 and are skipped or deleted for the read-only deploy. **Real effort: 2–3 engineering days + owner steps**, off the critical path.
3. **Shared backend.** Confirmed as the design intent, and the rev 2 wording was wrong about the mechanism: the PoC (`spikes/fl-assistant/`) never calls FL — its Node server drives a throw-away `spike_prepare_pack`/`spike_commit` MCP server through OpenAI's hosted-MCP tool, keeps the ticket server-side and calls commit itself. So F1 is a dashboard page **plus** `POST /assistant/*` routes in the FastAPI service, whose model tools are thin wrappers over `POST /resolve`, `/{action}/prepare`, `POST /tickets/{t}/commit`, `GET /receipts/*`; no hosted-MCP tool; no business logic in F1's backend or in the MCP adapter (§1.6 — it relays FL's `draft`/`warnings`/`blockers` verbatim). Both front ends hit the same handlers and get the same refusals. §1.8 and the F1 row are rewritten accordingly.
4. **Floor identity.** Facts in §4.4: `actors` = name/role/key_hash/active, no PIN/session/device; dashboard uses the shared key; nothing asks who you are. Recommendation (c) was recorded as OPEN; Michael decided the same day: PIN-only on any device (item 16, §4.4 mechanics).

### 11.1 Original recommendations (for the record)

1. Keep direct routes master-key-only from PR-A1 onward (internal-only), delete after Dec 11. Running two write paths for staff is how a rule gets bypassed.
2. 10 minutes for chat, 30 for dashboard forms. Commit re-validates anyway; a shorter life just means "say yes sooner".
3. Personal actor key per browser for write screens; Google sign-in on the dashboard in Phase 2 — the alternative delays every floor form past Nov 20.
4. (Superseded by Michael's list.) Seed with today's six adjust codes plus `found_during_count`, `unreceived_delivery`, `data_entry_error`, `merge` and prune to 8 in week 1; `other` always requires a note.
5. Owner + office on the dashboard only; chat never.
6. 48 h free, 48 h–14 d flagged, > 14 d owner-only; revisit after one month.
7. Post immediately with photo required; hold only when the photo is missing.
8. Ship A8 if the three critical tracks are green by Nov 6; otherwise defer. Price basis and yield credits not designed here.
9. End of day — BOL at commit is the hard gate, photo is the proof trail.
10. Fallback = Luz; gate (a)–(d) as above.
