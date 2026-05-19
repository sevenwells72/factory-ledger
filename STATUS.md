# Factory Ledger — File Status

Classification date: 2026-05-19
Branch: `docs/status-classification-2026-05-19`
Ground truth: `~/Downloads/Factory_Ledger_GPT_Config.docx` (main GPT) and `~/Downloads/Factory_Ledger_Floor_GPT_Config.docx` (Floor GPT).

## Ground-truth summary from the two GPT config docs

| GPT | API title | API version | OpenAPI | Operations | Instructions header |
|---|---|---|---|---|---|
| Factory Ledger (main) | Factory Ledger System | **3.4.0** | 3.1.0 | **30** | "Factory Ledger GPT — v3.6.0" |
| Factory Ledger — Floor | Factory Ledger — Floor & Fulfillment | **4.0.0** | 3.1.0 | **21** | shared-rules.md + floor-specific.md, "Built: 2026-04-22 15:44 UTC" |

Categories used below: **LIVE**, **HISTORICAL**, **SUPERSEDED**, **AUDIT**, **WORKING**.

---

## Schemas (`*.yaml`)

| File | Title / Version | Ops | Verdict | Reason |
|---|---|---|---|---|
| `openapi-gpt-v3.yaml` | Factory Ledger System / **3.4.0** | **30** | **LIVE** (main GPT) | Title, version, and full 30-operation list match the main GPT config doc exactly. |
| `gpt-configs/schemas/openapi-floor.yaml` | Factory Ledger — Floor & Fulfillment / **4.0.0** | **21** | **LIVE** (Floor GPT) | Title, version, and full 21-operation list match the Floor GPT config doc exactly. |
| `openapi-v3.yaml` | Factory Ledger System / 3.3.0 | 33 | **SUPERSEDED** | Predecessor of 3.4.0 — operations include `checkFulfillment`, `getCurrentInventory`, `getInventoryItem`, `productionDaySummary`, `productsMissingCaseSize` that have been removed/renamed in live 3.4.0. |
| `openapi-schema-gpt.yaml` | Factory Ledger API / 2.7.0 | 32 | **SUPERSEDED** | File's own description says `DEPRECATED — uses split /preview and /commit endpoints that no longer exist`. |
| `openapi-schema.yaml` | Factory Ledger API / 2.7.0 | 35 | **SUPERSEDED** | Same 2.7.0 split-preview/commit design, removed before 3.x. |

---

## Instruction docs (`*INSTRUCTIONS*.md`, `gpt-instructions*.md`)

| File | Verdict | Reason |
|---|---|---|
| `gpt-instructions-v3.md` (8175 B) | **LIVE** (main GPT) | Header is `Factory Ledger GPT — v3.6.0`. Contains the `PRE-FLIGHT — CUSTOMER` block and the `4xx detail.error_code + detail.suggestions` line that appear in the deployed main config. |
| `GPT_INSTRUCTIONS.md` (7831 B) | **SUPERSEDED** | Also claims `v3.6.0`, but is an older variant: missing `PRE-FLIGHT — CUSTOMER` and the 4xx-`detail` ERRORS line, and has a stray `"wrap up"/.../"daily summary"` entry in the routing block (live has it only under `DAY SUMMARY`). |
| `gpt-configs/sources/shared-rules.md` | **LIVE** | Source of the deployed Floor instructions (canonical, hand-edited). |
| `gpt-configs/sources/floor-specific.md` | **LIVE** | Source of the deployed Floor instructions (canonical, hand-edited). |
| `gpt-configs/dist/GPT_FLOOR_INSTRUCTIONS.md` | **LIVE** | Generated artifact (build time 2026-04-22 16:10 UTC) of shared+floor; matches deployed Floor config (deployed build is 2026-04-22 15:44 UTC — same content, deploy is one rebuild behind, not a different spec). |
| `gpt-configs/README.md` | **LIVE** | Documents the `gpt-configs/` directory layout / build flow. |
| `build_gpt_instructions.py` | **LIVE** | Generator script that produces `gpt-configs/dist/GPT_FLOOR_INSTRUCTIONS.md`. |

---

## Migrations (`migrations/*.sql`)

Per task rule: "Unreferenced one-time backfills → HISTORICAL." Grep of `main.py`, `dashboard/dashboard.js`, and `build_gpt_instructions.py` finds **no** code reference to any file in `migrations/`. The schema they define lives in the DB; the files are one-time-applied SQL. All → **HISTORICAL**.

(Note: `main.py` also contains inline idempotent versions of migrations 004–012 in its `startup()` event — those run on every boot. The `migrations/` files for the same numbers are still HISTORICAL one-shots; they predate the inline mirrors or were applied manually first.)

| File | Verdict | Reason |
|---|---|---|
| `migrations/003_notes_todos_reminders.sql` | HISTORICAL | Notes/todos schema; one-time DDL. |
| `migrations/004_production_scheduling.sql` | HISTORICAL | Production scheduling tables; one-time DDL (also mirrored inline in `main.py`). |
| `migrations/005_customer_aliases.sql` | HISTORICAL | Customer aliases table; one-time DDL (mirrored inline). |
| `migrations/012_pg_trgm_product_search.sql` | HISTORICAL | Enables `pg_trgm`; one-time. |
| `migrations/013_shipment_tables.sql` | HISTORICAL | Shipments / shipment_lines DDL; one-time. |
| `migrations/014_fix_so260217001_flake_overshipment.sql` | HISTORICAL | One-time data fix. |
| `migrations/015_fix_lot131_negative_balance.sql` | HISTORICAL | One-time adjustment. |
| `migrations/016_backfill_received_at.sql` | HISTORICAL | One-time backfill. |
| `migrations/017_transaction_status_and_void_ghosts.sql` | HISTORICAL | One-time DDL + void cleanup. |
| `migrations/018_backfill_pre_migration_shipments.sql` | HISTORICAL | One-time backfill. |
| `migrations/019_populate_case_size_lb.sql` | HISTORICAL | One-time backfill. |
| `migrations/020_numeric_precision.sql` | HISTORICAL | One-time column-type change. |
| `migrations/021_fix_so260217008_undershipment.sql` | HISTORICAL | One-time data fix. |
| `migrations/022_close_so260217001.sql` | HISTORICAL | One-time order close. |
| `migrations/023_reconcile_so260312005_dicarlo.sql` | HISTORICAL | One-time reconciliation. |
| `migrations/024_close_so260213001_juliette.sql` | HISTORICAL | One-time order close. |
| `migrations/025_set_supplier_lot_sprinkles_26-03-10-FOUN-001.sql` | HISTORICAL | One-time supplier-lot set. |
| `migrations/026_add_bs_8oz_granola_products.sql` | HISTORICAL | One-time product insert. |
| `migrations/027_add_sliced_almonds_products.sql` | HISTORICAL | One-time product insert. |
| `migrations/028_add_is_service_to_products.sql` | HISTORICAL | One-time DDL. |
| `migrations/029_rename_unknown_lot_to_25216.sql` | HISTORICAL | One-time rename. |
| `migrations/030_backfill_standalone_shipment_records.sql` | HISTORICAL | One-time backfill. |
| `migrations/031_relax_quantity_lb_check_for_service_items.sql` | HISTORICAL | One-time CHECK constraint relax. |
| `migrations/032_backfill_skus_and_merge_bs_cocoa.sql` | HISTORICAL | One-time SKU backfill + product merge. |
| `migrations/033_force_close_so260326002_ace_endico.sql` | HISTORICAL | One-time order close. |
| `migrations/034_force_close_so260414003_hannas.sql` | HISTORICAL | One-time order close. |
| `migrations/034a_fruit_nut_bom_bake_line_copack.sql` | HISTORICAL | Backfilled direct-SQL session (2026-05-18); applied. |
| `migrations/036_drop_bake_line.sql` | HISTORICAL | One-time DDL drop (2026-05-19); resolved 035 slot conflict per commit 1a74969. |

---

## Audits

| File | Verdict | Reason |
|---|---|---|
| `AUDIT_GPT_FABRICATION_2026-04-21.md` | **AUDIT** | Dated GPT fabrication audit. |
| `TRACEABILITY_AUDIT_2026-03-24.md` | **AUDIT** | Dated traceability audit. |
| `DASHBOARD_ACTIONABILITY_AUDIT.md` | **AUDIT** | Dashboard actionability audit (undated but audit-shaped). |
| `audits/2026-05/readonly-baseline-20260518T210745Z.json` | **AUDIT** | Read-only baseline snapshot (already under `audits/`). |

---

## Other root files

| File | Verdict | Reason |
|---|---|---|
| `main.py` | **LIVE** | FastAPI backend; Railway entrypoint. |
| `requirements.txt`, `runtime.txt` | **LIVE** | Deploy config (Railway). |
| `netlify.toml` | **LIVE** | Netlify deploy config (publish = `dashboard/`). |
| `pytest.ini` | **LIVE** | Test runner config. |
| `CLAUDE.md` | **LIVE** | Project guardrails (regression-guard rules, changelog protocol). |
| `CHANGE_LOG.md` | **LIVE** | Append-only change log mandated by `CLAUDE.md`. |
| `FACTORY_LEDGER_CHANGELOG.md` | **LIVE** | Regression-guard changelog explicitly referenced by `CLAUDE.md`. |
| `DEPLOYMENT.md` | **LIVE** | Current deployment guide (60 KB; matches live Railway/Netlify setup). |
| `CONTEXT.md` | **LIVE** | Full project context (referenced as the orientation doc). |
| `FOLLOWUPS.md` | **WORKING** | Open follow-up list — actively edited, not yet resolved. |
| `KEEPALIVE.md` | **LIVE** | Describes the `/health` ping that `scripts/daily-health-ping.sh` performs. |
| `GUIDE.md` | **SUPERSEDED** | Header is `Dummy Guide (v2.5.0)`. Live API is 3.4.0; this is a v2.5.0-era doc. |
| `SALES_API.md` | **SUPERSEDED** | Header is `Sales API Reference (v2.5.0)`. Live API is 3.4.0; references endpoints/shapes from the 2.x series. |
| `factory_ledger_reconciliation.sql` | **HISTORICAL** | One-time reconciliation report SQL, run-in-editor only; not referenced by code. |
| `keepalive.log` | **HISTORICAL** | Cron output log from the keepalive ping. (Should likely be `.gitignore`d going forward.) |
| `scripts/daily-health-ping.sh` | **LIVE** | Cron-callable health pinger referenced by `KEEPALIVE.md`. |
| `dashboard/index.html`, `process-flow.html`, `sankey.html`, `traceability.html`, `dashboard.css`, `dashboard.js`, `dashboard_config.json` | **LIVE** | Served by `main.py:9604` via `app.mount("/dashboard", StaticFiles(...))`. |
| `tests/__init__.py`, `conftest.py`, `test_resolve_customer.py`, `test_ship_order_service_line.py` | **LIVE** | Current test suite. |
| `.gitignore` | **LIVE** | Repo hygiene. |

### Items requested in the task that do not exist on this branch

The task references files that are not present in this branch (`claude/serene-panini-dc449d`). They exist on `main` but not here, so I did not classify them:

- `demand_plan.html`, `planner_v2.html` — both absent from this branch; no `git mv` proposal possible from here. (`main.py` does not register a route for either name in this checkout.)
- `render_planner_v2.py`, `render_demand_plan.py` — absent from this branch.

If you want them classified, run this prompt on `main` (where `find` showed them), or rebase this branch onto `main` first.

---

# Proposed `git mv` plan (not executed)

Goal: keep production deploy paths intact, move stale and archival material out of the root, gather audits in one place.

## What does NOT move (deploy-critical at repo root)

Moving any of these would break Railway / Netlify / Supabase / Custom GPT wiring:

- `main.py`, `requirements.txt`, `runtime.txt`, `pytest.ini`, `netlify.toml`, `.gitignore`
- `dashboard/` (Netlify publishes from this path; `main.py` mounts it from this path)
- `tests/` (pytest `testpaths = tests`)
- `scripts/` (cron path)
- `gpt-configs/` tree (already structured; `build_gpt_instructions.py` reads `gpt-configs/sources/`)
- `openapi-gpt-v3.yaml` — `CLAUDE.md` HARD RULE pins this filename ("30-operation limit"); Custom GPT Actions tab references this exact spelling. Leave at root.
- `build_gpt_instructions.py` — leave at root; reads `gpt-configs/sources/` via relative path.
- `gpt-instructions-v3.md` — currently the canonical LIVE main-GPT instructions text. Leave at root.
- `CLAUDE.md`, `CHANGE_LOG.md`, `FACTORY_LEDGER_CHANGELOG.md`, `DEPLOYMENT.md`, `CONTEXT.md`, `KEEPALIVE.md`, `FOLLOWUPS.md` — referenced by `CLAUDE.md` change-log protocol or used as top-level orientation.

## Moves (run from repo root)

```bash
# 1. Audits → /audits  (consolidate the dated audit reports beside audits/2026-05/)
mkdir -p audits/reports
git mv AUDIT_GPT_FABRICATION_2026-04-21.md   audits/reports/
git mv TRACEABILITY_AUDIT_2026-03-24.md      audits/reports/
git mv DASHBOARD_ACTIONABILITY_AUDIT.md      audits/reports/

# 2. Superseded OpenAPI schemas → /archive/superseded-schemas
mkdir -p archive/superseded-schemas
git mv openapi-v3.yaml          archive/superseded-schemas/   # 3.3.0 predecessor of 3.4.0
git mv openapi-schema-gpt.yaml  archive/superseded-schemas/   # 2.7.0 self-DEPRECATED
git mv openapi-schema.yaml      archive/superseded-schemas/   # 2.7.0 split preview/commit

# 3. Superseded instructions → /archive/superseded-instructions
mkdir -p archive/superseded-instructions
git mv GPT_INSTRUCTIONS.md      archive/superseded-instructions/   # older v3.6.0 variant
git mv GUIDE.md                 archive/superseded-instructions/   # Dummy Guide v2.5.0
git mv SALES_API.md             archive/superseded-instructions/   # Sales API Ref v2.5.0

# 4. Applied migrations + one-shot recon → /archive/migrations-applied
#    (keep migrations/ on main if you still want the canonical run-order;
#     this is the archival shape if you want them out of an active dir.)
mkdir -p archive/migrations-applied
git mv migrations                          archive/migrations-applied/sql
git mv factory_ledger_reconciliation.sql   archive/migrations-applied/

# 5. Working-in-progress notes → /working   (FOLLOWUPS only; everything else is LIVE)
mkdir -p working
git mv FOLLOWUPS.md   working/

# 6. /live is intentionally NOT created.
#    Every LIVE file is deploy-pinned to its current path. Creating /live and
#    moving production code into it would break Railway entrypoint, Netlify
#    publish path, GPT Actions schema URL, and the CLAUDE.md HARD RULE on
#    openapi-gpt-v3.yaml. If you want a "live" marker, add a top-level LIVE.md
#    that lists the LIVE paths instead of moving them.
```

## Side-effects to handle before/after the moves

1. **`keepalive.log`** — HISTORICAL, but should be added to `.gitignore` and removed from git, not moved. Suggest a follow-up: `git rm --cached keepalive.log && echo keepalive.log >> .gitignore`.
2. **Custom GPT Actions tab** still points at the deployed Railway URL, not a file path, so moving the local YAMLs does not affect the deployed schema. But anyone re-uploading from the repo must use `openapi-gpt-v3.yaml` and `gpt-configs/schemas/openapi-floor.yaml`.
3. **CLAUDE.md HARD RULE** references `openapi-gpt-v3.yaml` by name — confirms the no-move decision for that file.
4. **Migration history** — if `archive/migrations-applied/sql/` is the new home, update any team docs that say "drop new files in `migrations/`".
