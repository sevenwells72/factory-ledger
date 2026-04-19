# Category 2 — Dead Code & Orphans

Scope: unused functions, leftover stubs, unused imports, commented-out blocks, superseded migration files, and auth-hygiene orphans.

**Headline:** the monolith is remarkably clean. One truly dead function, zero unused imports, zero commented-out code blocks, zero TODO/FIXME markers. Most of the "deadness" is in doc/config files rather than Python.

---

### [F02-01] `bilingual_response` helper is dead
**Severity**: low
**Files**: [main.py:649](../../main.py#L649)
**Current behavior**: Defined alongside `validate_bilingual` but never called anywhere in main.py, dashboard/, or migrations. Its sibling `validate_bilingual` is used 11×.
**Risk**: Negligible — confusing cruft only.
**Recommended fix**: Delete the 7-line function.
**Effort**: small

---

### [F02-02] 12 preview/commit alias stubs are alive but never called by GPT
**Severity**: medium
**Files**: [main.py:5857–5923](../../main.py#L5857) — 12 endpoints decorated with `include_in_schema=False`. Each is 4–8 lines that sets `req.mode = "preview"` or `"commit"` and forwards to the unified endpoint.
**Current behavior**: These exist because an earlier OpenAPI contract had split `/receive/preview` + `/receive/commit` endpoints; those names are now aliases for back-compat. They are not in `openapi-gpt-v3.yaml` (the current GPT sees only unified endpoints), not called by the dashboard JS (per dashboard endpoint audit), and `openapi-schema.yaml` — the stale spec that still references them — is self-labeled DEPRECATED in its own description ([openapi-schema-gpt.yaml:5](../../openapi-schema-gpt.yaml#L5)).
**Risk**: The routes still register handlers at app startup. If any external client (a scheduled job, a curl bookmark, a third-party integration) still hits `/receive/preview`, deleting these breaks it silently.
**Recommended fix**: Two-step. (1) Add a one-line `logger.warning("deprecated alias called: %s", request.url.path)` to each stub and ship. (2) After 30 days with no hits in Railway logs, delete all 12. If any hits appear, update the caller before deleting.
**Effort**: small (telemetry), small (delete later)

---

### [F02-03] 5 legacy dashboard view endpoints are dead in practice
**Severity**: medium
**Files**: [main.py:1266–1314](../../main.py#L1266) — `/dashboard/inventory`, `/dashboard/low-stock`, `/dashboard/today`, `/dashboard/lots`, `/dashboard/production`. Each reads from a SQL view (`inventory_summary`, `low_stock_alerts`, `todays_transactions`, `lot_balances`, `production_history`).
**Current behavior**: Per dashboard endpoint audit, the Netlify frontend exclusively uses the `/dashboard/api/*` routes defined later at [main.py:6509+](../../main.py#L6509), not these. These 5 endpoints query DB views that may or may not still exist in Supabase (none are created by any migration file in-repo — they must be Supabase-console views).
**Risk**: Slot-bloat at the framework level, and the views they depend on are undocumented. If any of the referenced views was dropped from Supabase, the endpoint returns 500 — but nothing would notice because nothing calls it.
**Recommended fix**: Smoke-test each of the 5 endpoints against production (one-time). If any 500s, confirm no caller and delete. If all work, still consider deleting as they duplicate `/dashboard/api/*` coverage.
**Effort**: small

---

### [F02-04] Duplicate GPT-instructions file — byte-identical
**Severity**: low
**Files**: [GPT_INSTRUCTIONS.md](../../GPT_INSTRUCTIONS.md), [gpt-instructions-v3.md](../../gpt-instructions-v3.md). Both are 7,643 bytes; `diff` returns empty per OpenAPI-audit agent. Changelog entries (#11, #13, #16, #17, #18) reference both by name as if they were different — suggesting the two files have drifted in the past and been re-synced.
**Risk**: Next edit will update one but not the other → behavior regression (changelog entry #21 specifically notes "restore SEARCH FIRST + NEVER INSTRUCT rules" after a sync drift).
**Recommended fix**: Delete `GPT_INSTRUCTIONS.md`; keep `gpt-instructions-v3.md` as canonical (matches the v3 OpenAPI naming). Or symlink. Update CLAUDE.md's regression-guard section to reference one filename.
**Effort**: small

---

### [F02-05] Two stale OpenAPI specs still in the repo root
**Severity**: medium
**Files**: [openapi-schema.yaml](../../openapi-schema.yaml) (35 ops, v2.7.0, uses split `/receive/preview` + `/receive/commit` — now aliased-but-hidden endpoints), [openapi-schema-gpt.yaml](../../openapi-schema-gpt.yaml) (32 ops, self-labeled DEPRECATED at line 5).
**Current behavior**: Neither is loaded by the app. `openapi-gpt-v3.yaml` is the live GPT contract; `openapi-v3.yaml` is the superset reference. The other two are left over from v2.x.
**Risk**: Reader confusion; when a schema change is needed, someone may edit the wrong file. There are 4 files to keep in sync per changelog entry #13 ("Added format:date + description to requested_ship_date in **both** schemas").
**Recommended fix**: Delete `openapi-schema.yaml` and `openapi-schema-gpt.yaml`. Add a one-line header comment to `openapi-gpt-v3.yaml` ("CANONICAL — this is the file ChatGPT's Actions config points to") and to `openapi-v3.yaml` ("SUPERSET REFERENCE — not deployed; includes dashboard-only ops").
**Effort**: small

---

### [F02-06] `/audit/integrity` endpoint is unauthenticated
**Severity**: medium (auth/hygiene; could become high if the app is ever exposed beyond trusted scope)
**Files**: [main.py:9122](../../main.py#L9122) — `def audit_integrity():` has no `Depends(verify_api_key)`. Comment at [main.py:9125](../../main.py#L9125) explicitly says "No auth required — read-only diagnostic endpoint for dashboard."
**Current behavior**: Returns lot IDs, lot codes, product names, customer names, voided-transaction counts, and a score to any unauthenticated caller.
**Risk**: The Railway API URL is not secret; anyone with the URL can enumerate lot/product/customer metadata. Low-severity for now because data exposed is not PII, but violates the API-key contract of every other endpoint.
**Recommended fix**: Add `_: bool = Depends(verify_api_key_flexible)` (same pattern as the packing-slip endpoint). The dashboard JS already passes an API key.
**Effort**: small

---

### [F02-07] 14 `/dashboard/api/*` endpoints lack API-key auth
**Severity**: medium (same rationale as F02-06)
**Files**: [main.py:6509](../../main.py#L6509) (`dashboard_api_production`) through [main.py:7382](../../main.py#L7382) (`dashboard_api_notes_toggle`). None of the 14 have a `Depends(verify_api_key)` guard.
**Current behavior**: Dashboard-private data (production calendar, inventory snapshots, lot detail, search, notes CRUD) all accessible without auth. The `/dashboard/api/notes` write paths (POST/PUT/DELETE/toggle) allow anyone with the URL to create, edit, or delete operator notes.
**Risk**: Notes CRUD is write-enabled. An external actor could spam or wipe the notes board, or add misleading notes that operators follow.
**Recommended fix**: Add `Depends(verify_api_key_flexible)` to all 14. The dashboard JS already calls `fetchAPI` with headers — confirm the API key is included; if not, adjust `fetchAPI` at dashboard.js:131 to send `X-API-Key`.
**Effort**: small

---

### [F02-08] `bilingual_response` sibling is the only dead *function*; everything else defined is called
Grep-scan of all `def ` at module scope against their own name in the rest of the file returned exactly one zero-reference match: `bilingual_response` (F02-01). Every other helper has at least one call site. `resolve_order_id` appears 7× as a `Depends`; `resolve_pack_add_ins` is called once from `pack`; `_inventory_detail_for_products` is called from `inventory_lookup` and `get_current_inventory`; `_inventory_detail_for_product` (singular) is called once at [main.py:1660](../../main.py#L1660) as a fuzzy-fallback path in `get_inventory` — barely used but live.

---

### [F02-09] Zero unused imports
Every name imported in [main.py:1–22](../../main.py#L1) resolves to at least one usage below the import block (verified by grep). `StreamingResponse`, `StaticFiles`, `Union`, `io`, `re` each have exactly 1 use.

---

### [F02-10] Zero commented-out Python blocks longer than 1 line
Scanned for patterns like `# def `, `# if `, `# return`, `# for `. Only matches are section-banner prose comments and single-line explanatory comments (e.g. `# mode == "commit"` at [main.py:2166](../../main.py#L2166) — a label, not dead code).

---

### [F02-11] Zero `TODO`/`FIXME`/`XXX`/`HACK` markers
grep on `TODO|FIXME|XXX|HACK` in main.py returned nothing.

---

### [F02-12] Migration files that are superseded but not removed
**Severity**: low
**Files**: Per-order reconcile migrations are one-shot fixes for specific historical incidents:
- [migrations/014_fix_so260217001_flake_overshipment.sql](../../migrations/014_fix_so260217001_flake_overshipment.sql) — SO-260217-001, done
- [migrations/015_fix_lot131_negative_balance.sql](../../migrations/015_fix_lot131_negative_balance.sql) — lot 131, done
- [migrations/021_fix_so260217008_undershipment.sql](../../migrations/021_fix_so260217008_undershipment.sql) — SO-260217-008, done
- [migrations/022_close_so260217001.sql](../../migrations/022_close_so260217001.sql) — done
- [migrations/023_reconcile_so260312005_dicarlo.sql](../../migrations/023_reconcile_so260312005_dicarlo.sql) — done
- [migrations/024_close_so260213001_juliette.sql](../../migrations/024_close_so260213001_juliette.sql) — done (but leaves GAP-12 open; see [17-migration-integrity.md](17-migration-integrity.md))
- [migrations/025_set_supplier_lot_sprinkles_26-03-10-FOUN-001.sql](../../migrations/025_set_supplier_lot_sprinkles_26-03-10-FOUN-001.sql) — done
- [migrations/029_rename_unknown_lot_to_25216.sql](../../migrations/029_rename_unknown_lot_to_25216.sql) — done
**Current behavior**: These already ran; they're idempotent `IF NOT EXISTS`/guarded updates so re-running is safe. They do bloat `migrations/` and make it unclear which files still matter for a fresh environment.
**Risk**: None operationally; a new engineer looking at `migrations/` sees 23 files and assumes all are core schema when only ~10 are. Also if one of these hand-coded SQL fixes had a subtle bug that only manifests now, it re-runs on every Railway deploy indirectly through Supabase (though these are not actually executed by the app — they were manual one-shots).
**Recommended fix**: Move one-shot reconcile migrations to `migrations/archive/` (or a separate `reconciliations/` folder). Keep numbered schema-changing migrations (003–013, 016–020, 026–028, 030, 031) in `migrations/`.
**Effort**: small
