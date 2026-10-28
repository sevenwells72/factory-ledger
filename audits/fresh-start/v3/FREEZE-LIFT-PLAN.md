> **ALL STEPS DONE 2026-10-28** (production main `9a9a9536`). Current state: `MASTER-HANDOFF-2026-10-28.md`. The "Nothing below has been done" line below is the original Oct 27 text.

# Freeze lift and deployment plan — for owner approval (2026-10-27)

Reset: COMPLETE (owner confirmed 2026-10-27). Nothing below has been done. ★ = separate owner approval per step.
Every merge to `main` auto-deploys the backend (Railway FastAPI) and the dashboard (Netlify).

## Version check (read-only, 2026-10-27)
| Part | Live version | Note |
| --- | --- | --- |
| Backend (Railway FastAPI) | `9d5e888b` (Oct 20), healthy | as the freeze requires |
| Dashboard (Netlify) | current `main` `4150434b` (Oct 23: #142 screens + #149 wallpaper) | CSS/JS identical to main; HTML differs only by Netlify link rewriting |
| Floor GPT | v4.3.0 (pasted Oct 21; includes #145/#146 wording) | its new `no_supplier_lot` field is ignored by the live backend — no errors |

**Live mismatch to fix first:** the production Record entry screen (#142) calls `GET /entry/lots`, which the live
backend does not have (404; staging has it). Its lot picker is likely failing since Oct 23. Step 2 fixes it.

## Steps
1. DONE 2026-10-27 — ★ **Lift the freeze.** Remove the "PRODUCTION FREEZE: ACTIVE" block from `CLAUDE.md` and `AGENTS.md` (repo checkout),
   `~/.claude/CLAUDE.md`, `~/.codex/AGENTS.md` and `COUNT-DAY-RULES.md`. Fresh production backup first
   (`~/Backups/fl/fl-backup-and-verify.sh`).
2. DONE 2026-10-27 (deploy 71f7551d, checks passed) — ★ **Backend to current `main`** (ships #142: read-only `/entry/lots` + lot search; no migration). Railway "Redeploy" of `main`.
   Check: deployment SUCCESS on `4150434b`, `/health`, `/entry/lots` answers 401 without a key (exists), Record entry
   lot picker works in the browser. Rollback: redeploy `9d5e888b`.
3. DONE 2026-10-27 (merge d7bf203a; Railway ea92fdf6 + Netlify 6ae0fc0c, checks passed) — ★ **#145 — supplier lot on ingredient receipts** (no migration). Update the branch to `main` (it conflicts with the
   wallpaper change), full tests both modes, merge. Check: deploy SUCCESS, a real receive shows "Write this on the
   pallet tag" with the CNS lot; a receive without a lot still records (unidentified + follow-up). Rollback: revert.
4. DONE 2026-10-28 (merge 97674c0a; Railway 51519cdf; checks passed; live link check = next real shipment) — ★ **#146 — signed packing-slip links on ship** (no migration; stacked on #145). Retarget to `main`, tests, merge.
   Check: next real shipment returns a 10-minute packing-slip link. Rollback: revert.
5. DONE 2026-10-28 (088 applied + verified; merge b02982b4; Railway 1e84a828; checks passed; first real shipment confirms 1 receipt row) — ★ **#153 — duplicate-shipment guard.** Its "Deploy order": staging rehearsal → migration 088 on production
   (port 5432) and verified → update branch after #145/#146 → tests → merge. Check: no
   `SHIP_ORDER_RETRY_GUARD_UNAVAILABLE` in logs; one receipt row per shipment. Required before regular GPT shipping.
6. DONE 2026-10-28 (merge 9a9a9536; Railway 9a01f022; pooled-lot identity refused as draft blocker) — ★ **#154 — pooled-lot safeguard** (no migration). Update branch, tests, merge. Check: identifying the pooled Coconut
   Flake lot is refused (read-only preview of the correction, nothing posted).

One step at a time; the next starts only after the previous one is checked. Steps 2–4 fit in one morning; 5 needs the
staging rehearsal first.

## After deployment (each ★ when you are ready)
- Corrections: Dark Chocolate +28 lb; flax 50 lb and monk fruit 55 lb Found entries; #9 Regular (lot split decision);
  Almond Vanilla / PB and Banana Mix (product confirmation).
- Required note for processing-yield adjustments (small change).
- Open PRs to review separately: #147, #148, #144, #143, #141.
- API-key rotation about two weeks after stable operation.
