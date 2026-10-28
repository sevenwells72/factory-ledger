# Factory Ledger — MASTER HANDOFF (cutover closed 2026-10-28)

**Start here.** This document replaces `HANDOFF-2026-10-27.md` as the current state. Older plan and handoff files in
this folder are historical records; where they disagree with this file, this file wins.
Owner: Michael (production actor "Blubber"). Wants brief, nontechnical answers; every production change needs his
explicit approval for that step.

Location: in the GitHub repo `sevenwells72/factory-ledger` at `audits/fresh-start/v3/MASTER-HANDOFF-2026-10-28.md`
(docs-only PR "Cutover closeout records"; once merged the repo copy is canonical). On Michael's Mac the same folder also exists,
untracked, in `~/dev/fl-countday-wt/audits/fresh-start/v3/` with the full 136 MB evidence; a verified archive of all of it is in
`~/Backups/fl/2026-10-28-cutover-evidence/` (Mac only). The repo holds the Markdown records plus the key reconciliation/proof files.

Legend: **VERIFIED** = checked directly (read-only) on 2026-10-28 ~13:15 ET · **REPORTED** = seen only through the
owner's screenshots or statements · **PENDING** = not done yet.

---

## 1. Executive summary
- The October physical-count inventory reset is **complete** (RESET COMPLETE 2026-10-27, whole-reset proof passed).
- The production freeze was lifted 2026-10-27 and all four waiting updates are live: **#145, #146, #153, #154**
  (plus #142 backend). Every deploy had a verified backup first and passed its checks.
- Arturo's follow-up corrections are posted (Dark Chocolate, flax, monk fruit, #9 Regular, Almond Vanilla).
  PB & Banana (1,120 lb) is held for disposal and deliberately **not** in FL.
- What remains is operational: real-world first use by Arturo and Liz, two supplier-lot follow-ups, the PB & Banana
  disposal entry, and housekeeping (old PRs, key rotation, GPT→plugin migration before Dec 11).

## 2. Verified current production state (2026-10-28)
| Item | State | Status |
| --- | --- | --- |
| Backend (Railway `FastAPI`) | deployment `9a01f022` SUCCESS on main **`9a9a9536`** (#154); `/health` healthy | VERIFIED |
| Workers (`expire-tickets`, `exceptions-sweep`) | SUCCESS on `9a9a9536` | VERIFIED |
| Dashboard (Netlify `cns-factory-ledger`) | published build of `d7bf203a`; `dashboard/` + `netlify.toml` identical to main `9a9a9536` (later merges changed no dashboard files, Netlify cancels "no content change") | VERIFIED |
| Staging (Railway `FastAPI-staging`) | SUCCESS on `9a9a9536`, healthy; staging DB has migration 088 | VERIFIED |
| Database | migration 088 applied (`ship_order_commit_receipts`, RLS on, 0 rows); 0 lots negative on posted transactions; no held tickets, no live drafts | VERIFIED |
| Latest transaction | TX2891 (10 posted today, TX2882–2891, all by Blubber) | VERIFIED |
| Reset / final reconciliation | `final-reconciliation-complete/` says RESET COMPLETE, 0 differences, 0 open; whole-reset proof `whole-reset-oct26/proof-final/` | VERIFIED (files) |
| Open exceptions | 2: #66 Flax – Ground lot 26-10-28-FOUND-001, #67 Monk Fruit lot 26-10-28-FOUND-002 (UNIDENTIFIED_LOT, due 2026-11-07 UTC ≈ Nov 6 ET) | VERIFIED |
| Office GPT "Factory-Ledger 2.0" | instructions **v3.8.2**, schema **3.5.1** (30 actions), API-key auth; preview "no lot on label" → unidentified, not recorded | REPORTED (owner screenshots 2026-10-28) |
| Floor GPT "Factory Ledger - Floor 2.0" | schema **4.3.0** (includes #145 + #146), last edited Oct 23; same preview test passed | REPORTED (owner screenshots); full instruction text not inspected |

**Corrected balances (posted transactions only, VERIFIED):** #9 Regular (107) **1,682 lb**, all on lot OCT 16 2026 ·
Almond Vanilla (113) **475 lb** · Dark Chocolate (121) **1,078 lb** · Flax – Ground (27) **50 lb** · Monk Fruit (45)
**55 lb** · PB Banana Granola (122) **0** · Coconut Flake (12) **166,150 lb** (pooled 26-10-27-FOUND-011 159,250 + lot 6013 6,900).

> Query note: always sum `transaction_lines` joined to `transactions` with `status='posted'`. Summing all lines
> includes voided history and shows six false negative lots (e.g. Coconut Flake lot 6037 −2,400).

## 3. Completed work (chronological)
| Date | What | Evidence |
| --- | --- | --- |
| Oct 21–22 | Physical count (Arturo et al.) | `count-sheet-v3.*`, `count-inputs/`, `raw/` |
| Oct 22–27 | Reset groups reviewed/approved and posted (01, 02, final-g1, small groups A/B, provisional #9, 285/286) | `approved-*`, `review-*` folders |
| Oct 27 | Opening stock posted FND-261022-001…011 (TX2871–2881); Coconut Fancy as 4 supplier lots (6187, 6196, 6113, 6089); Coconut Flake as one POOLED lot | `found-oct26/`, `TRACEABILITY-EXCEPTIONS.md` |
| Oct 27 | 7 unidentified-lot follow-ups (FL exceptions 50, 51, 53, 55, 57, 59, 65) written off (`waived`); lots stay unidentified | `TRACEABILITY-EXCEPTIONS.md` |
| Oct 27 | Whole-reset proof FULL RESET SCOPE RECONCILED; final reconciliation RESET COMPLETE | `whole-reset-oct26/proof-final/`, `final-reconciliation-complete/` |
| Oct 27 | Backup + local restore verified; **freeze lifted** (CLAUDE.md/AGENTS.md notices) | `~/Backups/fl/2026-10-27/` |
| Oct 27 | Backend → main `4150434b` (ships #142, `/entry/lots`); Railway `71f7551d` | CHANGE_LOG 15:07 |
| Oct 27 | **#145** supplier lot required on ingredient receipts → main `d7bf203a`; Railway `ea92fdf6`; Netlify `6ae0fc0c` (manual build: squash message carried `[skip netlify]`); production preview checks: lot → identified, blank → 422, "no lot on label" → unidentified | backup `2026-10-27-pre145/` |
| Oct 28 | Office GPT re-pasted (v3.7.0 → v3.8.2, schema 3.5.1); Floor GPT confirmed 4.3.0; both preview tests passed | owner screenshots |
| Oct 28 | **#146** signed 10-min packing-slip links → main `97674c0a`; Railway `51519cdf`; forged/over-long/missing signatures → 401 on production; valid link opened as PDF on staging | backup `2026-10-28-pre146/` |
| Oct 28 | **Migration 088** on production (port 5432, one transaction, lock_timeout 5s); verified | backup `2026-10-28-pre088/` |
| Oct 28 | **#153** duplicate-shipment guard → main `b02982b4`; Railway `1e84a828`; staging rehearsal: same commit twice → 1 shipment, second `idempotent_replay: true` | backup `2026-10-28-pre153/` |
| Oct 28 | **#154** pooled-lot identity safeguard → main `9a9a9536`; Railway `9a01f022`; production prepare on pooled lot → blocker POOLED_LOT_NOT_IDENTIFIABLE (drafts 205/206 expired, never committed) | backup `2026-10-28-pre154/` |
| Oct 28 | Corrections (reason `physical_count`; `count_correction` does not exist): ADJ-261028-001 Dark Chocolate +28 (TX2882); FND-261028-001 flax 50 (TX2883, lot 1659); FND-261028-002 monk fruit 55 (TX2884, lot 1660) | `ARTURO-OCT27-CORRECTIONS.md` |
| Oct 28 | #9 Regular 7,883 → 1,682 lb, **provisional FIFO** (not read from labels): ADJ-261028-002…007 (TX2885–2891); 4 entries >500 lb held (exceptions 68–71) and approved with the owner key on Michael's chat approval | same |
| Oct 28 | Almond Vanilla = 113: FND-261028-003 475 lb, lot 26-10-28-FOUND-003 (1661), provisional lot note | same |
| Oct 28 | PB & Banana 1,120 lb: owner decision — keep segregated for disposal, record nothing until discarded | same |

Every deploy: backup with `~/Backups/fl/fl-backup-and-verify.sh` (dump, Storage, local restore, KEY CHECKS ALL MATCH),
merge with a custom squash subject (no skip markers), Railway SUCCESS on the merge commit, `/health`, `keys.sql`
fingerprints identical before/after. Detailed entries: `~/dev/fl-countday-wt/CHANGE_LOG.md` and the repo `CHANGE_LOG.md`.

## 4. Decisions and operating principles (owner, 2026-10-27/28)
1. Accurate inventory and traceability **going forward** beat reconstructing imperfect history; this cleanup was one-time.
2. Historical assumptions are written down as assumptions (e.g. "PROVISIONAL FIFO, not read from labels") — never presented as physically verified.
3. Transactions and audit history are never edited or deleted (append-only triggers are live); corrections are new records with a reason, receipt and named actor.
4. Adjustments need a valid reason; >500 lb needs a photo or owner approval; Found always needs owner approval for extra-kosher.
5. Supplier lots are recorded exactly as printed, or explicitly "No lot on label" (→ unidentified + 7-business-day follow-up). Never invent one.
6. Pooled inventory is never identified as one supplier lot; split it with Found + adjust corrections per supplier lot.
7. Production changes: one approved step at a time, verified backup first, full tests, verify after (deploy, health, fingerprints).
8. Arturo and Liz use their own production accounts/keys, never shared ones.
9. Simple, reliable daily workflows; no unnecessary complexity.
10. Material held for disposal is not entered as usable inventory; its Found + disposal pair is posted on the day it is discarded.

These are also summarised in the repo `CLAUDE.md` ("Current state" section) so every session sees them.

## 5. Outstanding-work register
| # | Item | Status | Priority | Owner | Next action | Blocks ops? | Date |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | PB & Banana Mix 1,120 lb (most likely product 122; alt. 291 WIP mix) | Segregated, NOT in FL | High | Arturo → Claude | Tag "DISPOSAL – DO NOT USE"; on discard day post Found 1,120 ("held for disposal") + adjust −1,120 `damage_disposal` with photo | No | When discarded |
| 2 | Flax lot 26-10-28-FOUND-001 / Monk fruit 26-10-28-FOUND-002 unidentified (exc. 66/67) | Open | Medium | Arturo | Read supplier lot off bags → supplier-lot correction; else waive with note | No | Nov 6 |
| 3 | Pooled Coconut Flake 26-10-27-FOUND-011 (159,250 lb) + lot 6013 (6,900 lb, older May lot, **not** covered by #154 guard) | Permanent register | Medium | Michael | Recall of supplier lots 6161/6182/6113/6153/6196/6114/6190/6127/6051/6125 covers both lots; split with Found+adjust as bags are confirmed | No | Ongoing |
| 4 | Six other unidentified opening lots (Cinnamon, Sugar 6X, Glycol, Choc Chips, Corn Starch, Honey) | Register, follow-ups waived | Low | Arturo | Identify when evidence appears or consume | No | Ongoing |
| 5 | Provisional lots: #9 OCT 16 2026 (FIFO assumption), Almond Vanilla 26-10-28-FOUND-003 | Documented | Low | — | None; use first | No | — |
| 6 | Unrecounted small items: Kookies & Kreme (5 cases Oct 22 unexplained), Sprinkles lot split 25132/25114, Classic Granola 25 LB 5 lb, White Choc Chips 0 | Accepted unverified | Low | Arturo | Include in weekly spot-counts | No | Next 2 weeks |
| 7 | First real ingredient receipt under #145 (CNS lot on pallet tag; "no lot on label" path) | Pending live use | High | Arturo | Do it with the Floor GPT; confirm receipt read-back | No | This week |
| 8 | First real Floor GPT order shipment: packing-slip link (#146) + one `ship_order_commit_receipts` row (#153) | Pending live use | High | Arturo → Claude | Ship normally; then check link opened and 1 receipt row | No | This week |
| 9 | Liz: dashboard Record entry + Office GPT receiving/orders; daily Exceptions check | Pending live use | High | Liz | Use for real work; report issues | No | This week |
| 10 | Arturo/Liz on their own production accounts and the production dashboard (not staging hosts) | REPORTED only | High | Michael | Confirm each person's key/actor and URL `cns-factory-ledger.netlify.app` | Partly | This week |
| 11 | Owner key ("Blubber") gets 403 on header-key `GET /sales/orders/{id}/packing-slip` (pre-existing permission) | Open | Low | Claude | Decide if owner should have it; small permissions fix + PR | No | — |
| 12 | Required note on processing-yield (`hydration_yield`) adjustments | Not started | Medium | Claude | Small backend change + PR | No | — |
| 13 | Open PRs: #148, #147, #144, #143, #141 (pilot UX), #129, #127, #126, #125, #117, #114, #113, #96, #91, #32, #29, #16 | Drafts/stale | Low | Michael | Review; several titles contain `[skip netlify]` — use a custom squash subject if any touches the dashboard | No | — |
| 14 | API-key rotation | Not started | Medium | Michael + Claude | Rotate after ~2 weeks stable use | No | ~Nov 11 |
| 15 | Custom GPTs → plugins (ChatGPT banner: GPTs not migrated stop after **Dec 11**) | Not started | High | Michael + Claude | Plan migration of Floor + Office GPTs | Yes after Dec 11 | **Dec 11** |
| 16 | Railway "Config as Code" deprecation (railway.json/toml stop working **2026-12-01**) | Warning seen | Medium | Claude | `railway config migrate` in a PR, test on staging | Yes after Dec 1 | **Dec 1** |
| 17 | Full reset evidence (136 MB review folders, raw data) and count-day tool edits (`fresh_start_v3.py` etc.) stay Mac-only (untracked folder + backup tgz); key records are in the docs PR | Mac-only | Medium | Michael | Merge docs PR; optionally copy `~/Backups/fl/` off-machine | No | — |
| 18 | Repo checkout `/Users/cns/Documents/factory-ledger` (detached, old `0fa518f6`): uncommitted `CHANGE_LOG.md` block (+~430 lines, many sessions); 84 untracked files incl. duplicate "… 2.png/.log" screenshot copies. CLAUDE/AGENTS notices are in the docs PR | Partly unsaved | Low | Michael | After the docs PR merges, decide whether to commit that CHANGE_LOG block; review duplicates before any cleanup | No | — |
| 19 | Old worktrees with local-only commits (fl-a1-wt 7, fl-o7-wt 7, fl-a1p2-wt 3, fl-recipes-wt 3, fl-a7-wt 2, fl-count-wt 1) and abandoned uncommitted alt-088 in `.claude/worktrees/sleepy-lamarr-654d57` | Not from this session | Low | Michael | Review before any cleanup; nothing deleted | No | — |
| 20 | Other findings of the operating-model review (OpenAI key spending cap, packing-slip path for the office, stale STATUS.md/CONTEXT.md/DEPLOYMENT.md, Office GPT duplicate-entry risk) | See `docs/operating-model/07-issues-and-recommendations.md` | Medium | Michael | Work through that list | No | — |
| 21 | Staging data: test shipment 21 on staging order SO-261015-003 (rehearsal) | Expected | — | — | None | No | — |

## 6. Known risks
- **Single-copy evidence:** reset evidence lives only in an untracked folder (+ local backup tgz). Fix: docs PR / off-machine copy.
- **GPT deadline Dec 11** and **Railway config deadline Dec 1** could break the GPTs / deploys if ignored.
- **First real use not yet observed** for #145/#146/#153 on production; tests and staging passed.
- **Recall scope:** Coconut Flake pooled lot + lot 6013 mean a supplier-lot recall covers ~166k lb.
- **Squash subjects:** PR titles with `[skip netlify]` silently skip dashboard publishing if used as the merge subject.

## 7. Architecture (current, brief)
| Part | Production | Staging |
| --- | --- | --- |
| Database | Supabase Postgres project `vrafvwcdpcijvxdvefpr` (MyFirstProject, us-east-1); use session pooler **port 5432** for scripts | Supabase `jygmyvxnxdjiiilhxseq` |
| Backend API | Railway `FastAPI` → https://fastapi-production-b73a.up.railway.app (auto-deploys main) | Railway `FastAPI-staging` → https://fastapi-staging-production-dd7b.up.railway.app (also builds main; `railway up` for branches) |
| Workers | Railway `expire-tickets`, `exceptions-sweep` | `*-staging` |
| Dashboard + FL Assistant | Netlify `cns-factory-ledger` → https://cns-factory-ledger.netlify.app (publishes main when `dashboard/` changes) | other Netlify hosts are staging |
| Floor GPT | "Factory Ledger - Floor 2.0", schema `gpt-configs/schemas/openapi-floor.yaml` 4.3.0, instructions `gpt-configs/dist/GPT_FLOOR_INSTRUCTIONS.md` | — |
| Office GPT | "Factory-Ledger 2.0", `openapi-gpt-v3.yaml` 3.5.1 (30-op cap), `gpt-instructions-v3.md` v3.8.2 | — |
| Storage | Supabase Storage bucket `purchase-documents` (6 files) | — |
Detailed architecture and employee workflows: `docs/operating-model/` in `/Users/cns/Documents/factory-ledger`
(by the "Factory Ledger architecture review" session; not duplicated here).

## 8. Next recommended actions
1. Watch the first real receipt and first Floor GPT shipment with Arturo (items 7–8), and Liz's first day (item 9).
2. Merge the docs-only PR (handoff, reset records, `docs/operating-model/`, CLAUDE/AGENTS notices, regression-changelog row 236). Large evidence stays in backups.
3. Plan the GPT→plugin migration (Dec 11) and Railway config migration (Dec 1).

## 9. How to verify state (read-only)
- `curl https://fastapi-production-b73a.up.railway.app/health`; `railway deployment list --service FastAPI` (run in `/Users/cns/Documents/factory-ledger`, the linked dir).
- DB: `psql "$(cat ~/.config/factory-ledger/db_url)"` (port 5432) with `BEGIN TRANSACTION READ ONLY;` — never session-level readonly on 6543.
- Fingerprints: `psql … -f ~/Backups/fl/2026-10-28-pre154/verify/keys.sql`, compare with `prod_keys_posttest.txt` there (changes once new transactions post — expected).
- Backup before any production change: `LC_ALL=en_US.UTF-8 FL_BACKUP_DIR=~/Backups/fl/<date>-<label> ~/Backups/fl/fl-backup-and-verify.sh`.

## 10. Supporting records (this folder unless noted)
`FREEZE-LIFT-PLAN.md` (all steps DONE) · `ARTURO-OCT27-CORRECTIONS.md` · `TRACEABILITY-EXCEPTIONS.md` ·
`UNVERIFIED-BALANCES.md` · `HANDOFF-2026-10-27.md` (superseded) · `final-reconciliation-complete/` ·
`whole-reset-oct26/` · `~/Backups/fl/` (backups 2026-10-27 … 2026-10-28-pre154, evidence tgz) ·
`~/dev/fl-countday-wt/CHANGE_LOG.md` · `~/change-log.md` · repo `FACTORY_LEDGER_CHANGELOG.md` rows 229, 230, 234, 235 and deploy record row 236 · staff guides and architecture: `docs/operating-model/`.
