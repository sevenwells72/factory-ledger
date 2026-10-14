# Hosted-staging count rehearsal — 2026-10-13/14 (one area)

**Plain English first.** The cutover count procedure was run for one area against the **deployed**
FastAPI-staging service on Railway, not a server on this Mac. It worked first time: 11 lots were
posted with receipts, the 680 lb correction was held and released under the signed list, the open
shortage was closed as *counted*, and the new extra-kosher lot was loaded and attested. Afterwards
every lot showed difference 0. Hosted posting takes **about 3 s per lot** (local run: about 1 s).
No production read or write. The synthetic owner is **not** Michael.

## What ran

| Item | Value |
|---|---|
| When | 2026-10-13 23:41–23:43 EDT (2026-10-14 03:41–03:43 UTC) |
| API | `https://fastapi-staging-production-dd7b.up.railway.app` (hosted; `FL_STAGING_API_URL` unset) |
| Deployed build | Railway FastAPI-staging deployment `6d169ef7`, SUCCESS from `d5e9fea` (main = #108 count tool `5dd7711` + #111 read-only key). `/openapi.json` exposes `resolves_shortage_ids`. |
| Database | guarded staging DB (`staging_safety.assert_staging_database`, port 5432); fixtures seeded there directly, all ids ≥ 10⁹ |
| Script | `scripts/count_rehearsal_staging.py --hosted` (new one-area mode on this PR) |
| Receipt | `docs/deployments/count-rehearsal-hosted-staging-receipt.json`, reference `STG-CNT-A13CF3` |
| Tools | this branch = main `5dd7711` count tools, unchanged (`reset_preview.py --v3 --area`, `apply_reset.py --target staging`, `verify_reset.py --v3`) |

Steps, as on cutover morning: fake filled sheet for area **Dry storage** (11 rows, counted 23:41:41) →
fresh preview → (no owner review needed: nothing held) → signed `approval.md`
(`Owner approves held entries: Y`, `Extra-kosher lots: 1000000141:STG-CNT-A13CF3-EK-1`) →
dry run → `--apply` with the phrase typed into a real terminal → verify.

## The fake count sheet and what FL did

| Row | Book | Counted | Change | Receipt | What it proved |
|---|---|---|---|---|---|
| Oats (estimated) | 400 lb | 395 lb | −5 | ADJ-261013-047 | ordinary lot, Estimated=yes note |
| **Sugar** | 1,200 lb | 520 lb | **−680** | ADJ-261013-048 | **>500 lb hold on the signed list**: LARGE_CORRECTION exception #53 held, approved by the owner key, `resolved/approved` |
| **Coconut** | −20 lb (short make MK-261013-022) | 25 lb | +45 | ADJ-261013-049 | **shortage lot**: SHORTAGE #52 named in the ticket, closed `resolved/counted`, fixing ticket = ADJ-261013-049, decision **XR-261013-014** (`resolve_exception`, committed) |
| **Extra-kosher batch** (synthetic, `kosher_tier=extra_kosher`) | no lot | 180 lb, written lower-case | +180 | FND-261013-006 | **extra-kosher line**: NEW lot `STG-CNT-A13CF3-EK-1`, `found_inventory`, `extra_kosher`, attested by the owner actor; KOSHER_ATTESTATION #54 `resolved/approved`; 180 lb on one ledger line |
| Granola 15 LB case | 300 lb | 17 cases | −45 lb | ADJ-261013-050 | case pounds: not held (45 lb, not 675) |
| Ingredients 1–6 | 100 lb each | 99…94 | −1…−6 | ADJ-261013-051 … -056 | ordinary lots |

Verify: all 11 lots **READY, remaining difference 0**, balances equal the counted figures. Exit
code 2 only because the rest of the staging catalog was not counted (full-scope sign-off), as in the
local runs. Rehearsal owner/floor actors deactivated afterwards; the staging owner key file removed.

## Real timings (hosted)

| Step | Seconds |
|---|---|
| New rehearsal keys accepted by hosted FL (actor cache, 60 s TTL) | 32.2 — set-up only; not part of a real pause |
| Fresh preview | 1.0 |
| Dry run | 0.7 |
| Apply incl. typed confirmation and read-back | 40.2 (executor: confirmed 0.6, all posted 39.1, read-back done 39.5) |
| Verify | 0.5 |
| **Machine time of the pause** | **42.4** |

Per lot, in posting order (seconds): oats 2.6 · **sugar 4.7** (prepare, held commit, owner approve) ·
**coconut 4.2** (shortage close + XR) · cases 2.7 · ingredients 2.7, 2.7, 3.1, 2.7, 2.8, 2.9 ·
**extra-kosher 6.5** (found, held commit, attestation approve). Ordinary lot ≈ 2.8 s; held/closing lot
4–7 s. Local-server runs on 2026-10-13 took about 1.0 s per lot, so the hosted round trips add about
2 s per lot.

**What this means for the real pause.** Each lot re-reads a fresh snapshot before writing. Staging's
snapshot takes about 1 s; production's took 4.0 s on 2026-10-13. Estimate per production lot ≈ hosted
2.8 s + 3 s extra snapshot ≈ **6 s** (held/kosher lots ≈ 8–10 s), plus about 15 s fixed for preview,
dry run and verify. A 40-lot area ≈ **4–5 minutes of machine time**, matching the READINESS estimate;
the rest of the 15–30 min pause is the owner checking the fresh preview against the sheet and signing.

## Choices made unattended (and why)

- Waited for the in-progress staging deploy of #111 (`d5e9fea`) to finish before running, so the
  rehearsal hit one stable build. #111 adds a read-only GPT key and does not touch the count paths.
- The extra-kosher line uses a **synthetic** extra-kosher lb product: every real lb candidate
  (283/285/288) already carries an earlier rehearsal's lot on staging, and reusing one would not test
  the new-lot path.
- Fixtures (products, lots, opening stock, actors) were seeded directly in the guarded staging DB,
  as in the earlier runs. Every posting and approval in the area went through the hosted API.
- Used `~/dev/factory-ledger/.venv-test` Python (has psycopg2); the system Python does not.

## Left behind on staging

Synthetic products/lots/actors labelled `STG-CNT-A13CF3` (ids ≥ 10⁹, actors inactive), and the
receipts above. Harmless; later staging tests of those ids must expect them.

## Michael's production dry run (preview + verify only, named key, no posting)

**Where:** Michael's Mac, in an up-to-date checkout of `main` (≥ `5dd7711`), from the repository
root, `python3` and `psql` on PATH. Production is read through `scripts/psql_ro.sh` (port 5432,
`BEGIN TRANSACTION READ ONLY`); the dry run makes **no API calls** and posts nothing.

**Needs first:**
1. `~/.config/factory-ledger/blubber_key` — his personal owner key, `chmod 600`, not a symlink.
   (On 2026-10-14 this Mac had only `~/.config/factory-ledger/db_url`; the key file was **absent**.)
2. `~/.config/factory-ledger/db_url` — production read credential (present).
3. A small real sheet: 3–5 lots he counts himself in one area, rows copied from
   `audits/fresh-start/v3/count-sheet-v3.csv` with real times, plus `coverage.csv` and an empty
   `moves.csv`, in a folder such as `~/fl-dryrun/`.

**Commands** (about **30 minutes** in all: ~15 min counting/typing, ~5 min signing, **under 30 s**
of machine time — production preview ≈ 5 s, dry run ≈ 5 s, verify ≈ 5 s):
```sh
python3 -B audits/fresh-start/reset_preview.py --v3 ~/fl-dryrun/count.csv --coverage ~/fl-dryrun/coverage.csv \
  --moves ~/fl-dryrun/moves.csv --area "Dry storage" --target production --output-dir ~/fl-dryrun/preview
# cp audits/fresh-start/approval.md ~/fl-dryrun/preview/ and fill it: Preview file reset-preview-v3.json + its SHA-256 (shasum -a 256), Approved product IDs = READY rows,
# Owner name + Named actor + Signature = Blubber, Date = today (New York), Opening entry classification: N,
# Owner approves held entries: N, Extra-kosher lots: none
python3 -B audits/fresh-start/apply_reset.py ~/fl-dryrun/preview/reset-preview-v3.json \
  --approval ~/fl-dryrun/preview/approval.md --count-csv ~/fl-dryrun/count.csv --target production   # NO --apply
python3 -B audits/fresh-start/verify_reset.py --v3 ~/fl-dryrun/count.csv --coverage ~/fl-dryrun/coverage.csv \
  --moves ~/fl-dryrun/moves.csv --area "Dry storage" --target production --output-dir ~/fl-dryrun/verify
```

**What to look at:**
- Preview: each row READY or HELD with a reason; counted vs book and the proposed change look right;
  any open shortage, >500 lb hold or extra-kosher row is flagged.
- Dry run prints `Target: PRODUCTION (https://fastapi-production-b73a.up.railway.app)`, one line per lot
  `current + (change) = planned` with its cutoff, and ends `DRY RUN ONLY: no API calls, no inventory
  changes. Named actor verified by read-only lookup.` A refusal such as "Key is not the owner-approved
  named actor" means the key file is wrong — stop.
- No `audits/fresh-start/apply-state/<sha>/journal.jsonl` was created, and the dashboard shows no new
  ADJ/FND receipts.
- Verify shows the same proposed differences (not 0 — nothing was posted) and exits 2 for the
  uncounted rest of the catalog. That is expected.
