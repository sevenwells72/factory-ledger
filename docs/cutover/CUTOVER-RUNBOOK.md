# MASTER CUTOVER RUNBOOK

## Key dates

All calendar dates are kept in this table. Use the milestone names below throughout this runbook. Times are America/New_York; checklist clock times are proposals for Michael to confirm (O-1).

| Milestone | Date | Required result |
|---|---|---|
| Pilot start | Wed Oct 21, 2026 | Arturo and Luz begin on **staging**, for part of each day. |
| Readiness week | Oct 26–30, 2026 | Five clean shift summaries **and the production dry run**; no production apply in the dry run. |
| Print packets | Fri Oct 30, 2026 | Print and register approved count packets. |
| Go/no-go | Mon Nov 2, 2026 | Michael decides from completed evidence. |
| Live count | Thu Nov 5, 2026 | Physical count while production keeps running. |
| Signed approval list | Evening of the Live count milestone | Michael signs the exact reset lists the evening before cutover. |
| Reset + cutover | Fri Nov 6, 2026 (`2026-11-06`) | Each area pauses 15–30 minutes for reset; whole-reset proof passes before release; master rotates; both GPTs become read-only. |
| A10 stage 1 | Mon Nov 9, 2026 | Retire staff legacy ledger writes through the reviewed rollout. |
| Fallback 1 cutover | Fri Nov 13, 2026 (`2026-11-13`) | Use only after Michael selects it and rebooks the dependent milestones. |
| Fallback 2 cutover | Fri Nov 20, 2026 (`2026-11-20`) | Use only after Michael selects it and rebooks the dependent milestones. |
| M0 + GPT retirement deadline | **Before Fri Dec 11, 2026** | M0 read-only plugin live and accepted; old GPTs retired with a working ChatGPT read path. |

**Status: DRAFT for Michael's review.** The settled decisions in this update supersede older schedule and OPEN-status text in linked sources. Original source baseline: merged `origin/main` `611d0c4`; later accepted O-2/O-3 decisions are recorded in PR #117. PRs [#112](https://github.com/sevenwells72/factory-ledger/pull/112) and [#113](https://github.com/sevenwells72/factory-ledger/pull/113) provide rehearsal and pilot context. This document does not claim that pending production acceptance has happened.

## Michael's one-page summary

Follow **Key dates**: part-day staging pilot, Readiness week, Print packets, Go/no-go, Live count, then Reset + cutover. A failed mandatory check means NO-GO. Select a listed fallback and agree its go/no-go, count, signing and A10 schedule; none of those fallback milestones is silently inferred.

Arturo counts while production continues. Each area records start/finish times and every movement during its window. Luz collects photographs, types the exact counts and checks that every location was searched. Blank rows never mean zero. Pre-cutover extra-kosher stock goes into the count as **283/284 batches and 285–290 finished goods**. Extra-kosher **received is refused; found is held until Michael approves from his phone**. Verify the paper batch log and approve each new lot. Packaging counts are information only.

At **Signed approval list**, sign each area's exact list. On **Reset + cutover**, each area pauses 15–30 minutes for reset. Claude Code checks freshness, runs the dry run, applies with your personal owner identity and reads every receipt back with Luz. Changed evidence requires a fresh preview and signature. An uncertain response is reconciled with the same ticket/journal, never a second reset.

**No receipt = not recorded.** Every write is **prepare → Record → receipt number**. A draft, a hold or an Assistant explanation is not a ledger entry. Staff sign in with their personal key once per shift and use an **FL session with a 180-minute idle timeout**. Background auto-refresh does not extend it. Re-sign in after expiry or sign-out; **PIN login stays OFF**. Luz handles office-authorized actions; Arturo or Michael handles floor actions under their own sign-in.

**FULL RESET SCOPE RECONCILED (exit 0)** is required before **any cutover release step**, including key rotation and flags (O-3 CLOSED). Then rotate the master **on cutover day**, rotate the scoped dashboard credential and make both GPTs read-only. The owner-only, note-required, pre-cutover “shipped, not recorded” close/reopen rules cover every path, including the master key (O-4 CLOSED). Sunshine bulk remains on the paper bin log until after cutover; A8 is deferred (O-2 CLOSED). Office ticket-screen work is O-8 CLOSED; production handoff checks and A10 stage 2 remain separately tracked.

Open release work includes production session/proxy acceptance (O-5), the complete key-dependent inventory (O-6), dashboard key delivery (O-7), and the real production dry run (O-10). The register names each owner and status. Check summaries, exceptions and paper each operating shift after cutover. Complete **both** M0 acceptance and old-GPT retirement before the **M0 + GPT retirement deadline**.

### Settled recipe and Assistant checks

- **Recipes:** oat bran stays in SS #1/#2/#3 at **29/28/28 lb**. Rolled oats may substitute pound-for-pound **without approval**; record the actual ingredient and lot. Fruit Nut is **328.3 lb = 276.3 lb Classic #9 + 13 lb each of four nuts/fruits**. Product **284 uses 4,000-count chips**. Verify these in pilot drafts/receipts; no recipe or stock change is executed by this docs task.
- **Assistant:** unsupported requests show a neutral English/Spanish card, never a raw error. A red card is only for a failed **Record**. Adjust/found without a reason shows reason buttons; missing details are listed in plain words. No receipt number means nothing is recorded, regardless of card color. Verify in both languages during the pilot.
- **Extra-kosher:** do not work around a refused receive or held found. Michael approves from his own phone; pre-cutover stock belongs in the signed count, not an invented receive. New make holds and regular-source pack refusals remain part of acceptance.

## How to execute and record this checklist

This is one sequence. Tick a step only when its expected result and verification are recorded. `OPEN O-n` refers to the decision register at the end. A blocked command is deliberately not invented. Michael approves each future production action separately, as required by [CLAUDE.md](../../CLAUDE.md) and design §10.1. This PR only writes documentation.

Use only the approved execution checkout. Commands below are future operator instructions, **not commands executed for this PR**. Do not use another worktree's count inputs. All count inputs, reviews, previews, approvals, output and journals must stay under that checkout's `audits/fresh-start/`; `~/fl-dryrun` from the earlier #112 example is rejected by the tool's path guard. Use a private operations copy; count evidence and journals are not PR attachments. Journals contain ticket credentials and must never be printed or published.

Define these non-secret variables once in the approved operator checkout. Copy the ISO value from the selected **Reset + cutover / Fallback** row of **Key dates** into the prompt. On fallback, select that row, rebook dependent milestones with Michael (O-1), and create fresh inputs and approvals:

```sh
cd ~/dev/fl-cutover-wt
set +x
umask 077
read -r 'CUTOVER?Selected cutover ISO date from Key dates (YYYY-MM-DD): '
API=https://fastapi-production-b73a.up.railway.app
RUN="audits/fresh-start/v3/cutover-$CUTOVER"
mkdir -p "$RUN/inputs" "$RUN/evidence" "$RUN/approved"
```

The proposed daily start is 07:00. Michael/Arturo must approve actual areas, order, staffing and clock times (O-1). An area group includes **all observations for its products/lots**; stock spanning areas cannot be forced into independent resets. Use repeated `--area` flags for an approved combined group. Rehearsal estimate: about 35 seconds fixed plus 6 seconds per ordinary lot, longer for held/kosher rows; 40 lots roughly 4–5 minutes of machine time. Keep 15–30 minutes for review/readback; split or regroup large scopes before Friday, preserving cross-area reconciliation.

For each step record scheduled/actual time, executor, Michael's approval reference, tested/deployed SHA or deployment ID, input/preview hashes, receipt/exception IDs, pass/fail and rollback decision. Never record key values, session/ticket tokens, DB URLs, bearer links or screenshots of secret fields. Generate secrets inside the password manager; transfer them only into masked fields/private files. The G1 checklist's secret-printing generation example is superseded here by this rule.

## Ordered, timed checklist

1. [ ] **Before Pilot start, proposed 09:00–10:00: confirm scope, people and fallback.**
   - **Owner:** Michael; Arturo and Luz supply the area plan; Codex records decisions.
   - **Action / click path:** open this runbook → **Key dates** and O-1; name the area captains, each area's posting contact and the approved execution checkout. Confirm part-day pilot slots with Arturo/Luz, the selected cutover and fallback rebooking procedure. O-2 is CLOSED: Sunshine bulk uses the paper bin log until after cutover; A8 is deferred. Review actual dedicated OpenAI project/key, cap/alerts and test-key revocation evidence (O-13); ask Michael for the readiness slot, not an invented deadline.
   - **Expected / verify:** dated staffing/area schedule, one owner acceptance slot each working day, current evidence links; Sunshine bulk bins on the paper bin log (O-2). No invented plant closure calendar; currently only weekends are skipped by deadlines.
   - **Rollback:** if scope or staffing is unavailable, defer cutover; continue the existing recording process and staging pilot. On fallback, rerun this step for changed staffing and holiday availability.

2. [ ] **Before Pilot start, proposed 10:00–12:00: reconcile merged code, deployments and safeguards.**
   - **Owner:** Claude Code; Codex reviews; Michael approves subsequent operator work.
   - **Action / click path:** GitHub → `main` → commit history; Railway → gleaming-solace → **FastAPI** → Deployments and Variables (masked); Netlify → **cns-factory-ledger** → Deploys. Match deployed commits against A1–A7, A9/A12, F1 #90, key sessions #103, count #108, G1 #111 and D1/D2 #107. Read the final rollout sections rather than the old “unmerged/pending” paragraphs. Inspect approved migration-marker/ownership evidence; do not rerun the test schema. A6's 074/private bucket/signer/sweep rollout is already recorded; verify it remains healthy. O-3/O-4 are CLOSED decisions; retain their regression evidence. Production acceptance and inventory remain OPEN O-5/O-6.
   - **Expected / verify:** signed code/deployment/migration matrix; private production `evidence` bucket, app/table ownership and dedicated `PACKING_SLIP_SIGNING_KEY` proved. Merge #103 (`5228885`) records 077 applied to production, with KEY_SESSIONS_ENABLED unset; merge #90 (`a5b0ff6`) records 068 applied, with ASSISTANT_ENABLED unset. Verify current markers/ownership/flags, rather than blindly reapplying either migration. Parked 073/A11 PIN is not a prerequisite. G1 production support must be present before issuing its key. Any missing rollout requires its own reviewed procedure and approval.
   - **Rollback:** leave new flags off and hold GO. Preserve additive evidence/receipt tables. No migration rollback or DB restoration is a shortcut to readiness.

3. [ ] **Before Go/no-go, proposed 13:00–14:00: inventory every key dependent and prepare replacements privately.**
   - **Owner:** Michael and Claude Code; Codex checks coverage.
   - **Action / click path:** password manager → create three distinct URL-safe ASCII generated entries (letters/digits/underscore/hyphen, no quotes, backslashes or newlines; use `flro_` prefix for read-only) for new `API_KEY`, new `DASHBOARD_API_KEY`, new `READONLY_API_KEY`; no stdout generation. Railway → FastAPI → Variables (masked) and all reference-variable consumers; Netlify → project configuration/deployment inputs; GPTs → My GPTs → each editor's Actions → Authentication. Complete this masked inventory with consumer, owner, credential kind, replacement path, probe, disable/rollback path and readiness state (O-6/O-7):

     | Dependent | Required disposition in rotation |
     |---|---|
     | FastAPI `API_KEY` | Replace server secret; old master must fail after deploy. |
     | Office and Floor GPT action auth | Replace old master with read-only key after server rotation; never give GPTs new master. |
     | FastAPI `DASHBOARD_API_KEY` | Replace with a separate scoped browser credential. |
     | `dashboard/dashboard.js` (`SALES_API_KEY`), `mini-calendar.js` (`SALES_API_KEY`), `process-flow.html`, `sankey.html`, `traceability.html` (`API_KEY`) | All five currently embed the shared dashboard value. Prepare reviewed delivery of the new scoped value to **both** Railway-served assets and Netlify; `dashboard_config.json` has no key field, and a Netlify variable alone does not rewrite static literals. No values in diffs/review notes. Implementation/delivery method is OPEN O-7. |
     | Master bootstrap/admin callers, including `scripts/mint_actor_keys.py`; manual `set_sweetened_coconut_yield.py` (`API_KEY`), `cleanup_void_reversals.sh` (`FACTORY_API_KEY`), `granola_writedown_0917.py` (Railway variable lookup) | Inventory and disable obsolete execution; update retained private callers. Do not run historical write tools as probes. |
     | External integrations, local secret files, Postman/curl presets, bookmarks, scheduled jobs, other machines/users | Michael must supply the complete inventory. Remove old key-bearing packing-slip bookmarks; use authenticated dashboard links. Unlisted consumers remain O-6. |
     | Personal actor keys/sessions, count owner's key file, OpenAI key, private storage credentials, slip signer | Different credentials: do not replace them with the new master/dashboard key. A master rotation does not revoke personal actor keys. |
     | `expire-tickets`, `exceptions-sweep`, public health keepalive | DB-reference/public-health dependencies, not master-key consumers; verify health without copying DB secrets. |
     | Future M0 actor mappings / read-only service | Per-user private keys and Google auth; no shared master in ChatGPT. |

   - **Expected / verify:** every consumer accounted for, retained callers have a prepared private replacement, obsolete callers disabled, both GPT editors ready. All new keys differ from each other and active actor keys. Save version references only.
   - **Rollback:** before deployment, cancel the rotation window and keep candidates private. After rotation, repair consumers or issue a further fresh key; do **not** restore the exposed old master.

4. [ ] **Before Go/no-go, working-day review slots: accept current rehearsal and safeguard evidence.**
   - **Owner:** Claude Code builds in separately authorized work; Codex reviews; Michael accepts.
   - **Action / click path:** GitHub → PR #112 → `HOSTED-REHEARSAL-2026-10-14.md` and hosted receipt; compare with merged [READINESS](../../audits/fresh-start/READINESS-2026-10-13.md) Parts 2/2b/3b. Require current hosted-staging evidence for shortage closure, >500 lb signed hold, extra-kosher found/attestation, duplicate/multi-area behavior and per-area readback. #112's hosted proof is merged rehearsal evidence (`2f11015`); refresh if runtime changes. Reconfirm the accepted O-3 whole-reset proof procedure and O-4 all-path close/reopen safeguards; both are CLOSED. Review the whole-reset success and failure evidence with the rehearsal; prepare reviewed A10 stages.
   - **Expected / verify:** named reviewers, current head/evidence, real hosted receipt IDs, zero differences on counted rehearsal lots; exit 2 for the uncounted rest of staging is explained, not called full sign-off. #112's fixture-date changes affect tests, not the production tool's date policy.
   - **Rollback:** reject stale/unreviewed evidence and continue staging practice; no synthetic stock or staging rehearsal script runs in production.

5. [ ] **Readiness week, each participating shift: collect five clean summaries.**
   - **Owner:** Arturo; Luz checks office actions; Michael reviews daily.
   - **Action / click path:** accepted staging FL Assistant → personal sign-in → real pilot scenarios; staging Shift Summary → Today → read all sections → **Confirm** or **Confirm with notes**. Test ambiguity/product/unit/lot confirmation, resolved suppliers on every RCV, substitutions through the accepted interface, BOL/photo policy, correction hold and one extra-kosher make+pack with Michael approving from his own device; deliberately attempt regular-source extra-kosher pack and record `KOSHER_SOURCE_REQUIRED` refusal. Run all four personal sign-ins, FL-session reuse across Assistant/forms/Shift Summary, 180-minute idle expiry while background auto-refresh continues, Sign out → `SESSION_EXPIRED`, and attribution checks. PIN remains OFF. Use PR #113's part-day pilot plan; Michael must answer its remaining practice/stock/coverage questions (O-9). Check the recipe and bilingual Assistant rules above, including reason buttons and plain missing-detail lists.
   - **Expected / verify:** five consecutive **working-day** staging shifts confirmed with **zero unexplained discrepancies**, every RCV with resolved `supplier_id`, owner-attested kosher flow/refusal, receipts attributed correctly; Michael, Arturo and one office user sign acceptance of A1–A6/F1/key sessions/D1/D2/G1 scripts. An explained staging fixture shortage needs its explanation and approved disposition, not an ignored red flag.
   - **Rollback:** mark a failing shift **Something is missing**, investigate and restart the five-day clean streak after correction. Do not compress a failed five-summary streak into the remaining slot. Michael rebooks readiness and go/no-go under the selected fallback. Staffing/closure treatment is O-1/O-15.

6. [ ] **Readiness week, slot to confirm: production count-tool dry run, no apply.**
   - **Owner:** Michael supplies/approves real evidence and personal key; Claude Code executes; Codex checks sanitized output.
   - **Action / command:** reserve a fresh local evidence folder under `audits/fresh-start/v3/production-dryrun-$CUTOVER/`. Luz copies/transcribes a small **real counted** area, coverage and moves; no fabricated positive production counts. Michael privately provisions `~/.config/factory-ledger/blubber_key`, directory 0700/file 0600, for the actual production owner actor (currently `Blubber`). Verify permissions with `stat -f '%Lp' ~/.config/factory-ledger/blubber_key`; never `cat` it. O-10 remains OPEN until performed. In a terminal from the execution checkout:

     ```sh
     DRY="audits/fresh-start/v3/production-dryrun-$CUTOVER"
     mkdir -p "$DRY/inputs" "$DRY/preview-1" "$DRY/approved" "$DRY/verify"
     python3 audits/fresh-start/reset_preview.py --v3 "$DRY/inputs/count.csv" \
       --coverage "$DRY/inputs/coverage.csv" --moves "$DRY/inputs/moves.csv" \
       --area "Dry storage" --target production --output-dir "$DRY/preview-1"
     ```

     Michael fills the generated owner-review template against actual evidence; rerun with `--review "$DRY/inputs/owner-review.json"` into a fresh `preview-2` directory. Sign `"$DRY/approved/approval.md"` for `preview-2/reset-preview-v3.json`, its full SHA and exact READY product set. Then:

     ```sh
     python3 audits/fresh-start/apply_reset.py "$DRY/preview-2/reset-preview-v3.json" \
       --target production --approval "$DRY/approved/approval.md" \
       --count-csv "$DRY/inputs/count.csv"
     python3 audits/fresh-start/verify_reset.py --v3 "$DRY/inputs/count.csv" \
       --coverage "$DRY/inputs/coverage.csv" --moves "$DRY/inputs/moves.csv" \
       --review "$DRY/inputs/owner-review.json" --area "Dry storage" \
       --target production --output-dir "$DRY/verify"
     ```

     Use the real area name, not a guessed label. These commands intentionally omit `--apply` and saved `--snapshot`.
   - **Expected / verify:** apply dry run exits 0 with `Target: PRODUCTION`, correct named actor and `DRY RUN ONLY: no API calls, no inventory changes`. Preview/verify take fresh transaction-scoped read-only snapshots via port 5432. Verify may return 2 for planned differences/incomplete scope; log each reason. No new receipt or API write; no apply journal. Record production timings and extrapolate the area schedule. Do not run the staging seeder against production.
   - **Rollback:** any target/key/hash/hold/permission failure stops the dry run; correct private provisioning or source evidence and obtain fresh review. No production inventory rollback is needed because nothing posted.

7. [ ] **Print packets, proposed 13:00–14:00: print and register blind count packets.**
   - **Owner:** Luz prints/registers; Arturo checks legibility; Michael approves packet version.
   - **Action / click path:** open `audits/fresh-start/v3/count-sheet-v3.pdf` in Preview → File → Print → US Letter → Landscape → Actual size/100% → single-sided → Print; inspect a sample before the full set. Current packet is 110 pages/850 rows/210 products including unidentified rows. Print the live instructions, sheet register, coverage and movement logs; add uniquely numbered spare pages. Copy blank CSVs to `"$RUN/inputs/"` as `count.csv`, `coverage.csv`, `moves.csv`; retain originals. Confirm current A12 names and separate lines for 283–290, lb for 285/288 and **Dup?** on every row. O-11 if catalog/labels have changed.
   - **Expected / verify:** Arturo reads printed lot/unit/row fields at full size; Luz accounts for each issued sheet/area and spare ID. No FL quantity or suggested stock balance is shown to counters. Suggested categories are not physical areas. Packaging/billing 102/176 remain information only; inactive products are physically counted but stay subject to reset holds.
   - **Rollback:** withdraw and mark an obsolete packet **VOID — replaced**, preserve it, reissue a versioned approved packet. The builder overwrites issued v3 files; any regeneration needs a separately reviewed packet copy, never an unnoticed overwrite during counting.

8. [ ] **Before Go/no-go, proposed 09:00–10:00: prepare manual fallback and count evidence archive.**
   - **Owner:** Luz and Arturo; Michael chooses access/retention and staffing (O-9/O-12).
   - **Action / click path:** print/approve a numbered bilingual late-entry form with action, product/SKU, lot and initialled last-four/full-code confirmation, supplier/lot, quantity/unit, happened time, person, reason, BOL/bin details and receipt box. Design §9 names `docs/manual-tests/late-entry-sheet.pdf`; verify the file exists and is accepted before relying on it (OPEN). Set a private evidence folder for sheet photographs, original CSVs, signed approvals and hashes; separate credential-bearing journals.
   - **Expected / verify:** tabletop outage drill: one event is uniquely referenced (`LE-…/row`), complete and ready to enter later with `entry_source='paper'`, `paper_ref`, original happened time and receipt written back. Office permissions remain enforced: Luz may receive/ship/correct supplier lots; office cannot make/pack/adjust/found/void. Arturo/Michael cover those actions under their own sign-in.
   - **Rollback:** stop the affected electronic entry and retain numbered paper; never borrow another person's key or use a GPT to write. Missing accepted form or authorized coverage blocks GO for that work.

9. [ ] **Go/no-go, meeting time to confirm: formal GO / NO-GO.**
   - **Owner:** Michael decides; Arturo/Luz sign operational acceptance; Claude Code/Codex present evidence.
   - **Action / click path:** cutover log → mark each gate line pass/fail and link evidence:
     - A1–A6, F1, key-to-session sign-in, D1/D2 and G1 accepted on current staging by Michael, Arturo and an office user; ambiguity, safety/ship/lot acceptance scripts passed.
     - All four personal actors exchange keys; 180-minute idle expiry despite background auto-refresh and Sign out/refusal passed; receipts show the actual person. PIN stays off; the 10,000-PIN sweep belongs to a later PIN-activation gate.
     - Five consecutive confirmed clean working-day summaries from **Readiness week**; every RCV has resolved supplier; extra-kosher owner make+pack and source refusal passed.
     - Current count/rehearsal packet accepted, hosted rehearsal reviewed, production dry run complete, accepted whole-reset procedure ready to require **FULL RESET SCOPE RECONCILED (exit 0)** before any release (O-3 CLOSED), live count/review/signing slots booked. **Real count quantities and exact reset approval remain a Thursday/Friday release gate**, not a fictional signed count at **Go/no-go**.
     - Owner-only, note-required, pre-cutover close/reopen guards accepted on every path, including master-key paths (O-4 CLOSED), key consumers/new dashboard delivery ready (O-6/O-7), production sessions/proxy/destination plan accepted (O-5), OpenAI cap/alerts and storage/signer/cron readiness proved (O-13).
     - **No open STOP problem:** false-green/missing receipt, duplicate event, wrong actor, draft/post mismatch, credential exposure or policy bypass. These require engineering repair/reverification, not an ordinary usability waiver.
     - O-8 office ticket-screen decision is CLOSED; verify the accepted office handoff, and record the A10 stage 2 rollout decision and any deferrals (O-9/O-14/O-15).
   - **Expected / verify:** Michael signs dated GO with evidence and only accepted nonblocking deferrals, or NO-GO with owners/dates. Design §11 item 10(a)'s old pre-gate master rotation is superseded by decision 35: prepare it now, **execute on cutover day**.
   - **Rollback:** any mandatory failure → NO-GO. Michael selects **Fallback 1** or **Fallback 2** from **Key dates**, rebooks dependent milestones and revalidates evidence. No conditional waiver of whole-reset proof or the all-path close/reopen rule. If neither fallback is viable, ask Michael for a new plan.

10. [ ] **After Go/no-go and before Live count: freeze execution package and prepare recovery.**
    - **Owner:** Claude Code; Codex checks; Michael accepts.
    - **Action / click path:** record the approved deployment/packet versions and private configuration version references; Railway → FastAPI → Deployments → identify the last healthy release; Netlify → Deploys → identify healthy assets. Supabase → project → Database → Backups → verify a recoverable backup/status under Michael's approved procedure; do not create/restore data for this docs PR. Print the command/checklist pages and verify local dependencies (`python3`, `psql`, tool requirements) without calling production. Schedule change approvals and people for Friday.
    - **Expected / verify:** restore decision owner, known healthy code/config references, evidence custody and one recovery terminal/operator. Read the count tool journal/resume procedure together; a DB restore is disaster recovery only, requiring reconciliation of everything since backup.
    - **Rollback:** if recovery or approvals are unavailable, postpone. Preserve the healthy release; never clear tables or replay archived reconciliation scripts.

11. [ ] **Live count, agreed first shift through proposed 15:00: count every area while production runs.**
    - **Owner:** Arturo counts/directs; Luz logs movements and collects evidence; Michael answers holds.
    - **Action / floor procedure:** start the area's sheet with actual area, date, counter and start HH:MM; record each actual location/container/printed lot/product/quantity/unit, row time, Estimated notes and Dup? tick where appropriate. Record **every** receive, consume/make, pack, ship and transfer during the start/end window with time, item/lot, native quantity, from/to location/area, tag and existing FL transaction/receipt if there is one. The movement CSV is for cross-area moves; ordinary receive/consume/pack/ship records and paper events also belong in the area's evidence and ledger review. Do not fabricate an FL adjustment for a physical location-only transfer. At finish write end HH:MM, sign and photograph the sheet. Luz marks issued/returned sheets in the register; Michael initials completed coverage only after every location for that product was searched.
    - **Expected / verify:** completed row times fall inside the recorded window; coverage has offset timestamps (`-05:00`); moved_at has an explicit offset; every sheet/row ID is unique. Spare sheets get new IDs. Blank unused rows are not zero; missing known lots can be certified absent only by completed product coverage. FL has whole-lot balances, not shelf balances. Multi-area lots require linked observations and no-unlogged-moves owner review.
    - **Rollback:** if an area is interrupted, record actual stop/restart windows and retain all sheets/moves; recount ambiguous stock or hold it. Keep production running during the count; Friday's pause concerns posting for the selected items. Unknown food lots remain held with seven-day follow-up, not discarded or silently created.

12. [ ] **Live count, proposed 15:00–16:00: reconcile extra-kosher paper log and stock.**
    - **Owner:** Michael attests; Arturo identifies stock; Luz transcribes.
    - **Action / click path:** paper extra-kosher batch log (date + printed lot + actual procedure evidence) → match each counted lot/photo to **283/284 batches and 285–290 finished goods**, separate from regular 107/108 and regular finished stock. 285/288 are lb; boxed goods use their confirmed unit conversion. For a new extra-kosher lot, review canonical lot code, product, exact quantity, count cutoff and owner-attestation evidence in the found proposal. Existing regular lots are never retagged; extra-kosher receive is refused, found is held for Michael's phone approval, and no historical stock correction is invented.
    - **Evidence / exact upload path:** preserve the paper/photo archive and, under the separately approved evidence-custody plan O-12, upload each count photograph through authenticated `POST /attachments` with multipart `kind=count_sheet`, `file=<photo>` and optional `ticket_id` for an already known ticket. Save the returned attachment ID/hash privately. The executor does not upload or bind these photographs; the approved plan must specify any ticket/evidence linking instead of claiming it happens automatically. Signed owner approval is the accepted hold-release alternative.
    - **Expected / verify:** every NEW lot is listed as `PRODUCT:LOT` in the exact signed preview's approval; approving owner attestation required. Procedure/identity uncertainty stays held. The approved reset loads stock before any normal extra-kosher pack consumes it; found holds go through owner approval rather than bypassing it.
    - **Rollback:** segregate questionable stock and hold its product/group until Michael resolves evidence. Preserve the original count; no guessed tier, receive workaround or regular-lot conversion.

13. [ ] **Signed approval list, proposed 16:00–18:00: transcribe, review and sign each exact list.**
    - **Owner:** Luz transcribes; Michael approves; Claude Code generates previews; Codex checks the register.
    - **Action / command:** preserve source photos/CSV blanks; transcribe exact quantities/ticks. For each approved group use a fresh directory and exact actual area flags:

      ```sh
      GROUP=dry-storage                    # approved safe directory label
      AREA='Dry storage'                   # exact count area; repeat --area for combined groups
      INPUTS="$RUN/groups/$GROUP/evening-inputs"
      PREVIEW="$RUN/approved/$GROUP-evening-1"
      mkdir -p "$INPUTS" "$PREVIEW"
      cp "$RUN/inputs/count.csv" "$INPUTS/count.csv"
      cp "$RUN/inputs/coverage.csv" "$INPUTS/coverage.csv"
      cp "$RUN/inputs/moves.csv" "$INPUTS/moves.csv"
      python3 audits/fresh-start/reset_preview.py --v3 "$INPUTS/count.csv" \
        --coverage "$INPUTS/coverage.csv" --moves "$INPUTS/moves.csv" \
        --area "$AREA" --target production --output-dir "$PREVIEW"
      ```

      Copy `"$PREVIEW/owner-review-template.json"` to `"$INPUTS/owner-review.json"` and fill that **new group-specific input review file**. Each group uses immutable copies of the full applicable count/coverage/move/review data; later edits to another group must not alter these sources. Michael answers before/after movements, late entries, corrections, duplicate dispositions, move endpoints/inclusion/allocation, multi-area no-unlogged-moves confirmations and certified-absence questions. Rerun with `--review "$INPUTS/owner-review.json"` into a **new** final directory; set `PREVIEW` to it. Review largest changes, original/rounded quantities, all row cutoffs, stock ownership and the “Owner-approved posts” list. Compute `shasum -a 256 "$PREVIEW/reset-preview-v3.json"`. Copy the unsigned [approval template](../../audits/fresh-start/approval.md) into that directory and fill each field exactly once:
      `Preview file`, full 64-character `SHA-256`, exact `Approved product IDs`, `Owner name`, `Named actor`, `Date`, matching `Signature`, `Opening entry classification: N`, `Owner approves held entries: Y|N`, and exact `Extra-kosher lots` list or `none`.
    - **Expected / verify:** Michael's signed paper/photo and matching approval file for each READY group, including every proposed nonzero change, >500 lb hold, shortages and kosher lot. `Owner name/Signature` can be Michael while `Named actor` is the authenticated production `Blubber`; confirm the current roster privately. Y authorizes only the exact listed predicted FL holds with the full preview hash; a reconciliation HOLD is never approvable. Dates cannot be in the future; an evening-before signature is allowed. Evidence/input changes invalidate the preview.
    - **Rollback:** hold unresolved movement/duplicate/identity/unit/ownership questions; retain counts and rerun from corrected inputs with a new preview and signature. Do not edit generated plans, delete evidence, or approve unready groups. An incomplete whole reset blocks cutover release. Only step 20's **FULL RESET SCOPE RECONCILED (exit 0)** permits downstream release.

14. [ ] **Reset + cutover, proposed 06:45–07:00: morning readiness check.**
    - **Owner:** Michael GO; Arturo/Luz present; Claude Code operates.
    - **Action / click path:** cutover log → verify the selected **Go/no-go** decision, complete signed Thursday register, actual count completeness, accepted O-3/O-4 safeguards and completed O-5/O-6/O-7 release evidence and overnight movements/paper. Railway → FastAPI deployment health; dashboard → recent entries/receipts; confirm named owner key file privately. Check backup/recovery references and responsible people. Reconfirm posting contacts and group order.
    - **Expected / verify:** signed morning release, no missing/unreviewed count scope, tool/runtime/packet consistent, no unresolved technical blocker. Define who calls “pause”, “resume” and “abort” for each group.
    - **Rollback:** NO-GO before any write → reschedule and continue existing recording. If partial stock has already posted, enter the stop/recovery procedure at step 19; the calendar fallback does not erase those receipts.

15. [ ] **Reset + cutover, 07:00 onward: open each 15–30 minute group posting pause and refresh approval.**
    - **Owner:** Arturo calls the pause; Luz keeps its paper event log; Michael reapproves; Claude Code refreshes.
    - **Action / floor + command:** tell every poster the precise product/lot scope and log pause start; stop electronic posting for those items and interacting consumption/packing/shipment/transfer events. Capture physical events on numbered paper. Production outside scope continues. Create a new group morning-inputs directory and preserve the evening inputs; copy in the full updated count/coverage/moves/review data. Set `INPUTS` to that new directory. Refresh preview using step 13 with those current inputs and a new `...-morning` directory, without `--snapshot`. If overnight activity changed evidence or baseline, update owner review and sign a **new** exact preview/hash; retain the evening version.
    - **Expected / verify:** fresh READY products, complete group including cross-area observations, Michael signature and no concurrent posts affecting them. Estimate machine time against remaining pause; review people/time dominate. Do not push a hurried signature because the pause is running out.
    - **Rollback:** if >30 minutes or unresolved change, abort this apply window, log actual stop time and pending paper events, release operational pause under Michael's direction and schedule a fresh one. No write until the new signed preview passes.

16. [ ] **Reset + cutover, within each pause, first 2–5 minutes: dry run the exact signed production group.**
    - **Owner:** Claude Code executes; Michael witnesses; Luz compares the list.
    - **Action / command:** restore `GROUP`, `AREA`, `INPUTS` and `PREVIEW` for this group from its signed register. `PREVIEW` is this morning's approved directory; input paths and hashes exactly match its JSON:

      ```sh
      python3 audits/fresh-start/apply_reset.py "$PREVIEW/reset-preview-v3.json" \
        --target production --approval "$PREVIEW/approval.md" \
        --count-csv "$INPUTS/count.csv"
      ```

    - **Expected / verify:** exit 0, PRODUCTION, authenticated approved owner via read-only actor lookup, exact products/lots/changes/holds/cutoffs, no API calls. The current v3 tool uses the **named personal owner key**, superseding design §10.1's historical “master key” wording. The **first live group is one small active product**, counted in whole pounds or its approved native unit, separately signed/approved. Complete steps 17–18 and check the recorded actor plus dashboard balance before Michael approves/apply proceeds for any remaining group. Do not cherry-pick rows out of a signed plan.
    - **Rollback:** drift, unexpected shortage, new lot, invalid signature or warning → stop; return to fresh review/signature. No blind retries or changing an approved file to make the dry run pass.

17. [ ] **Reset + cutover, within each pause, next 5–15 minutes: apply only that signed group.**
    - **Owner:** Michael gives discrete apply approval; Claude Code executes with Michael's named owner key; Luz records receipts.
    - **Action / command:** run in an interactive terminal:

      ```sh
      python3 audits/fresh-start/apply_reset.py "$PREVIEW/reset-preview-v3.json" \
        --target production --approval "$PREVIEW/approval.md" \
        --count-csv "$INPUTS/count.csv" --apply
      ```

      Type exactly the phrase printed by the tool: `APPLY OPENING BALANCE <latest count-cutoff NY date> <full preview SHA-256>`. The date is the plan's latest cutoff, **not automatically Friday**. Never pipe the phrase. Existing-lot ADJ and new extra-kosher FND use `physical_count`, true count cutoff and `backfill=true`; predicted >500 lb/kosher holds are approved only when the signed list authorizes them. Named open shortages close as *counted* with ADJ plus XR decision receipts. Zero-change shortage remains a HOLD: Michael resolves via Exceptions → shortage → counted resolution/XR, then refreshes/signs the plan.
    - **Expected / verify:** each posted lot prints `Read back ADJ-…` or `FND-…`, correct product/lot/TX/actor. Preserve private `audits/fresh-start/apply-state/<SHA>/journal.jsonl` and `apply-results/<SHA>/readback.csv`, `timings.json`. Check XR receipts for each closed shortage, approved-hold owner/hash and kosher tier/attester. An unlisted hold, changed open-shortage set or unexplained duplicate stops, preserving partial progress.
    - **Rollback:** no bulk undo. Stop remaining posts and use step 19. Already committed receipts remain; any compensating ticket needs a new reviewed count/evidence and Michael's separate approval. Never execute direct `/adjust`, SQL stock edits or historical posters.

18. [ ] **Reset + cutover, last 5–10 minutes of each pause: independent verification and resume.**
    - **Owner:** Claude Code verifies; Luz compares every receipt to sheets; Michael/Arturo end only that area's reset pause.
    - **Action / command:** get the full hash (non-secret) and prove this group's original signed plan/journal:

      ```sh
      SHA=$(shasum -a 256 "$PREVIEW/reset-preview-v3.json" | awk '{print $1}')
      VERIFY="$RUN/verification/$GROUP-$SHA"
      python3 audits/fresh-start/verify_reset.py --v3 "$INPUTS/count.csv" \
        --coverage "$INPUTS/coverage.csv" --moves "$INPUTS/moves.csv" \
        --review "$INPUTS/owner-review.json" --area "$AREA" --target production \
        --approved-preview "$PREVIEW/reset-preview-v3.json" \
        --approval "$PREVIEW/approval.md" \
        --journal "audits/fresh-start/apply-state/$SHA/journal.jsonl" \
        --output-dir "$VERIFY"
      ```

      Repeat the same area flags as the signed preview. Count/review files must still be identical to it. **No-change branch:** if this group has no reset posts and no prior opening entries, no journal is created. Run the same verify command **without** `--approved-preview`, `--approval` and `--journal`; still require every selected lot READY/difference zero and explain incomplete remainder. Do not create a dummy journal. Dashboard → header **Receipt #** field → enter number → **Look up** (`/receipt.html?number=<RECEIPT>` on Netlify; `/dashboard/receipt.html?number=<RECEIPT>` on Railway) → lookup every ADJ/FND/XR and compare product/lot/count-time/quantity/actor. Read independent stored kosher tier and owner attestation. Check readback balances = count brought forward by reviewed later movements, tolerance 0.0001 native unit. Keep each group's signed input/review copy immutable; enter later working transcriptions in new files and retain original paths/hashes for verification.
    - **Expected / verify:** every selected lot READY/difference zero, authenticated receipt/journal proof, signed end-of-pause time. This resumes existing production for the verified area only; it does not release keys, flags or the new cutover workflow before step 20. Exit 2 can mean **other scope remains uncounted/held**; list the actual reasons. “No full sign-off” is not an error to suppress. Late paper events are entered once, at their actual time, only after resume; receipts are written onto paper.
    - **Rollback:** do not resume ambiguous posting for the affected scope. Resolve discrepancies/uncertain writes via step 19; never declare the whole reset complete from an area report.

19. [ ] **Reset + cutover, on any failure, immediately: stop and reconcile partial/uncertain execution.**
    - **Owner:** Michael controls recovery; Claude Code investigates; Luz/Arturo retain paper.
    - **Action / click path:** stop new reset processes; preserve lock/journal/plan/approval and inspect receipt/ticket status privately. A timeout can still commit. Rerun **the same apply command with unchanged approved files and original journal** only after preflight proves the baseline; it reuses an uncertain ticket or returns its original receipt. A new ticket is allowed only when the first is proved dead by the tool; an old journal without its token needs manual reconciliation. Check existing receipts in Receipts and holds in Exceptions before any approval/rejection. Do not delete/tamper with the journal, fabricate an opening classification or concatenate journals.
    - **Expected / verify:** one receipt/ledger post per ticket, every partial post accounted for, remaining scope clearly paused/paper. Record cause, receipt IDs and Michael's resume/abort choice without credentials.
    - **Rollback:** prefer roll forward from proved receipts. Compensating corrections use normal owner ticket controls and fresh signed evidence. Whole-DB restoration requires separately authorized disaster recovery and reconciliation of every event since backup; it is not the count rollback. If FL fails, continue numbered paper and keep GPT writes disabled once rotated.

20. [ ] **Reset + cutover, after the last group, proposed 10:00–10:30: whole-reset release gate.**
    - **Owner:** Michael signs; Claude Code supplies proof; Codex checks completeness; Luz reconciles the register.
    - **Action / command:** after the last group, freeze the final working copy, then list every group in one manifest at `$RUN/whole-reset-manifest.json` (procedure from PR #120, merged 5a2d57e). Never concatenate journals and never edit a signed group file.

      ```sh
      FINAL="$RUN/groups/final-inputs"
      mkdir -p "$FINAL"
      cp "$RUN/inputs/count.csv" "$RUN/inputs/coverage.csv" "$RUN/inputs/moves.csv" "$FINAL/"
      cp "$RUN/inputs/owner-review.json" "$FINAL/owner-review.json"
      # write "$RUN/whole-reset-manifest.json" — format: audits/fresh-start/tool-guide.md → "Whole-reset proof"
      python3 audits/fresh-start/verify_reset.py --whole-reset "$RUN/whole-reset-manifest.json" \
        --output-dir "$RUN/verification/whole-reset"
      ```

      Manifest paths are **absolute, or relative to the manifest's folder (`$RUN`)**, never to the shell's working directory. With `RUN=audits/fresh-start/v3/cutover-$CUTOVER`:
      - `"target": "production"`, `"expected_reset_products": 178`.
      - `final_inputs`: `"count": "groups/final-inputs/count.csv"`, `"coverage": "groups/final-inputs/coverage.csv"`, `"moves": "groups/final-inputs/moves.csv"`, `"review": "groups/final-inputs/owner-review.json"`.
      - One `groups` entry per signed group, in apply order: `name`; `areas` (the exact `--area` flags); `approved_preview` and `approval` copied from the **exact final signed `PREVIEW` actually applied**, using its paths relative to `$RUN`; `"journal": "../../apply-state/<that applied preview's SHA>/journal.jsonl"`. Read these from the signed execution register. A refreshed morning plan replaces the evening plan for proof: never point the manifest at a superseded evening preview or mix its approval with a later hash.
      - A group with no journal is listed with `"no_change": true` and its signed preview and approval. That is a step-18 no-change group, **and also a group whose approval lists only zero-stock products**: its step-17 apply posts nothing, writes no journal and a header-only `readback.csv`, then verifies.
      - **Zero-stock products** (coverage certified, no counted or on-hand lot, coverage review confirmed by the owner) must each appear in exactly one group's `Approved product IDs`. The executor accepts them and posts nothing.

      Everything stays under `audits/fresh-start/`. The tool is read-only: one snapshot, no API calls.
    - **Expected / verify:** exit **0** and `whole-reset-proof.md` saying **FULL RESET SCOPE RECONCILED**.
      - **Per group:** the signature; the signed inputs unchanged; the journal chain and tickets. Every approved change is posted exactly once, nothing held or uncertain; a second ticket still held beside a posted reset fails the group. **Every** journal receipt is checked whatever its outcome: it must be an outcome the executor writes, and only an `ok` receipt may name a transaction/receipt, which must be the proved post on the same ticket. **Every owner hold a journal names** (on a `held` receipt or an `approve` intent) must be FL's actual exception: same id, kind, ticket, product and lot, a kind the signed plan lists, under `Owner approves held entries: Y`, and an `approved` release must be by the group's named actor (`HOLD_NOT_IN_FL`; the executor refuses the same journal before any API call).
      - **Across groups:** each of the 178 reset products, zero-stock ones included, is approved by exactly one group; every count row of an approved product (duplicate-excluded rows included) and its coverage row are unchanged from the copy signed with that group; each multi-area lot is reset once and held in the other groups; no lot is posted twice; no opening post and no `physical_count` post by **any actor** (ticket or not) inside the reset window that is not a proved post of a signed group; no owner hold still open or escalated on a scope product; no signed owner answer changed; the final whole-scope verification reconciles with zero unexplained differences.
      - Exit **2** lists typed gaps in `whole-reset-gaps.csv`, e.g. `MISSING_PRODUCT`, `OVERLAP`, `MISSING_OBSERVATION`, `CHANGED_OBSERVATION`, `INVALID_RECEIPT`, `RECEIPT_CONTRADICTS_FL`, `HOLD_NOT_IN_FL`, `POST_OUTSIDE_APPROVED_PLAN`, `OPEN_HOLD`, `HELD_NOT_RELEASED`, `UNEXPLAINED_DIFFERENCE`. Each gap blocks release until resolved and the proof rerun. For an open hold the owner either approves it (then it is proved) or declines it. Packaging completion stays informational, not part of stock reset sign-off. Current packet scope is 178 reset products + 32 packaging/billing products; a catalog change shows as `SCOPE_CHANGED` and must be reconciled explicitly.
    - **Rollback:** **GO criterion (O-3 CLOSED, accepted by Michael):** no release step — key rotation, flag changes or any other downstream live release — until this proof reports **FULL RESET SCOPE RECONCILED (exit 0)**. Any gap (exit 2) or tool error stops release: preserve partially posted state for Michael's recovery decision, resolve the gap and rerun the proof.

21. [ ] **Reset + cutover, 10:30–10:45: enforce the actual order cutoff, including Friday's bypass fix.**
    - **Owner:** Michael approves; Claude Code changes approved production configuration; Codex checks evidence.
    - **Action / click path:** Railway → gleaming-solace → **FastAPI** → Variables → set `ORDER_LEGACY_CLOSE_CUTOFF` to the selected `$CUTOVER` from **Key dates** → Apply changes/Deploy → wait Active/Success. Verify the accepted O-4 guard is in the deployed release before allowing close/reopen operations. O-4 is CLOSED; this step verifies production configuration and preserved protections. Tests on approved synthetic staging fixtures: `/sales/orders/{id}/close/prepare` → commit permits named-owner `shipped_not_recorded` only for immutable NY **created-before-cutover** orders, requires nonblank note; orders created on/after cutoff without recorded shipment return 403 even for owner. Every path attempting **`shipped_not_recorded` close or reopen** must enforce the named-owner, nonblank-note and pre-cutover-order restrictions or refuse the action. For this special close/reopen, master, dashboard, office and floor credentials are refused. Ordinary invoicing and reopening with a saved shipped/invoiced state retain their effective-recorded-shipment and normal role checks; status changes cannot bypass the special-close rule. Reopening a `shipped_not_recorded` order needs a **new nonblank note**, keeps the original close audit, and still requires immutable NY creation before the cutoff even if a shipment was recorded later. Accepted implementation: [PR #121](https://github.com/sevenwells72/factory-ledger/pull/121), merge `9bd30be`. Use existing real read-only order evidence in production; do not close/create a production order just to test.
    - **Expected / verify:** production variable/deployed SHA, staging cutoff/bypass proof and production startup/health evidence. `order_date` is not the boundary. Unset makes all orders pre-cutover; malformed dates refuse startup. The cutoff configuration must run together with the accepted all-path guard, not replace it.
    - **Rollback:** failed config deploy → retain healthy release and stop close/invoice operations while correcting it. After cutover do not remove the cutoff to bypass a refusal; keep paper/held work and repair the guarded path.

22. [ ] **Reset + cutover, 10:45–11:00: rotate server master and dashboard keys in one controlled window.**
    - **Owner:** Michael handles secrets/approves each deployment; Claude Code coordinates consumers.
    - **Action / click path:** freeze old shared-key writes and notify operators in person. Railway → FastAPI → Variables → Edit `API_KEY` and `DASHBOARD_API_KEY` using distinct password-manager candidates → Save staged changes; stage every retained server-side dependent/reference from step 3. Deploy FastAPI → wait Success/health. New master remains private server/admin-only. Retained private clients change next; obsolete ones remain disabled. Do not place new master into GPTs/static dashboard. Use a private prompt for old/new probes and emit **status codes only**; verify old master header authentication gets 403, new master read succeeds, personal owner remains valid. No write-handler probe with a valid master.
    - **Expected / verify:** old master dead on all workers, new master usable only by inventoried private consumers, actor-key count executor unaffected, signer/storage stable. Both GPTs temporarily fail until step 24; direct routes still exist until A10, so master custody matters.
    - **Rollback:** failing deploy → keep posting paused, restore healthy **code** with the fresh key configuration or rotate forward to another fresh key. Never reinstate the exposed old master, or weaken auth to save a client. Failure of a nonessential consumer is a disabled consumer, not permission to revive the old key.

23. [ ] **Reset + cutover, immediately after server rotation, 11:00–11:15: deliver new dashboard key and redeploy Netlify.**
    - **Owner:** Claude Code executes the approved O-7 delivery plan; Michael approves; Luz checks pages.
    - **Action / click path:** use the accepted all-five-embedding delivery procedure from step 3 for Railway `/dashboard/*` and Netlify. Netlify → cns-factory-ledger → Deploys → Trigger deploy → **Clear cache and deploy site** (or the approved prepared artifact deploy) → wait Published. Check canonical production site, Railway-hosted dashboard and a staging preview separately; hard reload. Record deploy ID/asset digest, never the browser key. Verify old dashboard key rejected, fresh scoped key reads work, and it cannot exchange into a personal session or prepare make/pack/adjust/found/ship tickets. The current shared dashboard receive-ticket grant still exists; inventory it explicitly and retire only through a reviewed change.
    - **Expected / verify:** all five pages/mini-calendar read correctly; no cached old-key 401s; new public dashboard key remains scoped. This docs PR does not replace embedded values; O-7 must supply the implementation before GO. Rolling back a Netlify build with old literals would break reads.
    - **Rollback:** republish healthy assets carrying the current fresh scoped key. If a scoped key is mishandled, rotate it forward in server and all bundles again. Never substitute master or restore old public credentials to work around build failures.

24. [ ] **Reset + cutover, 11:15–11:45: issue READONLY_API_KEY, perform both G1 editor pastes and retire old literals.**
    - **Owner:** Michael handles secret/editor access; Claude Code assists; Codex/Luz record results.
    - **Action / click path:** follow [G1 CUTOVER_CHECKLIST](../../gpt-configs/readonly/CUTOVER_CHECKLIST.md) A–D with the no-secret-printing override. Railway → FastAPI → Variables → add `READONLY_API_KEY` from password manager → Deploy → Success. Use its six status checks: customers/orders reads 200; empty `/make`, `/receive`, `/sales/orders` POSTs 403 before handlers; session exchange 401 when enabled or 404 when off. For `POST /products/resolve`, use the harmless G1 read-only resolution action and expect success. To keep keys out of process arguments, use zsh hidden input and curl headers via stdin:

      ```sh
      set +x
      read -r -s 'RO?READONLY key (hidden): '; printf '\n'
      printf 'X-API-Key: %s\n' "$RO" | curl -s -o /dev/null \
        -w 'read /customers -> %{http_code}\n' -H @- "$API/customers"
      printf 'X-API-Key: %s\n' "$RO" | curl -s -o /dev/null \
        -w 'read /sales/orders -> %{http_code}\n' -H @- "$API/sales/orders?status=open&limit=1"
      for route in make receive sales/orders; do
        printf 'X-API-Key: %s\n' "$RO" | curl -s -o /dev/null \
          -w 'write refused -> %{http_code}\n' -H @- -H 'Content-Type: application/json' \
          -X POST "$API/$route" -d '{}'
      done
      printf '{"actor_key":"%s"}' "$RO" | curl -s -o /dev/null \
        -w 'session exchange -> %{http_code}\n' -H 'Content-Type: application/json' \
        -X POST "$API/auth/session" --data-binary @-
      unset RO
      ```

      Office first, then Floor: chatgpt.com → GPTs → My GPTs → Edit (pencil) → Configure → Instructions → replace with `office-instructions.md` / `floor-instructions.md`; Actions → existing FL action → Schema → paste `openapi-office-readonly.yaml` / `openapi-floor-readonly.yaml`; Authentication → API Key → Custom → `X-API-Key` → paste read-only key privately → Save → back → Update → confirm Update, same sharing. Office must show **14** actions (13 GET + resolveProducts), Floor **11** GET, zero writes. Check instructions' final line and schema errors before saving.
    - **Expected / verify:** all G1 smoke rows in **new chats**: O1–O14 and F1–F11 (25 results), O-W/F-W no write action/claim; Recent Entries has no GPT entry. Record all six status codes, 25 read actions, two write refusals, editor names/time and deployment IDs. Full route refusal evidence is `tests/test_readonly_api_key.py`; do not execute app/DB tests on production. Shared read-only key cannot exchange into an actor session.
    - **Action / old-literal scrub:** Claude Code follows the separately reviewed rotation package (O-7) to remove obsolete key literals from the seven audit-listed tracked files: `CHANGE_LOG.md`, `DEPLOYMENT.md`, `gpt-instructions-v3.md`, `gpt-configs/dist/GPT_FLOOR_INSTRUCTIONS.md`, `gpt-configs/sources/floor-specific.md`, `archive/superseded-instructions/GPT_INSTRUCTIONS.md`, `audits/reports/DASHBOARD_ACTIONABILITY_AUDIT.md`. Verify with a redacting scanner that emits filename/pass-fail only, never matches or secret-bearing diff lines. Repository history still contains the retired value; rotation is the control. No history rewrite or literal edits are performed by this docs PR.
    - **Rollback:** mismatched read → repaste auth/schema/instructions, update and retry failing row. Unsafe refusal behavior → remove `READONLY_API_KEY` in Railway and redeploy; stop GPT actions and investigate. Leave GPTs read-only even if reads fail. Never restore write schemas/master access.

25. [ ] **Reset + cutover, 11:45–12:15: enable production key sessions and verify dashboard destination/proxies.**
    - **Owner:** Michael approves; Claude Code configures; Codex reviews; Arturo/Luz verify identity.
    - **Action / click path:** prove 077 ownership/marker/private RLS tables under the approved rollout; if absent, use the separately reviewed 077 migration recipe, **not** a staging acceptance/seeder script. Railway → FastAPI → Variables → `KEY_SESSIONS_ENABLED=1`, `PIN_LOGIN_ENABLED=0`, reviewed `FORWARDED_ALLOW_IPS` (O-5) → Deploy → Success. Trusted proxy addresses must be established from the actual Railway ingress/Uvicorn startup; never invent `*` or trust arbitrary forwarded headers. Dashboard hosting setting: `window.FL_SESSION_API_BASE` must identify the intended API **before** runtime/session/fetch scripts on any custom host. Current `dashboard-runtime.js` already chooses canonical Netlify → production, other Netlify previews → staging, Railway → own origin; preview isolation wins over override. A Netlify env var alone does not inject that window value. Netlify redeploy step 23 must include accepted destination/runtime assets; reload after server flag flips.
    - **Expected / verify:** public `GET /auth/session/config` shows enabled/180 minutes; network destinations are production only on canonical production pages and staging only on previews. Michael, Arturo, Luz and Miriam each sign in with their personal key once per shift; the FL session serves Assistant, forms and Shift Summary on that origin. Every sign-in shows the correct actor. Session-only browser writes, Sign out refusal/re-sign-in, reload and idle/revocation evidence pass; only tokens in sessionStorage, no personal key persistence. Background auto-refresh must not extend the 180-minute idle timeout. Separate origins require separate sign-in. Assistant tickets remain 10 minutes; dashboard tickets retain their own expiry.
    - **Rollback:** bad proxy/routing/config → stop affected writes, disable `KEY_SESSIONS_ENABLED` and redeploy only with Michael's approval, keep paper. D1/D2 forms/Exceptions are hidden when this flag is off or config fails; disabling it is a continuity loss, not an automatic safe raw-key fallback. PIN stays off. Do not activate parked #96/073 to solve this.

26. [ ] **Reset + cutover, 12:15–12:30: enable FL Assistant with production OpenAI controls.**
    - **Owner:** Michael configures key/spend policy; Claude Code deploys; Arturo checks first real entry.
    - **Action / click path:** OpenAI API Platform → dedicated FL production project → API keys → create/use its production key privately; Project settings → Limits → Spend → Edit spend limit → **$20/month**, **Enforce a hard limit ON** → Save; set email alerts at **50% ($10)** and **80% ($16)** to `michael@sevenwellsgranola.com`. Verify recipient/project and no remaining PoC/test key. Railway → **FastAPI** → Variables → private `OPENAI_API_KEY` and `ASSISTANT_ENABLED=1` → Deploy → Success. OPEN O-13 until account controls and evidence are verified. Alerts alone do not cap traffic; hard enforcement can lag slightly ([official OpenAI spend-limit guide](https://developers.openai.com/api/docs/guides/spend-limits)).
    - **Expected / verify:** production `/dashboard/fl-assistant.html` → Arturo personal sign-in → show correct name → draft the **next genuine event** → inspect lot/unit/supplier/BOL/tier/warnings → Record → receipt lookup proves quantity/actor and exactly one post. Voice/photo only fill drafts. Record read-only budget/alert evidence without key fields. Never spend $20 or seed fake stock to test the limit.
    - **Rollback:** OpenAI/cap/model failure → `ASSISTANT_ENABLED=0` and deploy under approval; use accepted dashboard forms with the same tickets/roles. Do not raise/remove the cap without Michael's decision. FL outage → numbered paper; uncertain Record → same draft/receipt check, never an invented second entry.

27. [ ] **Reset + cutover, 12:30–12:45: hand over the new recording rule and fallback.**
    - **Owner:** Michael explains; Arturo/Luz repeat it and test their permitted forms.
    - **Action / click path:** in-person shift briefing: “ChatGPT answers questions; entries go through the FL Assistant or dashboard forms.” Open canonical dashboard → personal Sign in → **Record an entry** (`/entry.html`; Railway `/dashboard/entry.html`) → permitted action → draft → Record → Receipts. Office coordination uses Luz, floor make/pack/adjust/found uses Arturo or Michael. Ship needs BOL, lot confirmation and photo due by 23:59 plant time; shortage means resolve the real missing movement/count, not fabricate stock. Extra-kosher receive is refused; make/found stays held until Michael approves from his own phone; pre-cutover stock goes through the signed count as 283/284/285–290; expired holds are rejected/reprepared, not presented as live approval.
    - **Expected / verify:** operators can find Assistant, forms, Receipts, Exceptions and Shift Summary on their devices; identity correct, Sign out practiced on handover. Current permission matrix is honored; accepted office handoff and any remaining substitution limitations posted (O-8 CLOSED; O-9 questions). For a wrong entry, receipt lookup → permitted owner/floor **Prepare void for entry #** → reviewed draft → **Record** → VD receipt; office asks Michael. Do not invent an offsetting entry. Staff repeat “No receipt = not recorded”: prepare → Record → receipt number. Show a neutral bilingual card for unsupported requests, reason buttons for adjust/found missing a reason, and plain-word missing details; reserve red for a failed Record.
    - **Rollback:** an unavailable action uses its approved authorized form or numbered paper; pause that workflow until corrected. GPTs never become a write fallback and keys are never shared.

28. [ ] **Reset + cutover, end of every shift (proposed 15:00 / 10 minutes): confirm the first production summary.**
    - **Owner:** Arturo confirms his shift; Luz checks office entries; Michael reviews all actors.
    - **Action / click path:** production Shift Summary (`/shift-summary.html`; Railway `/dashboard/shift-summary.html`) → reuse the active personal FL session → Today → compare every ledger receipt and separate `other_receipts` (ORD/ER/LOT and decision/metadata receipts) → read NOT ENTERED YET?, shortages, held entries, BOL proof, late entries and kosher evidence → Confirm / Confirm with notes / Something is missing. Reuse the active FL session from the shift's personal sign-in; a second raw-key flow is not the operating procedure. Re-sign in only after expiry/sign-out or on another origin/device. A page that cannot reuse the session fails acceptance and is logged for repair. All-actors mode is read-only. Open Receipt from any disputed row.
    - **Expected / verify:** today's person/date and fresh `summary_hash`, receipt list matches floor/paper; a stale 409 requires refreshed review. “Something is missing” creates/reuses one SHIFT_DISCREPANCY, assigned for Michael. Approval changes may stale an earlier confirmation; reconfirm. One make/pack receipt covers one planned product/day run; don't mark all similar runs complete.
    - **Rollback:** never click Confirm to make a red flag disappear. Correct/reconcile the event, attach notes or mark missing and assign follow-up. No receipt → investigate before re-entering.

29. [ ] **Morning after Reset + cutover, proposed 08:00–08:30: verify jobs and proof deadlines.**
    - **Owner:** Claude Code checks service health; Michael reviews exceptions; Luz supplies missing BOL evidence.
    - **Action / click path:** Railway → `exceptions-sweep` → latest scheduled run (`30 7 * * *` UTC = 02:30 EST, after 23:59 plant time) → verify successful sanitized counts; `expire-tickets` → latest scheduled run/success. Exceptions page → open/escalated BOL_PHOTO_MISSING, UNIDENTIFIED_LOT, SHORTAGE, PRE_MAKE_ADJUST, LARGE_CORRECTION, KOSHER_ATTESTATION and SHIFT_DISCREPANCY; Receipts → related shipment/ticket. Late BOL upload/link through the accepted shipping evidence path resolves its missing-photo exception; signed slip/truck photo alone does not.
    - **Expected / verify:** jobs point to production through approved DB references/host guards, no duplicate photo exceptions, proof_complete shipments remain clear, held/expired tickets handled by owner. Record only status/IDs/counts, never DB variables/log secrets. Review Unknown-reason and >10%-of-lot correction flags in shift summaries/receipts: UNKNOWN_REASON is a flag, not an Exceptions-page kind. PRE_MAKE_ADJUST flags a positive adjustment within 30 minutes before that same actor makes/packs. SHORTAGE is due in **two business days**; received UNIDENTIFIED_LOT in **seven business days**: count weekdays after local entry day, due 23:59 NY, with no holiday/closure skips currently. Separate unidentified count-sheet follow-up is **seven calendar days after actual count time**. If no shift runs, record no shift; do not fabricate a confirmation.
    - **Rollback:** failed job → repair/schedule its approved service and manually review deadlines until restored; never run a staging job at production. Missing evidence remains flagged; do not waive it solely to clear the dashboard.

30. [ ] **Before A10 stage 1, proposed 16:00–16:30: readiness and paper backlog.**
    - **Owner:** Michael with Arturo/Luz; Claude Code/Codex assess A10 evidence.
    - **Action / click path:** runbook evidence → first-day F1/form receipts → verify every post/readback; reconcile all numbered paper rows with receipt boxes and duplicate review. Railway/Netlify health and current keys → status only. Review the prebuilt A10 stage 1 PR's current head/review:merge evidence and exact allowlist diff; verify retained owner/reset/admin tooling and the already-live Friday close guard remain covered.
    - **Expected / verify:** no unexplained reset/receipt gap, authorized fallback usable, A10 stage 1 can remove staff legacy ledger routes without stranding work. Any pending paper row has person/actual happened_at/paper_ref and an assigned entry time. No accepted receipt is entered again.
    - **Rollback:** unresolved gaps → delay **A10 stage 1** under Michael's decision, keep new ticketed recording and read-only GPTs; hold the affected work and resolve evidence first.

31. [ ] **A10 stage 1, proposed 08:00–09:00: retire staff legacy ledger routes.**
    - **Owner:** Michael approves discrete rollout/merge in that separate task; Claude Code deploys; Codex reviews; Arturo/Luz accept.
    - **Action / click path:** after first-day receipt lookup passes, follow the separately approved A10 stage 1 rollout PR. Remove direct ledger writes from **both** shared dashboard and personal-actor allowlists: `/receive`, `/make`, `/pack`, `/adjust`, `/ship` and their legacy `/preview`/`/commit` aliases; `/inventory/found`, `/inventory/found-with-new-product`, `/void/{transaction_id}`; `/sales/orders/{order_id}/ship` and its legacy preview/commit aliases; direct lot rename/supplier-lot/received-at PATCHes, reassign and `/admin/lots/merge`; legacy `confirmation_code` surface. Preserve supported ticket prepare/commit paths, including A5 `/lots/{lot_id}/move/prepare`, plus receipt/trace reads. Keep allowed private master/admin operations only as explicitly reviewed; final deletion is later. Retire/tier-gate `scripts/inv_recon_post_2026_08_17.py`; never execute it after 075. Use accepted staged refusal/permission tests, then health/read-only route checks in production and the next genuine ticketed entry/receipt. OPEN O-14 until built/accepted.
    - **Expected / verify:** staff/shared keys cannot post directly; prepare/commit workflows and authorized personal sessions still work; extra-kosher previews use tier-safe ticket paths; unchanged Friday close bypass protection. Record rollout SHA, refusal evidence and genuine ticket receipts. For a fallback cutover, Michael must rebook this milestone (O-1).
    - **Rollback:** pause affected entries and use paper; revert only the approved retirement change with current fresh secrets and Friday close guard preserved. Michael decides any temporary exposure; no automatic old-key/GPT rollback, no dropped receipts/tables.

32. [ ] **After stage 1 acceptance, time to confirm: A10 stage 2 decision; separately approved deployment.**
    - **Owner:** Luz demonstrates; Michael decides; Claude Code executes a separate reviewed rollout; Codex reviews.
    - **Action / click path:** dashboard header → **Office / Oficina** (`/office.html` on Netlify; `/dashboard/office.html` on Railway) → **Office action / Acción de oficina** → Luz under her office session demonstrates create/lines/header/status/close/cancel and expected-receipt create/change/cancel through prepare/commit, role rules and ORD/ER receipt lookup on staging. O-8 is CLOSED: [PR #115](https://github.com/sevenwells72/factory-ledger/pull/115), merge `d962a1f`, supplies the office ticket screens. Their production controls are gated by key-session enablement; verify the deployed version and flag during handoff. A10 stage 2 remains OPEN O-14: it removes legacy order/expected-receipt writes from both allowlists only under its separately reviewed rollout. Keep the Friday direct-close gate in force while stage 2 is deferred.
    - **Expected / verify:** Michael signs either accepted stage 2 release/date or explicit deferred date/owner with safe office procedure. Office never loses required order editing without a working ticketed replacement. Synthetic staging tests prove refusal and replacement receipts; genuine production edits are verified by lookup.
    - **Rollback:** missing ticket UI or failed acceptance → **do not deploy stage 2**. After a failed release, restore approved scoped access only under Michael's decision, retaining cutoff/role/note/shipment guards and current keys; paper for unavailable work.

33. [ ] **First operating week, proposed 15:00–15:30 daily: stabilize and review.**
    - **Owner:** Michael owns review; Arturo/Luz bring summaries; Claude Code handles service faults; Codex tracks defects.
    - **Action / click path:** Shift Summary → today/person/all actors; Exceptions → open/escalated; Receipts → missing/held/late proof; production usage/spend/health dashboards → status/amounts only. Compare opening-count receipts, ordinary events, shortages and paper to the end-of-shift evidence. Assign every discrepancy a person/due time. Schedule each active shift's summary and daily 10-minute owner check through the first operating week.
    - **Expected / verify:** no unexplained inventory changes, no lost paper rows, true suppliers/lots, kosher preparer and attester separate, no duplicate receipts, shortage resolution uses real count/missing movement. Keep receipt IDs/notes for weekly review. Exact-run links, weekly UI and Spanish/alias improvements are OPEN O-9/O-15; never assume their reports exist.
    - **Rollback:** recurring fault → disable the faulty Assistant only, use ticket forms; FL failure → paper; role/cutoff/tier bypass → stop the affected workflow and repair before resuming. No new synthetic production writes to investigate.

34. [ ] **First operating week, each shift + 10-minute morning check: monitoring.**
    - **Owner:** Arturo confirms; Luz reconciles office/paper; Michael reviews/escalates; Claude Code/Codex investigate assigned defects.
    - **Action / click path:** repeat steps 28–29 on operating days: verify shift confirmation, open Exceptions by age/owner, shortages and their ADJ/XR evidence, unresolved supplier lots/seven-business-day identification, missing BOL after 23:59, >500 lb holds, >10%-of-lot corrections, Unknown-reason flags and PRE_MAKE_ADJUST exceptions, expired kosher holds, late entries/paper backlog, legacy closes and SHIP_NOT_SAME_DAY acknowledgements, downgraded kosher packs, NOT ENTERED YET? planned work, aliases/unmatched language issues and OpenAI alerts. A handwritten log is the fallback when weekly aggregation is unavailable. Do not count a receipt twice across count/paper/shift lists. Plant holidays remain undefined in FL; Michael must set actual staffing and deadline handling without silently changing app behavior.
    - **Expected / verify:** daily evidence/owner/due time, five clean first-week production shifts where operating, receipt boxes filled on every entered paper row, discrepancies explained or openly escalated. At the end of the first operating week, proposed 15:00: weekly review from available shift/receipt/exception evidence; A9 weekly view is OPEN, not required to exist to do the review. Review legacy closes, late/same-day policies, thresholds, kosher downgrades and Sunshine ownership/billing issues without inventing invoices/prices.
    - **Rollback:** one false-green/missing receipt, duplicate event, wrong actor, draft/post mismatch, credential exposure or policy bypass → stop that action immediately, preserve private evidence/paper and require repair/reverification before resume. Other unexplained discrepancies → hold affected scope, recount and use owner ticket controls; defer optional features. Preserve the audit trail. A cap alert triggers owner review; cap exhaustion uses forms, not an unapproved limit increase.

35. [ ] **Before M0 + GPT retirement deadline, slot to confirm: accept live M0 read-only plugin before retiring GPTs.**
    - **Owner:** Codex prepares M0; Claude Code reviews; Michael supplies owner setup and approves each rollout; Arturo/Luz test with Michael and the fourth office user.
    - **Action / click path:** use the separately reviewed M0 plan: reconcile `feature/mcp-server` with current main; implement `MCP_WRITES_ENABLED=false` to **hide and refuse** write tools and state read-only behavior. M0 deployment and switch acceptance remain OPEN O-16. Owner setup covers Google OAuth/client/allowlist, Railway `factory-ledger-mcp` service, private per-user `MCP_ACTOR_KEY_*` mappings and ChatGPT connector installation. Exact service/console setup commands and current tool list belong in the reviewed M0 deployment checklist before execution; they are not built here. Deploy read-only and run each of the expected **17 read tools**, per-user sign-in/role/unauthorized-user checks, and attempted writes proving hidden+refused; no production test writes.
    - **Expected / verify:** all four people can ask real FL questions in ChatGPT through M0, never hold master/actor keys in prompts or browser code, and cannot access a write tool via listing or direct call. Record sanitized identity/tool/refusal evidence, service SHA/deploy ID and approved rollback. Book the acceptance slot with Michael (O-16), leaving time to complete retirement before the **M0 + GPT retirement deadline**. No specific earlier acceptance date is assumed.
    - **Rollback:** disable connector/service on auth/scope/write refusal failure; keep both GPTs read-only while Michael decides the explicit deadline contingency. No write-capable MCP/GPT fallback. Failure cannot silently retire the only working ChatGPT read path.

36. [ ] **Before M0 + GPT retirement deadline, after M0 passes: retire both old custom GPTs.**
    - **Owner:** Michael retires editors/sharing; Codex checks replacement; Arturo/Luz confirm reads.
    - **Action / click path:** first verify all four users can use the accepted read-only plugin. ChatGPT → GPTs → My GPTs → Office / Floor → Edit → remove FL Actions → Update; change sharing/availability using the agreed retirement method (archive/unshare rather than destructive deletion unless Michael chooses it). Mark instructions retired with link/directions to the accepted plugin. Remove `READONLY_API_KEY` from Railway → FastAPI → Variables → Deploy **only after** inventory confirms no remaining consumer. Exact sharing/retirement method and any M0-late exception are OPEN O-17.
    - **Expected / verify:** old GPTs have no FL action capability, their read-only key is invalid when retired, all four people's plugin reads succeed, write attempts refused. ChatGPT itself remains usable; FL Assistant/dashboard remain write interfaces. Retirement before the **M0 + GPT retirement deadline** is distinct from possible legacy-route deletion, which needs its own reviewed A10 cleanup.
    - **Rollback:** M0 acceptance failure → escalate to Michael before the deadline and agree recovery that preserves read access. An extension is not approved by this runbook and does not satisfy the deadline. After retirement, restore only read-only schemas with a **fresh** scoped key if Michael authorizes; never restore old master/write actions.

## Decision and build register

The **status** column governs; a closed decision still has to be preserved and verified in the deployed cutover release. OPEN O-5/O-6/O-7/O-10 and all mandatory acceptance evidence block GO until complete. O-3's proof blocks every downstream release even though its procedure and criterion are CLOSED. Link sanitized evidence when an open item is completed; this docs update does not assert production readiness.

| ID | Status | Owner / reviewer | Remaining action or settled decision | Needed by / release effect |
|---|---|---|---|---|
| O-1 | **OPEN — schedule details** | Michael decides; Arturo/Luz schedule; Codex records | Confirm part-day pilot slots, area groups/captains, count shifts, 07:00 proposal, pause order and holiday staffing. If fallback is used, book its go/no-go, count, signing and A10 milestones. | Before the affected milestone in **Key dates**; no guessed fallback dates. |
| O-2 | **CLOSED — paper bins; A8 deferred** | Michael decided; Luz/Arturo operate | Sunshine bulk uses the [paper bin log](https://github.com/sevenwells72/factory-ledger/blob/main/docs/pilot/sunshine-bin-log.pdf) until after cutover. One line per bin: time, bin, product/lot, gross/tare/net lb, order, weigher; same-day normal FL shipment for net lb. Billing rules unchanged; invoicing trigger remains an O-15 question. | Preserve through cutover; revisit A8 afterward. |
| O-3 | **CLOSED — whole-reset GO criterion accepted** | Michael accepted; Claude Code built; Codex reviewed | Step 20: `verify_reset.py --whole-reset "$RUN/whole-reset-manifest.json"` must exit 0 and say **FULL RESET SCOPE RECONCILED** before **any release step**. Any gap stops release. [PR #120](https://github.com/sevenwells72/factory-ledger/pull/120), merge `5a2d57e`, [accepted evidence](https://github.com/sevenwells72/factory-ledger/tree/main/docs/deployments/whole-reset-proof-evidence/STG-CNT-08370B). | Cutover-day proof remains mandatory; no concatenation workaround. |
| O-4 | **CLOSED — every close/reopen path guarded** | Michael decision; Codex built; Claude Code reviewed | “Shipped, not recorded” closing/reopening is owner-only, needs a note, and is pre-cutover only. Enforce on every route, including direct/status/master-key paths; the master is refused; use a named owner. Reopening needs a new note. [PR #121](https://github.com/sevenwells72/factory-ledger/pull/121), merge `9bd30be`, supplies the guard; step 21 checks the selected cutoff and deployed protections. | Preserve before any order workflow release; do not defer to A10. |
| O-5 | **OPEN — production session/proxy acceptance** | Claude Code establishes; Codex reviews; Michael authorizes; Arturo/Luz accept | Confirm 077 ownership/marker, flag enablement, actual `FORWARDED_ALLOW_IPS` and Uvicorn ingress, production/preview destinations and custom-host runtime wiring. If using a custom domain, include it in `dashboard-runtime.js` `apiBase()`; an env variable alone does not wire static assets. Verify once-per-shift FL-session reuse and idle expiry despite refresh. | GO plan and cutover production verification; staging evidence is not production proof. |
| O-6 | **OPEN — full masked key-dependent inventory** | Michael supplies/decides; Claude Code inventories; Codex checks | Account for private/external callers, retained/retired owners, replacement paths and rollback probes. | GO blocker; before cutover-day rotation. |
| O-7 | **OPEN — scoped dashboard-key delivery** | Claude Code prepares; Codex reviews; Michael chooses/approves | Reviewed delivery to all five embeddings, both hosting surfaces and caches; no master in browser. Resolve custom-host routing with O-5. Separate concurrent implementation work is not declared complete here. | GO blocker; verify fresh assets after rotation. |
| O-8 | **CLOSED — office ticket workflow** | Michael decision; Codex/Claude Code implementation/review; Luz acceptance | [PR #115](https://github.com/sevenwells72/factory-ledger/pull/115), merge `d962a1f`: **Office / Oficina** → **Office action / Acción de oficina**; ticketed orders/expected deliveries, shared Shift Summary session and receipt-page void. Office controls require production session enablement. Verify deployment/flag at handoff; office has no void permission. | Closed decision; A10 stage 2 rollout is O-14. |
| O-9 | **OPEN — pilot handoff questions tracked in PR #113** | Michael decides; Arturo/Luz validate; Claude Code/Codex repair gaps | Confirm practice events/current production recording process, staging opening stock, permitted fallback coverage, staff Spanish review and substitution acceptance. Void already exists for permitted owner/floor users: receipt lookup → **Prepare void for entry #** → draft → **Record** → VD receipt; office calls Michael. No second raw-key Shift Summary flow or unsupported Assistant errors are accepted alternatives to the settled rules. | Before affected practice or GO; see [pilot plan](https://github.com/sevenwells72/factory-ledger/pull/113). |
| O-10 | **OPEN — real production dry run** | Michael provisions owner key/signs; Claude Code runs; Codex checks | Fresh real evidence, owner movement answers, no-apply command, timings and review are still required. Hosted rehearsal is not a replacement. | **Readiness week**, before GO. |
| O-11 | **OPEN — catalog/count packet acceptance** | Michael decides; Arturo counts; Luz reconciles; Claude Code/Codex preserve holds | Approve current names/labels/packet and reissues; resolve 183–186 legacy identities/units if present, unknown regular stock and customer-owned Sunshine stock. Pre-cutover extra-kosher is 283/284/285–290; 285/288 lb is settled. Recipe quantities above are settled; actual stock true-ups come from count. | Print packets, then signed reset; unresolved reset stock blocks affected scope/full release. |
| O-12 | **OPEN — evidence custody and late paper** | Michael decides; Luz administers; Claude Code verifies | Private photo/input/approval/journal storage, access, retention, accepted late-entry form and receipt linking. Executor does not upload count photos; exact signed approval is its hold-release alternative. | Before Live count. |
| O-13 | **OPEN — OpenAI readiness evidence** | Michael configures/decides; Claude Code verifies masked settings | Dedicated project/key, enforceable $20 cap, 50%/80% alert recipient, test-key revocation and production secret/flag evidence. Confirm readiness slot; planned settings are not proof. | Before GO/Assistant release. |
| O-14 | **OPEN — A10 rollout and retirement** | Claude Code/Codex prepare/review; Michael approves each action | Stage 1/2 PRs, exact legacy-route inventory/refusal evidence, archived SQL poster retirement/tier guard and later deletion. Stage 2 uses accepted O-8 office workflow plus production handoff checks. | **A10 stage 1**; stage 2 timing still to decide. |
| O-15 | **OPEN — follow-ups and business decisions** | Michael prioritizes/decides; Luz supplies evidence; Claude Code/Codex build | Weekly view, aliases/unmatched requests, exact run/receipt links, closure calendar, supplier cleanup, Sunshine custody/QBO conflicts, invoicing trigger and SO-260817-001 status. | Manual shift/receipt/exception review meanwhile; no guessed invoice or stock change. |
| O-16 | **OPEN — M0 live acceptance** | Codex prepares; Claude Code reviews; Michael owner setup/approval; Arturo/Luz test | Reconcile M0 branch, writes hidden/refused, OAuth/allowlist, hosted service, private per-user mapping, installation and all read/auth/refusal acceptance. Book slot early enough for subsequent retirement. | Before **M0 + GPT retirement deadline** and before retiring old GPTs. |
| O-17 | **OPEN — GPT retirement execution** | Michael decides; Codex/Claude Code verify | Exact sharing/retirement method, read-only-key consumer check, all-user replacement read acceptance and contingency if M0 is late. | Complete before **M0 + GPT retirement deadline**; no extension assumed, no write-capable fallback. |

### Questions for Michael

1. What part-day pilot slots/events, current production recording process, staging stock and fallback contacts should the pilot use? Confirm the linked PR #113 questions, without assuming double entry or loading a fixture.
2. Who counts and signs for each area, what are the clock times/pause order, and which dependent milestones should be rebooked if a listed fallback is selected (O-1)?
3. Where are the remaining private production session/proxy acceptance, key inventory/delivery and owner dry-run evidence (O-5/O-6/O-7/O-10)? Who supplies missing pieces before GO?
4. Which packet/catalog exceptions remain, where should private count evidence live, and which late-entry form/receipt-linking procedure is accepted (O-11/O-12)?
5. What readiness slot confirms the OpenAI project/cap/alerts/test-key removal, and what is the A10 stage 2 rollout plan (O-13/O-14)?
6. Which O-15 follow-ups take priority, including Sunshine invoicing/custody and weekly/holiday handling?
7. What M0 acceptance and GPT retirement slots, sharing method and deadline contingency should be booked (O-16/O-17)?

## Source and reconciliation notes

- [Design §10.1 and §11](../design/phase1-safe-operating-system.md): live-count option A, cutover-day key rotation, PIN-off personal sessions, $20 project policy, fallback operator, A10 stages and the read-only replacement deadline in **Key dates**. Decision 35 supersedes old gate(a)/PR-0 prerequisite; decision 40 fixes count procedure.
- [FOLLOWUPS](../../FOLLOWUPS.md): P1.16–P1.24 plus resolution-receipts coordination; P1.20 historical direct-close work (now O-4 CLOSED), P1.22 tier-safe legacy retirement, A9/weekly/office gaps. Historical “not merged” text does not override current ancestry.
- [All merged deployment notes and receipts](../deployments/): A1/A2/A3b/A5/A6/A7/A9/A12, recipe 078, F1, key sessions, resolution, count and G1 evidence. [A6 final rollout](../deployments/a6-ship-gate.md), [F1](../deployments/f1-assistant.md), [key sessions](../deployments/key-sessions.md), [key-session contract](../contracts/key-sessions.md), [D1/D2 validation](../validation/dashboard-d1-d2.md) distinguish source/test/hosted/production evidence. F1 #90, count #108, key sessions #103, G1 #111 and D1/D2 #107 were merged at the original baseline; later [#115](https://github.com/sevenwells72/factory-ledger/pull/115) supplies office/void/shared-session screens and [#121](https://github.com/sevenwells72/factory-ledger/pull/121) closes the owner close/reopen bypass; A11 #96/073 PIN remains parked.
- [G1 editor checklist](../../gpt-configs/readonly/CUTOVER_CHECKLIST.md) and its four paste files: complete 25-read/two-write-refusal smoke, dedicated read-only auth, rollback; secret-printing example replaced with masked generation here.
- [Fresh-start README](../../audits/fresh-start/README.md), [tool guide](../../audits/fresh-start/tool-guide.md), [apply guide](../../audits/fresh-start/apply-guide.md), [approval](../../audits/fresh-start/approval.md), [READINESS](../../audits/fresh-start/READINESS-2026-10-13.md), [v3 Stage 2](../../audits/fresh-start/v3/STAGE-2.md), packet/scope/ownership/recount/backlog/validation and tool implementations: current ticket reset and date/path/journal guards supersede historical direct-adjust/readiness gaps. Remaining build-backlog proposals are not claimed as live features.
- [PR #112](https://github.com/sevenwells72/factory-ledger/pull/112), head `c6feef6`: hosted count rehearsal 11 READY lots/diff 0, 42.4-second machine pause and sanitized receipt; tests use NY plant dates. Merged as `2f11015`; its earlier outside-audit dry-run paths are corrected here. [PR #113](https://github.com/sevenwells72/factory-ledger/pull/113): updated part-day pilot, bilingual guides and go/no-go evidence. Older missing-screen/API-only-approval/second-sign-in assumptions are superseded by #107/#115 and the settled decisions here; office permissions still limit Luz's fallback. Neither PR is merged by this task.
- [Current OpenAI hard-limit documentation](https://developers.openai.com/api/docs/guides/spend-limits): configure enforcement separately from alerts; account acceptance remains O-13. No OpenAI account changes were made.

**Document validation for this PR:** source/CLI/path/permission/date review and Markdown/link/coverage checks only. No application tests, DB connections, count-tool execution, staging or production operations, key generation/rotation or merge are part of authoring this runbook.
