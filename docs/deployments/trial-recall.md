# Trial recall and BOL photo links — what they are and how to switch them on

Code: `google_evidence.py`, `receiving_form.py`, `trial_recall.py`, `extraction.py::def extract_bol`,
`scripts/receiving_form_sync.py`, `scripts/trial_recall_drill.py`, `scripts/google_oauth_setup.py`,
migration `097_trial_recall.sql`.

**Status: built on branch `feat/trial-recall`, NOT merged, NOT deployed, switched off.** Migration 097 was
applied to **staging** on 2026-11-18 (owner-approved) and is **not applied to production**. The Google read-only
sign-in was created on 2026-11-18 and is saved only on Michael's Mac; it is not set on any Railway service. Merged code changes nothing by itself: the two jobs are not started until
their switches are set on the cron service, a receipt shows no "BOL photo" button and the owner routes answer
503 until the migration marker exists, and no email is sent until `TRIAL_RECALL_EMAIL_ENABLED=1`.

## The two sources (independent evidence, never changed)

- Google Sheet **"Receiving# (Responses)"** (`1_zHWbCa0SHlmqVFhI07qxcD9ea35T66iHMMDJIapXJs`), tab
  **"Form Responses 1"**. Columns are found by their heading, never by position: Timestamp, Name, lot stickers,
  "What lot# did you use?" (the CNS internal lot), Truck condition, Approved Supplier?, BOL#, CARRIER,
  PICTURE OF BOL (a Drive link), and up to six product columns typed as `Name-case size-cases`
  (`Sugar 6X-50lbs-225`). If a needed heading is missing the job stops; it never guesses a column.
- The **BOL photos** in Google Drive that the sheet links to.

`google_evidence.py` can only read: every Sheets and Drive request is a GET, and a sign-in that carries any
permission beyond the two read-only ones is refused. No photo is ever copied or stored.

## Part 1 — BOL photo on the receipt

**Nightly** (`scripts/receiving_form_sync.py`, started by the `exceptions-sweep` cron when
`RECEIVING_FORM_SYNC_ENABLED=1`):

1. Read the tab. Every row dated on or after **2026-11-11** (`RECEIVING_FORM_START_DATE` overrides) is saved
   as it was seen in `receiving_form_rows`. A row that is later edited (or moves) in the sheet is saved again as
   a new version; nothing is ever updated.
2. Match each row to its FL receipt by **date + internal lot + product**:
   - **lot**: the lot typed on the Form equals the FL lot code (letters and digits only, so
     `26-11-12-STAR-001` = `261112star001`). The Form often leaves off the trailing sequence number
     (`261113DUTC` for `26-11-13-DUTC-001`): that counts as the same lot (owner, 2026-11-18) **only when exactly
     one FL receipt fits by date and product**. An exact lot code always wins; `-001` and `-002` that both fit
     are not chosen between; and a short lot together with product wording that does not confirm the product is
     not linked (two leniencies on one link). Only the sequence is forgiven, never the date or supplier part;
     the link records `lot: lot code without its sequence number`;
   - **date**: the Form date and the receipt's business date are at most **1 day** apart
     (`DATE_TOLERANCE_DAYS`; the Form is sometimes filled the evening of or the morning after);
   - **product**: matched generously (owner, 2026-11-18). A confident fit is: same words, or the shared FL
     product resolver (names and learned aliases) names that product, or the Form wording is less specific
     ("Sugar") and that product is one it could mean. When the wording is not a confident fit but lot and date
     leave exactly **one** FL receipt line and **one** Form product unaccounted for, they are still linked, and
     the link says `product: not_confirmed`. That line is UNVERIFIED wherever it is used; wording alone never
     fails anything. With two or more left over on either side nothing is guessed. The basis is saved on the link.
   - Only posted, not-voided receipts count (`ledger_current_transactions`, `effective_status='posted'`).
3. **Clear match** (exactly one FL receipt line fits a Form product) → a link record in `receipt_bol_links`
   (Form row + Drive file id + what agreed).
   A receipt shows only the links the **latest completed** nightly read still stands behind
   (`receiving_form_sync_runs.summary.link_ids`): a row edited back to an earlier wording, a row removed from
   the sheet, or a receipt voided since, drops off the receipt without anything being updated or deleted.
4. **Anything else is never guessed.** It is listed for the owner as **"Receiving form row with no matching FL
   receipt"** with the reason: no receipt uses that lot; the dates are N days apart; lot and date match but the
   product differs; more than one receipt fits; no lot or no product on the row; unreadable timestamp.

**Owner list.** `GET /verification/receiving-form/unmatched` (owner, office, master key) returns the list from
the latest nightly read (plus `product_wording_unconfirmed`: rows that were linked by lot and date although the
Form's product wording did not confirm the FL product), and says so if a later read failed. The same list is in section 4 of every trial
recall email. The dashboard has no Weekly tab yet (design §7.2 is not built); when it is built it can read this
route. See "Open questions".

**Receipt page.** For a receive receipt with a link, the page shows **"BOL photo"** under "From the receiving
form". Pressing it streams the photo from Drive through the API:

- `GET /receipts/{receipt_number}/bol-photos` — which photos are linked (never the Drive file id).
- `GET /receipts/{receipt_number}/bol-photos/{photo_id}/file` — the photo, streamed. `Cache-Control: private,
  no-store`; nothing is written to disk, Storage or the database.

Both need a **signed-in named person** (a personal key, a key/PIN session, or a name tapped on an approved
browser). The shared dashboard key, the read-only GPT key and the master key are refused. Only a file that is
linked to that receipt is ever served, only if Drive says it is an image or a PDF, not in the trash, and at most
15 MB. Optional extra lock: set `RECEIVING_FORM_PHOTO_FOLDER_ID` to the Form's upload folder id and files
outside that folder are refused.

## Part 2 — Trial recall

`scripts/trial_recall_drill.py --scheduled`, started by the `exceptions-sweep` cron when
`TRIAL_RECALL_ENABLED=1`. The cron fires every night at 07:30 UTC (**02:30 Eastern in winter, 03:30 in
summer**); the command itself runs only on **Monday, Wednesday and Friday** (plant time) and only once a day.

**Strictly read-only on the ledger.** Everything read from Factory Ledger is read inside `READ ONLY`
transactions (transaction-scoped, never a session setting, so it is safe on the 6543 pooler). The one write is
a single INSERT into `trial_recall_runs`, on a separate connection, after the drill has finished.

**Cutoff 2026-11-11.** Only Form rows and FL receipts dated on or after it are tested. The backward test only
picks finished lots first made on or after it; an older ingredient in the chain is shown **"before cutoff - not
tested"** and is left out of the flags. If nothing is eligible the test says **"No eligible lot"**; that is not a
failure.

**Test 1, forward.** A random Form row (on/after the cutoff) that has a BOL photo.

1. The photo is read by the existing extraction model (`extraction.extract_bol`, same model and key as document
   intake): product, supplier, manufacturer, carrier, quantity, lot numbers, BOL and reference numbers. Only
   what is printed; nothing inferred.
2. **Unreadable or not openable → UNVERIFIED, never PASS** (photo gone or in the trash, a format the model
   cannot read, the model says it cannot read the lines, the model call fails). **HEIC / HEIF phone photos are
   converted to JPEG in memory and then read** (`extraction.heic_to_jpeg`, library `pillow-heif`); if the
   conversion fails the photo is UNVERIFIED. A readable BOL that
   cannot confirm the product or the quantity is UNVERIFIED too. **Every** photo on the Form row has to be
   readable: one good page never covers a missing or unreadable one. Up to 6 photos are read; a row with more is
   UNVERIFIED. A page photographed twice (the whole reading identical) is counted once; one line repeated on two
   different pages is kept and shown, and the row cannot pass on it.
3. **Three-way comparison, Sheet vs BOL vs FL**: product, quantity, internal lot, supplier lot, BOL/reference
   number (whole numbers only: `4412` is found in `PO 4412`, never inside `944120`), supplier (checked on every
   matched receipt), other products on the BOL (a product the BOL shows that FL has no receipt for within a day, or a
   line FL's product list does not recognise, is UNVERIFIED), manufacturer, carrier. Every value from every source is shown for every field. A
   difference in quantity, lot, reference number or supplier is a FAIL of the receiving check and the email shows
   all three values; **FL is never assumed to be the right one**. **Product wording never fails anything**: the
   product either agrees with confidence on all three records or the line is UNVERIFIED.
   Quantities agree within 0.5 lb, or case counts exactly. A case size written with no unit ("25") is never
   assumed to be pounds. The BOL's quantity for a product is used only when every one of its lines can be
   added up in one unit (a line in pallets, pieces or with no quantity, or bags on one line and drums on
   another, leaves it unconfirmed). The Form's "number of cases" is a count of whatever package the product
   came in, so one kind of package on the BOL is compared with it. A quantity counts as
   confirmed only when **FL's own quantity was compared** as well as
   the BOL's; the Form and the BOL agreeing with each other proves nothing about FL (UNVERIFIED). Supplier lots
   on the product's own BOL line must be the **same list** as FL's: a lot printed on the BOL and missing from FL
   is a disagreement.
4. **Trace forward** from the received lot: consumption → batches → finished lots → packing → shipments →
   customers (`lot_trace.forward`, the existing complete forward trace).

**Test 2, backward.** A random shipped finished lot (made on/after the cutoff) → every ingredient lot
(`lot_trace_backward.BackwardTrace`) → each ingredient lot's Form row and BOL photo (the file is actually
opened from Drive to check it is on file; **not read by the model** in this test) → forward again from those
ingredient lots to **every other customer** they reached. A batch inside the finished lot that was not only
made (stock also received or found straight into it) is flagged at every level, not just on the finished lot;
a batch made only before the cutoff is shown as not tested. **Each receipt into an ingredient lot is checked on
its own** (its own Form row, its own photo): a lot topped up by a second truck, or by found stock, is not covered
by the first truck's paperwork. "Before cutoff" needs a known date: a lot with no dated entry at all is tested
and flagged as unresolved, never skipped as old. An older receipt into a lot that also took stock after the
cutoff is shown "before cutoff - not tested" beside the newer one.

**Data rules.** Posted current-ledger data only (`ledger_current_transactions` +
`ledger_current_transaction_lines`, `effective_status='posted'`); the existing trace code is reused, not
copied. Nothing is assumed about which stock was used first (no FIFO): the traces follow recorded lot lines.

**Flags** (on the lots of the two chains; in the backward test that includes every lot the tested ingredient
lots went into, not only the picked finished lot; lists over 40 are cut in the email and saved whole with the run):

| Flag | Raised when | Level |
| --- | --- | --- |
| `LOT_ATTRIBUTION_UNRESOLVED` | A lot is unidentified or pooled; has an open shortage / unidentified-lot / negative-balance exception; an ingredient lot (or part of one) did not come in on a supplier receipt; a made lot was not only made (stock also received or found straight into it). | **Makes it NO** |
| `MADE_REMOVED_BY_ADJUSTMENT` | A made lot was taken out by adjustment and never packed, used or shipped. | **Makes it NO** |
| `ORDER_SHIPPED_NOT_RECORDED` | An order for a finished product in the chain was closed "shipped, not recorded" on/after the cutoff. | **Makes it NO** |
| `UNKNOWN_ADJUSTMENT_INTO_PRODUCT` | Stock was added with reason "unknown" (or none) to a lot that then went into packed or shipped product. | **Makes it NO** |
| `NEGATIVE_BALANCE` | A lot in the chain is below zero (more was used than the books ever held). | **Makes it NO** |
| `TRACE_GAP` | A consumed lot has no identified output lot. | **Makes it NO** |
| `PRODUCTION_INPUTS_MISSING` | Backward test: a make or pack entry in the chain records no ingredient lots (or leads back to itself). | **Makes it NO** |
| `SHIPMENT_CUSTOMER_UNKNOWN` | A shipment in the chain names no customer and has no order that names one. | **Makes it NO** |
| `MISSING_SUPPLIER_LOT` | A received ingredient lot has no supplier lot (also when staff chose "No lot on label"). | Receiving FAIL |
| `RECEIVING_EVIDENCE_MISSING` | Backward test: an ingredient lot's receipt has no Form row. | Receiving FAIL |
| `SHIPMENT_NO_BOL` | A shipment in the chain has no BOL number (the line also says if there is no photo). | Warning |
| `SHARED_KEY_POSTING` | An entry in the chain (on/after the cutoff) was made with the shared login. | Warning |

**Results.** Each test is `PASS`, `FAIL`, `UNVERIFIED` or `No eligible lot`.

**Full recall possible** (owner, 2026-11-18):

- **NO** when any affected customer or shipment cannot be identified: unresolved lot attribution, made product
  removed with no destination, an order closed "shipped, not recorded", a consumed ingredient lot with no
  receipt, or a broken lot link (the "Makes it NO" flags above). The reasons are listed.
- **Receiving mismatches** (quantity, lot, reference, supplier; a missing supplier lot; a missing Form row) show
  as **FAIL in the receiving section** and do not by themselves make it NO. Unreadable evidence is UNVERIFIED
  and does not either. The email says which receiving checks did not pass, next to the answer.
- **A missing BOL photo or number, and entries under the shared login, are warnings only.**
- **YES** otherwise. **NOT TESTED** when nothing could be traced (no eligible lot, or a Form row with no FL
  receipt to trace from and no eligible finished lot).

Two readings of mine inside your rule, say if you want them moved: stock added with reason "unknown" that went
into product, and a lot below zero, both count as unresolved lot attribution (NO).

**Email** (the daily reconciliation's Resend setup: `ALERT_EMAIL_TO`, `RESEND_API_KEY`,
`RECONCILIATION_EMAIL_FROM`), sections in order:

1. Receiving verification (Sheet / BOL vs FL)
2. Inventory and consumption reconciliation (every lot in the chain: received, made, used, shipped, adjusted, on hand)
3. Forward traceability
4. Missing or conflicting records (the flags, and "Receiving form row with no matching FL receipt")
5. Full recall possible: YES / NO

with links to the sheet row and the BOL photo. **Every run is saved** in `trial_recall_runs` (results, the whole
comparison and the email as sent). A drill that could not run is saved as `failed`, so a missing email is never
mistaken for a pass.

- `GET /verification/trial-recall/runs?limit=30` and `/runs/{id}` — owner, office, master key.

## Variables

| Variable | Service | Meaning |
| --- | --- | --- |
| `GOOGLE_OAUTH_CLIENT_ID`, `GOOGLE_OAUTH_CLIENT_SECRET`, `GOOGLE_OAUTH_REFRESH_TOKEN` | `FastAPI` and `exceptions-sweep` | The read-only Google sign-in (steps below). Never logged. |
| `RECEIVING_FORM_SYNC_ENABLED` | `exceptions-sweep` | `1` starts the nightly Form read. **Default off.** |
| `TRIAL_RECALL_ENABLED` | `exceptions-sweep` | `1` starts the Mon/Wed/Fri drill. **Default off.** |
| `TRIAL_RECALL_EMAIL_ENABLED` | `exceptions-sweep` | `1` sends the email. **Default off** (the run is still saved). |
| `ANTHROPIC_API_KEY` (optional `EXTRACTION_MODEL`) | `exceptions-sweep` | The extraction model that reads the BOL photo. Already on `FastAPI`; the cron service needs it too. |
| `ALERT_EMAIL_TO`, `RESEND_API_KEY`, `RECONCILIATION_EMAIL_FROM` | `exceptions-sweep` | Already set for the daily reconciliation. |
| `RECEIVING_FORM_START_DATE` | `exceptions-sweep` | Optional; default `2026-11-11`. |
| `RECEIVING_FORM_PHOTO_FOLDER_ID` | `FastAPI` | Optional; only photos in this Drive folder are served. |

## Google access — what only Michael can do

Sign in as **michael@sevenwellsgranola.com** for every step. About ten minutes, once.

**A. Create the project**

1. Open <https://console.cloud.google.com/>.
2. Click the project name at the top left (next to "Google Cloud") → **New project**.
3. Project name: `Factory Ledger Evidence`. Organization: **sevenwellsgranola.com** (it must not say
   "No organization"). Click **Create**, wait for the notification, then click **Select project**.

**B. Switch on the two read APIs**

4. Menu (☰) → **APIs & Services** → **Library**.
5. Search `Google Drive API` → click it → **Enable**.
6. Back to the Library, search `Google Sheets API` → click it → **Enable**.

**C. Consent screen, INTERNAL**

7. Menu (☰) → **APIs & Services** → **OAuth consent screen** (the page is titled "Google Auth Platform").
   Click **Get started**.
8. App name: `Factory Ledger read-only evidence`. User support email: your address. **Next**.
9. Audience: choose **Internal**. **Next**. (If "Internal" cannot be chosen, the project is not inside the
   sevenwellsgranola.com organization: stop here and tell Claude.)
10. Contact email: your address. **Next** → tick the policy box → **Continue** → **Create**.
11. Left menu → **Data Access** → **Add or remove scopes**. In the filter box type `drive.readonly` and tick
    `.../auth/drive.readonly`; then type `spreadsheets.readonly` and tick `.../auth/spreadsheets.readonly`.
    Tick nothing else. **Update** → **Save**.

**D. The client**

12. Left menu → **Clients** → **Create client**.
13. Application type: **Desktop app**. Name: `FL setup`. **Create**.
14. In the box that appears click **Download JSON** (it saves `client_secret_….json` to Downloads). **OK**.

**E. The one-time "Allow"**

15. Tell Claude the file is downloaded. Claude runs, in a Factory Ledger checkout on your Mac:

    ```bash
    .venv-test/bin/python scripts/google_oauth_setup.py ~/Downloads/client_secret_*.json --railway
    ```

16. Your browser opens a Google page. Choose **michael@sevenwellsgranola.com**. The page must list exactly two
    permissions, both starting with "See": *See and download all your Google Drive files* and *See all your
    Google Sheets spreadsheets*. Click **Allow**. The tab then says "Done".

The script checks the sign-in is limited to those two read-only permissions, reads the Form tab once to prove
it works, and saves the three values in `~/.config/factory-ledger/google_oauth.env` (readable only by you). It
never prints the token and never contacts Railway.

**F. Railway variables** — a production change: needs your approval for that step. The script prints the two
exact commands (one per service); Claude runs them only after you say yes.

If Google ever withdraws the sign-in (you remove the app's access, or a Workspace policy expires it), the jobs
fail with `AUTH_FAILED`, the failure is saved, and step 15–16 are repeated. To withdraw access yourself:
<https://myaccount.google.com/permissions> → "Factory Ledger read-only evidence" → Remove access.

## Rollout — each step needs Michael's approval

1. **Google** steps A–E above.
2. **Staging:** apply `migrations/097_trial_recall.sql` (port 5432, one transaction); set the Google variables on
   the staging services; `python scripts/receiving_form_sync.py --dry-run`, then a real run; then
   `python scripts/trial_recall_drill.py --dry-run`, then a real run with `TRIAL_RECALL_EMAIL_ENABLED=1` and a
   test address. (Staging has no real receipts for these lots, so expect "no matching FL receipt" and
   "No eligible lot": that proves the plumbing, not the matching.)
3. **Production:** verified backup (`~/Backups/fl/fl-backup-and-verify.sh`, must end `BACKUP OK`) → merge →
   apply migration 097 → set the Google variables on `FastAPI` and `exceptions-sweep`, `ANTHROPIC_API_KEY` on
   `exceptions-sweep` → a production `--dry-run` of each command (reads only) → `RECEIVING_FORM_SYNC_ENABLED=1`
   → after the first night's list looks right, `TRIAL_RECALL_ENABLED=1` and `TRIAL_RECALL_EMAIL_ENABLED=1`.
4. Record the deploy in `FACTORY_LEDGER_CHANGELOG.md`.

Rollback: unset the two `*_ENABLED` switches (the button disappears when no link is found; to hide it at once,
unset `GOOGLE_OAUTH_REFRESH_TOKEN` on `FastAPI`, which makes the photo route answer 503). The tables can stay;
`migrations/down/097_trial_recall_down.sql` refuses to drop them while they hold rows.

## The commands

```
python scripts/receiving_form_sync.py               # the nightly Form read
python scripts/receiving_form_sync.py --dry-run     # same read and matching; prints the result; writes nothing
python scripts/trial_recall_drill.py                # run the drill now and save it
python scripts/trial_recall_drill.py --scheduled    # the cron form: Mon/Wed/Fri plant time, once a day
python scripts/trial_recall_drill.py --dry-run      # same drill; prints the email; saves nothing, sends nothing
```

Both read only environment variables and use the same database guard as `scripts/exceptions_sweep.py`
(`DATABASE_URL`, `ENVIRONMENT`, `PRODUCTION_DATABASE_HOST`). Exit codes: `0` done (or not a recall day, already
run today, or another run in progress), `1` the job failed, `2` (drill only) the job worked but the email did
not send. Either one failing fails that night's cron run, like the reconciliation.

## What this check cannot do

It tests the records, not the shelves. It reads one Form row and one finished lot per run, chosen at random, so
a clean result is evidence about those two chains, not the whole plant. It cannot see product that left with no
entry at all, and it cannot read a BOL better than the photo allows.

## Decisions (Michael, 2026-11-18)

1. **Product wording:** match generously; a match that is not confident is UNVERIFIED for that line; never FAIL
   on wording alone.
2. **Date tolerance:** one day either side (the Form is sometimes submitted the next day).
3. **Start date:** 2026-11-11, for the nightly matching and the recall cutoff.
4. **HEIC photos:** converted and read; if conversion fails, UNVERIFIED.
5. **Full recall possible:** NO only when an affected customer or shipment cannot be identified; receiving
   mismatches FAIL their own section; a missing BOL photo and shared-login entries are warnings (rules above).
6. **Anthropic key on the cron service:** reuse the existing key.
7. **Numbering:** changelog row 306, migration 097.
8. **Short lot numbers on the Form:** a Form lot that is the FL lot code without its trailing sequence number
   is the same lot, if only one FL receipt fits by date and product.

## Still open

1. **Owner weekly view.** It is designed (§7.2) but not built. The lists are served by a route and printed in
   every recall email; a Weekly tab is a separate change.
2. **Backward test does not read every BOL.** It opens each ingredient lot's photo to check it is on file.
   Reading all of them would cost one model call per ingredient lot per run.
3. **HEIC on the receipt page.** The model reads a converted copy; the "BOL photo" button still streams the
   original file, which Safari shows and other browsers open as a download.
4. **New library.** `pillow-heif==1.8.1` is added to `requirements.txt` (needs pillow 11.1 or newer, which
   reportlab already brings in). The Railway build with it has not been run yet; prove it on staging first.
5. **Optional Drive folder lock.** `RECEIVING_FORM_PHOTO_FOLDER_ID` limits served photos to the Form's upload folder.
