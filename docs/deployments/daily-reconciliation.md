# Daily inventory reconciliation — what it is and how to switch it on

Scope and hand run: [`docs/verification/daily-reconciliation-scoping.md`](../verification/daily-reconciliation-scoping.md).
Code: `reconciliation.py`, `scripts/daily_reconciliation.py`, migration `094_daily_reconciliation.sql`.

**Status: merged to main 2026-11-10 (PR #198), switched off. Migration 094 is not applied to staging or
production.** The merged code changes nothing in production by itself: the job is not called until
`RECONCILIATION_ENABLED=1` is set on the cron service, the endpoints answer 503 until the
migration marker exists, and email is off until `RECONCILIATION_EMAIL_ENABLED=1`.

## What runs each night

1. **Snapshot** every lot's posted, void-aware balance (`v_lot_quantities`) into `inventory_snapshots`.
2. **Check the business day** (America/New_York). The window is every entry and every void made since the
   previous snapshot, so a missed night or a weekend is simply a longer window.
   - **Make vs recipe.** Batches = output ÷ (batch size × yield). Expected pounds = recipe × batches. Water
     (`exclude_from_inventory`) is skipped. A batch used as an ingredient (Batch Classic #9 inside Fruit Nut)
     is checked like any other ingredient. A recorded substitution counts as correct.
     The expected quantities are saved with the run (`reconciliation_runs.summary.makes`), because
     `batch_formulas` keeps no history.
   - **Balance roll-forward per lot.** Previous snapshot + what was entered (makes at recipe quantities) must
     equal tonight's balance.
   - **Packed more than made**, per batch lot, and **pack with no make behind it**.
3. **Save flags** to `reconciliation_flags`, **send the email**, and write one row to `reconciliation_runs`
   with the delivery result. All of it is one transaction.

It never posts, voids, adjusts or corrects anything, and it never writes to an existing table.
(Applying migration 094 adds one row to `migration_markers`, like every migration; the job itself does not.)

## Rules

| Rule | Behaviour |
| --- | --- |
| Tolerance | The greater of **5 lb** or **2% of that day's movement for the lot** (not 2% of the balance). For a make ingredient: 2% of the recipe quantity. |
| Tolerance for packed-vs-made | **Different on purpose:** the greater of **5 lb** or **2% of all the pounds ever made into that batch lot**, compared with all the pounds ever packed out of it. It is measured on the lot's whole life, not on one day's movement, because a lot is often made one day and packed over several. Example: 350 lb made allows 7 lb; 375 lb packed is 25 lb over and is flagged. |
| Lot below zero | **Always** listed, every night, until it is corrected: under "New today" the night it goes below zero or gets worse, then under "Still not fixed" with the number of days it has been open. The one exception is the coconut rule below. |
| Adjustment with reason "unknown" (or none), or with no named person (shared key) | **Always** listed. |
| Coconut packed above made, up to 2% | Its own "Normal coconut difference" section. Not an alarm. A coconut batch lot that is below zero **only** because of such an over-pack (its minus balance is no bigger than the over-pack, and the over-pack is within 2% of what was made) is listed there too (`COCONUT_BELOW_ZERO`, yield), never as a below-zero alarm. Any other lot below zero is always listed. Above 2% (and above 5 lb) it is an alarm like any other product. Coconut = the four coconut bulk batch products, by product ID (125 Fancy, 126 Flake, 127 Medium, 128 Toasted Flake; `COCONUT_YIELD_PRODUCT_IDS` in `reconciliation.py`). It is a fixed list, not a name match, because `products.product_category` is empty for every product. **A new coconut bulk batch product must be added to that list**; until then a small over-pack on it is simply inside tolerance and gets no line. |
| Voids and late entries | Listed for information. They do not change the result. |

**New or still open.** Every alarm and review finding carries `new` (also `detail.new_today`). Only a lot below
zero can be still open: it is new when it was not below zero at the start of the window (the last delivered
snapshot at night, the start of the day in a replay) or is lower now; otherwise it is still open.
`detail.below_zero_since` is the entry time after which the lot's current posted lines first stay below zero,
and `detail.days_open` counts plant days from then. Every other finding comes from the window's own entries, so
it is always new.

**Result** is `attention` when there is anything **new** (over tolerance, a lot newly below zero or worse, an
adjustment to review, or an earlier finding never emailed), otherwise `all_clear`, even when older lots are still
below zero. `attention_count` (= `summary.new_count`) counts new items; `summary.still_open_count` the rest.
**Job status** (`completed` / `failed`) is a separate field: a job that ran and found problems is `completed`.

Flag kinds: `BALANCE_DIFFERENCE`, `MAKE_OFF_RECIPE`, `MAKE_INGREDIENT_MISSING`, `MAKE_UNLISTED_INGREDIENT`,
`MAKE_NO_RECIPE`, `PACK_NO_MAKE`, `PACK_EXCEEDS_MAKE`, `NEGATIVE_LOT` (alarm); `UNEXPLAINED_ADJUSTMENT` (review);
`COCONUT_YIELD`, `COCONUT_BELOW_ZERO` (yield); `VOID_OF_EARLIER_ENTRY`, `VOID_SAME_DAY`, `LATE_ENTRY` (info).

## Email

Sent **every night when enabled, including "all clear"**. No email means the job did not run.

Written for a reader with no technical or accounting background, reading quickly on a phone. The first line is
one sentence a person could act on ("Nothing new today; 2 older items still need a stock count (listed below)."
or "1 new thing to check today: find out where the extra 25 lb of … came from."). The subject counts only new
items, e.g. `Factory Ledger daily check, Fri Nov 6: Nothing new today. 2 still open.` A day with nothing new
never says it needs attention. Sections, in order (empty ones left out, except "New today"):

1. **Found earlier, not emailed until now** — saved findings of undelivered runs (below).
2. **New today** — each item leads with what happened and what to do, then the numbers; transaction numbers at
   the end of the line. A pack above make shows Lot / Made / Packed / Difference / Covered by (the hand stock
   change that covered it) and an "Ask …" line naming who packed it and who added the stock.
3. **Still not fixed** — one line each, with days open.
4. **Normal coconut difference** — today's coconut over-packs and older coconut lots still below zero from one.
5. **Cancelled or late entries** — information only.
6. **What we checked** — every make with its batches and "all N ingredients match the recipe" (small
   differences named), the exact differences, or why it could not be checked; batch lots packed from; the
   stock add-up.
7. **How to read this** — the plain notes ("No name on it" = made with the shared login, the allowed small
   difference, coconut, the check cannot see the shelves, no email means no run).

Each flag stores its plain parts in `detail` (`headline`, `facts`, `ask`, `refs`, `summary`) and the joined
line in `detail.message`.
A second run for the same business day saves its results but does not send a second email.
If sending fails, the run and its flags are still saved with `email_status='failed'` and the error; running
the command again retries the send.

**Nothing saved goes unreported.** Each night starts from the last run whose findings were *delivered*
(email sent, or email switched off). A run whose email failed, a night whose job failed, or an extra run
saved without an email (a manual daytime run) is never the starting point: the next email covers that
period again and says so in a `Note:` line near the top, for example "Note: the email for Mon Nov 9 did not
go out. This email covers that day too." Those runs are listed in `summary.undelivered` of the run
that covered them. Whether a run has been delivered is never worked out from run numbers or times: a
delivered run lists, in `summary.acknowledged`, every earlier run it reported, and any saved run not on
such a list (and not itself emailed) is picked up by the next email, whatever order the runs were saved in.
One exception keeps the email quiet: a repeat run for a day that was already emailed
(for example the command run twice in one night) gets no `Note:` line unless it found something the sent
email did not include.

**Saved findings are shown as saved.** Covering the period again is not enough by itself: a lot that was
at −1 lb when the email failed and was corrected the next morning would no longer show. So the next email
also lists the flags those undelivered runs *saved*, unchanged, in their own section at the top
("Found earlier, not emailed until now"), each with a line on what has happened since: still the case,
numbers changed, entry cancelled, lot corrected, recipe changed, or no longer found. Tonight's own findings
follow. A saved alarm or adjustment counts as new (the owner has not seen it) unless tonight's own new findings
already count it, so it makes the result `attention`, never "all clear". A run's flags count as
delivered only after a send that included them succeeds; until then every later email repeats them (each
once). They are stored in `summary.carried` of the run that showed them, not saved again as flags, so
`flag_count` is tonight's findings only and `attention_count` also counts carried ones.

**A failed job keeps its place.** When the job itself fails, the saved `failed` row records the window it
was attempting. If nothing was ever delivered before (a failure on the very first night), the next run
starts from the earliest undelivered day: a failed check for Nov 9 followed by the Nov 11 run covers
Nov 9 and Nov 10.

| Variable | Meaning |
| --- | --- |
| `RECONCILIATION_EMAIL_ENABLED` | `1` to send. **Default off** (`email_status='disabled'`). |
| `ALERT_EMAIL_TO` | Recipient. Several addresses may be separated by commas. |
| `RESEND_API_KEY` | Resend API key. Never logged or stored. |
| `RECONCILIATION_EMAIL_FROM` | Optional sender, e.g. `Factory Ledger <checks@sevenwellsgranola.com>`. Needs the domain verified in Resend (three DNS records). Without it the job uses Resend's test sender, which only delivers to the Resend account's own address. |

## The command

```
python scripts/daily_reconciliation.py                               # the nightly run
python scripts/daily_reconciliation.py --dry-run                     # same check; prints the email; writes nothing
python scripts/daily_reconciliation.py --dry-run --date 2026-11-09   # replay one past day; needs no migration
```

The business day checked is always "yesterday" in Eastern time at the moment the command starts, so it
must run after midnight Eastern (both 02:30 and 03:30 do). A manual run started before midnight would label
the day before.

`--date` replays one calendar day as **midnight up to, but not including, the next midnight** (Eastern;
23 or 25 hours on the two clock-change days). An entry made at exactly midnight belongs to the day that
is starting, and the opening and ending balances are those just before each midnight. The nightly run is
different on purpose: it covers everything after the previous snapshot up to and including the moment of
tonight's, so consecutive nights meet with no gap and no overlap.

**One run at a time.** A real run takes a lock before anything else and keeps it until its results
are committed. A second run started while the first is still going stops at once: it prints "Another
reconciliation run is in progress. This run stopped: nothing saved, nothing sent." and exits `0`. Start
it again once the first has finished; it will then see that tonight's email already went out and will
not send a second one.

How the lock works:

- It is a Postgres **transaction-level advisory lock** (`pg_try_advisory_xact_lock`), released
  automatically at commit or rollback, even if the process dies. That kind works through the Supabase
  transaction pooler on port 6543, where a session-level advisory lock would not hold. It locks no table
  and blocks no ledger entry.
- The command takes it on a small separate **guard connection** first, and only then opens the
  connection the job runs on. The job reads the books as of one fixed moment (its first statement), so
  it must not read anything before the lock is held: otherwise it could miss a run that finished a
  moment earlier and email the same day twice. The guard is released only after the job has committed.
- So a real run uses two database connections for a few seconds. The job transaction also takes a
  second lock of the same kind as a backstop.
- `--dry-run` and the preview route read only and never take a lock.

It reads only environment variables (`DATABASE_URL`, `ENVIRONMENT`, `PRODUCTION_DATABASE_HOST` — the same
guard as `scripts/exceptions_sweep.py` — plus the email variables above). `--dry-run` runs inside a
`READ ONLY` transaction (transaction-scoped; never a session setting, so it is safe on the 6543 pooler).
Exit codes: `0` done (or stopped because another run was in progress), `1` job failed, `2` job worked but the email did not send.

**Railway today:** the existing `exceptions-sweep` cron (`30 7 * * *`, fixed in UTC: **02:30 Eastern in
winter, 03:30 Eastern in summer**) keeps its start command
`python scripts/exceptions_sweep.py`. When `RECONCILIATION_ENABLED=1` is set on that service, the sweep
starts `python scripts/daily_reconciliation.py` as a separate process after it finishes. The sweep's own
transaction is already committed, so the reconciliation cannot undo or block it. Either one failing fails
the cron run.

**Render later:** create a Cron Job with the command `python scripts/daily_reconciliation.py` and the same
variables. Nothing in the script changes. Turn `RECONCILIATION_ENABLED` off on Railway at the same time
so it does not run twice (a second run would not email again, but it is tidier).

## Read-only endpoints

Owner and office (named keys) and the master key. Not in either GPT schema (`include_in_schema=False`;
the office schema stays at 30 of 30). The dashboard key and the read-only GPT key are refused.

- `GET /verification/reconciliation/runs?limit=30` — recent runs with counts and email status.
- `GET /verification/reconciliation/runs/{id}` — one run with its flags and stored summary.
- `GET /verification/reconciliation/preview` — owner only. Tonight's check as of now; saves nothing, sends nothing.

**What "read-only" means here.** The three handlers only read. Signing in with a named key still does its
normal bookkeeping, exactly as on every other route: it updates `actors.last_used_at`, and a browser
session may update `key_actor_sessions`. The owner accepts those two writes (decision 2026-11-10). Nothing
else in any existing table changes; a test fingerprints every existing table around the three requests
and allows only those two.

## Rollout — each step needs Michael's approval

1. **Staging:** apply `migrations/094_daily_reconciliation.sql` (port 5432, one transaction), run
   `--dry-run`, then a real run with `RECONCILIATION_EMAIL_ENABLED=1` and a test address.
2. **Resend:** create the account/key; verify sevenwellsgranola.com if a branded sender is wanted.
3. **Production:** verified backup (`~/Backups/fl/fl-backup-and-verify.sh`) → merge → apply migration 094 →
   set `ALERT_EMAIL_TO`, `RESEND_API_KEY`, `RECONCILIATION_EMAIL_ENABLED=1` and `RECONCILIATION_ENABLED=1`
   on the `exceptions-sweep` service. The first night saves the starting snapshot and emails the day's
   checks; the balance roll-forward begins the second night.
4. Record the deploy in `FACTORY_LEDGER_CHANGELOG.md` (row 273).

Rollback: unset `RECONCILIATION_ENABLED`. The tables can stay; `migrations/down/094_daily_reconciliation_down.sql`
refuses to drop them while they hold rows.

## What this check cannot do

It compares the books with themselves. It will fire on edits to earlier days, voids, late entries,
off-recipe makes and packs that exceed what was made. It will not see physical loss, a miscount or
stock used without an entry. Only a count can. Makes are currently posted from the recipe, so the
make-vs-recipe check stays quiet until real weights are entered.

Out of scope for this PR: dashboard page, SMS, lot-attribution checks, any inventory correction.
