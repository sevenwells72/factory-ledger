# Daily inventory reconciliation — what it is and how to switch it on

Scope and hand run: [`docs/verification/daily-reconciliation-scoping.md`](../verification/daily-reconciliation-scoping.md).
Code: `reconciliation.py`, `scripts/daily_reconciliation.py`, migration `094_daily_reconciliation.sql`.

**Status: built, not deployed. Migration 094 is not applied to staging or production.**
Merging this code changes nothing in production by itself: the job is not called until
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

## Rules

| Rule | Behaviour |
| --- | --- |
| Tolerance | The greater of **5 lb** or **2% of that day's movement for the lot** (not 2% of the balance). For a make ingredient: 2% of the recipe quantity. For packed-vs-made: 2% of the pounds made. |
| Lot below zero | **Always** flagged, every night, until it is corrected. |
| Adjustment with reason "unknown" (or none), or with no named person (shared key) | **Always** listed. |
| Coconut packed above made, up to 2% | Its own "Coconut yield" section. Not an alarm. Above 2% (and above 5 lb) it is an alarm like any other product. Coconut = a batch product with "Coconut" in its name. |
| Voids and late entries | Listed for information. They do not change the result. |

**Result** is `all_clear` or `attention` (anything over tolerance, any lot below zero, any adjustment to review).
**Job status** (`completed` / `failed`) is a separate field: a job that ran and found problems is `completed`.

Flag kinds: `BALANCE_DIFFERENCE`, `MAKE_OFF_RECIPE`, `MAKE_INGREDIENT_MISSING`, `MAKE_UNLISTED_INGREDIENT`,
`MAKE_NO_RECIPE`, `PACK_NO_MAKE`, `PACK_EXCEEDS_MAKE`, `NEGATIVE_LOT` (alarm); `UNEXPLAINED_ADJUSTMENT` (review);
`COCONUT_YIELD` (yield); `VOID_OF_EARLIER_ENTRY`, `VOID_SAME_DAY`, `LATE_ENTRY` (info).

## Email

Sent **every night when enabled, including "all clear"**. No email means the job did not run.
A second run for the same business day saves its results but does not send a second email.
If sending fails, the run and its flags are still saved with `email_status='failed'` and the error; running
the command again retries the send.

**Nothing saved goes unreported.** Each night starts from the last run whose findings were *delivered*
(email sent, or email switched off). A run whose email failed, a night whose job failed, or an extra run
saved without an email (a manual daytime run) is never the starting point: the next email covers that
period again and says so in a `NOTE:` line near the top, for example "the email for Mon Nov 9 was not
delivered. This email covers that day as well." Those runs are listed in `summary.undelivered` of the run
that covered them.

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

It reads only environment variables (`DATABASE_URL`, `ENVIRONMENT`, `PRODUCTION_DATABASE_HOST` — the same
guard as `scripts/exceptions_sweep.py` — plus the email variables above). `--dry-run` runs inside a
`READ ONLY` transaction (transaction-scoped; never a session setting, so it is safe on the 6543 pooler).
Exit codes: `0` done, `1` job failed, `2` job worked but the email did not send.

**Railway today:** the existing `exceptions-sweep` cron (02:30 ET, `30 7 * * *`) keeps its start command
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
