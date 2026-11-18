# Adjustment rules: destination on removals, no hold over 500 lb, owner email

Owner decision 2026-11-18. Code: `exceptions_enforcement.py`, `adjustment_alerts.py`, `ticket_actions.py`,
`write_tickets.py`, `main.py` (two posting cores), `scripts/exceptions_sweep.py`, `dashboard/entry.js`,
`dashboard/ledger-ui.js`. Migration: `migrations/096_adjust_destinations.sql`.

**Status: branch `feat/adjust-destination`, not merged, not deployed. Migration 096 is not applied to staging or
production.** It has only been applied to throwaway local test databases.

## What changes

| Rule | Before | After migration 096 |
| --- | --- | --- |
| Adjustment that removes stock | Any of the eight reasons | Must pick where it went, one pick: Shipped but not recorded (also the sales order), Sent to Sunshine as bulk, Used in production but not recorded, Damaged or thrown out, Count correction — cause unknown (note required). Also accepted, each with a required note: Wrong lot used, Data-entry error (the stock was never on this lot) |
| Over 500 lb (adjust, Found, counted shortage close) | Photo, or held for the owner | Recorded at once. Photo optional. The owner gets an email |
| Adding stock, Found | Unchanged | Unchanged (the two retired reasons were removal-only) |

Unchanged: who may adjust (keys and roles), the name on every entry, the 10% and "unknown" highlighting,
"never add stock to cover a shortage", and what closes a shortage. The owner's count ticket that closes a lot's
shortages keeps the reason *physical count*.

Retired for new entries only: `damage_disposal` and `unrecorded_usage`. Their rows, labels and every old
transaction that carries them are untouched.

Wrong lot used and Data-entry error keep their rows unchanged apart from the new `removal_allowed` flag. The
required note applies only when they remove stock; on an addition, a void or a lot fix the note stays optional.
Physical count, Missing production and Unknown are not accepted on a removal.

"Shipped but not recorded" names the order as the explanation. It does not ship, allocate or close the order;
closing the order stays the office's step.

## The switch is the migration marker

Everything is gated on the marker `096_adjust_destinations`:

- code deployed, migration not applied: the old rules stay in force, whole (old reasons, photo-or-hold);
- migration applied, old code: the old code ignores the new columns and table.

So either order is safe. The dashboard form does assume the new rules; if it meets an API still on the old rules
the API refuses the new reason, sends its own list, and the form switches to that list. To avoid that window,
apply the migration first, then merge (Railway and Netlify deploy from the merge).

## The owner email

- Written in the posting transaction: one `adjustment_alerts` row, inside a savepoint. If that fails the
  correction still posts and the failure is logged (`ADJUSTMENT_ALERT_QUEUE_FAILED`).
- Sent afterwards, on separate connections, through the Resend sender in `reconciliation.py`
  (`RESEND_API_KEY`, `ALERT_EMAIL_TO`, optional `RECONCILIATION_EMAIL_FROM`). One alert is one email.
  A failed send leaves the row `pending` with `last_error` and `attempts`.
- Retry: the web process tries again after 5 s, 1 min, 5 min and 15 min; the nightly cron
  (`scripts/exceptions_sweep.py`, `send_adjustment_alerts`) sends whatever is still pending, and repeats an
  attempt that got no answer up to twice more in the same run (after 45 s and 90 s).
- **A retry never sends a second copy.**
  - Each row has a fixed `idempotency_key` (`fl-adjustment-alert/<uuid>`), sent on every attempt as Resend's
    `Idempotency-Key` header. For 24 hours Resend answers a repeat of the same key and body with its first
    answer and sends nothing again.
  - The email is built from the ledger at the first attempt and stored on the row (`message`). Every retry
    sends that stored text, so the body never differs. A photo or a void that arrives after the first attempt
    shows on the receipt, not in the email.
  - The row becomes `sent` only when Resend confirms with a 2xx answer. A send that was interrupted (no
    answer, a crash, or the status write failing) stays `pending`; the retry repeats the same key, Resend
    returns the original answer, and that is recorded.
  - `unconfirmed_since` marks the earliest attempt whose answer never arrived. If nothing has confirmed the
    email 23 hours later, the row becomes `unconfirmed` and is **not sent again**, because Resend may have
    forgotten the key and a repeat could be a second copy. The log line is `ADJUSTMENT_ALERT_EMAIL
    unconfirmed`, and the nightly job prints the count. A person checks whether the email arrived:
    `SELECT id, transaction_id, queued_at, last_error FROM adjustment_alerts WHERE status = 'unconfirmed';`
    To send it again on purpose, accepting a possible repeat:
    `UPDATE adjustment_alerts SET status = 'pending', unconfirmed_since = NULL, last_attempt_at = NULL WHERE id = <id>;`
  - A refusal from Resend (4xx: wrong key, unverified domain, rate limit) with nothing in doubt is retried
    without any time limit.
- **The web service has no Resend settings today** (they are on the `exceptions-sweep` cron service). Until
  `RESEND_API_KEY` and `ALERT_EMAIL_TO` are also set on the FastAPI service, the email goes out with the
  nightly run, not within seconds. Setting them is a credential change and needs Michael's approval.
- The email holds: product, lot, pounds, person, reason, destination (and the sales order), the note, the
  receipt number, and whether a photo is attached with a link to the receipt page (`DASHBOARD_URL`, default the
  Netlify dashboard). The photo itself needs a dashboard sign-in, so the link opens the receipt, not the file.
- Entries through the older direct routes (`POST /adjust`, `/inventory/found`) are emailed too.

## Holds made before the switch

A correction already waiting for approval can still be approved or rejected. If its preparer commits it again it
posts without the photo and the exception closes as `hold_lifted`.

A hold is judged by the reason it was held with, not by today's list. That covers a removal held with
"Damage/disposal" or "Unrecorded usage", which this migration retires for new entries: it is approved,
committed again or rejected as before, and it posts with its original reason and no destination. Only that one
code is let through, and only for that held ticket (`held_reason` in `exceptions_enforcement.resolve_reason`,
taken from the ticket's stored draft); a new draft with a retired reason is refused.

Not changed: found inventory for an extra-kosher product still waits for the owner's approval
(`write_tickets.py`, the kosher attestation gate). That gate is separate from the 500 lb rule.

## Deploy order (each step needs Michael's approval; staging first)

1. Verified backup (`~/Backups/fl/fl-backup-and-verify.sh`, must end `BACKUP OK`).
2. Apply `migrations/096_adjust_destinations.sql` on port 5432 in one transaction (header of the file).
   Check: 13 rows in `correction_reasons`, 7 with `removal_allowed`, 5 of those with `removal_destination`,
   2 inactive, table `adjustment_alerts` empty (with `idempotency_key`, `message`, `unconfirmed_since`), marker present.
3. Merge. Railway and Netlify deploy.
4. Optional, for email within seconds: set `RESEND_API_KEY` and `ALERT_EMAIL_TO` on the FastAPI service.
5. Verify: a small removal lists the five destinations, then Wrong lot used and Data-entry error; `GET /exceptions?kind=LARGE_CORRECTION` shows no new holds.

## Rollback

Deleting the marker switches the application back to the old rules at once, no deploy needed: the old reason
check, the photo-or-hold over 500 lb, no email. It does **not** by itself bring back "Damage/disposal" and
"Unrecorded usage" for new entries; those two stay retired until they are restored:
`UPDATE correction_reasons SET active = true WHERE code IN ('damage_disposal', 'unrecorded_usage');`

`migrations/down/096_adjust_destinations_down.sql` does both and removes the rest. It refuses while any
transaction carries a destination reason or an owner email is still pending.

## Tests

`tests/test_adjust_destinations.py`, `tests/test_pilot_entry_ui.js`, `tests/test_adjust_destinations_browser.js`
(`tests/visual/run-adjust-destinations.mjs`, real browser, throwaway local database).
