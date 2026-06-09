# Read-only ship_order failure — diagnosis 2026-05-27

**Order:** SO-260514-001
**Reporter:** Arturo (via ChatGPT GPT, dispatch attempt)
**Error returned to user:** `Ship commit failed: cannot execute INSERT in a read-only transaction`
**Investigating branch:** `feat/planner-v2` (HEAD `2d665d3`)
**Production branch:** `main` (HEAD `06b6f4a`)

---

## 1. Did the read-only tripwire fire?

**No.** The tripwire that PR #6 (commit `c9f58d8`, merged as `49a22e3`) added is
wired into exactly one route — `PUT /sales/orders/{order_id}` — and `ship_order`
is a different handler with its own `except Exception` block that does not call
the probe.

Evidence on `main`:

- Tripwire helpers defined at [main.py:523](main.py:523) (`_is_readonly_error`,
  `_capture_readonly_diagnostics`).
- Tripwire invoked at [main.py:5722](main.py:5722) inside `update_order_header`
  — the only call site.
- `ship_order` commit branch at [main.py:6158](main.py:6158) does:
  ```python
  except Exception as e:
      logger.error(f"Ship order commit failed: {e}")
      return JSONResponse(status_code=500, content={"error": str(e)})
  ```
  No probe, no `READONLY_TRIPWIRE:` log line, no `diagnostics` payload.
- `ship_order` preview at [main.py:5979](main.py:5979) has the same pattern.

PR #6's commit message called this out explicitly under "Scope note":

> this patches only the PUT /sales/orders/{order_id} handler. Broader UPDATE
> endpoints (/ship/commit, /make/commit, /adjust) use the same `except Exception`
> pattern but are not yet wired to the tripwire. A real
> `@app.exception_handler(Exception)` would cover them in one shot; left for follow-up.

So: the message the user saw — `{"error": "cannot execute INSERT in a read-only
transaction"}` — is the raw psycopg2 error stringified through the generic
handler. **No `READONLY_TRIPWIRE:` log line will exist in Railway logs for this
incident.** Confirm by greping `READONLY_TRIPWIRE` over the relevant window if
desired, but expect zero hits.

Note on this branch: `feat/planner-v2` diverged at `ecfd7fa` — *before* the PR #6
merge — so even the `update_order_header` tripwire is absent here. Production
runs `main`, where it is present (just not wired to `ship_order`).

---

## 2. State of deferred reliability work (`stash@{0}` on `interesting-lehmann`)

Still stashed, untouched. None of it has been merged on any branch.

`git stash list`:
```
stash@{0}: On interesting-lehmann: WIP: global exception handlers, pool size 50,
           retry logic, GPT action error docs (stashed 2026-04-19)
```

What's in the patch (`git stash show -p stash@{0}`):

| Change | File | Effect |
|---|---|---|
| `@app.exception_handler(pool.PoolError)` → 503 JSON | `main.py` | Clean JSON instead of unhandled exception when pool is exhausted |
| `@app.exception_handler(psycopg2.OperationalError)` → 503 JSON | `main.py` | Same, for DB connection errors |
| `@app.exception_handler(Exception)` → 500 JSON | `main.py` | Catches every other unhandled exception — would have caught Arturo's read-only error |
| Pool size `maxconn=20` → `maxconn=50` | `main.py:164` | Larger headroom under concurrent GPT load |
| `get_db_connection()` retries `getconn()` once after a 1s sleep on `PoolError` | `main.py` | Briefly weathers exhaustion bursts |
| `get_db_connection()` guards `putconn(conn)` with `if conn is not None` | `main.py` | Don't blow up on cleanup when acquire failed |
| `/health` exposes `pool.in_use` and `pool.available` | `main.py` | Observability for pool saturation |
| GPT action error-handling doc | `GUIDE.md` | Tells GPTs to retry on 503 after checking `/health`, never retry mutating ops |

Confirmed merged since stash was created (2026-04-19) — checked via
`git log --all --since=2026-04-19 -i --grep "readonly\|read-only\|pool\|retry\|exception_handler\|tripwire"`:

- `c9f58d8` PR #6 tripwire (does NOT overlap the stash; tripwire is per-route,
  stash adds a global handler).
- `60e25aa` changelog row for PR #6.
- That's it.

Pool config in current `main` is still `minconn=2, maxconn=20`. No
`@app.exception_handler` of any kind exists in `main.py` on either `main` or
`feat/planner-v2`.

**Would any of this have prevented today's failure?** No — the stash adds a
global exception handler that would have given the request a clean JSON shape
and (if reshaped slightly) could have invoked the readonly probe for any
endpoint. But the underlying read-only error would still happen. The retry-once
on `PoolError` doesn't help here either; this wasn't a pool exhaustion event.
The value the stash *would* have added: cleaner error envelope + diagnostics on
every endpoint, not just `update_order_header`.

---

## 3. Most likely root cause

Three candidates, weighed against what PR #6 was designed to catch and the
baseline it captured (`audits/2026-05/readonly-baseline-20260518T210745Z.json`
on `main`: `default_ro=off`, `txn_ro=off`, `is_replica=false`, `pg_version=17.6`):

| Layer | Candidate | Plausibility | Why |
|---|---|---|---|
| Session | Middleware/conn-string forced `default_transaction_read_only=on` | **Low** | No such middleware in `main.py`; baseline shows `default_ro=off`. Conn string doesn't include `options=...read_only`. |
| Connection | Pool handed out a stale connection now bound to a node in recovery | **Possible** | `psycopg2.pool.ThreadedConnectionPool` does not health-check checked-in connections. After a Supabase HA event, an existing TCP session could be the *same connection number* now routed to a read-only node. Long-lived pooled connections are a known footgun for this. |
| Infra | Supabase failover / pooler routed transactions to a replica briefly | **Most likely** | Matches PR #6's original hypothesis ("transient Supabase failover window"), matches the intermittent symptom, matches the fact that 30/30 manual UPDATEs reproduced healthy state right after PR #6 shipped. The Transaction Pooler at port 6543 is supposed to multiplex onto primary connections, but during a failover/promotion event routing can briefly land on the wrong node. |

**Best-supported call:** an infra-level Supabase HA event whose connection-level
manifestation was a pooled connection (or freshly-acquired one) bound to a
read-only node at the moment Arturo's commit ran its first `INSERT`. The
intermittent, transient character — and the inability to reproduce on demand —
points away from configuration/middleware (which would be persistent) and toward
something the pooler/database transparently did in the background.

The candid honest answer, though: **we still can't name it precisely, because
the tripwire didn't fire on this endpoint.** The whole point of capturing
`inet_server_addr`, `pg_is_in_recovery`, `current_user`, and `txn_ro` at the
moment of failure was to discriminate between these three. Today's incident
produced no such receipt.

---

## 4. DATABASE_URL — has it drifted?

Local `.env` (redacted, host/port only):

```
DATABASE_URL=postgresql://[REDACTED]@aws-1-us-east-1.pooler.supabase.com:6543/...
```

✓ Matches the canonical Transaction Pooler endpoint on `aws-1-us-east-1`, port
`6543`. Not a read replica URL, not a Session Pooler (5432), not the direct
endpoint (`db.<project>.supabase.co:5432`).

**Production Railway value: not verified in this pass.** `railway variables`
failed with `Token refresh failed: invalid_grant — please run railway login
again`. To verify, re-login and run:

```
railway variables --kv | grep DATABASE_URL=
```

Expected: `aws-1-us-east-1.pooler.supabase.com:6543`. If it shows port `5432` or
a `db.<project>.supabase.co` host, that's drift worth fixing — though note that
even a misrouted Session Pooler URL wouldn't *normally* produce a read-only
error; it would produce different failure modes (long-held idle connections,
prepared-statement collisions). So URL drift is a long-shot explanation.

Minor doc cleanup worth noting: [DEPLOYMENT.md:90](DEPLOYMENT.md:90) still
references `db.vrafvwcdpcijvxdvefpr.supabase.co:5432` (direct endpoint, not
pooler). Almost certainly stale text rather than a live misconfig, but flagging
in case it's used as a reference somewhere.

---

## Recommended next moves

1. **Wire the tripwire to every UPDATE endpoint, not just `update_order_header`.**
   The cleanest path: take the global `@app.exception_handler(Exception)` from
   `stash@{0}` and add a one-liner to call `_capture_readonly_diagnostics()` and
   emit `READONLY_TRIPWIRE:` when `_is_readonly_error(exc)` is true, before
   returning the generic 500. That handles ship, make, adjust, pack, receive,
   and anything else in one place. Without this, the next occurrence will also
   produce no receipt.

2. **Re-login to Railway and confirm:**
   - `DATABASE_URL` host = `aws-1-us-east-1.pooler.supabase.com`, port = `6543`.
   - Grep recent logs for `READONLY_TRIPWIRE` (expected: zero hits for
     today's incident; confirms diagnosis above).
   - Grep for `Ship order commit failed: cannot execute INSERT in a read-only`
     to pin the exact timestamp/log entry from Arturo's attempt.

3. **Decide on the stash.** It's been sitting since 2026-04-19; `main.py` has
   moved underneath it. Realistic options:
   - **Re-derive the global exception handlers fresh on `main`** (cleanest;
     stash is mostly small additive blocks and a pool-size bump that should
     rebase cleanly but are also easy to re-type).
   - **Rebase the stash onto `main` and PR it.** Likely to merge-conflict on
     `get_db_connection()` and `health_check`; not painful but not free.
   - **Cherry-pick just `maxconn=50` + the three `@app.exception_handler`
     blocks** as the minimum-viable reliability change and drop the rest.

4. **(Speculative, only if it recurs.)** If Supabase failover is confirmed as
   the trigger once the tripwire produces a receipt, the durable mitigation is
   either: (a) replace `ThreadedConnectionPool` with a pool that validates
   connections on checkout (e.g. SQLAlchemy `pool_pre_ping`), or (b) add a
   one-shot retry on `psycopg2.errors.ReadOnlySqlTransaction` after discarding
   the offending connection. Not needed yet; do not implement without a captured
   tripwire payload.
