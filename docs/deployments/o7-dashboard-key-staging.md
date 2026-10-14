# O-7 staging rotation proof — 2026-10-14

**Staging only.** Production is untouched. The procedure followed is
`docs/cutover/O-7-dashboard-key-rotation.md`. No key values appear here: only
HTTP statuses and the first 12 hex characters of each key's SHA-256.

| What | Value |
|---|---|
| Branch / head | `feat/dashboard-key-delivery` @ `dab8c76` (PR #122), based on main `fd0e00b` |
| API | FastAPI-staging, `https://fastapi-staging-production-dd7b.up.railway.app` |
| Netlify host | deploy preview `https://deploy-preview-122--cns-factory-ledger.netlify.app/` (calls the staging API) |
| Old staging key | sha256 `e5addd5647ef`, 64 chars. This was *not* the pre-O-7 public literal. |
| New staging key | sha256 `c543da8ada2c`, 47 chars, in `~/Documents/fl-secrets/staging-dashboard-key.txt` (mode 600). It differs from the old key, `API_KEY` and `READONLY_API_KEY`. |

## Timeline (UTC)

| Time | Step |
|---|---|
| ~17:40 | `railway up` of the branch → deployment `c5ac4ef3` **SUCCESS**. `/health` 200, `/auth/dashboard-key` 200. |
| 17:44–17:46 | Baseline with the old key: probe **ALL PASS**; page check **ALL PASS** on both hosts. |
| 17:47:49 | `DASHBOARD_API_KEY` set to the new value. Railway redeployed as `bb760e54` **SUCCESS**. |
| 17:50 | The probe found `/auth/dashboard-key` **404**. `bb760e54` had built the service's GitHub source (`main` `fd0e00b`), not the uploaded branch. Every other probe line passed: new key 200, old key 403, no session, no prepare. Runbook updated (`--skip-deploys`, then re-upload). |
| ~17:51 | `railway up` again → `a2c97d4c` **SUCCESS** (17:52:59). The new variable persisted. |
| 17:53 | Probe and page checks after rotation, below: **ALL PASS**. |

## Probe after rotation (`scripts/check_dashboard_key_rotation.py`)

```
PASS  GET /auth/dashboard-key: 200, cache-control=no-store, served sha256 c543da8ada2c vs new c543da8ada2c
PASS  new key GET /auth/whoami: 200, key_kind=legacy_dashboard
PASS  old key GET /auth/whoami: 403 (want 403)
PASS  new key POST /auth/session: 401 (want 401, or 404 with sessions off)
PASS  new key POST /make/prepare: 403 (want 403)
PASS  new key POST /pack/prepare: 403 (want 403)
PASS  new key POST /adjust/prepare: 403 (want 403)
PASS  new key POST /inventory/found/prepare: 403 (want 403)
PASS  new key POST /ship/prepare: 403 (want 403)
PASS  new key POST /ship/bulk/prepare: 403 (want 403)
PASS  new key POST /sales/orders/999999999/ship/prepare: 403 (want 403)
ALL PASS
```

## Pages after rotation (`scripts/check_dashboard_pages.mjs`, fresh browser)

Railway-served `/dashboard/`:

```
PASS  index.html: page 200, /auth/dashboard-key 200 (sha256 c543da8ada2c), keyed reads 9, refused 0, page errors 0
PASS  process-flow.html: page 200, /auth/dashboard-key 200 (sha256 c543da8ada2c), keyed reads 5, refused 0, page errors 0
PASS  sankey.html: page 200, /auth/dashboard-key 200 (sha256 c543da8ada2c), keyed reads 3, refused 0, page errors 0
PASS  traceability.html: page 200, /auth/dashboard-key 200 (sha256 c543da8ada2c), keyed reads 2, refused 0, page errors 0
PASS  history.html: page 200, /auth/dashboard-key 200 (sha256 c543da8ada2c), keyed reads 1, refused 0, page errors 0
PASS  runs.html: page 200, /auth/dashboard-key 200 (sha256 c543da8ada2c), keyed reads 1, refused 0, page errors 0
PASS  entry.html: page 200, /auth/dashboard-key none (sha256 -), keyed reads 0, refused 0, page errors 0
PASS  shift-summary.html: page 200, /auth/dashboard-key 200 (sha256 c543da8ada2c), keyed reads 1, refused 0, page errors 0
ALL PASS
```

Netlify deploy preview: the same build as before the rotation, with **no rebuild
and no cache clear**.

```
PASS  index.html: page 200, /auth/dashboard-key 200 (sha256 c543da8ada2c), keyed reads 9, refused 0, page errors 0
PASS  process-flow.html: page 200, /auth/dashboard-key 200 (sha256 c543da8ada2c), keyed reads 5, refused 0, page errors 0
PASS  sankey.html: page 200, /auth/dashboard-key 200 (sha256 c543da8ada2c), keyed reads 3, refused 0, page errors 0
PASS  traceability.html: page 200, /auth/dashboard-key 200 (sha256 c543da8ada2c), keyed reads 2, refused 0, page errors 0
PASS  history.html: page 200, /auth/dashboard-key 200 (sha256 c543da8ada2c), keyed reads 1, refused 0, page errors 0
PASS  runs.html: page 200, /auth/dashboard-key 200 (sha256 c543da8ada2c), keyed reads 1, refused 0, page errors 0
PASS  entry.html: page 200, /auth/dashboard-key none (sha256 -), keyed reads 0, refused 0, page errors 0
PASS  shift-summary.html: page 200, /auth/dashboard-key 200 (sha256 c543da8ada2c), keyed reads 1, refused 0, page errors 0
ALL PASS
```

`entry.html` makes no keyed read before a personal sign-in while key sessions
are on, so it never asks for the dashboard key. That is expected.

## Open tabs across the rotation

One `traceability.html` tab stayed open on each host for the whole rotation, with
no reload. Every 20 s it called `FLDashboard.key()` plus one keyed read
(`GET /transactions/history?limit=1`):

```
17:47:46 fastapi-staging-producti key e5addd5647ef read 200
17:47:46 deploy-preview-122--cns- key e5addd5647ef read 200
17:49:31 fastapi-staging-producti key e5addd5647ef read 200
17:49:32 deploy-preview-122--cns- key e5addd5647ef read 200
17:50:07 fastapi-staging-producti key e5addd5647ef read 403
17:50:07 deploy-preview-122--cns- key e5addd5647ef read 403
17:52:30 fastapi-staging-producti key e5addd5647ef read 403
17:52:30 deploy-preview-122--cns- key e5addd5647ef read 403
17:52:50 fastapi-staging-producti key e5addd5647ef read 403
17:52:50 deploy-preview-122--cns- key e5addd5647ef read 403
17:53:13 fastapi-staging-producti key c543da8ada2c read 200
17:53:14 deploy-preview-122--cns- key c543da8ada2c read 200
17:53:35 fastapi-staging-producti key c543da8ada2c read 200
17:53:35 deploy-preview-122--cns- key c543da8ada2c read 200
```

* **Before 17:50:** old key, 200.
* **17:50–17:52:50:** reads got 403. The redeploy from `main` had no route, so
  each tab kept its last good key. That key was already dead on the server.
* **From 17:53:13:** once the route served the new key, both tabs switched to it
  on their own, with 200 reads.

In production the route exists before and after the variable deploy. Tabs
therefore switch within the 60 s refresh (KEY_TTL_MS) after the deploy's
**Success**.

## Literal guard with the live values

`FL_KEY_GUARD_VALUES=<new + old staging key> python3 scripts/check_no_key_literals.py`:

```
key-literal guard: 0 finding(s); 2 live value(s) checked
```

## State left behind

* FastAPI-staging runs this branch (`a2c97d4c`) with the **new** key.
* The next `main` deploy of staging will serve pages without
  `/auth/dashboard-key` until O-7 merges. Those are pre-O-7 pages carrying the
  old public literal, and staging already rejected that literal before this
  proof. So they lose nothing that worked before.
* The old staging value is kept privately in
  `~/Documents/fl-secrets/staging-dashboard-key.old.txt` until its fingerprint
  joins `RETIRED_FINGERPRINTS`.
