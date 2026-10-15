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

## Fix round 1 re-check (Codex 6175101774) — 2026-10-14

Head `bb77c1d` (rebased on main `5a2d57e`). This time the runbook order was
followed: `railway variable set DASHBOARD_API_KEY --stdin --skip-deploys`, then
`railway up`.

**Why the review's hosted checks failed (environment only):** merging #120 at
18:01:44Z redeployed FastAPI-staging from GitHub `main` as `879c4de7`. That
replaced the branch upload, so `/auth/dashboard-key` returned 404 throughout the
review (the review comment is timestamped 18:11:15Z). The 502s fell in that
redeploy window. The code did not change.

| Time (UTC) | Step |
|---|---|
| 18:29:57 | Rotated `c543da8ada2c` → `b94a405680dc` with `--skip-deploys`. No deploy was triggered; the latest deployment stayed `879c4de7`. |
| 18:36:29 | `railway up` → `ec4a2065` **SUCCESS**. Probe **ALL PASS**; pages **ALL PASS** on both hosts (preview rebuilt for `bb77c1d`). |
| 18:38:49 | With a read-only Shift Summary view open on each host, rotated forward again: `b94a405680dc` → `4769877cf319`, `--skip-deploys`, then `railway up` → `43034346` **SUCCESS** (18:42:10). |

Probe after the first rotation:

```
PASS  GET /auth/dashboard-key: 200, cache-control=no-store, served sha256 b94a405680dc vs new b94a405680dc
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

Pages, Railway `/dashboard/`:

```
PASS  index.html: page 200, /auth/dashboard-key 200 (sha256 b94a405680dc), keyed reads 9, refused 0, page errors 0
PASS  process-flow.html: page 200, /auth/dashboard-key 200 (sha256 b94a405680dc), keyed reads 5, refused 0, page errors 0
PASS  sankey.html: page 200, /auth/dashboard-key 200 (sha256 b94a405680dc), keyed reads 3, refused 0, page errors 0
PASS  traceability.html: page 200, /auth/dashboard-key 200 (sha256 b94a405680dc), keyed reads 2, refused 0, page errors 0
PASS  history.html: page 200, /auth/dashboard-key 200 (sha256 b94a405680dc), keyed reads 1, refused 0, page errors 0
PASS  runs.html: page 200, /auth/dashboard-key 200 (sha256 b94a405680dc), keyed reads 1, refused 0, page errors 0
PASS  entry.html: page 200, /auth/dashboard-key none (sha256 -), keyed reads 0, refused 0, page errors 0
PASS  shift-summary.html: page 200, /auth/dashboard-key 200 (sha256 b94a405680dc), keyed reads 1, refused 0, page errors 0
ALL PASS
```

Pages, Netlify deploy-preview-122:

```
PASS  index.html: page 200, /auth/dashboard-key 200 (sha256 b94a405680dc), keyed reads 9, refused 0, page errors 0
PASS  process-flow.html: page 200, /auth/dashboard-key 200 (sha256 b94a405680dc), keyed reads 5, refused 0, page errors 0
PASS  sankey.html: page 200, /auth/dashboard-key 200 (sha256 b94a405680dc), keyed reads 3, refused 0, page errors 0
PASS  traceability.html: page 200, /auth/dashboard-key 200 (sha256 b94a405680dc), keyed reads 2, refused 0, page errors 0
PASS  history.html: page 200, /auth/dashboard-key 200 (sha256 b94a405680dc), keyed reads 1, refused 0, page errors 0
PASS  runs.html: page 200, /auth/dashboard-key 200 (sha256 b94a405680dc), keyed reads 1, refused 0, page errors 0
PASS  entry.html: page 200, /auth/dashboard-key none (sha256 -), keyed reads 0, refused 0, page errors 0
PASS  shift-summary.html: page 200, /auth/dashboard-key 200 (sha256 b94a405680dc), keyed reads 1, refused 0, page errors 0
ALL PASS
```

Open read-only Shift Summary views across the second rotation, Refresh every 20 s
(item 2 of the review):

```
18:38:21 fastapi-staging-producti shift-summary key b94a405680dc status 200
18:38:23 deploy-preview-122--cns- shift-summary key b94a405680dc status 200
18:41:22 fastapi-staging-producti shift-summary key b94a405680dc status 200
18:41:22 deploy-preview-122--cns- shift-summary key b94a405680dc status 200
18:41:55 fastapi-staging-producti shift-summary key b94a405680dc status 502
18:42:02 fastapi-staging-producti shift-summary key 4769877cf319 status 200
18:42:03 deploy-preview-122--cns- shift-summary key 4769877cf319 status 200
18:42:25 deploy-preview-122--cns- shift-summary key 4769877cf319 status 200
18:42:25 fastapi-staging-producti shift-summary key 4769877cf319 status 200
18:42:42 fastapi-staging-producti shift-summary key 4769877cf319 status 200
18:42:43 deploy-preview-122--cns- shift-summary key 4769877cf319 status 200
```

Both views switched to the new key without a reload, with **no 403 window**: with
`--skip-deploys`, the old deployment kept serving the old key until the new one
went live. The single 502 is Railway's container swap.

Guard with both live values: `key-literal guard: 0 finding(s); 2 live value(s) checked`

## State left behind

* FastAPI-staging runs the branch upload `43034346` with key `4769877cf319`.
  The value is in `~/Documents/fl-secrets/staging-dashboard-key.txt`; the
  previous value `b94a405680dc` is in `.old.txt`.
* The next `main` deploy of staging serves pages without `/auth/dashboard-key`
  until O-7 merges. Those pre-O-7 pages carry the old public literal, which
  staging already rejected before this work, so they lose nothing that worked
  before.
