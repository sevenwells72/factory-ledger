# O-7 — Dashboard key delivery and rotation

Closes cutover-runbook item **O-7** (steps 3 and 23). After this change, rotating
the scoped dashboard key is a Railway variable change only. You don't edit code,
rebuild Netlify or clear any cache.

## The method, and why

**Runtime delivery from the API.** Every dashboard page already loads
`dashboard/dashboard-runtime.js` first. That file already chooses the API the page
talks to:

| Page host | API it calls |
|---|---|
| `cns-factory-ledger.netlify.app` | production FastAPI |
| any other `*.netlify.app` (deploy previews) | FastAPI-staging |
| Railway `/dashboard/*` | the same Railway service (same origin) |

`FLDashboard.key()` asks *that same API* for its scoped key:
`GET /auth/dashboard-key` → `{"dashboard_key": …}`, `Cache-Control: no-store`.
Every keyed request sends `X-API-Key: await FLDashboard.key()`. No tracked file
holds a key value.

Why not build-time injection:

* Netlify would need a build command, plus a production variable and a different
  deploy-preview (staging) variable. Railway would need its own templating of
  `dashboard/` at start-up. That makes two copies of the secret, and they can
  disagree.
* Every rotation would need a Netlify rebuild and cache clear. The runbook's
  rollback warning ("rolling back a Netlify build with old literals would break
  reads") would still apply on every rotation.

With runtime delivery, the service that **checks** the key is the only place that
**holds** it, so the key a page gets always matches the API it calls. That is
true on all three hosts.

**Scope does not widen.** The route returns only `DASHBOARD_API_KEY`. That value
was already public, because it was hard-coded in the served JS. The route returns
**503**, never another key, if the dashboard key is unset or equals `API_KEY` or
`READONLY_API_KEY`. The served key still:

* passes only `DASHBOARD_KEY_ALLOWLIST`;
* cannot open a personal session (`POST /auth/session` → 401);
* cannot prepare make/pack/adjust/found/ship (403).

Tests: `tests/test_dashboard_key_delivery.py`. Probe:
`scripts/check_dashboard_key_rotation.py`.

**Open tabs** re-read the key every 60 s (`KEY_TTL_MS`). If a re-read fails, the
tab keeps the last good key. If the first read fails, nothing is cached and the
next request retries. Within a minute of a rotation, open tabs send the new key
and need no reload. One exception: a Shift Summary viewer who chose *Read-only*
holds the key for that view. They press Sign out → Read-only again, or reload.

**Literals stay out.** `scripts/check_no_key_literals.py` fails if any of these
appear in a tracked file:

* a retired key (matched by SHA-256 fingerprint);
* a hard-coded `X-API-Key` value or `*API_KEY = '…'` in `dashboard/**`;
* any live value passed in `FL_KEY_GUARD_VALUES`.

It prints only `path:line` and never the value. It runs in pytest
(`test_repository_has_no_key_literals`). It is also ready for GitHub Actions:
`docs/cutover/key-literal-guard.workflow.yml` becomes `.github/workflows/key-literal-guard.yml`
once the pushing token has the `workflow` scope.

## First production rollout of this change (one time)

The pages need the new route before they can read. **Railway must be live before
Netlify publishes.** Until then the old key is unchanged, so the order below is
safe.

1. Netlify → cns-factory-ledger → Deploys → **Lock to stop auto publishing** on
   the current published deploy. Record its deploy ID.
2. Merge (with Michael's approval). Railway → FastAPI → wait for **Success**.
   Then check the route; this prints the status code only:
   `curl -s -o /dev/null -w '%{http_code}\n' https://fastapi-production-b73a.up.railway.app/auth/dashboard-key`
   → `200`.
3. Netlify → Deploys → the merge's deploy → **Publish deploy**, then **Unlock**.
4. `FL_ALLOW_PRODUCTION=1 node scripts/check_dashboard_pages.mjs https://cns-factory-ledger.netlify.app/`
   and `… https://fastapi-production-b73a.up.railway.app/dashboard/` → `ALL PASS`.

Rollback before any rotation: re-publish the deploy recorded in step 1 and revert
the merge. The old literal still works until step 3 of the rotation below.

## Rotation procedure (each environment; staging first)

Do these steps in a private terminal. Every command prints status codes or
fingerprints, never a value. `<SVC>` = `FastAPI-staging` or `FastAPI`;
`<API>` = that service's URL; `<ENV>` = `staging` or `production`.

1. **Prepare.**
   * Save the current value privately, for the probe and the retired list:
     `railway variables --service <SVC> --json | python3 -c "import json,sys;print(json.load(sys.stdin)['DASHBOARD_API_KEY'])" > ~/Documents/fl-secrets/<ENV>-dashboard-key.old.txt`
   * Generate the new value from the password manager, or with
     `python3 -c "import secrets;print('fld_'+secrets.token_urlsafe(32))" > ~/Documents/fl-secrets/<ENV>-dashboard-key.txt`.
   * Run `chmod 600` on both files. The new value must differ from `API_KEY`,
     `READONLY_API_KEY` and every actor key.
2. **Set it on the server.** Railway → `<SVC>` → Variables → `DASHBOARD_API_KEY`
   → paste → Deploy. From the CLI:
   `railway variables --service <SVC> --set "DASHBOARD_API_KEY=$(cat ~/Documents/fl-secrets/<ENV>-dashboard-key.txt)"`.
   Wait until the deployment shows **Success**.
3. **Hosts.** No action is needed.
   * Netlify serves no key, so it needs no rebuild and no cache clear.
   * Railway `/dashboard/*` is served by the same deployment.
   * The route is `no-store`. Open tabs pick up the new key within 60 s.
4. **Verify. Every line must print PASS:**
   * `python3 scripts/check_dashboard_key_rotation.py --base <API> --new-key-file ~/Documents/fl-secrets/<ENV>-dashboard-key.txt --old-key-file ~/Documents/fl-secrets/<ENV>-dashboard-key.old.txt`
     (add `--allow-production` for production). This checks that the route
     serves the new key, the new key reads, the old key gets 403, and the new key
     still cannot open a session or prepare.
   * `node scripts/check_dashboard_pages.mjs <site>` for each host. Production:
     the Netlify site and `<API>/dashboard/`. Staging: `<API>/dashboard/` and a
     deploy preview, `https://deploy-preview-<N>--cns-factory-ledger.netlify.app/`.
     The script loads index (with the mini-calendar), process-flow, sankey,
     traceability, history, runs, entry and shift-summary. Each page must show
     route 200, zero 401/403 on keyed reads and zero page errors.
   * `FL_KEY_GUARD_VALUES="$(cat ~/Documents/fl-secrets/<ENV>-dashboard-key.txt)" python3 scripts/check_no_key_literals.py`
     → `0 finding(s); 1 live value(s) checked`.
5. **Retire the old value.** Add its fingerprint to `RETIRED_FINGERPRINTS` in
   `scripts/check_no_key_literals.py` in a small reviewed PR. Get the fingerprint
   with
   `python3 -c "import hashlib,sys;print(hashlib.sha256(sys.stdin.read().strip().encode()).hexdigest())" < ~/Documents/fl-secrets/<ENV>-dashboard-key.old.txt`.
   Then delete the `.old.txt` file.

**Rollback.**

* *A bad deploy:* Railway → `<SVC>` → Deployments → redeploy the previous good
  deployment. Variables persist, so the new key stays.
* *A mishandled key:* rotate forward to another fresh value (steps 1–4).
* In **production**, never set `DASHBOARD_API_KEY` to the master key or to the
  exposed pre-O-7 literal. The route refuses the master with 503 anyway.
* On **staging** only, restoring the previous value is acceptable, so that other
  lanes' previews built before O-7 keep reading.
* *After* a rotation, do not roll Netlify or Railway back to a pre-O-7 build. Those
  assets carry the old literal and every keyed read would fail. Roll forward
  instead.

## Notes for the master runbook (#117)

* **Step 3:** the "five embeddings" row was actually **nine** served files, all
  now removed: `dashboard.js`, `mini-calendar.js`, `process-flow.html`,
  `sankey.html`, `traceability.html`, plus `history.js`, `runs.js`, `ledger-ui.js`
  and `shift-summary.js`. The disposition becomes: "served by FastAPI from
  `DASHBOARD_API_KEY`; nothing to prepare beyond the new value".
* **Step 23:** replace "Trigger deploy → Clear cache and deploy site" with
  rotation steps 3–4 above. Netlify needs no rebuild.
* **Old-literal scrub (master key, seven files):** after it, add the retired
  master fingerprint to `RETIRED_FINGERPRINTS`. The guard then enforces that
  scrub too, still with redacted output.
* A future custom dashboard domain still needs an `apiBase()` entry (O-5,
  FOLLOWUPS P1.28). Otherwise the page would ask its own origin for the key.
