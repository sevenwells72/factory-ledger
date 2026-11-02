# PR #173 production rollout plan (prepared 2026-11-02, NOT executed)

Formula-driven Make entry, Phase A (+ product search fix, phone layout).
PR: https://github.com/sevenwells72/factory-ledger/pull/173, head `0afb794`, base `main` `87f500f`.
Owner decisions in force: keep warning-and-tick exceptions (no owner approval) for this
release; no recipe, yield, inventory-calculation or approval-rule changes.

**Nothing here has been run.** The merge is also the production deploy: one approval
covers both.

## 0. What one merge does (automatic, no separate deploy step)

Merging into `main` immediately triggers:

| System | Trigger | Effect |
|---|---|---|
| Railway **FastAPI** (production API) | `railway.json` watch patterns (`/*.py` …); PR changes root `.py` files | Builds and deploys the new API (about 2–4 min). Replaces `5e17b70d` (commit `a4a5c99`). |
| Railway **expire-tickets**, **exceptions-sweep** (production crons) | Same watch patterns | Rebuild; next runs 07:15 / 07:30 UTC on the new code. |
| **Netlify** production dashboard (cns-factory-ledger.netlify.app) | Auto-publish from `main` (`stop_builds: false`) | Publishes the new `dashboard/` (Make screen, `ledger-ticket.js` v13 …), usually within 1–2 min. |
| Railway **FastAPI-staging** and staging crons | Source = `main` | May rebuild from `main`, which **removes the staging TEST MODE** sign-in (it is only on the local `staging/pr173-signin-mobile` build). Staging keeps working with personal keys/PINs. No production effect. |

**Mismatch window (fail-safe):** if Netlify publishes before Railway finishes, the new
Make screen calls the old API. The old `/make/prepare` rejects the new fields (422) and
`/entry/make-products` is 404, so Make shows an error / "Try again". **Nothing posts.**
Old cached pages against the new API keep working (legacy make payload kept). So: no
Make entries until step 3 says "ready".

## 1. Before merge (all read-only, ~5 min)

1. Backup: `~/Backups/fl/2026-11-02-pre173` verified (KEY CHECKS: ALL MATCH). Re-run if
   there is >1 h or any production write between backup and merge.
2. PR #173: OPEN, MERGEABLE/CLEAN, head `0afb794`, checks green; `main` still `87f500f`.
3. Production: FastAPI latest SUCCESS `5e17b70d`; `/health` 200, database connected;
   `/auth/session/config` = key sessions on, PIN on, **no `test_sign_in`**; Netlify
   published `6ae78fe1…` (`39988d1`).
4. Arturo is not mid-Make (no open Make draft). Choose a quiet time.
5. Record a read-only fingerprint: `max(transactions.id)`, `max(transaction_lines.id)`,
   count of posted make transactions today.

## 2. Merge (needs explicit approval: merge = production deploy)

- Method: **squash merge** (same as #168/#170): `gh pr merge 173 --squash` with the PR
  title as the subject. Expected result: one new commit on `main` whose parent is
  `87f500f` and whose tree equals PR head `0afb794`. Verify:
  `git diff 0afb794 origin/main --stat` → empty.
- Do not delete the branch until the deploy is verified.

## 3. Verify the deployment (read-only)

1. `railway deployment list --service FastAPI` → new deployment SUCCESS, commit = the
   squash commit. Same for expire-tickets and exceptions-sweep (SUCCESS or SKIPPED).
2. Netlify: published deploy `commit_ref` = squash commit; `entry.html` loads
   `entry.js?v=13`, `make-formula.js?v=4`, `ledger-ticket.js?v=13`, `ledger-forms.css?v=14`.
3. API: `/health` 200 + database connected; `/auth/session/config` unchanged
   (key sessions on, PIN on, no `test_sign_in`); `/auth/test-sign-in/users` → 404;
   `/auth/whoami` without key → 401; `/openapi.json` contains `GET /entry/make-products`.
4. `GET /entry/make-products?query=classic` with the dashboard key (from
   `/auth/dashboard-key`) → 200 and lists Batch Classic Granola #9 (90002), 323 lb per pan.
5. Logs: `railway logs --service FastAPI` for 10 min → no Traceback / 5xx.
6. Fingerprint unchanged (no transactions created by the deploy).

## 4. Make screen smoke test (no inventory transaction)

On a phone, Michael signed in with **his own** account (PIN or key):

1. Open Record an entry → Make. No red errors; layout fits at phone width.
2. Product search "classic" → choose Classic #9 (90002) → 2 pans.
3. **Prepare draft** → ingredient cards with oldest-first bags; one line for
   unconfirmed bags; 2 × 323 = 646 lb.
4. **Do not press Record.** Tap Cancel.

Note: step 3 writes one *draft ticket* row (`write_tickets`, status prepared, expires in
10 min). No transaction, no lot, no inventory change. If you want strictly zero writes,
stop after step 2; the first real Make by Arturo is then the first draft.

Then **Michael authorizes the first real Make** (step 6).

## 5. Rollback

**Application code (minutes, no data loss):**

- Preferred: `git revert` the squash commit on `main` (needs approval; push to `main`).
  Railway and Netlify auto-deploy the revert, keeping API and dashboard consistent.
- Fastest: Railway dashboard → FastAPI → deployment `5e17b70d` → **Rollback**; Netlify →
  Deploys → `6ae78fe1…` → **Publish deploy**. Both are production setting changes
  (approval). Then revert `main` too, or the next push redeploys PR #173.
- Verify as in step 3 with the old versions (`entry.js?v=10`, no `/entry/make-products`).

**Database:** PR #173 has **no migration**, so rollback needs no schema change.

- Make transactions recorded by the new code are normal posted ledger rows. Keep them;
  they stay valid under the old code. Corrections only by void/new record with reason and
  named actor. Never delete or edit.
- `transaction_substitutions` "not used" rows stay (history).
- Open Make drafts prepared by the new code: let them expire (10 min). The old code may
  refuse to commit them; the worker prepares again.
- Restore from `2026-11-02-pre173` **only** for data corruption. It would discard every
  later production write, so it is Michael's decision, after a fresh backup.

## 6. First day monitoring (read-only)

- **Errors:** Railway logs: Traceback, 5xx on `/make/prepare`, `/tickets/*/commit`,
  `/entry/make-products`.
- **Incomplete drafts:** `write_tickets` with action make: prepared vs committed vs
  expired today. Many expired drafts mean a confusing step; look at their blockers.
- **Attribution:** every posted make today has `entered_by_actor_id` (Arturo/Michael), no
  shared-key makes from the dashboard.
- **Duplicate warnings:** `POSSIBLE_DUPLICATE` on make tickets (expected for identical
  batches within 45 min). Count how often Arturo has to tick it.
- **Recipe flags:** shift summary `INGREDIENT_OFF_RECIPE` / `NOT_USED` / `SWAP` rows;
  shortage flags.
- **Inventory sanity:** posted-only on-hand for the ingredients used vs the receipts.
- Check at end of first Make, mid-day, end of shift; write results into the deploy record.

## 7. First-day note for Arturo

**EN:** New Make screen today. Choose the product, type the number of pans, tap Prepare
draft. Each ingredient shows the bags to use. Type the code from each bag label. If you
change pounds or bags, tap **Check bags and pounds** at the bottom, then **Record**. If
an ingredient was not used, tap Not used and choose a reason. If you see "A matching make
was posted N minutes ago", tick it only if this is really a new batch. If anything looks
wrong, don't record; call Michael.

**ES:** Pantalla nueva de producción hoy. Elige el producto, escribe el número de
bandejas y toca Preparar borrador. Cada ingrediente muestra las bolsas a usar. Escribe el
código de la etiqueta de cada bolsa. Si cambias libras o bolsas, toca **Verificar bolsas y
libras** abajo y luego **Registrar**. Si no usaste un ingrediente, toca No se usó y elige
el motivo. Si ves "Una operación similar se registró hace N minutos", márcalo solo si de
verdad es otra tanda. Si algo se ve mal, no registres; llama a Michael.

## 8. After verification

Record the deploy: `FACTORY_LEDGER_CHANGELOG` row (DEPLOYED, squash commit, Railway and
Netlify deploy ids, backup path, checks), via a docs-only PR like #157/#169/#171.
