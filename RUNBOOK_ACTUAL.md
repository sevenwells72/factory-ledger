# Factory Ledger — Production Planner Runbook

What actually works on this machine, in the order you'd run it on a Monday morning.

## Primary weekly workflow

The 3-week rolling planner is the main view. It's a static, self-contained
HTML report — open in any browser, no server needed.

```bash
cd /Users/cns/Documents/factory-ledger
python3 render_planner_v2.py
open planner_v2.html
```

Expected stdout:
```
Rendered W1-W3: <X> granola, <Y> coconut batches (gross) | Beyond: <A> granola, <B> coconut (<date range>) | <N> SOs across <M> open lines -> planner_v2.html
```

What the report shows:
- **Weeks 1–3** (Tue → Mon work-week, past-due rolls into Week 1) — granola
  and coconut batch totals, full per-batch breakdown, finished-goods cases
  to pack. Batches are **gross requirements** (computed from order quantity
  against YAML `target_batch_lb`, ignoring on-hand inventory).
- **Beyond Week 3** — totals plus the top 3 future production weeks by batch
  count. No upper cap; everything past Week 3 lands here.
- **Sales orders panel** — every open SO line, past-due first, then ascending
  ship date. Zebra-striped rows; past-due rows in red.

Note: Friday is a short day (8:00am–2:30pm). The Tue→Mon working week treats
it as a normal day for batch counting; account for the reduced hours when
sequencing actual production.

## Secondary view: inventory-aware drill-down

When you need to know "what's actually short after on-hand," use the v1
renderer. Same data source, but it subtracts FG and base-batch on-hand
before computing batches-to-run.

```bash
python3 render_demand_plan.py
open demand_plan.html
```

This is the original three-tier dashboard: capacity summary → batch plan →
per-SO detail. Use it to confirm a flagged shortage is real before
scheduling production. Do **not** use it as the primary planning surface —
the gross view in v2 reflects raw demand and is easier to plan against.

## One-time setup

### Python dependencies

System Python 3.9.6 is the only interpreter on this machine (no brew, no
pyenv). Use `--user` so the installs don't touch system site-packages.

```bash
python3 -m pip install --user psycopg2-binary python-dotenv pyyaml jinja2
```

pip will print a warning that `~/Library/Python/3.9/bin` is not on PATH.
Harmless — we import these modules from Python, we never call the `dotenv`
CLI directly. Ignore.

Verify:
```bash
python3 -c "import psycopg2, dotenv, yaml, jinja2; print('ok')"
```

Both renderers use PEP 604 union syntax (`dict | None`) in **type
annotations** under `from __future__ import annotations`, which stringifies
annotations at parse time. 3.9 imports the modules fine. Do not use union
syntax in runtime expressions (e.g. `isinstance` checks) — that would
require 3.10+.

### Database connection

Both renderers read `DATABASE_URL` from `.env` at the repo root. The file is
gitignored — copy from a known-good source (1Password / earlier laptop) if
missing. The current value points at the production Supabase project
(`MyFirstProject`, us-east-1).

## Smoke test

`psql` is **not on PATH** on this machine — there's no Postgres CLI client
installed. Use a psycopg2 one-liner instead:

```bash
python3 -c "
from dotenv import load_dotenv; load_dotenv()
import os, psycopg2
c = psycopg2.connect(os.environ['DATABASE_URL'])
cur = c.cursor()
cur.execute('SELECT NOW()')
print('db ok:', cur.fetchone()[0])
"
```

If that prints a current timestamp, the renderers will run.

## When to update the YAML

Update `production_planning_knowledge.yaml` when:
- A new SKU is added to the plant
- A batch target weight changes permanently
- A new batch-to-batch dependency is discovered
- Weekly capacity ceilings change (new equipment, shift change)

Re-running the renderer picks up the new values automatically — no code
changes required.

## Files this runbook references

- `render_planner_v2.py` — primary planner (3-week rolling, gross math)
- `planner_v2.html` — primary planner output
- `render_demand_plan.py` — secondary inventory-aware view (v1)
- `demand_plan.html` — v1 output
- `demand_planning_v1.sql` — shared SQL query, **do not modify**
- `production_planning_knowledge.yaml` — capacity + dependency knowledge,
  **do not modify** without owner sign-off
- `.env` — `DATABASE_URL` only; gitignored
- `CHANGE_LOG.md` — append-only history of changes to renderers, SQL,
  migrations, dashboard
- `BACKLOG.md` — deferred items and known follow-ups
- `discrepancies.md` — open data-quality issues

## Troubleshooting

- **`ERROR: DATABASE_URL not set in .env`** — restore `.env` at the repo root.
- **`psycopg2.OperationalError: SSL connection has been closed`** — Supabase
  occasionally drops idle connections. Re-run the renderer; it opens a fresh
  connection each time.
- **`FATAL: password authentication failed for user "postgres"`** — Supabase
  pooler strips the tenant suffix from the error message. The actual user
  is `postgres.<ref>`, not `postgres`. Real cause: stale or rotated password.
  Pull a fresh `DATABASE_URL` from Railway → FastAPI service → Variables.
- **Stderr `WARN: target_batch_lb drift for SKU…`** — YAML and DB disagree on
  a batch's target lb. Planner uses YAML. Investigate via
  `git log -p production_planning_knowledge.yaml` for the SKU; reconcile
  with owner if the DB was updated more recently.
- **Stderr `WARN: unknown upstream rule shape for base SKU…`** — a new
  batch has gained an upstream batch dependency that v2's two known rules
  (95005, 90008) don't cover. Add a rule in `upstream_rule_map` before the
  next run.
