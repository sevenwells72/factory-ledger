# Factory Ledger staging

Staging is isolated in Supabase project **fl-staging** and Railway service
**FastAPI-staging**, in project **gleaming-solace**.

- URL: https://fastapi-staging-production-dd7b.up.railway.app
- Railway service ID: `0d957be1-8787-41e5-ab38-de71287c30ce`.
- Railway environment ID: `f4d219df-2fea-45e8-85de-466b36a86c07`.
  Its platform name is `production`, but this is a separate staging service with
  `ENVIRONMENT=staging`; the existing FastAPI service is not changed.
- Source: `sevenwells72/factory-ledger`, branch `main`.
- Local implementation: branch `infra/staging` in `~/dev/fl-staging-wt`.
- Staging DB URI: `~/Documents/fl-secrets/staging-db-url.txt`, mode 600.
- Staging API key: `~/Documents/fl-secrets/staging-api-key.txt`, mode 600.
  Never print either secret, put it in a command-line argument, or commit it.

## Isolation and startup

Before schema loading and deployment, the production host from the existing
local configuration and the staging service's actual DATABASE_URL were compared:

| Database | Host |
|---|---|
| Production | aws-1-us-east-1.pooler.supabase.com |
| Staging | aws-0-us-east-1.pooler.supabase.com |

`staging_safety.py` refuses staging startup if the host equals production, if
the production Supabase project appears through another connection endpoint,
or if ambiguous routing overrides are present. Staging requires an explicit
`PRODUCTION_DATABASE_HOST`. The deployed service also pins
`STAGING_DATABASE_HOST` and `STAGING_DATABASE_PROJECT_REF`.
The guard runs before opening the app's pool and again before any startup
migration or data sweep. Other environments keep their existing behavior.

The staging service deploys **main**, as requested, while this infrastructure PR
remains unmerged. Its service-specific start command embeds the exact same guard,
then starts uvicorn. This protects current main even before main contains the
new module. To regenerate the non-secret command:

```sh
python scripts/staging_start_command.py
```

Set the resulting command on **FastAPI-staging only**. Do not put it in the shared
`railway.json` or update production settings. Regenerate this embedded command
when changing the guard. The service listens on `PORT=8000` and checks `/health`.

All variables were set independently: DATABASE_URL, API_KEY, a separate random
DASHBOARD_API_KEY, ENVIRONMENT, the three database identity settings above,
PORT, RAILPACK_PYTHON_VERSION, and PYTHONUNBUFFERED. No production variables were
copied; no AI/storage/integration credentials were added. The dashboard key lives
only in this service's variables.

## Schema and seed

The checked-in `tests/schema/schema.sql` (including migrations 051–057) was
loaded once into an empty staging database in one transaction with
`ON_ERROR_STOP`, after creating `pg_trgm` in public. Anonymous/authenticated
Data API table/sequence privileges and public function execution were revoked
in staging. Do not reload this schema over an existing populated database.

Run the seed using an environment with the repo's Python requirements installed:

```sh
python scripts/seed_staging.py
```

The script reads the protected staging URI file; it does not fall back to
DATABASE_URL. It checks isolation before connecting and uses one transaction
plus an advisory lock and durable markers, so reruns do not duplicate fixtures.

Fixtures derive from `seed_database` and `seed_write_database` in
`mcp_server/tests/ledger_harness.py` at commit
`d74f747d2766393c75d07e2ca295f6ada49b6c74` on `feature/mcp-server`.
That committed file was read with `git show`; the branch was not changed.
The fixture IDs start above 1,000,000,000 to avoid both master-data IDs and the
app's historical hard-coded startup mappings.

The synthetic scenario contains:

- Three products: STAGING Test Almonds, STAGING Test Batch, STAGING Pallet Charge.
- One customer and one supplier, both prefixed STAGING.
- One customer product alias and one supplier product alias, with unit conversions.
- One batch formula, two lots containing 250 lb each, and one receive transaction.
- Order SO-STAGING-TEST with two 100 lb product lines and two pallet charges.
- One inactive synthetic actor with an unrecoverable random key hash.
  No published harness keys are deployed.

No production catalog or business records were copied during setup.

## Optional real catalog

Only on an explicitly authorized run, supply a separate mode-600 production
URI file:

```sh
python scripts/seed_staging.py --copy-master-data \
  --production-db-url-file /secure/path/production-db-url.txt
```

This function is the only code path that opens the source database. It runs
SELECT statements inside an explicit repeatable-read, read-only transaction and
ends with ROLLBACK. It never changes source session read-only defaults.

The fixed allowlist is customers, suppliers, products,
customer_product_aliases and supplier_product_aliases. The schema has no separate
units table: product units and alias unit/conversion columns travel with those
rows. Generated alias keys are regenerated by PostgreSQL. Source IDs are
preserved; sequence values are advanced only in staging. Source IDs overlapping
the synthetic range cause a refusal.

Orders, lots, transactions, inventory, actors, credentials, formulas and all
other tables are excluded from the copy. Import errors roll back staging
changes; existing staging master-data collisions are errors rather than silent
overwrites. A durable marker prevents a completed import from being repeated.

## Validation

The targeted suite runs against a disposable **local** PostgreSQL database:

```sh
python -m pytest tests/test_staging_safety.py tests/test_seed_staging.py \
  tests/test_startup_migrations.py
```

Set TEST_DATABASE_URL only to a disposable local test database using the normal
repo setup. The existing test-suite guard deliberately rejects all Supabase
hosts, including staging. The seed tests verify idempotency and units/generated
aliases, and instrument the source so any non-allowlisted SQL fails.
The safety tests prove neither the pool nor sweeps can run on a forbidden host,
including when using the exact generated Railway launcher.

Setup validation: **44 tests passed**. Live smoke results are recorded below. Production records are not queried to check absence: the
authorized production-read scope is limited to the optional master-data copy.
Isolation is established from the service's own URI, host/project guard, API
routing and direct staging record verification.

## Live staging verification — 2026-10-07 12:02 ET

- Railway deployment: `1ce1986c-e204-4220-b04b-06661f365b43`, SUCCESS.
- Deployed branch: main; commit: `d8d8081b58465b3033a1912b4481160da4c00d93`.
- Product search: HTTP 200; returned STAGING Test Almonds.
- Order create: HTTP 200; order ID `1000000002`, 10 lb STAGING Test Batch.
- Receive: HTTP 200; transaction ID `1000000002`, lot ID `1000000003`, 25 lb almonds.
- Unique smoke reference: `STG-SMOKE-5BB2511D5EA3`.
- Direct staging SELECTs verified order references, receipt reference, lot identity
  and the 25 lb transaction line. The API order-detail read also succeeded.
- After smoke: 3 products, 1 customer, 1 supplier, 3 lots, 2 transactions,
  2 orders and 4 order lines. The smoke records are intentionally retained.
- No production queries, catalog copies, service changes or variable changes.

This PR is not merged. The hosted staging deployment uses the guarded
service-specific launcher while continuing to track main.
