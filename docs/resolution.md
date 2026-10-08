# A4 part 1: resolution and approved alias seeds

Implements the read portion of `docs/design/phase1-safe-operating-system.md`
§1.8, §3, §10 A4 and §11, based on `d6ead6c` and rechecked against revision 3.4
on `cef1c16`, with Michael's 2026-10-08 review corrections recorded in §3.2/§10.
This PR is **A4 part 1 (resolution + seeds)**. **A4 part 2**, scheduled after
A1 merges, adds `resolution_log` and `/aliases` write/prepare/deactivate endpoints.
The approved pouch rule from FOLLOWUPS P1.2 is now explicit in §3.2.

## HTTP contract

`POST /resolve` uses existing `X-API-Key` authentication. Master, dashboard and
all active actor roles can read it; actor access inherits the dashboard
allowlist. The F1 backend, dashboard and future MCP adapter use the same handler.
No client source, model confidence, chosen ID or caller-supplied aliases are
accepted. This branch does not change the MCP branch or its SAFE_POSTS list.

```json
{
  "kind": "product",
  "query": "Sunshine 9",
  "context": {"action": "make", "group": "floor"}
}
```

Kinds: `product`, `customer`, `supplier`, `order`, `lot`, `unit`. `context`
defaults to `{}` and accepts action, group, product/customer/supplier/order IDs,
customer address, and order state/status. Quantity is an optional positive,
finite decimal, at most 1 trillion. Query is required, at most 500 characters;
an empty unit query explicitly asks for a unit.

The response always has `outcome` (`match`, `ambiguous`, `none`), nullable
`match`, ranked `candidates`, `confidence`, `query_normalized`,
`expansions_applied`, `needs_clarification`, and `ask`. Candidates include ID,
label, score, tier, reason, context boost and relevant catalog/lot/order metadata.

- Exact identifiers/full names and exact aliases form the first tier. Multiple
  exact identities are ambiguous. An exact hit is not diluted by fuzzy neighbors.
- Otherwise, keyword hits score 0.8 and trigram hits use PostgreSQL similarity.
  Two or more candidates at least 0.5 are **always ambiguous**, even with a large
  score gap. Only one plausible identity can match.
- Candidates from 0.25 to below 0.5 are suggestions requiring clarification,
  never matches. Below 0.25 the outcome is `none`, with up to three `near_misses`
  kept outside `candidates`. No confident match is stated explicitly.
- Body fields `limit` (default 8, range 1–25) and `offset` (default 0, range
  0–2147483647) page candidates. SQL deduplicates all alias spellings before
  counting and paging: `candidate_count` is exact and `has_more` is page-aware.
  A one-item or empty page cannot change the global outcome or chosen identity.
- Equal product scores sort by latest activity within 180 days: a posted
  transaction occurrence (using current corrected ledger records) or an open,
  unfulfilled order line. Context breaks remaining ties, then name/id. Recency
  and context never change scores, bypass the exact tier, or resolve ambiguity.
- Alias substitution uses normalized whole whitespace-delimited spans, in both
  directions. It never substitutes inside a word and never fuzzy-searches alias
  spellings. Short and numeric search terms also have word boundaries: `SS`
  cannot find `Classic`, and `9` cannot find `19`. Conflicting token expansions
  remain alternatives. Expansion overflow asks for an exact identifier.
- Context boosts only rank; they do not change scores. The make domain is
  batches/ingredients, with ingredients first for floor make. For pack with a
  source product, only that batch's finished children are eligible. Order favors
  finished goods/services; receive favors ingredients/packaging. In the current
  SS #9 catalog, make has two batch candidates; pack with source 283 has its
  three children (285–287), and source 284 has its three (288–290).
- Product aliases from existing customer/supplier tables require that exact
  customer/supplier context and exact item code/description. Customer aliases
  are also exact-only; address agreement only ranks an ambiguous list.
- Lots use normalized exact code (case/whitespace/trailing `LOT`), supplier lot
  codes, then last four within a required product scope, then similar codes.
  Balances use posted-only ledger views; empty/negative lots are excluded except
  for adjust/void. An exact but ineligible lot returns no match with the reason;
  it never falls through to another lot. Exact supplier codes retain a literal
  trailing LOT. An exact merged lot returns no match and its merge destination.
- Orders try exact order number/PO first; customer-scoped open-order phrases
  list newest first. Unknown order numbers never fall back to a convenient open
  order. State/status filters apply; exact closed order numbers remain readable.
- Inactive products, customers and suppliers are excluded. Pseudo-suppliers from
  migration 042 (FOUND, UNKNOWN, count/intake/correction sentinels) are excluded
  even if accidentally active or padded with tabs/newlines; a lot prefix is never
  a supplier identifier.

Revision 3.2's persisted kosher tier, attestation, renames and tier-specific
family expansion are assigned to A12 (§5.3's build-order paragraph), whose
catalog mapping still awaits owner confirmation. A4 already keeps `SS 9` within
the SS family and `9` ambiguous across regular/SS batch names. A12 must add the
stored tier field/labels and expand a bare `9` to regular finished products
whose names do not contain `9`. A4 does not invent tier labels or apply that
unconfirmed catalog mapping.

`POST /products/resolve` keeps the successful legacy per-name envelope and bulk
summary. Ambiguous/unconfident names now return `match: null` plus the new
candidate/clarification fields. Existing write resolvers are untouched.

`GET /aliases?kind=token&active=true` lists active rows by default; `active=false`
lists inactive rows. Both reads report `alias_table_available=false` when migration
060 has not been applied; ordinary catalog resolution continues without aliases.

## Pouch drafts

```json
{
  "kind": "unit", "query": "pouches", "quantity": 24,
  "context": {"product_id": 145}
}
```

For a finished pouch SKU (`pack_format=bagged`), a verified 12-pouch case returns
`draft: {product_id: 145, quantity: 2, unit: "cases"}`. A 25-pouch request returns
`outcome: "ambiguous"`, `code: "NEEDS_CLARIFICATION"`, `match: null`, no draft.
Fractional pouch counts, missing quantity, a nonpouch SKU, missing conversion
information or conflicting metadata likewise require clarification.

The ratio comes from `bags_per_case`, `units_per_case`, or case weight divided
by `retail_bag_oz`. Older rows may use one explicit `12x10 OZ` pack specification
in the catalog name, only if its total weight exactly agrees with `case_size_lb`.
All available factors must agree on a positive integer. Quantities are divided
with Decimal and never rounded into a whole case. The strict remaining unit
enum is lb/cases/bags/boxes/each/oz, constrained by product metadata; a missing
unit is always a question, including on each-only service products.

## Seeded migration and future import format

`migrations/060_search_aliases.sql` is idempotent and has **no startup hook**.
It adds `search_aliases`, normalized exact lookup, target/uniqueness constraints,
actor attribution columns and RLS, plus the five owner-approved token seeds:
`SS ⇄ Sunshine`, `BS ⇄ Blue Stripes`, `CLS ⇄ Classic`, `choc ⇄ chocolate`, `#9 ⇄ 9`.
Reruns do not overwrite maintained aliases or reactivate deactivated rows. Existing
customer/supplier alias tables remain intact. The read module does not apply it.

Future seed files use versioned JSON. This is a format example, not the real
shorthand list and not an approved seed:

```json
{
  "version": 1,
  "aliases": [
    {"kind": "token", "alias": "EXAMPLE", "expansion": "Example Brand", "language": "any"},
    {"kind": "product", "alias": "example item", "product_id": 123, "language": "es", "active": true}
  ]
}
```

`resolution.validate_alias_seed(document)` validates and returns insertable rows
without database access. Exactly one target is required: expansion for token,
product_id for product, customer_id for customer, supplier_id for supplier.
Language is any/en/es, active defaults true. Normalized duplicate targets,
unknown fields and missing/conflicting targets are rejected. Database FKs also
validate targets when a future authorized importer inserts the rows. Identity,
normalization and audit fields are generated/set by the database and future
authenticated write path, never trusted from a seed document.

There is no alias import/write/deactivate endpoint in A4 part 1. No unapproved
Spanish/shorthand draft is loaded. Alias writes and resolution logging are
A4 part 2, scheduled after A1 merges; this resolver issues only SELECT statements. Existing
authentication can still perform its usual actor last-used bookkeeping.

## Verification and staging boundaries

Run the Python suite on a **fresh disposable local database** loaded from
`tests/schema/schema.sql` with pg_trgm. The existing staging-seed tests advance
sequences to the synthetic ID range and some existing tests leave committed
fixtures, so reusing an earlier full-suite database can fail that seed test.

```sh
TEST_DATABASE_URL=postgresql://localhost:5432/factory_ledger_a4_test \
  /path/to/python3.12 -m pytest
node --test tests/test_*.js
python scripts/check_resolution_staging.py
```

The staging check validates the protected staging URI and opens an explicit
READ ONLY transaction, invoking the actual resolver/router against staging's
persisted aliases. It never imports `main` or runs startup hooks. The explicit
`--apply-migration` flag applies **only migration 060** in a separate transaction
before acceptance. It never connects to production or deploys the application.

Regression coverage includes ineligible exact lots, alias substrings of every
length, all pseudo-suppliers under mixed whitespace, ingredient bags, literal
supplier LOT suffixes, bulk request length validation, both Classic #9 tiers,
pagination across overlapping aliases, and posted/open-order recency.
Final test counts and staging evidence are recorded in the PR and CHANGE_LOG.

Acceptance verified 2026-10-08 at 10:31 ET after applying migration 060 to
**staging only**: Classic → ambiguous, 11 total, first five IDs
108 / 107 / 284 / 283 / 136; Sunshine 9 → ambiguous, eight SS products
283–290; #9 → ambiguous, regular 107/108 plus SS 283–290; CLS Specialty →
none. Pouch 24 → two cases; 25 → clarification. Checks used the local
core/router against staging in READ ONLY, not a hosted app deployment.
**1,563 Python tests (91 A4) and 69 JavaScript tests passed**; OpenAPI remains
at 30 operations. No production access or merge.
