# Visual / mechanical design audit

Renders every screen in [`docs/design/audit/00-screen-inventory.md`](../../docs/design/audit/00-screen-inventory.md)
in a real browser and runs the clauses of the design standards that a rendered page can settle without
judgement. It writes the matrix to [`docs/design/audit/06-browser-check.md`](../../docs/design/audit/06-browser-check.md).

```sh
npm install
npx playwright install chromium
npm run test:visual
```

Narrow a run while iterating:

```sh
npm run test:visual -- --screens S-25,S-42       # named screens only
npm run test:visual -- --variants 1440-dark      # one viewport/theme
npm run test:visual -- --headed --concurrency 1  # watch it drive
```

## What it does

* Serves `dashboard/` from a throwaway local static server — the real files, unmodified.
* Answers every Railway API call from `fixtures/`. **No network request leaves the machine and no database
  is touched.** Writes (`POST`/`PATCH`/`PUT`/`DELETE`) are acknowledged by the stub and applied nowhere.
* Captures each screen at **390px** (screens the inventory marks Mobile *Yes* or *Partial*), at **1440px**
  (every screen), each in light and dark, plus **1440px at 200% zoom** for the Operations, Sales Orders,
  Expected Receipts and Supplies tabs.
* Runs thirteen mechanical checks per capture:
  * **Layout and access** — hit-target size (TOUCH-003), WCAG AA contrast (ACCESS-008), horizontal overflow
    (LAYOUT-003 / ACCESS-001), fixed-bar occlusion (LAYOUT-011), and layout shift on the app's own refresh
    (LAYOUT-020).
  * **Status & data display** (category 17) — coloured nominal badges (STATUS-002), more than one alarm in a
    row (STATUS-004), the chip explanation hook (STATUS-005), number formatting (STATUS-006), orphan dash
    placeholders (STATUS-007), row height at desktop width (STATUS-008), developer vocabulary (STATUS-010),
    and repeated disclaimers (STATUS-011).

  The other six STATUS rules — STATUS-001, -003, -009, -012, -013, -014 — are manual review. They are not in
  `RULES`, do not appear in the matrix, and are never reported as passing.

  **STATUS-005 fails on every chip today, on purpose.** The rule defines the hook (`data-explain` naming the
  explanation element, `aria-describedby` naming the same id, and a focus stop); the product does not have it
  yet. The check is written against the markup the Sales Orders redesign will introduce, so its count is the
  size of that work, and it turns green when the redesign lands rather than needing to be written then.

It reports findings. It fixes nothing and changes no application file.

## Layout

| Path | What it is |
|---|---|
| `run-visual-audit.mjs` | Entry point: variants, job list, verdict thresholds |
| `lib/screens.mjs` | The 91 screens and the recipe that drives the app into each state |
| `lib/checks.mjs` | The in-page measurements |
| `lib/stub.mjs` | Endpoint → fixture routing, and the `{{TODAY}}` date tokens |
| `lib/server.mjs` | Static server over `dashboard/` |
| `lib/report.mjs` | Renders `06-browser-check.md` |
| `fixtures/*.json` | Stubbed API payloads |

## Fixtures

Fixture dates are written as tokens — `{{TODAY}}`, `{{TODAY-3}}`, `{{TODAY+4}}`, `{{TODAY_NAME}}`, `{{NOW}}` —
resolved once per run against the plant timezone (America/New_York). Overdue flags, the rolling 5-day
calendar and "today" are therefore correct on any run day, while the payload stays byte-identical within a
run — which is what LAYOUT-020's before/after comparison depends on.

Adding an endpoint means adding a rule to `ROUTES` in `lib/stub.mjs` and a fixture beside it. An endpoint
with no rule is answered 404 and listed as `unmatchedEndpoints` in the run output.

## Output

* `docs/design/audit/06-browser-check.md` — the matrix. Committed.
* `docs/design/audit/screenshots/<variant>/<S-id>.png` — one viewport capture per screen per variant, with
  the screen's region scrolled into view. Build output, git-ignored, regenerated on every run.
* `docs/design/audit/screenshots/results.json` — every raw measurement, including the findings the matrix
  truncates. Git-ignored.
