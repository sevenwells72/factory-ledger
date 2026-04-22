# gpt-configs/

Holds the canonical sources, generated instruction files, and OpenAPI schemas
for every Factory Ledger Custom GPT. Structured to scale from the current
single-GPT setup to the planned three-GPT split (Floor & Fulfillment, Sales &
Admin, Trace & Recall).

## Layout

- **`sources/`** — Canonical, hand-edited Markdown. `shared-rules.md` plus one
  `{role}-specific.md` per GPT (`floor-specific.md`, later `sales-specific.md`,
  `trace-specific.md`). Edit these.
- **`dist/`** — Generated `GPT_{ROLE}_INSTRUCTIONS.md` files, produced by
  `build_gpt_instructions.py`. **Never hand-edit.** Any manual change will be
  clobbered the next time the script runs — fix the sources instead.
- **`schemas/`** — Per-GPT OpenAPI YAML (`openapi-floor.yaml`, later
  `openapi-sales.yaml`, `openapi-trace.yaml`). Pasted into the GPT's Actions
  panel.
- **`archive/`** — Reserved for retired GPT artifacts. Empty until the current
  production GPT is decommissioned.

## Workflow

1. Edit the appropriate file in `sources/` (either `shared-rules.md` or a
   `{role}-specific.md`).
2. From the repo root, run:
   ```
   python build_gpt_instructions.py floor
   ```
   (or `sales` / `trace` once those sources exist).
3. Copy the contents of `dist/GPT_{ROLE}_INSTRUCTIONS.md` into the GPT's
   Instructions field in the OpenAI admin panel.
4. If the schema changed, paste the corresponding `schemas/openapi-{role}.yaml`
   into the GPT's Actions panel.

The script enforces the 8,000-char OpenAI instructions limit (soft warning at
7,500, hard fail at 8,000) and prepends a `<!-- GENERATED FILE — do not edit. -->`
header to every output.

## Current GPT status

- **Floor & Fulfillment** — live. Sources and schema in this folder.
- **Sales & Admin** — planned, not yet built.
- **Trace & Recall** — planned, not yet built.

## Legacy production GPT

The currently-deployed single GPT is still driven by files at the repo root,
not by anything in this folder:

- `openapi-gpt-v3.yaml` — live schema (30-operation cap, do not exceed).
- `GPT_INSTRUCTIONS.md`, `gpt-instructions-v3.md` — legacy instruction files.

These stay at the repo root until the three new GPTs fully replace the current
one. At that point they move into `archive/` and this folder becomes the only
source of truth.
