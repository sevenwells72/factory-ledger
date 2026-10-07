# V3 live-count office guide

**Plain-English summary:** Use the v3 floor packet now. Production continues. The office compares each item/lot/location row at its own time, reviews every relevant movement, and prepares a reset as soon as the count is reviewed. There is no fixed waiting period or apply-age deadline. The October 5 ingredient-total approach is superseded; no ingredient-total reset mode was built.

Moves are caught by the tool only when logged, tagged, or noted; for anything else, the H1 multi-area hold is the control. A note or tag is evidence for the transcriber and owner to review, not automatic detection of an unlogged physical move.

## Transcriber: duplicate tick column (option A)

Use the explicit `duplicate` tick column on the count sheet. Tick `x`, `X`, `✓`, `✔` or `☑` only on rows suspected of being duplicate observations; leave other rows blank. Notes, including “possible duplicate”, do not mark duplicates, and sharing a tag does not mark a row. Preserve notes as evidence and ask the owner to tick any intended duplicate on the source sheet. Do not normalize or interpret phrases as ticks.

Generate a new count-sheet CSV template, including the duplicate tick column, without a database snapshot:

```bash
python audits/fresh-start/fresh_start_v3.py \
  --write-count-template audits/fresh-start/v3/count-inputs/count-template.csv
```

Create the inputs folder first. The command refuses to overwrite an existing file. Previously issued CSVs without the column still load, with every duplicate tick treated as blank. Add the column to a transcription copy when needed; retain the original. The older PDF/material builder does not issue this new column: use the template command above and add a labeled duplicate tick column to paper copies.

This replaces v2's single-cutoff, non-production-day workflow. The old v1/v2 sheets and dated reports remain historical evidence, not current floor instructions. All 210 catalog products remain in v3's floor-count scope, including inactive products formerly excluded as 171/209. The reset covers 178 products; the 32 packaging/billing products are information only, including Pallets 102 and Pallet Charge 176. Inactive/service or unresolved identities cannot receive adjustments. The earlier owner-approved direct database write archiving product 185 remains documented in `archive-185-execution.txt`; it was not repeated or changed here.

## Files to fill

Copy issued blanks into a new `audits/fresh-start/v3/count-inputs/` folder before typing. Keep the original sheets/photos. Do not edit issued blank files or any generated preview.

- `count-sheet-v3.csv`: fill actual area, date, counter, start/end and the row fields. Repeat a sheet's same header values on its used rows. Row time is HH:MM (or explicit offset timestamp), falling back to sheet end. Use a separate row for each lot/location/container observation. Assign new unique sheet/row IDs for extra pages. Leave unused rows blank; they never mean zero.
- `sheet-register-v3.csv`: track all issued/returned sheets and actual areas. The office checks this register against photos and product coverage; it is a completeness aid, not a quantity-adjustment input.
- `count-coverage-v3.csv`: mark `all_locations_searched=yes`, actual `completed_at` (ISO timestamp with offset), owner initials and notes only when **every location for that product has been searched**. This certifies absence for unlisted known lots at that completion time. A reset-scope product absent or incomplete here stays held; nobody has to guess zero on blank floor rows.
- `moved-during-count-v3.csv`: enter every logged cross-area move, including actual from/to area and location, item, lot, native quantity/unit, time and tag. `moved_at` uses an explicit offset timestamp. Leave `fl_transaction_id` blank for a location-only move if FL has no appropriate event; do not manufacture a receipt/adjustment. A referenced internal-move transaction must net to zero for that lot.

Move-log rows with a blank or non-catalog product ID appear in `move-follow-up.csv` and the preview/verification's move-log follow-up section, with the source CSV row reference and “move-log row has no valid product_id”. Correct the source row and rerun. These follow-ups do not create global or unrelated-product holds; they do not certify the unidentified move as reviewed.

The tool and its snapshot guard resolve paths from the checked-out source file, so a clone or Git worktree uses its own `audits/fresh-start/` inputs. The committed `v3/delivery-sha256.json` covers the delivered fresh-start files (excluding the manifest itself); historical scratch reports are not part of this package.

**Packaging is information only.** Keep packaging observations on the same floor sheets. Preview and verification write a separate `packaging-information-only.md` / `.csv` with the entered quantities, units, lots and locations, plus `packaging-moves-information-only.csv`. Missing quantities remain not counted; no packaging balance or adjustment is inferred. Packaging coverage and movement problems do not hold up the reset. Pallets 102 and Pallet Charge 176 are excluded billing items. For uncatalogued labels or other confirmed packaging, the office may set `row_kind=PACKAGING_INFO`, leaving product ID blank and explaining the classification in notes on an input copy. Preserve the original sheet; never assign that mark to an unidentified food item. Catalogued food cannot bypass reconciliation this way.

FL has no current location balance. A row's FL reference is the whole-lot balance at that row time, not a fictitious balance for that shelf. Separate area observations are combined only after relevant movements are classified and allocated once.

## Create and review a preview

Run from the authorized checkout. The following is an example using future filled files, not a command run in production during this preparation:

```sh
python3 -B audits/fresh-start/reset_preview.py --v3 \
  audits/fresh-start/v3/count-inputs/count.csv \
  --coverage audits/fresh-start/v3/count-inputs/coverage.csv \
  --moves audits/fresh-start/v3/count-inputs/moves.csv \
  --output-dir audits/fresh-start/v3/review-01
```

The default reads fresh FL through `psql_ro.sh`, port 5432, wrapped in `BEGIN; SET TRANSACTION READ ONLY; ... ROLLBACK;`. No credential is written or printed. An optional `--snapshot` accepts a saved read-only snapshot for offline review, clearly dated in the report. Do not approve stale saved evidence without the executor's fresh recheck.

The report prominently shows **time since count**. Outputs include per-area reconciliation, per-row cutoffs, complete later movement lists, proposed per-lot adjustments, unidentified-lot seven-day follow-ups, and a versioned `owner-review-template.json`. Copy the template to the inputs folder and complete the owner's answers there. Rerun with `--review path/to/owner-review.json`. No template rerun overwrites existing answers.

Movement review rules:

1. **Every movement inside the sheet start/end window or within 30 minutes either side of a row time** requires owner classification `before count` or `after count` for each affected review row. Posted lines determine balances; voided lines remain visible for review with no quantity effect.
2. For a lot observed more than once at different times, all movements between earliest/latest count times are reviewed. Because FL does not identify source/destination shelves, give `allocations` mapping row IDs to signed native quantities, summing exactly to the movement quantity. Use the matching before/after answers. Never allocate the full amount to every row. Outside ambiguous windows, later whole-lot events are carried forward once.
3. Late-entry clause 1 uses **entry time**: `entered >= count_start` and `entered - occurred_at > 15 minutes`. An occurrence before count start can therefore be late; entry before count start or a gap of exactly 15 minutes does not meet clause 1. Independently, clause 2 reviews entries typed after sheet end whose occurred time is no later than sheet end plus the post-sheet margin. That margin is the maximum of the existing margin (30 minutes), 30 minutes, and half the sheet's duration: a 10-minute sheet gets 30 minutes, a 3-hour sheet gets 90 minutes, and a larger existing margin wins. Set `late_confirmed=true` only after owner confirmation. The movement template includes `type`, `adjust_reason` and `operator_id`. Corrections during/after count need explicit owner confirmation too. Fingerprints bind answers to exact evidence; changing it requires review again.
4. For each move log entry, identify `source_row`, `destination_row`, and whether each count physically included the moved amount. A counted-to-uncounted move counted at both ends subtracts the duplicated amount; an uncounted-to-counted move counted at neither adds the omitted amount. The table exposes the correction. These are local reconciliation calculations, never FL transfer/adjustment writes. Unresolved endpoints, quantities or locations hold the product. **A move-log row whose destination is a consumed lot stays held across reruns until that row is removed from the move log.** Owner notes cannot override a missing destination count or a consumed destination. Preserve the original evidence and retain the actual consumption in the ordinary FL movement review; do not invent a destination count to clear the hold.
5. Only rows with a duplicate tick require duplicate disposition: distinct stock, explicitly excluded ticked row(s), or a link to the resolved move involving those same rows. Tags group the evidence; untagged rows group by product/lot. At least one row always remains in each group. If the owner excludes every row, the first row is retained and the group stays held with that fact recorded. Source count rows remain unaltered. Unidentified lots stay held; do not create or guess lots. Follow up within seven days of their actual count.
6. Every lot observed in more than one area gets **one per-lot hold**, listing all observed row references: `owner confirms no unlogged moves between rows R1/R2`. Complete that lot's `multi_area_reviews` entry with `owner` and `no_unlogged_moves=true` only after checking every row. Logged/reviewed moves do not clear this separate control. Changes to the row evidence invalidate the answer.
7. Row problems, including blank products and unusable identities, appear in `sheet-follow-up.csv` and the corresponding sheet's `follow_up` list. Rows on another sheet do not enter that sheet's follow-up or `general`. Sheets with only invalid rows still appear, and unresolved sheet follow-ups block full-scope sign-off. Unknown-product rows hold lots counted on the same sheet; other sheets keep their own follow-ups. Known-product row errors also hold that product's lots. A product certified covered with no usable count rows gets its own `coverage_reviews` entry, `coverage certified but no count rows`; the owner must explicitly confirm it, even if the product has no lots.
8. Every matching `OPENING BALANCE … | v3 … L{lot_id}` adjustment is held as `prior opening without journal proof`, including entries with an older count hash. Only the existing journal verification can prove an executor opening; an ordinary movement answer cannot clear this hold.

For Sunshine pouch products, the office-only `ownership_reviews` answer must cite evidence and confirm all counted stock is CNS-owned before an ordinary reset can be approved. If any is Sunshine-owned or unresolved, retain the physical count and hold the product for supported ownership treatment; never change the count to hide it.

The owner is responsible for factual classifications; initials are an attestation, not proof of handwriting or evidence accuracy. Preserve photos, movement log and receiving/production documents with the signed preview.

## Quantities and approvals

Ingredients and batches use lb, finished cases use approved case weights, and product 291 uses containers. Packaging observations retain their entered units only in the separate information report; they never enter reset arithmetic. The legacy internal `_lb` calculation-field names in the JSON are retained for the shared executor, but `ledger_unit` explicitly governs the quantity; public CSV/Markdown tables label native units. No packaging value enters an adjustment or pound total. Approved Blue Stripes case weights and existing unresolved package holds remain in force.

`Estimated` carries from the sheet into the lot preview and exact adjustment reason. An unresolved or unusable lot/unit holds the product. Multiple location observations yield **one adjustment per existing lot**, dated to its latest row count; its target is computed from **all underlying row cutoffs**, reviewed allocations, and internal-move corrections. It is not calculated from a universal cutoff or ingredient-total allocation rule.

A pre-decision v3 preview must be regenerated and reapproved; apply rejects an obsolete scope-policy marker or any packaging/billing reset row, including zero-change rows. Fresh apply preflight checks the current catalog exclusion as well.

A signed v3 JSON preview includes the exact source hashes, owner answers, per-row cutoffs, rounded changes, native units and intentional historical posting (`backfill=true`). Historical posting preserves true count times if review takes longer; it is not permission to fabricate dates. The current application already supports this flag. No application/migration change was made.

Use `apply-guide.md` and a fresh signed `approval.md`. The only production-capable tool is `apply_reset.py`; it was not run against production in this preparation. Full reset scope is not signed off until all 178 reset products are covered and every reset hold is resolved. Packaging/billing counts are excluded from this verdict; completion of the informational floor count is tracked separately. No unknown stock is silently dropped. Sunshine custody/ownership review remains separate; see `v3/sunshine-ownership-reconciliation.md`.

## Verify

Automatic verification follows the executor's proved journal entries. Independent verification remains read-only and requires the original signed preview and journal so administrative opening entries are not mistaken for new physical movements:

```sh
python3 -B audits/fresh-start/verify_reset.py --v3 \
  audits/fresh-start/v3/count-inputs/count.csv \
  --coverage audits/fresh-start/v3/count-inputs/coverage.csv \
  --moves audits/fresh-start/v3/count-inputs/moves.csv \
  --review audits/fresh-start/v3/count-inputs/owner-review.json \
  --approved-preview audits/fresh-start/v3/approved/reset-preview-v3.json \
  --approval audits/fresh-start/v3/approved/approval.md \
  --journal audits/fresh-start/apply-state/PREVIEW_SHA256/journal.jsonl \
  --output-dir audits/fresh-start/v3/verification
```

Replace every example path/hash with the actual files. Before any apply, verification without journal flags is a valid read-only comparison. After an apply, unproved opening entries cause a hold. Verification returns 0 only for full reset scope with no unexplained differences; otherwise 2. Newly entered ordinary movements remain visible and must be reviewed when they meet the live-count rules. A new read-only report does not authorize another write.
