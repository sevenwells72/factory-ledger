# Ingredient use, batch output and packing yield: recommendation

Status 2026-11-02: **recommendation only — nothing implemented.** No recipe, yield,
inventory calculation or approval rule changes until Michael decides. Context:
[formula-entry-staging-followups-2026-11-02.md](formula-entry-staging-followups-2026-11-02.md) §5.

## How FL counts today

| Step | What leaves stock | What enters stock | Where the number comes from |
|---|---|---|---|
| **Make** (formula entry, PR #173) | Ingredient lots: the **actual pounds** the worker confirms per bag | Batch (WIP) lot: **pans × `default_batch_lb`** (323 for Classic #9) | Inputs are entered; output is a standard, never weighed |
| **Pack** | Batch lot: **cases × case weight** | Finished lot: the cases | Case weight from the product (10 or 25 lb) |

Three numbers that should agree, but are never compared:

1. **Ingredient consumption**: what left ingredient lots (MK-261102-001: 670.2 lb).
2. **Batch output**: what entered the batch lot (MK-261102-001: 646 lb = 2 × 323).
3. **Packing yield**: what the batch became in cases (Classic #9: 32 × 10 lb = 320 per pan, or 13 × 25 lb = 325).

So today every difference either disappears (input vs output: 24.2 lb on MK-261102-001 is
counted nowhere) or piles up silently in the batch lot (output 323 vs 10-lb pack 320:
+3 lb per pan stays as WIP that does not physically exist).

## Recommendation

**Principle:** record what was measured. Label every standard number as a standard, not a
weighing. Make each difference a named, reasoned number someone reviews. Don't let gaps
hide in a lot balance. Inventory stays the sum of posted lines, and nothing is ever edited.

### 1. Treat the three differences separately

| Difference | Meaning | How FL should handle it |
|---|---|---|
| **Recipe variance**: inputs vs recipe × pans | Worker used more or less of an ingredient | Already built (PR #173): reason or tick, flag on shift summary. Keep. |
| **Bake yield**: batch output vs total inputs | Moisture/bake loss, fines, spills | A **reported metric per make**, not an inventory movement: inputs leave their lots, the output lot gets what was produced, and FL shows the difference against an **expected bake yield %** per recipe. Outside the tolerance it gets a flag (like the 10% rule). |
| **Pack variance**: batch lot used vs cases × case weight | Giveaway, fines, rework, scale drift | When a batch lot is **finished**, the remainder (+ or −) is closed with an explicit, reasoned adjustment ("pack loss / fines", "overweight giveaway"), so WIP never keeps phantom pounds. |

### 2. Settle one set of standards per recipe (owner decision, master data)

For Classic #9 today: recipe 322.6, pan weight 323, 10-lb pack 320, 25-lb pack 325. These
should become one consistent, versioned set per recipe:

- **Expected inputs** = the recipe (stays as is).
- **Expected batch output per pan** = what one pan really weighs after baking (measure a few
  pans; may be below 322.6 if there is bake loss).
- **Cases per pan** follow from the output (e.g. 320 lb → 32 × 10 lb; 25-lb packing then
  plans 12 cases + 20 lb rework, not 13).

Store these with the recipe versions planned for formula-entry Phase B, so old makes keep the
standards they were recorded against.

### 3. Optional measured output on Make

Keep "pans × standard" as the default (no extra work on the floor). Add an optional
**"actual batch weight"** field the floor fills when a batch is weighed or clearly differs.
The receipt shows which it was ("323 lb standard" vs "318 lb weighed"). That's an assumption
labeled as an assumption, per the operating principles.

### 4. Review, not blocking

Like the 10% ingredient rule, yield and pack differences are **flags on the shift summary /
weekly review**, with the reason and who recorded it. They never block posting. Thresholds
are owner-set (e.g. bake yield outside expected ±3%, pack remainder over 1% of the lot).

## Phased path (each phase needs your approval; staging first)

| Phase | What | Changes inventory logic? |
|---|---|---|
| **1. Report only** | A daily/weekly yield report from existing posted data: per make, inputs vs output; per batch lot, output vs packed vs remaining. Shows the size of the problem with real numbers. | No |
| **2. Standards** | One owner-approved set of pan weight / cases per pan per recipe (master-data change, backup first). | No (numbers only) |
| **3. Measured output** | Optional actual batch weight on Make; standard stays the default. | Yes, small: output can be a weighed value |
| **4. Batch close-out** | At the last pack from a batch lot, FL proposes the reasoned remainder adjustment for the floor/owner to confirm. | Yes: new reasoned adjustments, never edits |

Suggested first step: **Phase 1**. It is read-only, needs no rule change, and will show
whether the gaps are a few pounds per pan or more before anything is decided.

## Questions for you

1. What does one baked pan of Classic #9 actually weigh? Can the floor weigh a few?
2. Is ~1% loss from batch to cases normal and expected, or a sign a number is wrong?
3. Should a batch lot's leftover pounds be closed automatically at the last pack (with a
   reason), or reviewed by you first?
4. Do you want Phase 1 (report only) built next?
