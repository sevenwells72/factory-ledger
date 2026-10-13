# Reset approval - UNSIGNED TEMPLATE

**Plain-English summary:** This blank template grants no approval. After reviewing the exact v3 preview, its largest changes, row cutoffs, movement classifications and source evidence, the owner signs only the intended product group. A held reset item is not approvable. Packaging and billing items Pallets 102 / Pallet Charge 176 are excluded and cannot be approved for adjustment.

Preview file:
SHA-256:
Approved product IDs:
Owner name:
Named actor:
Date:
Signature:
Opening entry classification: N
Owner approves held entries:
Extra-kosher lots:

Fill every field once. **Owner approves held entries** is `Y` or `N`: `Y` lets the executor, using the named actor's own key, approve exactly the FL holds this preview lists (corrections over 500 lb and extra-kosher loads), each with a note quoting this preview's hash; with `N` a plan containing such entries is refused. **Extra-kosher lots** lists every new extra-kosher lot in the plan as `PRODUCT:LOT` (comma separated, e.g. `283:EK-1014`), or `none`; it must match the plan exactly. Listing a lot is your kosher attestation that the counted stock is genuinely extra-kosher; FL records you as the attesting owner on the lot and its FND receipt. Preview file is the generated JSON filename. SHA-256 is its complete hash. Product IDs are comma separated. Date is YYYY-MM-DD. Signature repeats Owner name exactly. Named actor is the exact FL person associated with the private personal key; shared/master keys are refused. Preserve the signed paper/photo as appropriate; typed text is an attestation, not authentication of handwriting.

Every reset entry posts as an FL ticket under the named actor with reason `physical_count` and a receipt number (ADJ- for existing lots, FND- for new extra-kosher lots). A lot with an open shortage closes it as *counted* in the same post.

Signing approves the original and rounded quantities (four decimal places, ties to even), units, all row/lot cutoffs, Estimated notes, owner movement/duplicate/correction answers, and the source hashes in that exact preview. It also explicitly approves intentional historical posting (`backfill=true`) at the true per-lot count cutoff, and classification N for only the administrative opening entries proved by the executor journal and ledger. It does not approve unrelated new movements or unrecorded lot creation. New activity or changed evidence needs a fresh preview/review.

The floor count covers all inventory, while the reset excludes the 32 packaging/billing products and covers 178 products in the saved catalog; inactive/service and unknown-lot holds still apply. Sunshine ownership must be reconciled after the physical count using records supplied to the office. Coordinate only the approved products during the brief posting/readback interval; production elsewhere may continue. Follow `apply-guide.md`, including the separate small first-live-use check.
