# High-priority findings: live recheck

September 8, 2026, approximately 3:32 PM ET. Rechecked the public deployment at https://cns-factory-ledger.netlify.app/ after a browser reload, using 850×666, 1440×900, and representative 390×844 views. Findings refer to the numbering in the original audit.

**Result: none of the previously high-priority findings can be closed from the version served during this recheck.** This does not establish whether the changes exist locally, on another deployment, or in a newer build not served to this session. Updated receipt data was visible.

| Original finding | Status | Fresh evidence |
|---|---|---|
| 1. Quantity/unit mismatch | Still present | Recent Entries: Honey Nut shipment `−1,000 25 lb case`; Activity: same product/time `1,000 lb · 40 units`. Kookies & Kreme remains `−3,000 25 lb case` versus `3,000 lb · 120 units`. |
| 2. Narrow desktop header | Still present | At 850 px wide, global search still measures 26 px wide and Refresh is clipped at the right edge. |
| 3. Phone navigation | Still present | At 390 px, seven dashboard sections remain in a horizontally scrolling strip, with later destinations offscreen. |
| 4. Task-first mobile home | Recommendation still open | Calendar, header, search, tabs, and attention cards occupy the first screen; Today So Far is below it. No frequent floor-action shortcuts appear. This is a role-dependent recommendation, not proof that this dashboard must itself implement every floor workflow. |
| 5. Target sizes | Still present | Order expand buttons measure 22×22 px. Trace recent-lot chips remain 21 px high. |
| 6. Global search keyboard selection | Still present | After result for SKU 70060 appeared, Arrow Down then Enter did not open a dialog or navigate. Result still appears as generic content. |
| 7. Trace recent-lot shortcuts | Still present | Sample recent-lot elements remain spans without role/tabindex. SEP 03 2026 appears for two products differentiated by title tooltips. |
| 8. Accessible labels | Still present in checked examples | Order row checkboxes are unnamed in the accessibility tree; visible readiness heading is blank. Trace +/− controls still lack descriptive aria-labels and title tooltips. Notes controls were not separately repeated in this focused pass. |
| 9. Empty receipt Save | Still present | New Expected Receipt has blank product, supplier, and quantity; Save reports enabled. No submission was attempted, so this is specifically a gating failure, not evidence that the server accepts invalid records. |
| 10. READY/BLOCKED clarity | Still present | SO-260814-002 displays READY and BLOCKED. Expanded SO-260629-003 still says shipping is not gated by the computed status. |
| 11. Dispatch attention count | Still present | On returning to Operations after order data loaded, Dispatch blocked remains a dash rather than a resolved count. |
| 12. Health score | Still present | Header still displays bare 80; its tooltip contains internal issue names rather than a visible, actionable issue summary. |
| 13. Trace contrast | Still present | Completed trace button remains white text on rgb(96,165,250), approximately 2.54:1 contrast. |
| 16. Narrow sales-order layout | Still present | At 850 px, SO identifiers wrap to three lines, customer content wraps heavily, and right-hand columns are outside the visible region. |
| 31. Material Flow readability | Still present | At 1440×900, finished-goods and customer labels collide. At 390×844, column headings collide and much of the graph extends beyond the viewport. |

The current live behavior supports retaining these findings. Deployment/version verification is the next useful step if the intended updates address these specific items. No business records were changed; the dashboard and original viewport were restored.
