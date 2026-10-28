# C. Interface capability matrix (verified 2026-10-28)

## Legend

| Mark | Meaning |
|---|---|
| ✅ | **Works in production today** |
| 👁 | **Read-only** (can look, cannot change) |
| 🔍 | **Preview only** (shows what would happen, records nothing) |
| 🟡 | **Built but switched off in production**; works in **staging** |
| ❌ | **Not available** in this interface |
| ❓ | **Not verified** |
| ⚠ | Works, but with an important caveat (see notes) |

"Dashboard" means the Netlify website. Its **main tabs** are visible in production today. Its **Record entry / Office / Exceptions / Receipt-void / Shift Summary** screens are 🟡: hidden until personal sign-in is switched on.

## Matrix

| Task | Floor GPT | Office GPT | FL Assistant | Dashboard | Other ways |
|---|---|---|---|---|---|
| Receive ingredients / supplies | ✅ | ✅ | 🟡 | 🟡 Record entry → Receive | — |
| Record granola production | ✅ `make` | ✅ `make` | 🟡 | 🟡 Record entry → Make | — |
| Record coconut production | ✅ `make` (same action, coconut product) | ✅ | 🟡 | 🟡 Record entry → Make | No coconut-specific form anywhere |
| Record finished-goods packing | ✅ | ✅ | 🟡 | 🟡 Record entry → Pack | — |
| Ship customer orders | ✅ preview → commit, duplicate guard | ✅ ⚠ preview → commit | 🟡 | 🔍 capacity preview on the order today; 🟡 Record entry → Ship | — |
| Create / print packing slips | ⚠ 10-minute link returned on commit (not yet seen on a real shipment) | ⚠ static link containing a key | ❌ link created but not displayed | ❌ no packing-slip button (only "attach signed slip photo") | Back end has a packing-slip page per order |
| Physical inventory counts | ⚠ only as an adjustment | ⚠ only as an adjustment | 🟡 as an adjustment | 🟡 Adjust ("Physical count") or Exceptions → "Counted" | **No count feature.** Count day used a Google Form, loaded by Michael. |
| Inventory adjustments | ✅ ⚠ no owner-approval rule | ✅ ⚠ no role or approval rule | 🟡 | 🟡 Record entry → Adjust | — |
| Record Found inventory | ❌ | ❌ | 🟡 | 🟡 Record entry → Found | Today only through an engineering session using Michael's owner key |
| Correct / void a wrong entry | ✅ void (by transaction ID) | ❌ | ❌ sends you to the Receipt page | 🟡 Receipt page → "Prepare void" | Corrections otherwise = an adjustment |
| Review inventory by product and supplier lot | 👁 | 👁 | 🟡 👁 | ✅ 👁 Operations stock panels, lot detail, Supplies, Traceability | — |
| Review sales orders | 👁 | ✅ review and edit | 🟡 review; create (office/owner) | ✅ Sales Orders tab ⚠ edits saved under the shared dashboard identity | — |
| Create / change sales orders | ❌ (status only) | ✅ ⚠ no preview, auto-creates from uploaded PO | 🟡 create; changes sent to Office | ✅ ⚠ from PO photo, edit, close, cancel (shared identity) · 🟡 Office screen (named) | — |
| Review expected receipts | ❌ | ❌ (create only) | ❌ sends you to Office | ✅ Expected Receipts tab (create, edit, cancel; shared identity) | — |
| View production schedule | ❌ | ❌ | ❌ | ✅ Runs page (live, **no sign it is used**; 0 schedule rows) · 👁 Production Calendar = what was made | Scheduler board: offline simulator, not linked, not live data |
| Review exceptions and alerts | ❌ | ❌ | ❌ | ✅ 👁 "Needs Attention" chips · 🟡 Exceptions page (resolve / approve) | Michael's 65 approvals on Oct 22–27 were done through engineering sessions |
| Review daily production and inventory activity | 👁 day summary, history | 👁 day summary, history | 🟡 👁 "Today's entries" | ✅ 👁 Activity, Recent Entries, History, Receipt "Today" | — |
| End-of-day reconciliation | ❌ | ❌ | ❌ | 🟡 Shift Summary page (confirm / "something is missing"); not linked in any menu | No count-vs-ledger reconciliation screen exists |

### Notes on the ⚠ marks

1. **Office GPT shipping.** The new duplicate-shipment guard (#153) protects repeated "commit" calls on order shipments. That is verified for the Floor GPT's `commitShipOrder`. The Office GPT uses the same order-ship route with `mode: commit`, and the guard code sits on that shared commit path, but **this was not separately tested (❓)**.
2. **Adjustments through the GPTs** use the older direct route. That route **does not apply** the rule "over 500 lb needs a photo or owner approval", the role limits, or the late-entry exception. With the Office GPT, the office can therefore adjust inventory, which its role would forbid on the dashboard.
3. **Dashboard tab edits today** are saved with the shared dashboard key, so History will not show who made them.
4. **Packing slip:** see Issue F-6 in [07](07-issues-and-recommendations.md).

## Do all the interfaces create the same records?

**Yes, at the ledger level.** Every interface ends up in the same back-end "posting" code:

- a receive from the Floor GPT and a receive from the FL Assistant produce the same kind of transaction, lot and receipt number;
- both appear in the same History and stock figures;
- both apply supplier-lot capture (#145), kosher rules and the pooled-lot guard (#154).

**No, at the control level.** There are two doors into the same posting code:

| | "Direct" door (Floor GPT, Office GPT) | "Draft → Record" door (FL Assistant, Record entry, Office, Exceptions, Receipt void) |
|---|---|---|
| Who is recorded | `legacy-shared-key` (nobody) | The signed-in person |
| What protects against an accidental write | The GPT's instructions ("preview, then commit"). The server does not enforce a confirmation step. | The server issues a draft; nothing posts until the same person presses Record within 10 minutes (30 on the dashboard) |
| Role limits (floor / office / owner) | Not applied: the master key can do everything | Applied when the draft is made and again at Record |
| Over 500 lb correction → photo or owner approval | Not applied | Applied; held for the owner if there is no photo |
| Late entry (2–14 days) → owner exception | Not applied | Applied |
| Possible-duplicate warnings that must be ticked | Order shipments only (#153 retry guard) | Receive (same product, lb or lot within 24 h), make/pack (45 min), adjust/found (2 h), shipment (24 h), orders (duplicate customer PO) |
| A repeated "Record" press | Order shipments: replays the first receipt (#153). Other entries: **no guard (❓)** | Always returns the same receipt ("Same entry, same receipt") |

The plan was to retire the direct door ("A10", target Nov 23) once nobody depends on it. Until then, the GPTs remain the least-protected way to write.
