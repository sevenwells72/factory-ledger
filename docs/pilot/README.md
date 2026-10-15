# Pilot playbook

## Key dates

| Milestone | Date (2026, New York) | What happens |
|---|---|---|
| Pilot start | Wed Oct 21 | Arturo and Luz practice on staging for part of each day. |
| Clean-summary and dry-run window | Oct 26–30 | Five clean shift summaries plus the production dry run. |
| Packet printing | Fri Oct 30 | Print approved count packets. |
| Go / no-go | Mon Nov 2 | Michael decides from the evidence. |
| Physical count and signing | Thu Nov 5 | Count while production runs; Michael signs the approval list that evening. |
| Reset and cutover | Fri Nov 6 | Each area pauses 15–30 minutes; whole-reset proof before release; rotate master key; GPTs become read-only. |
| A10 stage 1 | Mon Nov 9 | Retire the approved old staff write paths. |
| First fallback | Fri Nov 13 | Cutover only after renewed readiness approval. |
| Second fallback | Fri Nov 20 | Cutover only after renewed readiness approval. |
| M0 and GPT retirement deadline | Before Fri Dec 11 | M0 read-only plugin live; old GPTs retired. |

The **Key dates** table above is this document's schedule. The other documents each keep their dates in their own top table; all instructions refer to milestone names.

| File | For | What |
|---|---|---|
| [Pilot plan](PILOT-PLAN.md) | Michael | Part-day staging practice, acceptance checks, readiness gate and open questions |
| [Floor guide](ARTURO-GUIDE.md) | Arturo | Short English / Spanish instructions side by side |
| [Office guide](OFFICE-GUIDE.md) | Luz and Miriam | Short English / Spanish instructions and office limits |
| [Daily check](MICHAEL-DAILY-CHECK.md) | Michael | Ten-minute review and escalation rules |
| [Floor quick card](cards/arturo-quick-card.pdf) · [Office quick card](cards/office-quick-card.pdf) | Print | Two-page Letter cards, English / Spanish side by side |

The owner's settled decisions govern these guides. They describe required behavior; the pilot must verify it on the accepted deployment. Historical PR comments about an unbuilt office screen, a second shift-summary sign-in or the old schedule do not override these instructions. Production readiness is tracked in [the master runbook PR](https://github.com/sevenwells72/factory-ledger/pull/117).

No receipt = not recorded. Every write is **prepare → Record → receipt**. PIN login stays off; personal sign-in lasts until sign-out or 180 minutes of inactivity. Background refresh does not extend it.

Rebuild both PDFs after any card edit, from `docs/pilot/cards/`:

```sh
for card in arturo-quick-card office-quick-card; do
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --no-pdf-header-footer \
    --print-to-pdf="$PWD/$card.pdf" "file://$PWD/$card.html"
done
```

Render and inspect **every PDF page**, including Spanish accents and page edges. Keep each card at two readable pages; do not shrink it into an unreadable single page.
