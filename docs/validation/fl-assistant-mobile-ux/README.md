# FL Assistant mobile UX round (2026-10-16)

Michael's iPhone testing on staging. Investigated first: code, then WebKit at 375/390/393/430 px against a local
server over a disposable copy of staging data (`fl_mobile_*` databases), using Michael's own 17-turn staging chat.

## Findings and changes

| # | Broken or confusing | Root cause | Change |
|---|---|---|---|
| 1 | **Broken.** The chat opened at the oldest message: scroll 0 of about 32,000 px (around 38 screens) in all four widths. The first thing shown was Oct 15's failed “classic number nine” answer. Reload resumed the same chat, so history was kept, but always at the top. | The page appended the saved turns and never scrolled. The header scrolled away with the page. | The page opens at the latest message, and every reply is brought into view (a reply taller than the screen opens at its start). A **Jump to latest ↓** button appears only after scrolling up more than 300 px. Plant-time dividers (**Yesterday**, **Yesterday · 7:03 PM**, **Today · 7:51 AM**, EN/ES) mark each day and any gap of an hour or more. `/assistant/resume` now returns each turn's `created_at`. Chats still resume by saved id; only **New chat** creates a new one. The header stays on screen on phones. |
| 2 | **Broken (looked like nothing happened).** Today *did* load (147 entries on the copy), but added its card about 10,000 px below the screen. | Same missing scroll: the result was appended off screen. | Tapping **Today** shows the loading card and then the entries on screen, from the start of the card. The entries are part of the chat, so returning is just scrolling or **Jump to latest ↓**. The header was not crowded enough to justify moving Today into the Menu. |
| 3 | **Confusing, plus two real failures.** The upload itself works: Michael's 2.8 MB `image.jpg` reached staging at 07:50:57. But (a) the type came from a dropdown pre-set to *Lot tag*, so a BOL photo could silently go up as a lot tag, and only BOL/truck/packing-slip photos are attached to a shipment; (b) a photo over 5 MB failed with “Photo upload failed. Please try again”, which a retry can never fix; (c) after sending “Photo”, the reply “I can only record or look up…” said nothing about the photo; (d) Send with a photo and no words silently did nothing. | The upload started on pick, with no progress and no states. The server's 5 MB limit gave the same message as a network failure. | Pick → thumbnail plus **What does this photo show?** with plain buttons (Lot tag, BOL (bill of lading), Truck, Signed packing slip, Count sheet, Correction, Other). Nothing uploads before a type is chosen. Then **Uploading 45%…** → **Attached ✓ · goes with your message**, or **Upload failed — tap to retry** (network or server), **Photo too large (max 5 MB)…** or **Not a photo…**. × removes the photo. Photos over 5 MB, or in a type FL can't store, are re-encoded on the phone to JPEG (at most 3000 px), so a 15.4 MB photo arrives as 2.6 MB. Send is held with a plain message while a photo has no type or is still uploading. A reply that is not a draft reminds the worker that the photo is still attached and what to type. The draft shows the photo as a thumbnail (tap to enlarge) before Record. All text is EN and ES. |
| 4 | **Historical; no change.** | The failures are from Oct 15, 13:59–14:37 EDT. Round C merged at 15:41 EDT; the same question in the same chat succeeded at 19:03. | None. On the staging copy, with the real model (`gpt-6.1-sol`) and 3 fresh chats, all 3 runs answered “How much classic number nine do we have?” with Batch Classic Granola #9 (SKU 90002) and its finished products (the model searched “Classic #9”, about 3 s each). |

Unchanged backend behavior, reported only: a **Lot tag** (also Count sheet / Other) photo stays with the chat and its
draft but is **not** linked to the receipt. FL links BOL/truck/signed-slip photos to shipments and one correction photo to
adjust/found (`evidence_fields`). On the local copy the receive was recorded as RCV-261016-054 and the lot-tag
photo's `ticket_id` stayed empty.

## Validation

- WebKit, real local server, staging-data copy: all four widths for items 1–3. Full photo path: real upload →
  real-model receive draft with the photo → Record **on the local copy only** → receipt. Errors: 15 MB (shrunk and
  attached), fake JPEG (Not a photo), network drop (retry succeeds), server 500 (retry), proxy 413 (too large),
  Spanish. Screenshots are kept locally, not committed, because they show staging data.
- New fixture runner `tests/visual/run-fl-assistant-mobile-ux.mjs`: WebKit, both session modes, EN/ES,
  375/390/393/430 px, **144 checks**; it fails on unchanged main. Screenshots for sessions-on are in `browser/`.
- New tests: `tests/test_assistant_mobile_ux.js` (6), `tests/test_assistant_mobile_ux.py` (4).
- `tests/visual/run-fl-assistant.mjs` photo steps now pick first, then choose the type.
