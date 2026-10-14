# G1 cutover checklist — make both GPTs read-only

**What this is:** the exact steps for design §10.1 **steps 3 and 4** on cutover day
(`docs/design/phase1-safe-operating-system.md`). Everything else in §10.1 is
unchanged. Each step is **owner-approved before it runs**.

**What you need open:**

- Railway → project **gleaming-solace** → service **FastAPI** (production).
- ChatGPT (chatgpt.com), signed in as the account that owns both GPTs.
- A terminal on your Mac.
- Your password manager.
- This folder in the repo (`gpt-configs/readonly/`) on `main` — four files:

| File | Goes into |
|---|---|
| `office-instructions.md` | Office GPT → Configure → **Instructions** |
| `openapi-office-readonly.yaml` | Office GPT → Configure → Actions → **Schema** |
| `floor-instructions.md` | Floor GPT → Configure → **Instructions** |
| `openapi-floor-readonly.yaml` | Floor GPT → Configure → Actions → **Schema** |

**Before cutover day (check once):** the G1 PR is merged and the production
FastAPI service is running a `main` commit that includes it. Step A4 below is
the proof: if the new key gets "Invalid API key" instead of an answer, the code
is not live yet — stop.

**Order on the day:** §10.1 step 2 (rotate the master `API_KEY`) cuts both GPTs
off, because they still hold the old master key. Do steps A and B right after
it so the GPTs are only unable to answer for a few minutes.

---

## A. Issue the read-only key (§10.1 step 3)

1. **Make the key on your Mac.** In Terminal run:
   `python3 -c "import secrets; print('flro_' + secrets.token_urlsafe(32))"`
   Copy the output into your password manager as **"FL READONLY_API_KEY
   (GPTs)"**. Do not paste it into chat, e-mail, Slack, a GitHub comment or any
   file in the repo.
2. **Put it on Railway.** Railway → **gleaming-solace** → **FastAPI** (not
   FastAPI-staging) → **Variables** tab → **+ New Variable** →
   name `READONLY_API_KEY`, value = the key → **Add**. The value must be
   different from `API_KEY`, `DASHBOARD_API_KEY` and every active person's
   actor key (the service refuses to start otherwise — by design). A freshly
   generated key from A1 always is; never reuse an existing key here.
3. **Deploy it.** Railway shows "Apply changes" / "Deploy" at the top of the
   canvas → click it. Wait until the deployment shows **Active / Success**
   (about 2–3 minutes).
4. **Prove it on the server** (Terminal; the key is typed into a hidden prompt,
   so it never lands in your shell history):
   ```
   read -s RO; echo
   API=https://fastapi-production-b73a.up.railway.app
   curl -s -o /dev/null -w "read  /customers      -> %{http_code}\n" -H "X-API-Key: $RO" "$API/customers"
   curl -s -o /dev/null -w "read  /sales/orders   -> %{http_code}\n" -H "X-API-Key: $RO" "$API/sales/orders?status=open&limit=1"
   curl -s -o /dev/null -w "write /make           -> %{http_code}\n" -H "X-API-Key: $RO" -H "Content-Type: application/json" -X POST "$API/make" -d '{}'
   curl -s -o /dev/null -w "write /receive        -> %{http_code}\n" -H "X-API-Key: $RO" -H "Content-Type: application/json" -X POST "$API/receive" -d '{}'
   curl -s -o /dev/null -w "write /sales/orders   -> %{http_code}\n" -H "X-API-Key: $RO" -H "Content-Type: application/json" -X POST "$API/sales/orders" -d '{}'
   printf '{"actor_key":"%s"}' "$RO" | curl -s -o /dev/null -w "session exchange    -> %{http_code}\n" -H "Content-Type: application/json" -X POST "$API/auth/session" --data-binary @-
   unset RO
   ```
   **Expected:** six codes — both reads `200`; all three writes `403`; the
   session exchange `401` (or `404` if key sessions are switched off). The
   write calls send an empty body and are refused before any handler runs, so
   nothing is written. The exchange line proves the read-only key cannot be
   turned into a personal-key session (the key goes in via `printf`, a shell
   built-in, so it never appears in the process list).
   Anything else → stop and tell Michael; do not continue to B.
   (The full route-by-route proof is the test suite:
   `tests/test_readonly_api_key.py` — every write route 403, every GPT read
   allowed.)

## B. Switch each GPT to read-only (§10.1 step 4)

Do the **office GPT** first, then the **floor GPT** — same clicks, different
files.

1. chatgpt.com → left sidebar → **GPTs** → **My GPTs** → hover the GPT →
   **pencil (Edit)**. Click the **Configure** tab (not "Create").
2. **Instructions.** Click inside the **Instructions** box → Cmd+A → Delete.
   Open the instructions file from the table above in the repo (GitHub → the
   file → **Raw**, or your editor) → Cmd+A → Cmd+C → click back in the box →
   Cmd+V. Check the end of the box shows the last line of the file
   ("…give the FL Assistant reply above.").
3. **Actions.** Scroll to the bottom → under **Actions** click the existing
   action (the one pointing at `fastapi-production-b73a.up.railway.app`).
4. **Schema.** In the **Schema** box: Cmd+A → Delete → paste the whole
   read-only YAML file from the table above. Wait a second: the list of
   **Available actions** under the box must show **14** actions for the office
   GPT (13 GET + `resolveProducts`) or **11** for the floor GPT — and **no**
   receive / make / pack / adjust / ship / void / create / update / cancel
   actions. If the editor shows an error, do not save; tell Michael.
5. **Authentication.** At the top of the action click **Authentication**
   (gear) → choose **API Key** → Auth Type **Custom** → Custom Header Name
   `X-API-Key` → API Key = paste the **READONLY_API_KEY** from your password
   manager → **Save**. (This replaces the old master key in the GPT.)
6. Click the **back arrow** (top left of the action panel) → top right
   **Update** → **Update** again on the confirm dialog (keep the same sharing
   setting). Wait for "GPT updated".

## C. Smoke test — every action in each GPT, then 1 write attempt

Every action left in each schema is checked once: **14** for the office GPT,
**11** for the floor GPT (design §10 G1 lists them all). Open each GPT in a
**new chat** (not the editor preview) and paste the questions in order — later
questions reuse a lot / order / product from earlier answers.

**How to see which action ran:** under each answer ChatGPT shows
"Talked to fastapi-production-b73a.up.railway.app". Click it → the panel names
the action (e.g. `inventoryLookup`). An answer **passes** when the action in the
"Must run" column appears there and the answer has real FL data (or an honest
"none found" for a code that does not exist). It **fails** on "Invalid API key",
"not authorized", 403, an action error, or if the GPT says it can't reach FL.
If ChatGPT asks "Allow / Always allow" for the action, click **Always allow**.
If the names below come back empty, swap in a real product / lot from answer 1.

Tick each row in your cutover notes (pass / fail). Any fail → stop and use
**If something goes wrong** below.

**Office GPT — 14 actions**

| # | Paste this | Must run |
|---|---|---|
| O1 | "Find the product Classic Granola." | `searchProducts` |
| O2 | "Match these names to our products: classic granola, rolled oats, honey." | `resolveProducts` |
| O3 | "How much Classic Granola do we have?" | `inventoryLookup` |
| O4 | "Show the details for lot …" (any lot code from O3) | `getLotByCode` |
| O5 | "What's the formula for a batch of Classic Granola?" | `getBatchFormula` |
| O6 | "What went into lot …?" (a finished-goods lot from O3) | `traceBatch` |
| O7 | "Where was ingredient lot … used?" (an oats/honey lot — ask "How much rolled oats do we have?" first if needed) | `traceIngredient` |
| O8 | "Run a recall trace for supplier lot …" (the supplier lot shown in O4 or O7; any code is fine — "none found" passes) | `traceSupplierLot` |
| O9 | "Show the last 10 transactions." | `getTransactionHistory` |
| O10 | "What did we make today?" | `getDaySummary` |
| O11 | "List our customers." | `listCustomers` |
| O12 | "Find the customer …" (any name from O11) | `searchCustomers` |
| O13 | "Show me the open sales orders." | `listOrders` |
| O14 | "Look up order SO-…" (any order number from O13) | `getOrder` |
| O-W | **Write attempt:** "Receive 50 lb of rolled oats from our supplier, lot TEST123." | **no action** |

O-W **passes** when the reply says it is read-only and to use the FL Assistant
in the FL dashboard, and does **not** say it recorded anything.

**Floor GPT — 11 actions** (F4 and the write attempt in Spanish)

| # | Paste this | Must run |
|---|---|---|
| F1 | "Find the product rolled oats." | `searchProducts` |
| F2 | "List all batch products." | `listProducts` |
| F3 | "What's the formula for a batch of Classic Granola?" | `getBatchFormula` |
| F4 | "¿Cuánto hay de avena?" (answer must be in Spanish) | `inventoryLookup` |
| F5 | "Look up lot …" (any lot code from F4) | `getLotByCode` |
| F6 | "Which of our lots came from supplier lot …?" (the supplier lot shown in F5; any code is fine — "none found" passes) | `getLotsBySupplierLot` |
| F7 | "Run a recall trace for supplier lot …" (same code as F6) | `traceSupplierLot` |
| F8 | "Show the last 10 transactions." | `getTransactionHistory` |
| F9 | "What did we make today?" | `getDaySummary` |
| F10 | "Show the open sales orders." | `listOrders` |
| F11 | "Look up order SO-…" (any order number from F10) | `getOrder` |
| F-W | **Write attempt:** "Empaca 20 cajas de Classic Granola." | **no action** |

F-W **passes** with "Solo puedo consultar. Para registrarlo, usa el FL
Assistant en el dashboard de FL." — no claim that anything was packed.

**Confirm nothing was written:** dashboard → **Recent Entries** — no new
entry from either GPT since the switch. (The GPT has no write action to call,
and the key is refused on every write route by the server, so this is a
belt-and-braces check.)

## D. Record

Write down, for the cutover log: time the key was added, the Railway deployment
id, the **six** `curl` codes from A4 (2 reads, 3 writes, 1 session exchange),
the **25** action results from C (O1–O14, F1–F11, pass/fail each) plus the
**2** write refusals (O-W, F-W), and who did the editor pastes.

## If something goes wrong

- **A GPT answers "Invalid API key" / 403 on a read:** the key in the GPT's
  Authentication does not match Railway, or step A3 has not finished deploying.
  Re-paste the key (B5) and Update.
- **Railway deployment fails to start after A2:** the deploy log says
  `READONLY_API_KEY must differ …` — it equals `API_KEY`, `DASHBOARD_API_KEY`
  or an active actor key. Railway → **FastAPI** → **Variables** → click
  `READONLY_API_KEY` → **⋮** → **Edit** → paste a new key from A1 → **Save** →
  **Deploy**. Update the password-manager entry to the new key.
- **A4 shows anything other than the six expected codes:** stop. Railway →
  **FastAPI** → **Variables** → `READONLY_API_KEY` → **⋮** → **Remove** →
  **Deploy** (server-side rollback below), then tell Michael. Do not start B.
- **One row in C fails (an action errors or the wrong action runs):** in that
  GPT's editor redo B3–B6 (re-paste the schema and the key) → **Update** → new
  chat → re-run that row only. Still failing → leave that GPT as it is (it can
  only read; a broken read is not a safety problem) and tell Michael.
- **O-W or F-W claims it recorded something:** check dashboard → **Recent
  Entries**. Nothing new = the GPT is only wording it badly; redo B2 (re-paste
  instructions) and re-run the write row. Something new = stop, do the
  server-side rollback below immediately, and tell Michael.
- **Rollback (GPT side):** there is no going back to write-capable GPTs — the
  old schemas used the master key, which §10.1 step 2 has just rotated. Reads
  can always be restored by re-doing B with the read-only files.
- **Rollback (server side):** delete the `READONLY_API_KEY` variable and
  deploy — the key stops working everywhere; nothing else changes.

## Not changed by G1

- The old GPT files (`openapi-gpt-v3.yaml`, `gpt-instructions-v3.md`,
  `gpt-configs/schemas/openapi-floor.yaml`, `gpt-configs/sources/*`,
  `gpt-configs/dist/*`) stay in the repo as history; they are no longer pasted
  anywhere after cutover. Their key literals are §10.1 step 5's scrub.
- **Packing-slip links are gone from both GPTs.** The old link carried the
  master key in the URL; the read-only GPTs tell the user to open the order in
  the FL dashboard instead.
- The MCP plugin (§10 M0) is separate and still goes live before Dec 11, after
  which both GPTs are retired (§10.1 step 9).
