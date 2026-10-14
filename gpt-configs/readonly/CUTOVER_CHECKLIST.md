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
   different from `API_KEY` and `DASHBOARD_API_KEY` (the service refuses to
   start otherwise — by design).
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
   unset RO
   ```
   **Expected:** both reads `200`; all three writes `403`. The write calls send
   an empty body and are refused before any handler runs, so nothing is written.
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

## C. Smoke test — 3 questions per GPT, then 1 write attempt

Open each GPT in a **new chat** (not the editor preview). Use real names from
today's stock if these come back empty — the point is that each question gets
a real FL answer, not an error.

**Office GPT**

1. "How much Classic Granola do we have?" → stock by lot (inventoryLookup).
2. "Show me the open sales orders." → list of SOs (listOrders).
3. "Look up order SO-…" (any order number from answer 2) → order with its lines
   (getOrder).
4. **Write attempt:** "Receive 50 lb of rolled oats from our supplier, lot
   TEST123." → **Expected:** a reply that it is read-only and to use the FL
   Assistant in the FL dashboard. It must **not** say it recorded anything.

**Floor GPT** (do one in Spanish)

1. "¿Cuánto hay de avena?" → stock, answered in Spanish (inventoryLookup).
2. "What did we make today?" → day summary (getDaySummary).
3. "Look up lot …" (any lot code from answer 1) → lot details (getLotByCode).
4. **Write attempt:** "Empaca 20 cajas de Classic Granola." → **Expected:**
   "Solo puedo consultar. Para registrarlo, usa el FL Assistant en el
   dashboard de FL." — no claim that anything was packed.

**Confirm nothing was written:** dashboard → **Recent Entries** — no new
entry from either GPT since the switch. (The GPT has no write action to call,
and the key is refused on every write route by the server, so this is a
belt-and-braces check.)

## D. Record

Write down, for the cutover log: time the key was added, the Railway deployment
id, the four `curl` codes from A4, the 6 answers + 2 refusals from C (pass/fail
each), and who did the editor pastes.

## If something goes wrong

- **A GPT answers "Invalid API key" / 403 on a read:** the key in the GPT's
  Authentication does not match Railway, or step A3 has not finished deploying.
  Re-paste the key (B5) and Update.
- **Railway deployment fails to start after A2:** `READONLY_API_KEY` equals
  one of the other two keys. Generate a new one (A1) and replace it.
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
