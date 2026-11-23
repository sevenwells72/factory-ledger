# Home-Mac master-key and production-API audit

Audit date: **2026-10-14, America/New_York**. Baseline for canonical repository references: `~/dev/fl-a10-wt`, branch `docs/a10-plan`, `origin/main` at `5592cdc`. This is a static, read-only audit. No production or staging API request, database connection, test, credential rotation, or cleanup was performed. **No key values are included in this report.**

## Result and counting rules

There are **5 source-identified master-key use sites**: one script that retrieves the server's production `API_KEY`, three current GPT instruction files that embed a credential in a packing-slip link, and one deployment guide containing that same documented credential. There are **2 additional untyped-key script consumers** that could receive the master key; their actual supplied key is unknown.

These are source locations, not five proven active integrations. The three GPT files include a source/build-output relationship. The credential in their links equals the value documented as `API_KEY` in the deployment guide, but **its current production validity was not tested**. The script that retrieves Railway variables can acquire the *new* master key after rotation; changing the key alone does not retire that access.

Separately, the documented legacy credential is stored in **7 logical repository paths across 13 permitted checkouts: 91 physical files, 247 occurrences**. The shell history adds **1 file and 123 occurrences**: **92 physical credential-storage files and 370 occurrences** in the checked locations. This storage count includes documentation, archives and duplicate worktrees; it is not an active-consumer count. Nine dashboard source files embed a different shared dashboard credential and are listed separately below.

The storage count is a lower bound. Protected worktrees, inaccessible Shortcut actions, Git history, browser storage and conversation logs were not exhaustively inspected. During this task, an earlier broad caller-search output returned an existing credential-bearing line before redaction was tightened. The value is not reproduced here or in the PR text; this task's tool/session history may therefore hold an additional copy that is **not included in 92**. Treat the credential's present validity as unknown and include that history in the planned post-rotation cleanup review.

## Coverage and exclusions

The initial pass listed 67,256 paths and inspected 22,849 text files under `~/dev`, `~/Developer` and `~/Projects`. Follow-up scans checked key-input code, the known credential's equality without displaying it, shell history, selected files in `~/Documents/factory-ledger`, and the generated GPT instruction file excluded by the general build-directory filter. No additional copies of the documented legacy credential were found elsewhere in the inspected text files under those three roots.

| Surface | Finding / limit |
|---|---|
| Shell startup | `~/.zshrc`, `~/.zprofile`, `/etc/zshrc`, `/etc/zprofile`, `/etc/profile` inspected; no Factory Ledger API key or production API reference found. Other common Bash/Zsh/Fish startup paths were absent. No additional `source` path was found in the existing user startup files. |
| Shell process environment | No `API_KEY`, `DASHBOARD_API_KEY`, `READONLY_API_KEY`, or `FACTORY_`/`LEDGER_`/`FL_` environment name in this audit process. This does not inspect other open terminals. |
| `~/.config/factory-ledger` | Only `db_url` was present. It is a database-connection configuration, not an API master-key use. No connection was attempted. |
| `.env` files | Three files inspected: `~/Projects/pattern-lab/.env`, `~/dev/fl-m0-wt/mcp_server/.env.example`, and `~/Documents/factory-ledger/.env`. The original checkout's `.env:1` names `DATABASE_URL`, not an API key. The MCP example's line 5 names `MCP_LEDGER_API_URL`; it is an example, not a populated production credential. No populated Factory Ledger API-key environment file was found. |
| Launchd | All 28 plist files in `~/Library/LaunchAgents`, `/Library/LaunchAgents` and `/Library/LaunchDaemons` inspected as structured data; no Factory Ledger master-key or production API reference found. This is configuration inspection, not proof of all running processes' arguments or inherited environment. |
| Cron | `crontab -l` reports no user crontab; `/etc/crontab` is absent. No crontab was created or changed. |
| GitHub workflows | No `.github/workflows` directory exists in the 13 inspected Factory Ledger checkouts. No Factory Ledger key/production-host reference was found in workflow files encountered in the other permitted projects. GitHub's remote secret settings and Actions execution history were not inspected. |
| Shortcuts | `shortcuts list` returned 14 shortcuts; none has Factory or Ledger in its name. `~/Library/Shortcuts` is permission denied, so action bodies and personal automations remain **unverified**. A generic shortcut name cannot establish that it is unrelated. No shortcut ran. |
| Codex configuration | `~/.codex/config.toml:64,91` refers to project trust/path configuration. `~/.codex/rules/default.rules:8,12-18,20,24,29` contains repository/command permission references; no Factory Ledger API credential or production-host invocation was found. No `automation.toml` was found under `~/.codex/automations`. |
| Claude configuration | `~/.claude/settings.json` and `~/.claude.json` inspected for credential/production-host references. Factory Ledger matches in `.claude.json` were project/usage metadata, not a configured API consumer. These files can change concurrently, so its metadata line numbers are not treated as durable findings. |
| Other projects | No Factory Ledger production-host or documented legacy-credential match found in inspected text under `~/Developer` or `~/Projects`. This does not classify unrelated services' API keys. |

**Not opened:** `~/dev/fl-o7-wt`, `~/dev/fl-f1chat-wt`, `~/dev/fl-f1chat-verify-wt`, `~/dev/fl-cutover-wt`, `~/dev/fl-pilot-wt`, and `~/dev/fl-pilot-docs-wt`. The last two have PR #113's `docs/pilot-playbook` branch. Git worktree metadata was used to identify exclusions; their contents were not audited. The broad scan also excluded Git internals/bare mirrors, dependencies, caches and build directories; it skipped 27 files over 3 MB and two binary files. The known GPT build artifact was checked separately. No claim is made about Keychain items, browser local/session storage, other user accounts, mounted backups, unlisted mobile devices, live Railway variables, or installed GPT Action credentials.

## Master-linked use sites and rotation outcome

Paths in the next tables are relative to **`/Users/michaelgross/dev/fl-a10-wt/`**, unless an absolute or home-relative path is shown. Line numbers refer to this audit snapshot. The copy inventory below supplies every inspected checkout root.

| ID | Path:line and variable / field | What it uses the key for | Does production master rotation break it? | Replacement |
|---|---|---|---|---|
| M1 | `scripts/granola_writedown_0917.py:212,219,256-258`; Railway variable `API_KEY`; request header at `121` | Fetches production Railway variables, selects the master key, expects a shared identity, then calls direct `/adjust` at `265,287`. | **Not necessarily.** A new invocation retrieves the rotated value if its Railway access remains. Existing identity/write restrictions can independently reject it; this was not run. A10 closes its direct write route. | **Retire** this one-off writedown script. Any repeat operation needs a named personal key/session and prepare → commit receipt flow; read-only checks use a read-only key. Do not refresh the master-key dependency. |
| M2 | `gpt-instructions-v3.md:104`; URL query parameter `key` | Creates a production packing-slip URL containing the documented legacy credential. | **Yes if that credential is currently accepted and is the key being rotated.** Existing links become stale; validity is unverified. | **Retire** credential-bearing links. Open packing slips through an authenticated dashboard flow; use a read-only credential only through an approved read client, not embedded in a shared URL. |
| M3 | `gpt-configs/sources/floor-specific.md:50`; URL query parameter `key` | Source instruction for the same packing-slip link. | Same conditional break as M2; rebuilding from this source can restore the obsolete credential. | **Retire** the embedded credential at the source; direct users to the dashboard. |
| M4 | `gpt-configs/dist/GPT_FLOOR_INSTRUCTIONS.md:92`; URL query parameter `key` | Generated Floor GPT instructions containing the same link. | Same conditional break as M2; a deployed GPT may retain a copied version. | **Retire** this link form and regenerate from a corrected source in a later implementation change. Check the installed GPT separately. |
| M5 | `DEPLOYMENT.md:91,124,133,141,157,172,180,195,205,213,220,308`; `API_KEY`, `x-api-key`, URL `key` examples | Deployment setup/manual API examples contain the same documented legacy credential. | Copied commands using that value fail after it stops being accepted. The guide itself does not run. | **Retire** literal credentials and direct-write examples. Read commands use a read-only key; legitimate writes use a personal key and tickets. Server secret setup belongs in a secret store, not a committed value. |
| U1 | `scripts/cleanup_void_reversals.sh:25,31-32,44-45,90`; `FACTORY_API_KEY` → `API_KEY`; `FACTORY_API_BASE` → `API_BASE` | Defaults to production, posts `/void/${id}`, then reads lots. Actual supplied key type is unknown. | **Yes if** the user supplies the old master key. Regardless of key rotation, A10 closes the direct void call. | **Retire** the one-off repair entry point or replace with a personal-key ticket workflow; inspection-only lot reads can use a read-only key. |
| U2 | `scripts/set_sweetened_coconut_yield.py:9-10,16,26,36`; environment `API_KEY` | Reads product search and sends `PUT /admin/products/{product_id}` for a historical yield correction. No stored key value in the script. | **Yes if** its supplied value is the rotated master. The admin mutation's retirement/migration is an independent A10 question. | **Retire** the one-off script. This route is master-only in the inspected source; there is no drop-in personal-key substitute. Build an approved catalog ticket workflow for any future change, or retire the operation; do not substitute a read-only key for writes. |

Across the thirteen inspected roots listed below, M1 and U2 each exist in **12 roots** (every root except `/Users/michaelgross/dev/factory-ledger`), and U1 exists in **all 13 roots**. Thus each additional script path is its listed root plus the relative M1/U1/U2 path. The key input references are `API_KEY` line 219 for M1, `FACTORY_API_KEY` lines 25/32 and assigned `API_KEY` line 32 for U1, and `API_KEY` line 10 for U2 in every inspected copy. All copies of these three scripts are byte-identical to A10, so their full canonical line citations apply in every listed root. The five M rows are the requested headline count; U1/U2 are kept outside it because a generic variable name does not prove master-key use. Production logs and the installed GPT configurations are needed to determine which sources are still active.

## Literal credential storage and every inspected copy

The seven paths below contain the same documented legacy credential. The field/variable is shown, never its value. **All seven paths are present in all thirteen roots listed after the table.** Therefore every physical path is `root` + `relative path`; this explicitly includes the audit's own A10 worktree.

| Relative path | Lines (A10 unless noted) | Field / use | Occurrences per checkout | Required handling after rotation |
|---|---|---|---:|---|
| `gpt-instructions-v3.md` | 104 | Packing-slip URL `key` | 1 | Retire embedded key; replace installed copies. |
| `gpt-configs/sources/floor-specific.md` | 50 | Packing-slip URL `key` | 1 | Correct source before rebuilding. |
| `gpt-configs/dist/GPT_FLOOR_INSTRUCTIONS.md` | 92 | Packing-slip URL `key` | 1 | Regenerate/replace deployed artifact. |
| `DEPLOYMENT.md` | 91,124,133,141,157,172,180,195,205,213,220,308 | `API_KEY`, request header/query examples | 12 | Replace examples with variable names and least-privilege workflows. |
| `CHANGE_LOG.md` | 5198; other roots below | Historical credential comparison | 1 | Historical storage, not a scheduled caller. Review secure redaction/history treatment separately; do not reuse as a credential source. |
| `audits/reports/DASHBOARD_ACTIONABILITY_AUDIT.md` | 179,471 | Historical authentication/URL discussion | 2 | Historical storage; retire literals and avoid replaying old examples. |
| `archive/superseded-instructions/GPT_INSTRUCTIONS.md` | 99 | Archived packing-slip URL `key` | 1 | Retired instructions; prevent copying back into a live GPT. |

| Checkout root | `CHANGE_LOG.md` occurrence line |
|---|---:|
| `/Users/michaelgross/dev/factory-ledger` | 1540 |
| `/Users/michaelgross/dev/fl-a10-wt` | 5198 |
| `/Users/michaelgross/dev/fl-count-wt` | 4030 |
| `/Users/michaelgross/dev/fl-d12-wt` | 4352 |
| `/Users/michaelgross/dev/fl-d2office-wt` | 4699 |
| `/Users/michaelgross/dev/fl-g1-wt` | 4030 |
| `/Users/michaelgross/dev/fl-m0-wt` | 4177 |
| `/Users/michaelgross/dev/fl-m111-wt` | 4051 |
| `/Users/michaelgross/dev/fl-p125-wt` | 4051 |
| `/Users/michaelgross/dev/fl-post108-wt` | 4044 |
| `/Users/michaelgross/dev/fl-r107-claude` | 4391 |
| `/Users/michaelgross/dev/fl-triage-wt` | 4538 |
| `/Users/michaelgross/Documents/factory-ledger` | 1810 |

`/Users/michaelgross/.zsh_history` has 123 occurrences of that same documented credential at lines:

`71,75,79,83,86,91,94,95,96,97,98,99,100,102,105,108,110,112,116,126,129,130,131,133,136,139,140,141,143,146,147,148,150,153,157,161,163,194,198,200,204,208,220,226,232,238,244,250,255,261,267,273,279,285,289,301,305,324,330,335,341,346,359,365,371,374,378,382,387,393,399,411,417,423,429,434,440,446,451,465,479,490,494,498,501,505,508,511,515,518,522,527,533,539,545,547,549,552,553,554,555,556,557,559,562,563,566,567,568,569,588,589,592,596,600,602,611,613,625,631,632,633,634`.

These are historical command/header/query storage, not 123 current integrations. Replaying an old credential-bearing command fails when its credential is invalidated; old direct-write commands also fail under A10. Retire these command forms and review secure history/log cleanup after rotation under a separate authorized change. The same history has 149 production-host-reference lines and six occurrences of the different shared dashboard credential at `1137,1193,1195,1197,1206,1221`. No command text was copied into this report.

## Other production clients: do not miscount these as master-key consumers

Read-only replacement guidance is conditional on route support: `READONLY_API_KEY` covers the G1 schema reads, not every GET or preview endpoint. Planning, notes, shift summaries and M0 previews may require named personal/session read access. Keep their current authorization; this audit does not authorize widening the read-only key.

The nine dashboard sources below contain an equal shared dashboard credential that differs from the documented legacy master value. Source equality does not establish how the deployed server currently classifies either credential. **Rotating only `API_KEY` should not break a separately configured dashboard key; retiring or rotating `DASHBOARD_API_KEY` can.** Any cached browser code/local storage must be checked independently. A10 route closure can still break a caller even when its credential remains valid.

| Path:line | Variable / field | Use and replacement |
|---|---|---|
| `dashboard/dashboard.js:2157` (production bases at `5,2156,5410`) | `SALES_API_KEY` | Shared dashboard requests; personal session for writes, least-privilege reads. |
| `dashboard/mini-calendar.js:8,233` (base `7`) | `SALES_API_KEY` | Scheduling client; personal session for mutations, read-only key/session for read views. |
| `dashboard/sankey.html:478,533` (base `477`) | `API_KEY` | Read visualization; read-only key or authenticated dashboard read. |
| `dashboard/process-flow.html:325,396` (base `324`) | `API_KEY` | Read visualization; read-only key or authenticated dashboard read. |
| `dashboard/traceability.html:415,427` (base `414`) | `API_KEY` | Traceability reads; read-only key or authenticated dashboard read. |
| `dashboard/history.js:13` | `x-api-key` literal | History reads; read-only key or authenticated dashboard read. |
| `dashboard/runs.js:38` (base `4`) | `x-api-key` fallback | Run screens; named session for writes, read-only access for viewing. |
| `dashboard/shift-summary.js:5` (base `6`, header `202`) | Shared-key constant | Shift summary; read-only key/session where permitted. |
| `dashboard/ledger-ui.js:83` | Shared-key fallback | Common client authentication fallback; migrate to session/read-only handling as appropriate. |

The same dashboard literal also appears in `tests/visual/run-runs-interactions.mjs:62` and `tests/visual/run-production-trace.mjs:28`. These are source/test fixtures; no test ran and no live browser session was inspected. Replace production credentials in tests with test-only values and intercepted/local fixtures in a later change. Historical dashboard-key references occur in `FACTORY_LEDGER_CHANGELOG.md:152,199,215,227,231,232`, `CHANGE_LOG.md:3675,5177,5178,5196,5198,5205`, `docs/manual-tests/044-so-allocations-pr6-dashboard.md:7,22,26,29,32,83`, `docs/design/phase1-safe-operating-system.md:408`, `docs/design/so-state-model-findings.md:1545`, `docs/design/runs-screen-spec.md:17`, `docs/designs/045-write-foundation-design.md:118`, `docs/design/audit/IMPROVEMENTS-MASTER.md:803`, `docs/design/audit/05-access-icon-other.md:705`, and `docs/design/audit/02-actions-input-touch.md:779`. These are stored references, not extra active clients.

| Other executable/configuration path:line | Variable / use | Rotation outcome and replacement |
|---|---|---|
| `audits/fresh-start/apply_reset.py:29,202-208,300,493,735,932`; `audits/fresh-start/fresh_start_common.py:39` | Production API base and operator-supplied actor key. Legacy backend calls `/adjust`; newer code also has `/adjust/prepare` at `731,822`. | A valid personal key does not depend on rotating the master. The **direct backend still breaks when `/adjust` closes**. Retain only the validated personal-key ticket path and receipt/reconciliation checks; retire the legacy backend. |
| `scripts/mint_actor_keys.py:39-40,160` | Personal-key verification example references production. Executable code generates random personal keys and SQL locally; it does not fetch/use a master key. | No master-rotation dependency. Keep personal-key provisioning through the approved process; do not run it as part of this audit. |
| `dashboard/dashboard-runtime.js:4`; `dashboard/shell.js:50` | Production API URL/bootstrap references, no master literal | URL alone does not break on rotation; session/bootstrap compatibility must be checked in the dashboard cutover. |
| `dashboard/fl-assistant.js:56` | Header/client authentication integration | Use named personal session and ticket receipts. No master literal identified here. |
| `openapi-gpt-v3.yaml:7,15`; `gpt-configs/schemas/openapi-floor.yaml:7,15` | Production server URL plus `x-api-key` security declaration; no credential value in the schema | Installed Action key type is external and unknown. Read-only GPTs use a read-only key and read-only schema. Retire direct-write Actions under A10 regardless of key replacement. |
| `gpt-configs/readonly/openapi-office-readonly.yaml:10,18`; `gpt-configs/readonly/openapi-floor-readonly.yaml:10,18` | Production read-only Action schema and `x-api-key` declaration | Keep on a dedicated read-only key. Rotating only the master does not require changing a distinct read-only credential. Verify installed GPT configuration separately. |
| `~/dev/fl-m0-wt/mcp_server/src/factory_ledger_mcp/identity.py:20,33,268`; `~/dev/fl-m0-wt/mcp_server/src/factory_ledger_mcp/adapter.py:69,111-119`; `~/dev/fl-m0-wt/mcp_server/.env.example:5` | MCP identity/adapter code references `API_KEY`, `DASHBOARD_API_KEY`, `MCP_TEST_API_KEY`, and `MCP_LEDGER_API_URL`; source has shared-key exclusions and actor mapping | No populated local MCP key configuration found. Do not give M0 the new master. Keep its approved read-only capability/identity configuration; installed/deployed configuration was not inspected. |

## Staging and test references

These files contain key inputs or request headers, but are staging/test tooling, not evidence of a production master-key installation. No credential was fetched and none of these scripts ran. Keep their staging credentials separate; never repair them by copying the new production master key. Where the check is for authorized writes, use a staging personal actor/session and tickets; use a staging read-only credential for read checks. Preserve explicit rejected-credential tests with test-only values.

| Path:line | Input / purpose |
|---|---|
| `scripts/check_assistant_staging.py:99,103,111,131-141,197` | Supplied API key/header checks for assistant behavior |
| `scripts/check_direct_order_close_staging.py:156,193,199,274-275,312,376,506` | Staging `API_KEY`, `DASHBOARD_API_KEY`, `READONLY_API_KEY` cases and key identity checks |
| `scripts/check_exceptions_staging.py:85-86,119,169,214` | Staging `API_KEY` / `DASHBOARD_API_KEY` |
| `scripts/check_key_sessions_staging.py:54,107` | Key-session check input/header |
| `scripts/check_kosher_tiers_staging.py:63-64,92` | Staging `API_KEY` / `DASHBOARD_API_KEY` |
| `scripts/check_lot_confirmation_staging.py:69-70,92` | Staging `API_KEY` / `DASHBOARD_API_KEY` |
| `scripts/check_office_void_staging.py:54,66` | Staging `API_KEY` / `DASHBOARD_API_KEY` |
| `scripts/check_order_tickets_staging.py:61-62,89,132` | Staging `API_KEY` / `DASHBOARD_API_KEY` and ticket checks |
| `scripts/check_pilot_triage_staging.py:46,65` | Staging key/header input |
| `scripts/check_readonly_key_staging.py:34,37,64,87,90,120` | `READONLY_API_KEY`; production hostname is an explicit target guard, not a production request |
| `scripts/check_recipe_corrections_staging.py:208,213,329` | Staging `API_KEY` / `DASHBOARD_API_KEY` |
| `scripts/check_resolution_receipts_staging.py:58,88,94` | Staging header/key input and receipt checks |
| `scripts/check_shift_summary_staging.py:61,83,120` | Staging `DASHBOARD_API_KEY` |
| `scripts/check_ship_gate_staging.py:56,82,98,102` | Staging key/header input |
| `scripts/check_supplier_lot_staging.py:57,80,122` | Staging key/header input |
| `scripts/count_rehearsal_staging.py:101,135,722,746` | Staging `API_KEY`, `DASHBOARD_API_KEY`, `DATABASE_URL`; rehearsal tooling |

Production URL strings also occur in test/stub sources `tests/test_readonly_api_key.py:123`, `tests/visual/run-d12-preview.mjs:4`, `tests/visual/run-run-types.mjs:14`, `tests/visual/lib/stub.mjs:13`, `tests/visual/run-runs-interactions.mjs:48,60`, and `~/dev/fl-m0-wt/mcp_server/tests/test_auth.py:822`. These are not counted as installed master-key uses. Test-only API-key variables throughout the test suite are likewise not production credentials; the A10 test replacement plan is in `a10-plan.md`.

## URL-only documentation and archived schemas

The following canonical A10 files reference the production host without establishing an additional installed credential. A URL reference does not itself break on rotation. Retire archived write instructions, move active read examples to read-only authentication, and move active write examples to personal-key tickets. The excluded/private runtime settings remain separate verification work.

| Path | Production-host reference line(s) |
|---|---|
| `CLAUDE.md` | 100 |
| `CONTEXT.md` | 600,613 |
| `FOLLOWUPS.md` | 635 |
| `week1-scoring-extract-2026-08-13.md` | 13 |
| `docs/scheduling/coconut-yield-forward-only.md` | 34 |
| `docs/manual-tests/044-so-allocations-pr6-dashboard.md` | 8 |
| `gpt-configs/readonly/CUTOVER_CHECKLIST.md` | 53,86,109 |
| `audits/fresh-start/HOSTED-REHEARSAL-2026-10-14.md` | 122 |
| `archive/superseded-schemas/openapi-schema.yaml` | 7; `x-api-key` declaration at 16 |
| `archive/superseded-schemas/openapi-v3.yaml` | 7; `x-api-key` declaration at 15 |
| `archive/superseded-schemas/openapi-schema-gpt.yaml` | 7; `x-api-key` declaration at 16 |
| `archive/superseded-instructions/SALES_API.md` | 5 |
| `CHANGE_LOG.md` | 5672,5742,5778,5779,6171; historical record |
| `FACTORY_LEDGER_CHANGELOG.md` | 256; historical record |

The credential-bearing instruction/audit files and executable production clients already listed above are deliberately not counted again as URL-only locations.

## Cutover actions and open questions for Michael

1. **Stop credential-bearing links and shared-key retrieval at their sources.** Confirm which installed Office/Floor GPT instructions match these files; replace packing-slip links and revoke their old configured Action credentials as part of cutover. Retire the granola one-off. It retrieves the current master dynamically, so rotation alone can leave it working with excessive access.
2. **Separate credential changes from route changes.** A personal key does not save a script that still posts to `/adjust` or `/void/{id}`. Check the fresh-start legacy backend, repair scripts and dashboard consumers against the A10 route list. Reads should use a read-only credential; writes need named identity and tickets/receipts.
3. **Close the audit gaps before declaring the Mac clean.** Review the 14 Shortcut action bodies/personal automations with the required macOS access, other active terminal environments, browser-stored dashboard/GPT keys, excluded worktrees with their owners, and installed cloud configurations. Then review secure cleanup of old credentials in files, Git history, shell history and agent/tool logs after rotation. Nothing was deleted or rewritten here.

Questions that source inspection cannot answer: Is the documented legacy credential still the production master? Which GPTs or bookmarked packing-slip links are in active use? Has anyone used U1/U2 with a master key recently? Is any generic Shortcut a Ledger automation? Is the deployed dashboard key distinct from the master and scheduled for its own retirement? Who owns the post-rotation history/log cleanup and the check of the protected worktrees? Verify these from owners and redacted logs, without pasting secrets into this document or chat.
