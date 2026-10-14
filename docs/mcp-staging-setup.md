# Try Factory Ledger in ChatGPT Business — M0 staging

This connector answers questions from **staging data only**. It cannot receive,
make, pack, adjust, ship, change an order, or save anything. Shipment previews
show what FL says would happen; they do not ship goods. Enter real work through
the existing FL Assistant or dashboard.

**Connector URL:** https://factory-ledger-mcp-staging-production.up.railway.app/mcp

**Current activation status:** the service is configured to start locked. Google
sign-in cannot be enabled until the Google client and verified personal staging
actor mappings below are supplied. A locked service returns 401 for connector
requests; that is intentional. No real user is authenticated yet.

## What Michael needs to supply once

1. In Google Cloud's **Google Auth Platform**, select a suitable project and
   create a **Web application** OAuth client for **Factory Ledger M0 Staging**.
   Configure the consent audience for the intended testers; if External and
   Testing, add each tester's Google account. Request only OpenID/email identity.
   Add this exact **Authorized redirect URI**:
   `https://factory-ledger-mcp-staging-production.up.railway.app/oauth/google/callback`.
   This is Google's callback to FL, not ChatGPT's callback. Save the client ID and
   secret directly into the new Railway service's Variables panel. Do not paste
   secrets into a chat, PR or setup document.
2. Independently confirm the Google email and FL **staging** actor for each
   tester. Collect each person's own staging actor key, actor ID, exact actor
   name and role. Never reuse a master, dashboard, read-only shared, production,
   or another person's key. Do not guess identities from variable names. The
   setup run found no protected Google OAuth credential file and no verified
   email-to-person mapping; existing synthetic staging credentials were not
   treated as Michael's identity.
3. In Railway, select project **gleaming-solace** →
   **factory-ledger-mcp-staging**, service ID
   `ef21f85c-e0a1-44b3-a6f0-49a238d34be1`. Set these variables on **this service
   only**, using fresh staging values:

   | Variable | What to enter |
   |---|---|
   | `MCP_ENV` | `staging` |
   | `MCP_WRITES_ENABLED` | `false` |
   | `MCP_LEDGER_API_URL` | `https://fastapi-staging-production-dd7b.up.railway.app` |
   | `MCP_PUBLIC_URL` | `https://factory-ledger-mcp-staging-production.up.railway.app` |
   | `MCP_GOOGLE_CLIENT_ID` | The new Google web client ID |
   | `MCP_GOOGLE_CLIENT_SECRET` | Its secret, entered privately |
   | `MCP_TOKEN_SECRET` | A newly generated random secret of at least 32 characters |
   | `MCP_ACTOR_KEY_PERSON` | That person's staging actor key; use a separate variable per person |
   | `MCP_ALLOWED_USERS` | The JSON list described below |
   | `MCP_AUTH_MODE` | Change `locked` to `google` only after completing the preceding fields |

   The underlying Railway environment is named **production**, ID
   `f4d219df-2fea-45e8-85de-466b36a86c07`; this is also where the existing
   **FastAPI-staging** service lives. The environment's label does not make this
   a production FL deployment. The new service is explicitly staging, its FL
   target is pinned in code, and the existing **FastAPI** production service is
   untouched. Do not copy variables from it. Do not set `DATABASE_URL`, shared
   API keys, local-stub settings or a production ledger URL here.

The allowlist contains configuration, **not inline keys**. Replace every
placeholder with independently verified staging identities:

```json
[
  {
    "email": "person@example.invalid",
    "role": "floor",
    "actor_key_env": "MCP_ACTOR_KEY_PERSON",
    "actor_id": 123,
    "actor_name": "Synthetic Person",
    "actor_role": "floor"
  }
]
```

Use `actor_role` exactly as FL returns it: `owner`, `office` or `floor`.
`role` is the inherited connector grouping: `admin_office_floor` for owner/office,
`floor` for floor. All these users have **reads only** in M0. FL's `/auth/whoami`
does not contain email, so the operator must establish the email-to-person link;
the server then checks that the key names that exact actor before it starts.
A missing/inactive/shared/swapped key or mismatched name, ID or role stops startup.

Redeploy **only the new MCP service**, with one replica, after entering the
values. Open its `/health` URL: `auth` should be `google`, `google_oauth_ready`
and `actor_mappings_verified` should be true, and `writes_enabled` should be
false. An unauthenticated `/mcp` request should still be 401, now with an OAuth
discovery challenge. If startup fails, fix the mapping; do not weaken validation
or substitute a shared key.

## What Michael clicks in ChatGPT

Once activation above is complete:

1. Open ChatGPT on the web and select the intended **Business workspace**.
2. Open [Plugins](https://chatgpt.com/plugins), click **+ → Add custom MCP server**.
3. Name it **Factory Ledger — STAGING (read only)**. Paste the connector URL above,
   including `/mcp`. Choose **OAuth**, acknowledge the warning, then click
   **Create as a plugin**. The server supports dynamic client registration; the
   Google client secret belongs in Railway, not in ChatGPT's client-secret field.
4. Sign in with the specific Google account in the FL allowlist. Install the
   resulting plugin. If a workspace permission blocks creation or installation,
   the Business admin must enable it under **Workspace apps / Permissions & roles**
   for the intended testers. Exact availability depends on the workspace.
5. Start a new chat, type **@**, choose the staging plugin and ask:
   “Search Factory Ledger staging for products named STAGING.” Then try an
   inventory or order lookup. The connection must discover exactly **17** read
   tools. Asking it to save a receipt or ship an order must not make a change.

If ChatGPT presents a callback URI, use that exact ChatGPT callback for its
registration; it is separate from Google's callback above. M0 accepts the
published `https://chatgpt.com/connector/oauth/<callback_id>` path and the stable
`https://chatgpt.com/connector_platform_oauth_redirect` path, with issuer
identification. It rejects arbitrary third-party redirect URLs.

After a deploy/restart, reconnect/sign in again: M0 intentionally invalidates
access and refresh tokens when its in-memory replay state is lost. After a tool
change, open the connection, click **Refresh**, then start a new chat. Keep the
service at **one process / one replica**.

Current instructions were checked against official OpenAI documentation:
[custom MCP server](https://developers.openai.com/api/docs/guides/custom-mcp-server),
[connect and test](https://developers.openai.com/plugins/deploy/connect-chatgpt),
[OAuth](https://developers.openai.com/plugins/build/auth), and
[workspace controls](https://learn.chatgpt.com/docs/enterprise/apps-and-connectors).
Google's own setup reference is
[web-server OAuth](https://developers.google.com/identity/protocols/oauth2/web-server).

## Redeploy staging safely

The service is a CLI upload of the `mcp_server` directory, with its own Dockerfile
and `railway.json`. It has no GitHub auto-deploy link and does not run FL's startup
migrations. The repository-root `railway.json` is unchanged.

From the M0 worktree, check disk space first and stop heavy work below 5 GB free:

```sh
df -h /
railway up mcp_server --path-as-root \
  --project 2206e070-d160-4528-a4f1-86a587ad88c3 \
  --environment f4d219df-2fea-45e8-85de-466b36a86c07 \
  --service factory-ledger-mcp-staging --detach
```

Use the explicit service on every variable/deploy command. Never run a bare
`railway up`, copy production variables, merge the draft PR, or deploy this branch
to the existing FastAPI service. To disable the connector, change **this service's**
`MCP_AUTH_MODE` back to `locked` and redeploy it; all MCP requests will be denied.

See [the deployment and test report](deployments/m0-readonly-mcp.md) for the
actual hosted status, evidence, and remaining owner steps.
