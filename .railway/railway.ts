import { defineRailway, preserve, project, service } from "railway/iac";

// Local review scaffold only; no Railway plan/apply has been run.
//
// Ownership: this file is a NAMED PARTIAL. Railway records which partial owns each
// resource, and a partial only ever creates, changes or deletes the resources it
// declares. The existing "FastAPI" service (and every other service, database,
// volume and variable in the project) is outside this partial and stays exactly as it
// is. Do NOT remove the `partial` export: without it the file would describe the whole
// project, and a plan would then propose deleting every undeclared service.
export const partial = "factory-ledger-mcp";

// The one project this partial may ever be applied to. Any other linked project fails
// before a plan is produced.
const EXPECTED_PROJECT = "gleaming-solace";

// Service variables managed in the Railway dashboard. `preserve()` tells the CLI to
// keep whatever value is already set on the service and never to write a value from
// this repository. No secret value exists in the repo; each of these is set (and
// rotated) by hand in the dashboard before the first deploy.
//
// Every user in MCP_ALLOWED_USERS names one MCP_ACTOR_KEY_* variable holding that
// person's existing Factory Ledger named-actor key (never API_KEY/DASHBOARD_API_KEY).
// A variable that exists on the service but is NOT listed here may show up as a
// deletion in `railway config plan`: add each new person's variable to this list
// before applying.
const DASHBOARD_MANAGED = [
  // Required
  "MCP_AUTH_MODE", // "google" for hosted sign-in; "locked" keeps both endpoints at 401
  "MCP_PUBLIC_URL", // https origin of this service's Railway domain (no path)
  "MCP_GOOGLE_CLIENT_ID",
  "MCP_GOOGLE_CLIENT_SECRET",
  "MCP_TOKEN_SECRET", // >= 32 random characters; signs codes, tokens and registrations
  "MCP_ALLOWED_USERS", // JSON list of {email, role, actor_key_env}; no emails in the repo
  "MCP_LEDGER_API_URL", // https://... or http://<service>.railway.internal:<port>
  "MCP_ACTOR_KEY_MICHAEL",
  "MCP_ACTOR_KEY_LUZ",
  "MCP_ACTOR_KEY_MIRIAM",
  "MCP_ACTOR_KEY_ARTURO",
  // Optional
  "MCP_GOOGLE_HOSTED_DOMAIN", // restrict sign-in to one Google Workspace domain
] as const;

export default defineRailway((ctx) => {
  if (!ctx.projectName) throw new Error("An explicit Railway project context is required");
  if (ctx.projectName !== EXPECTED_PROJECT) {
    throw new Error(
      `This partial is scoped to project "${EXPECTED_PROJECT}"; refusing "${ctx.projectName}"`,
    );
  }

  const mcp = service("factory-ledger-mcp", {
    root: "/mcp_server",
    build: {
      builder: "DOCKERFILE",
      dockerfilePath: "Dockerfile",
      watchPatterns: ["/mcp_server/**", "/.railway/railway.ts"],
    },
    deploy: {
      startCommand: "/app/.venv/bin/factory-ledger-mcp",
      healthcheckPath: "/health",
      healthcheckTimeout: 30,
      restartPolicyType: "ON_FAILURE",
      restartPolicyMaxRetries: 3,
    },
    env: {
      // The only literal: production hardening (refuses every test-only setting).
      MCP_ENV: "production",
      // MCP_AUTH_MODE is deliberately NOT hard-coded here any more, so the dashboard
      // value decides between "locked" and "google". The container image still
      // defaults to locked when the variable is absent.
      ...Object.fromEntries(DASHBOARD_MANAGED.map((name) => [name, preserve()])),
    },
  });

  // No repository/branch, domain, database or secret value is wired by this file.
  return project(ctx.projectName, { resources: [mcp] });
});
