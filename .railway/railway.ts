import { defineRailway, project, service } from "railway/iac";

// Local review scaffold only; no Railway plan/apply has been run.
// Own only the new MCP service. Existing services and their legacy config are
// outside this partial and must retain their current settings exactly.
export const partial = "factory-ledger-mcp";

export default defineRailway((ctx) => {
  if (!ctx.projectName) throw new Error("An explicit Railway project context is required");

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
      MCP_ENV: "production",
      MCP_AUTH_MODE: "locked",
    },
  });

  // No repository/branch, domain, database or secret is wired by this scaffold.
  return project(ctx.projectName, { resources: [mcp] });
});
