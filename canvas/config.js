// FL Canvas configuration. Read-only: the site only sends GET requests.
//
// apiKey: leave empty. The canvas then fetches the dashboard key at load from
// GET {apiBase}/auth/dashboard-key, exactly like the existing dashboard
// (dashboard/dashboard-runtime.js). Committing a key literal would fail the
// repo's key-literal-guard CI check, and the key rotates.
window.FL_CANVAS_CONFIG = {
  env: 'staging',
  apiBase: 'https://fastapi-staging-production-dd7b.up.railway.app',
  apiKey: '',
  // Where task buttons open the existing dashboard pages (same environment as apiBase).
  dashboardBase: 'https://fastapi-staging-production-dd7b.up.railway.app/dashboard',
  timeZone: 'America/New_York'
};

// Production, for reference only (do not switch without a decision):
//   apiBase:       'https://fastapi-production-b73a.up.railway.app'
//   dashboardBase: 'https://cns-factory-ledger.netlify.app'
