# Netlify preview smoke

Preview: https://deploy-preview-45--cns-factory-ledger.netlify.app

PR: https://github.com/sevenwells72/factory-ledger/pull/45

Verified 2026-09-11 against implementation commit `4b1d85b`:

- Index and all six relevant assets returned HTTP 200.
- `dashboard.css?v=43`, `dashboard.js?v=60`, `so-list.css?v=1`, `so-list.js?v=1`, `so-list-actions.css?v=1` and `so-list-actions.js?v=1` matched local bytes.
- The deployed index differed only in Netlify's four pretty-URL link rewrites and injected deploy toolbar; application script/style versions matched.
- Chromium loaded both new modules and rendered nine fixture orders with zero page errors.

API calls were stubbed, other external requests were blocked, and no real backend writes or Ready/exit interactions were performed. This verifies deployed frontend assets and startup; it does not verify production API compatibility. The branch's additive backend fields and Ready state guard must deploy before production use.
