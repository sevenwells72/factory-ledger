# Read-only production browser audit

Run from `tests/visual`:

```sh
npm ci
npm run browsers
npm run audit:live
node live/supplement.mjs
node live/hover.mjs
node live/find.mjs
node live/report.mjs
npm test
```

The runner visits only https://cns-factory-ledger.netlify.app and follows the page's own GET requests. Chromium and WebKit each reuse the loaded page across its inventory states. API reads are serialized across engines to avoid request bursts; responses are forwarded unchanged. No fixture library or API cache is used. Each engine has a fresh, disposable browser context, not a user's existing browser profile.

`live/safety.mjs` denies every non-GET/HEAD/OPTIONS network request and mutating URL segments. Capture-phase guards deny commit controls and form submission; the script never deliberately clicks them. Native dialogs are dismissed. Enter is tested only in text/date/number inputs and textareas, never on a button, checkbox, select, or file picker. No business draft is saved. The scheduler's own initialization may persist its default state inside the disposable context; no user scenario is loaded, changed, or imported.

The existing fixture audit (`../run-visual-audit.mjs`) remains unchanged. **Do not run the fixture audit to reproduce this live report:** it overwrites `06-browser-check.md` with different results.

Examples:

```sh
node live/run.mjs engine=chromium screens=S-22,S-45
node live/run.mjs engine=webkit variant=390x844
node live/run.mjs resume=true
```

`resume=true` skips evidence files with eight variants. For a fresh live run omit it. Do not use parallel shards against the production API. Screens are prepared at 1280×800, then resized through 390×844, 844×390, 768×1024 and 1280×800, light/dark. This checks responsive rendering in a desktop browser, not iOS/Android soft keyboards, OS text scaling, safe areas or real-device rotation.

Evidence is in `docs/design/audit/live-evidence/`. Each screen JSON records the engine, version, times, sanitized request paths, response status, guard interventions, raw axe-core contrast nodes, hit-target measurements, field attributes, keyboard sequence, wrapping and overflow. Screenshots are viewport captures. A `gap` explicitly means the screenshot is context for an unavailable state, not a capture of that state. Reports must not score such a parent page as the missing screen.

Contrast passes cover axe-core's `color-contrast` rule only. `incomplete` results are not passes. Hit regions reuse the repository's geometry checker, including wrapping labels and absolutely positioned hit-area pseudo-elements; the element's own box is retained when the hit area is extended. Tab checks report reachability and modal escape, not successful submission or business-process ordering. Enter outcomes on blank forms cannot establish how a valid filled form submits; guarded outcomes remain unverified. Color-only cues and wrapping candidates need human review against the screenshots and the master list.

Screenshots contain the operational information visible in production. Logs omit credentials, request query strings, request/response headers and API response bodies. Do not add authentication dumps or browser storage to evidence.
