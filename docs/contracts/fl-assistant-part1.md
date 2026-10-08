# FL Assistant part 1

Builder: Codex. Reviewer: Claude Code. Draft PR; do not merge.
Base: origin/main 1f3f098. Implements phase1-safe-operating-system §1.8,
F1 and §11 for staging only. A2 #89 and A5 #88 remain external dependencies;
rebase after they merge, preserving their shared FL validators.

## Boundaries

- `/dash/fl-assistant` is served by FastAPI; every API request uses that
  page's origin. Personal actor keys use existing authentication. A11 sessions
  can use the same header once supported by FL; PIN login is deferred.
- `/assistant/*` authenticates the caller and relays allowlisted requests to
  the same FastAPI application through ASGI, including FL authentication.
  There is no alternate ledger implementation, privileged relay key or MCP.
- OpenAI Responses uses function tools, `tool_choice=required`, sequential
  calls and no text escape tool. Model prose is not rendered. Resolved IDs
  come from FL matches or explicit candidate selections, never model guesses.
- Only the separate Record endpoint can commit. Durable, actor-bound draft
  IDs refer to the original server-held ticket and payload hash. Neither
  reaches the model/browser. Retry always commits that exact ticket.
- Choice, draft, blocker and read cards use FL JSON. Green receipts require
  an actual successful commit response with a receipt number. Failures retain
  a red NOT recorded state and the retryable draft.
- A5 annotations on `draft.input_plan` drive last-four/full-code inputs.
  Evidence passes unchanged to commit; FL validates it. Before A5 merges,
  suggested lots remain visible without pretending confirmation is enforced.
- Photos are bounded private attachments, stored durably with actor/session
  ownership and optionally linked to a draft. Their bytes never reach a model.
  Audio goes only to OpenAI transcription; editable text never auto-sends.
- Unmatched resolver queries are stored for review. No automatic alias learning.
- Correction choices come from FL's eight-row `correction_reasons` catalog.
  F1 does not define correction, inventory, permission or lot-validity rules.

## Delivery checkpoints

1. Contract and draft PR.
2. Backend transport, durable state, function tools, attachments and tests.
3. Bilingual chat, dictation, choice/draft/receipt cards and browser checks.
4. Complete Python/JavaScript suites; explicit staging migration and deployment;
   live staging acceptance, changelogs and reviewer handoff.

Migration `068_fl_assistant.sql` is additive F1 state only; 062–066 belong
to A5/A2 and 067 is left available to A3b. No startup migration is added.
`ASSISTANT_ENABLED=1` and `OPENAI_API_KEY` are set only on FastAPI-staging.

Part 2: photo-to-draft extraction and A6 evidence storage integration; A11 PIN
login; A9 shift-summary tool when its endpoint exists; full A5 acceptance after
merge; production readiness including owner-managed spend cap and email alerts.

API references: [function calling](https://developers.openai.com/api/docs/guides/function-calling)
and [transcription](https://developers.openai.com/api/docs/guides/speech-to-text).
