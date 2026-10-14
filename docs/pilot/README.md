# Pilot playbook (Nov 2 – Nov 20, 2026)

| File | For | What |
|---|---|---|
| [PILOT-PLAN.md](PILOT-PLAN.md) | Michael | Dates, who does what, the 10-day test script, success criteria, problem reporting, Nov 17 readiness gate, open points |
| [ARTURO-GUIDE.md](ARTURO-GUIDE.md) | Arturo | FL Assistant step by step (EN / ES) |
| [OFFICE-GUIDE.md](OFFICE-GUIDE.md) | Luz, Miriam | Office tasks, lookups, exceptions, what office can't do and why (EN / ES) |
| [MICHAEL-DAILY-CHECK.md](MICHAEL-DAILY-CHECK.md) | Michael | 10-minute morning routine and what to escalate |
| [cards/arturo-quick-card.pdf](cards/arturo-quick-card.pdf) · [cards/office-quick-card.pdf](cards/office-quick-card.pdf) | print | One-page EN/ES quick cards (Letter) |

Written against `origin/main` @ `d5e9fea`. 🟡 marks features that wait for PR #107; ⛔ marks things with no screen or PR yet.

Rebuild the PDFs after editing the card HTML (headless Chrome, from `docs/pilot/cards/`):

```sh
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --no-pdf-header-footer \
  --print-to-pdf="$PWD/arturo-quick-card.pdf" "file://$PWD/arturo-quick-card.html"
```

Each card must stay one page.
