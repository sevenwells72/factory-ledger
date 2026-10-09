# A11 PIN login and FL sessions

Builder: Codex. Reviewer: Claude Code. Draft; do not merge or deploy to production.
Base: `origin/main` at `ce7bb57`. Migration 072 reserved after checking open PRs:
066 #91 held, 067 #93 A7, 068 #90 F1; 069–071 already on main.

Migration 072 is additive and contains no credentials or seeded PINs. Apply as
app owner on staging, in one transaction with a 5-second local lock timeout.
Rollback the application first; retain identity/security evidence tables.
Do not drop them as part of application rollback.

Implementation in progress: durable PIN throttles, sessions, per-action owner
PIN verification, dashboard sign-in/admin and acceptance tests.

F1 #90 remains on its own branch. This PR will document its exact integration
changes against the completed shared browser-session API; no other worktree is
modified. Existing tickets continue to use actor identity, independent of the
session used to prepare them.
