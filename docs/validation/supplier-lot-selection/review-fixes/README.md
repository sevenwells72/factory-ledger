# PR147 review fixes on the combined receive / signed-slip branch

This revision addresses [review 6349465333](https://github.com/sevenwells72/factory-ledger/pull/147#issuecomment-6349465333) and the owner's conditional over-draw decision. The branch is rebased onto PR146 at `282204be`, which includes PR145 at `ffc405e8`.

## Stock rule

When the ingredient has enough usable stock overall, a worker cannot put more pounds on a chosen lot than that lot has. The server refuses in English and Spanish and asks the worker to choose the right lot or split the pounds. It checks at prepare and again at commit after locking the relevant lots, including frozen input plans. When total usable stock is genuinely short, the existing post-and-flag policy remains. Today and Assistant show the same conditional quantity limits; Floor uses the same server validator.

Kosher-ineligible source lots cannot make an extra-kosher Pack look sufficiently stocked. Missing supplier identities such as `UNKNOWN` are displayed as missing, not offered as real supplier codes; meaningful commingled aliases remain usable.

## Combined Floor contract

- Schema **4.4.0**, **22 operations**; every description is at most 300 characters.
- Generated instructions: **7,987 characters including header**, **7,857 body characters**. All ten permanent rules and the ledger-view guard remain.
- The exact PR145 no-lot / unreadable-lot rule uses `no_supplier_lot: true` and an empty supplier code; it never substitutes UNKNOWN or N/A. The redundant RECEIVE pre-flight sentence is trimmed to “Search first per shared rules.”
- PR146's signed `packing_slip_url` rule and server behavior remain intact. No key is inserted into a packing-slip link.
- Shared cache versions are above PR145: `ledger-forms.css` 10, `ledger-ticket.js` 11, `fl-assistant.css` 10 and `fl-assistant.js` 18. Upstream receive assets and tests are preserved.

## Future rollout prerequisites — nothing executed here

The production freeze remains active. No merge, deploy, production write or Railway setting change is authorized by this work.

**Migration 087 must be explicitly applied before the PR147 application deployment**, after migrations 062 and 063. Both supplier-lot confirmation inserts and substitution/exclusion audit inserts depend on it.

The server switch `REQUIRE_MAKE_PACK_LOT_EVIDENCE` defaults **OFF**. Leave it OFF until the combined Floor GPT schema and instructions have been pasted and checked in a separately approved rollout. Once enabled, legacy `/make` and `/pack` commits must supply both non-null evidence arrays; empty arrays cannot confirm FIFO. Previews use the checked selection flow. The master key cannot identify Floor separately from older integrations, so enabling this switch applies to all callers of those two legacy routes.

Full combined-branch test results and cleanup evidence will be saved here before pushing. All testing uses disposable local PostgreSQL databases and local browser fixtures. The protected count worktree and `audits/fresh-start/` are untouched.
