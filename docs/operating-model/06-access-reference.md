# E. Production access reference (verified 2026-10-28)

No passwords, keys or PINs are included here, and none should ever be put in a document, chat or URL.

## Links

| Interface | Production | Staging (test copy) | Verified how |
|---|---|---|---|
| Dashboard | `https://cns-factory-ledger.netlify.app/` | Any other Netlify address, for example `https://deploy-preview-<N>--cns-factory-ledger.netlify.app/` | Production page loads (HTTP 200). The page code sends every non-production Netlify address to the staging back end. |
| Record entry | `https://cns-factory-ledger.netlify.app/entry.html` | `…--cns-factory-ledger.netlify.app/entry.html` | Page loads; **entry form hidden in production** (sign-in off) |
| Office | `https://cns-factory-ledger.netlify.app/office.html` | as above | Page loads; **hidden in production** |
| Exceptions | `https://cns-factory-ledger.netlify.app/exceptions.html` | as above | **Hidden in production** |
| Shift Summary | `https://cns-factory-ledger.netlify.app/shift-summary.html` | as above | Not in any menu |
| Receipt lookup / Today | `https://cns-factory-ledger.netlify.app/receipt.html` (`?view=today`) | as above | — |
| History | `https://cns-factory-ledger.netlify.app/history.html` | as above | — |
| Traceability | `https://cns-factory-ledger.netlify.app/traceability.html` | as above | — |
| FL Assistant | `https://cns-factory-ledger.netlify.app/fl-assistant.html` | `…--cns-factory-ledger.netlify.app/fl-assistant.html`, or `https://fastapi-staging-production-dd7b.up.railway.app/dash/fl-assistant` | The page loads, but **production refuses every Assistant request** ("not enabled"). Railway's production `/dash/fl-assistant` gives 404; staging gives 200. |
| Back end (not for people) | `https://fastapi-production-b73a.up.railway.app` | `https://fastapi-staging-production-dd7b.up.railway.app` | Both `/health` say healthy and database connected |
| Floor GPT | ChatGPT, "Factory Ledger - Floor 2.0" | none | ❓ The GPT link is not recorded in the repo. Get it from ChatGPT → My GPTs. |
| Office GPT | ChatGPT, "Factory-Ledger 2.0" | none | ❓ as above |

### What each one connects to

| Interface | Back end | Database |
|---|---|---|
| Production dashboard / Assistant / both GPTs | Railway **FastAPI** (production) | Supabase **MyFirstProject** (production) |
| Any Netlify preview address | Railway **FastAPI-staging** | Supabase **fl-staging** (test data, names start with "STAGING") |

The staging server refuses to start if it is ever pointed at the production database. That guard was checked when staging was set up on Oct 7.

### How to tell production from staging

- **Look at the address bar.** Production is exactly `cns-factory-ledger.netlify.app`. Anything with `deploy-preview-` or `--` in front, or any other `.netlify.app` name, is **staging**.
- **Look at the data.** Staging products and customers are named "STAGING Test …", and staging uses a test owner called "Michael (staging)".
- ⚠ **The page itself has no banner** saying PRODUCTION or STAGING. This is a real risk (issue F-7).

## Accounts in production (read from the database, 2026-10-28)

| Account name | Role | Active | Ever used? | Intended person |
|---|---|---|---|---|
| Blubber | owner | yes | yes (last 2026-10-28 12:42 ET) | Michael |
| Arturo | floor | yes | **never** | Arturo |
| Luz | office | yes | **never** | Luz (Liz?) |
| Miriam | office | yes | **never** | Miriam |

- **There is no "Liz" account.** If Liz is not Luz, an owner must create one.
- No account has ever started a dashboard sign-in session or an Assistant chat in production. Every routine entry since August is recorded as **"legacy-shared-key"**: 235 entries in the last 30 days, the latest on Oct 23.

### Roles

| Role | Who | Summary |
|---|---|---|
| owner | Michael | Everything, including approvals, kosher attestation, voiding anyone's entry and backdating over 14 days |
| floor | Arturo | Receive, make, pack, adjust, found, ship against orders, mark ready, void own same-day entries, lot fixes, resolve shortages and unidentified lots |
| office | Luz, Miriam | Customers, orders, expected receipts, receive, all shipping, lot fixes. No make, pack, adjust, found, void or exception resolution. |

Roles are enforced **only on the personal-sign-in routes** (dashboard forms and FL Assistant). The GPTs use the master key, which bypasses roles.

## How people sign in

| Interface | Today in production | After switch-on |
|---|---|---|
| Dashboard tabs (read, plus office edits) | Nothing to type: the page fetches the shared dashboard key itself | Same for reading |
| Record entry / Office / Exceptions / Receipt void / Shift Summary confirm | Hidden | Paste the **personal actor key** once. A shift session starts; it ends after 3 hours idle or on **Sign out**. |
| FL Assistant | Refused | Same personal key / shift session |
| Floor GPT, Office GPT | ChatGPT login; the GPT itself holds the **master key** | Should become read-only with a separate read-only key (prepared, not done) |
| PIN | not available | Built on a branch (PR #96), dormant by choice |

**Giving staff their key:**
- Hand it over in person, or through a password manager.
- Store it in the phone's or browser's password manager.
- **Never** send it in WhatsApp, email or a GPT chat, and never put it in a link.
- If it is ever exposed, the owner deactivates that account's key and issues a new one.

## Identity risks

1. **No names on GPT entries.** The biggest accountability gap: anyone with access to either GPT records as "legacy-shared-key".
2. **Shared dashboard key edits.** Order and expected-receipt edits on the dashboard tabs today carry no name either.
3. **The master key has been exposed.** It appeared in GPT instructions published Aug 13 and in repository history. Rotation is planned "about two weeks after stable operation". Until then, anyone holding it can write anything as anyone.
4. **Staging confusion.** Without a banner, someone given a preview link could record real work into the test system. That work would simply be missing from production.
5. **Key sharing.** If Liz and Luz are different people sharing the "Luz" key, accountability is lost. Each person needs their own account.
6. **Shared phones.** Always Sign out. The session lasts up to 3 hours idle.
