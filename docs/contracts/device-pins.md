# Phone PIN sign-in (on top of personal-key sessions)

Branch `feat/pin-device-signin`. Builds on the live key sessions ([key-sessions.md](key-sessions.md), migration 077). Not a second sign-in system: PR #96 (A11) stays parked and is not used.

## What staff do

1. **Once per phone:** sign in with the personal key, open **My PIN**, type the personal key once more and choose a 4-digit PIN. The phone is now "enrolled" to that person.
2. **Every shift:** open Factory Ledger on that phone, type the PIN in the usual sign-in field. The sign-in lasts **12 hours** and survives closing the browser.
3. **Change PIN:** My PIN → Change my PIN (needs the current PIN, on that phone).
4. **Forgot / locked:** the owner presses **Reset PIN** on *Staff & phones*; the person chooses a new PIN on My PIN within 60 minutes.

The personal key keeps working everywhere ("Use personal key instead" under the PIN field). The Floor and Office GPTs (master key) are untouched.

## Switches

| Setting (FastAPI) | Effect |
|---|---|
| `KEY_SESSIONS_ENABLED=1` | Required (already on in production). |
| `PIN_SIGNIN_ENABLED=1` | Turns phone PINs on, but only once migration 089 exists (checked live, cached once found), so setting it early cannot break ordinary sign-ins. Default/unset/0: every PIN route returns 404, nothing reads the 089 tables, and open phone sessions end on their next request. Personal-key sessions are unaffected. |

`GET /auth/session/config` adds `pin_sign_in` and `pin_session_hours: 12`. The dashboard shows PIN links (`[data-fl-pin-ui]`) only while `pin_sign_in` is true.

## How it is secured

- **Device secret.** Enrollment returns 32 random bytes (`device_secret`) once; the phone keeps it in `localStorage` (`fl-device`). The server stores only its SHA-256 (`key_devices.secret_hash`). A PIN is useless without this secret, so a PIN only works on the enrolled phone.
- **PIN hash.** PBKDF2-SHA256, 200,000 rounds, random 16-byte salt **plus the device secret** as salt material. The database alone cannot be brute-forced back to PINs. PINs are never stored in the browser.
- **Simple 4-digit PINs allowed** (owner decision 2026-10-29, e.g. 1111, 2222). The PIN's job is to say *who* is entering on a phone, not to keep people out; there is no PIN-strength requirement. A PIN still only works on its own enrolled phone (device secret), and the per-phone lockouts are unchanged.
- **Enrollment** requires a session opened with the personal key (`POST /auth/session`), not a PIN session, a raw actor key, or any shared key, **and the personal key typed again** (checked against that person, counted in the personal-key sign-in limits), so a signed-in screen left open cannot be used to add a phone. That key session becomes the phone's 12-hour session. At most 3 phones per person; re-enrolling the same phone replaces it.
- **Rate limits per phone, never shared between people:** 5 wrong PINs in a row lock that phone for 15 minutes; 10 block it (and sign it out) until the owner resets the PIN. A correct PIN resets the count. Requests with unknown, revoked or malformed device secrets go into one separate bucket (30 per 10 minutes for all such phones together); that bucket never touches an enrolled phone. The personal-key exchange keeps its existing limits.
- **Sessions.** PIN sessions are rows in `key_actor_sessions` with `device_id`, a fixed 12-hour `expires_at` (activity does not extend it), the same token format (`fls_`), the same role and attribution as the personal key. They end at once when the phone is revoked/reset/blocked, when the person is deactivated, when their personal key is rotated (the phone is retired too) or when `PIN_SIGNIN_ENABLED` is turned off.
- **Errors never echo** a PIN or device secret (422s on `/auth/pin*` and `/auth/device*` are replaced by a fixed message).

## Routes

| Route | Who | Purpose |
|---|---|---|
| `POST /auth/device/enroll {actor_key,pin,label,replace_device_secret?}` | personal-key session + key again | Enroll this phone; returns `device_secret` + a 12-hour session |
| `POST /auth/pin {device_secret,pin}` | public | PIN sign-in → session |
| `POST /auth/pin/reset {device_secret,new_pin}` | public, only inside an owner reset window | Choose a new PIN → session |
| `POST /auth/pin/change {device_secret,current_pin,new_pin}` | that phone's PIN session | Change own PIN (wrong current PIN counts as a failure) |
| `POST /auth/device/forget {device_secret}` | own session | Remove this phone |
| `GET /auth/devices` | owner session | People and their phones: label, status, times. No secrets or hashes |
| `POST /auth/devices/{id}/revoke` · `/reset-pin` | owner session | Remove a phone · open a 60-minute new-PIN window. Works on non-owners' phones and the owner's own (e.g. a lost one), never another owner's |
| `POST /auth/actors/{id}/remove-access` | owner session | Revoke all phones and end all sessions of a non-owner (personal key still works) |
| `POST /auth/actors/{id}/active {active}` | owner session | Deactivate/reactivate a non-owner person (role unchanged) |

Session-only routes are listed in `key_sessions.ROUTES`; the master, dashboard and read-only keys and raw personal keys get 403 there. Every route that ends sessions takes the same per-person advisory lock as `key_sessions.resolve`, first, so they cannot deadlock.

**Owner note:** a PIN reset lets whoever holds that phone choose a new PIN for 60 minutes. For a lost or stolen phone use **Remove phone**, not Reset PIN.

**Security review (2026-10-28, independent pass):** no critical/high findings. Fixed: owner actions on other owners; enrollment from an unattended signed-in screen (key now asked again); switch-before-migration breaking sign-ins; phone marker not carried through internal re-authorisation; lock order. Accepted: one shared bucket for unrecognised phones (worst case a revoked phone sees "try later" instead of being forgotten); attempts tables are not pruned (same as 077); the down migration drops the device audit table (export it first if needed).

## Migration 089

Additive, idempotent: `key_devices`, `key_device_attempts`, `key_device_audit` (RLS on, no policies, revoked from PUBLIC/anon/authenticated) and `key_actor_sessions.device_id`. No change to `actors`, roles, tickets or ledger rows. Down: `migrations/down/089_device_pins_down.sql` (set `PIN_SIGNIN_ENABLED=0` first).

## Staging plan (each step needs Michael's approval)

1. Apply `migrations/089_device_pins.sql` to the **staging** database (port 5432, as the app/table owner); verify the marker and RLS.
2. Deploy this branch to **FastAPI-staging** only (`railway up --service FastAPI-staging` from the branch worktree), then set `PIN_SIGNIN_ENABLED=1` on FastAPI-staging. Production is not touched.
3. Push the branch and open a **draft** PR: its Netlify deploy preview talks to the staging API automatically and shows the red STAGING banner.
4. Make sure Arturo and Luz have **staging** keys (staging accounts are separate from production).
5. Run the staging test below, then decide on production.

### Staging test (about 20 minutes, on their own phones, preview address with the STAGING banner)

**Arturo (floor), in FL Assistant:** sign in with the staging key → Menu → My PIN → type the key once more and choose a PIN → close the browser completely → reopen the preview → type the PIN → signed in as Arturo · floor. Record one practice receive (staging data). Type a wrong PIN once (shows attempts left), then the right one. Change the PIN on My PIN; sign out; sign in with the new PIN.

**Luz (office), on the dashboard:** same set-up from the dashboard **Sign in** button (or Office). Reopen the browser → PIN → open an order and print its packing slip. Use "Use personal key instead" once to confirm the key still works.

**Michael (owner):** sign in, open My PIN → *Manage staff phones* → both phones listed. Reset Luz's PIN → Luz picks a new one on My PIN within 60 minutes. Remove Arturo's phone → his next request asks him to sign in, and the PIN no longer works until he sets the phone up again with his key. On Arturo's phone type 5 wrong PINs → locked for 15 minutes, while Luz still signs in normally.

**Pass when:** every person signs in with a PIN after reopening the browser; names and roles are right; wrong PINs lock only that phone; owner reset/remove work; personal keys still work.

## Production rollout (later, each step approved)

Verified backup → merge (code is inert with the switch off) → apply 089 to production → deploy → set `PIN_SIGNIN_ENABLED=1` → enroll each phone in person. Rollback: `PIN_SIGNIN_ENABLED=0` (phone sessions end, personal keys keep working); migration down only if needed.
