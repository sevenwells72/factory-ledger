#!/usr/bin/env python3
"""One-time Google sign-in for the receiving evidence (docs/deployments/trial-recall.md).

Run by Michael, once, on his own Mac, AFTER the Google Cloud steps in the doc:

    python3 scripts/google_oauth_setup.py ~/Downloads/client_secret_XXXX.json

What it does:

  1. opens the Google "Allow" page in the browser, asking for exactly two
     permissions: see Google Drive files (read-only) and see Google Sheets
     (read-only). Nothing that can change, share or delete anything;
  2. after "Allow", receives Google's answer on this Mac only (127.0.0.1) and
     exchanges it for a refresh token;
  3. checks the token is limited to those two read-only permissions and that it
     can read the receiving Form sheet;
  4. saves the three values in ~/.config/factory-ledger/google_oauth.env
     (readable only by this Mac user). It never prints the token.

With --railway it then prints the exact `railway variables` commands to put
the three values on the two Railway services. It does not run them: setting a
production variable is a production change that needs Michael's approval.

Nothing here touches Factory Ledger's database.
"""
import argparse
import hashlib
import http.server
import json
import os
from pathlib import Path
import secrets
import sys
import urllib.parse
import webbrowser

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import google_evidence
import receiving_form

AUTH_URL = 'https://accounts.google.com/o/oauth2/v2/auth'
OUT = Path.home() / '.config' / 'factory-ledger' / 'google_oauth.env'
ACCOUNT_HINT = 'michael@sevenwellsgranola.com'


def read_client(path):
    data = json.loads(Path(path).read_text())
    client = data.get('installed') or {}
    if not client.get('client_id') or not client.get('client_secret'):
        raise SystemExit('That file is not a "Desktop app" OAuth client. In Google Cloud: Clients -> Create client '
                         '-> Application type "Desktop app", then download its JSON.')
    return client['client_id'], client['client_secret']


def wait_for_code(state):
    """Listen on 127.0.0.1 for Google's redirect. Returns (code, redirect_uri)."""
    answer = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            if query.get('state', [''])[0] == state:
                answer['code'] = query.get('code', [None])[0]
                answer['error'] = query.get('error', [None])[0]
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(b'<p style="font:18px sans-serif">Done. You can close this tab and go back to the '
                             b'Terminal.</p>')

        def log_message(self, *args):       # never log the code
            pass

    server = http.server.HTTPServer(('127.0.0.1', 0), Handler)
    return server, answer, f'http://127.0.0.1:{server.server_port}'


def main(argv=None):
    parser = argparse.ArgumentParser(description='One-time Google sign-in (read-only Drive + Sheets)')
    parser.add_argument('client_json', help='the "Desktop app" OAuth client JSON downloaded from Google Cloud')
    parser.add_argument('--railway', action='store_true', help='also print the Railway commands (does not run them)')
    args = parser.parse_args(argv)
    client_id, client_secret = read_client(args.client_json)
    state, verifier = secrets.token_urlsafe(24), secrets.token_urlsafe(64)
    challenge = __import__('base64').urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b'=').decode()
    server, answer, redirect = wait_for_code(state)
    url = AUTH_URL + '?' + urllib.parse.urlencode({
        'client_id': client_id, 'redirect_uri': redirect, 'response_type': 'code',
        'scope': ' '.join(sorted(google_evidence.SCOPES)), 'access_type': 'offline', 'prompt': 'consent',
        'state': state, 'code_challenge': challenge, 'code_challenge_method': 'S256', 'login_hint': ACCOUNT_HINT})
    print('Opening Google in your browser. Sign in as', ACCOUNT_HINT, 'and press Allow.')
    print('If the browser does not open, paste this address into it:\n\n' + url + '\n')
    webbrowser.open(url)
    while 'code' not in answer and 'error' not in answer:
        server.handle_request()
    server.server_close()
    if not answer.get('code'):
        raise SystemExit(f'Google did not give permission ({answer.get("error") or "no answer"}). Nothing was saved.')
    response = httpx.post(google_evidence.TOKEN_URL, timeout=30, data={
        'grant_type': 'authorization_code', 'code': answer['code'], 'client_id': client_id,
        'client_secret': client_secret, 'redirect_uri': redirect, 'code_verifier': verifier})
    body = response.json() if response.status_code == 200 else {}
    refresh = body.get('refresh_token')
    if not refresh:
        raise SystemExit(f'Google did not return a refresh token (answer {response.status_code}). Nothing was saved.')
    granted = set(str(body.get('scope') or '').split())
    if not granted or not granted <= google_evidence.SCOPES:
        raise SystemExit('Google granted more than the two read-only permissions. Nothing was saved. '
                         'Check the scopes on the consent screen (Data access).')
    env = {'GOOGLE_OAUTH_CLIENT_ID': client_id, 'GOOGLE_OAUTH_CLIENT_SECRET': client_secret,
           'GOOGLE_OAUTH_REFRESH_TOKEN': refresh}
    # Prove it works, read-only, before saving: read the receiving Form tab.
    sheet = google_evidence.Client(env=env).sheet(receiving_form.SHEET_ID, receiving_form.TAB)
    rows = receiving_form.parse_rows(sheet['values'], sheet.get('time_zone'))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    handle = os.open(OUT, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(handle, 'w') as out:
        out.write(''.join(f'{name}={value}\n' for name, value in env.items()))
    os.chmod(OUT, 0o600)
    print(f'\nAllowed: read-only Drive and read-only Sheets, nothing else.')
    print(f'Checked: the tab "{receiving_form.TAB}" was read ({len(rows)} answered rows).')
    print(f'Saved:   {OUT} (only you can read it). The token was not printed.')
    if args.railway:
        print('\nTo put the values on Railway (each is a production change; run only with approval), from the '
              'linked project folder:\n')
        for service in ('FastAPI', 'exceptions-sweep'):
            print(f'  set -a; . {OUT}; set +a; railway variables --service {service} '
                  '--set "GOOGLE_OAUTH_CLIENT_ID=$GOOGLE_OAUTH_CLIENT_ID" '
                  '--set "GOOGLE_OAUTH_CLIENT_SECRET=$GOOGLE_OAUTH_CLIENT_SECRET" '
                  '--set "GOOGLE_OAUTH_REFRESH_TOKEN=$GOOGLE_OAUTH_REFRESH_TOKEN"')
    return 0


if __name__ == '__main__':
    sys.exit(main())
