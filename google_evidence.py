"""Read-only Google access to the receiving evidence (docs/deployments/trial-recall.md).

Two independent sources are read, never changed:

  * the Google Sheet "Receiving# (Responses)" (the receiving Form's answers);
  * the BOL photos in Google Drive that the sheet links to.

Access is OAuth as Michael's own Google account, through an INTERNAL app whose
only scopes are drive.readonly and spreadsheets.readonly. Three variables hold
the credentials (set on the Railway services, never in the repo):

    GOOGLE_OAUTH_CLIENT_ID, GOOGLE_OAUTH_CLIENT_SECRET, GOOGLE_OAUTH_REFRESH_TOKEN

This module can only read: every Sheets/Drive request is a GET, the one POST is
the token refresh, and a refreshed token that carries any scope beyond the two
read-only ones is refused (`SCOPE_TOO_WIDE`). Nothing is stored: a photo is
handed to the caller as bytes or as a stream and never written anywhere.
No database, no application imports.
"""
import os
import re
import threading
import time

import httpx

TOKEN_URL = 'https://oauth2.googleapis.com/token'
SHEETS_URL = 'https://sheets.googleapis.com/v4/spreadsheets'
DRIVE_URL = 'https://www.googleapis.com/drive/v3/files'
SCOPES = frozenset({'https://www.googleapis.com/auth/drive.readonly',
                    'https://www.googleapis.com/auth/spreadsheets.readonly'})
ENV_NAMES = ('GOOGLE_OAUTH_CLIENT_ID', 'GOOGLE_OAUTH_CLIENT_SECRET', 'GOOGLE_OAUTH_REFRESH_TOKEN')
MAX_PHOTO_BYTES = 15 * 1024 * 1024          # same cap as attachments.py and extraction.py
TIMEOUT = 30.0
_FILE_ID = re.compile(r'^[A-Za-z0-9_-]{10,200}$')
_FILE_FIELDS = 'id,name,mimeType,size,parents,trashed,webViewLink'


class GoogleError(Exception):
    """A read that did not work. `code` is one of: NOT_CONFIGURED, AUTH_FAILED,
    SCOPE_TOO_WIDE, NOT_FOUND, FORBIDDEN, TOO_LARGE, BAD_FILE_ID, UNAVAILABLE.
    The message never contains a credential."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def configured(env=None):
    env = os.environ if env is None else env
    return all((env.get(name) or '').strip() for name in ENV_NAMES)


def valid_file_id(file_id):
    return isinstance(file_id, str) and bool(_FILE_ID.match(file_id))


_token_lock = threading.Lock()
_token_cache = {}       # refresh token -> (access token, expires at)
_shared_http = None     # one connection pool for the whole process (httpx.Client is thread-safe)


def _http():
    global _shared_http
    with _token_lock:
        if _shared_http is None:
            _shared_http = httpx.Client(timeout=TIMEOUT)
        return _shared_http


class Client:
    """One read-only client. `http` is a seam for tests (an httpx.Client-like object)."""

    def __init__(self, env=None, http=None):
        self.env = os.environ if env is None else env
        self.http = http or _http()

    # -- token ---------------------------------------------------------------

    def _token(self):
        if not configured(self.env):
            raise GoogleError('NOT_CONFIGURED', 'Google access is not set up (the three GOOGLE_OAUTH_* variables).')
        refresh = self.env['GOOGLE_OAUTH_REFRESH_TOKEN'].strip()
        with _token_lock:
            cached = _token_cache.get(refresh)
            if cached and cached[1] > time.time() + 60:
                return cached[0]
            try:
                response = self.http.post(TOKEN_URL, data={
                    'grant_type': 'refresh_token', 'refresh_token': refresh,
                    'client_id': self.env['GOOGLE_OAUTH_CLIENT_ID'].strip(),
                    'client_secret': self.env['GOOGLE_OAUTH_CLIENT_SECRET'].strip()})
            except httpx.HTTPError as exc:
                raise GoogleError('UNAVAILABLE', f'Could not reach Google sign-in ({type(exc).__name__}).') from None
            try:
                body = response.json()
            except ValueError:
                body = {}
            if response.status_code != 200 or not isinstance(body, dict) or not body.get('access_token'):
                # invalid_grant = the Allow was withdrawn or the token expired: Michael must allow again.
                reason = str(body.get('error') or response.status_code) if isinstance(body, dict) else str(response.status_code)
                raise GoogleError('AUTH_FAILED', f'Google refused the saved sign-in ({reason[:60]}).')
            granted = set(str(body.get('scope') or '').split())
            if not granted or not granted <= SCOPES:
                # Never work with a token that could change anything.
                raise GoogleError('SCOPE_TOO_WIDE', 'The saved Google sign-in is not limited to the two read-only '
                                                    'permissions. Refusing to use it.')
            _token_cache[refresh] = (body['access_token'], time.time() + float(body.get('expires_in') or 0))
            return body['access_token']

    def _get(self, url, params=None):
        try:
            response = self.http.get(url, params=params, headers={'Authorization': f'Bearer {self._token()}'})
        except httpx.HTTPError as exc:
            raise GoogleError('UNAVAILABLE', f'Could not reach Google ({type(exc).__name__}).') from None
        self._check(response)
        return response

    @staticmethod
    def _check(response):
        if response.status_code == 200:
            return
        if response.status_code == 404:
            raise GoogleError('NOT_FOUND', 'Google has no such file or sheet (or it is not shared with this account).')
        if response.status_code in (401, 403):
            raise GoogleError('FORBIDDEN', 'Google did not allow this account to read it.')
        raise GoogleError('UNAVAILABLE', f'Google answered {response.status_code}.')

    # -- the sheet -----------------------------------------------------------

    def sheet(self, sheet_id, tab):
        """One tab as shown on screen: {'time_zone', 'gid', 'values': [[cell text, ...], ...]}."""
        if not valid_file_id(sheet_id):
            raise GoogleError('BAD_FILE_ID', 'That is not a Google Sheet id.')
        meta = self._get(f'{SHEETS_URL}/{sheet_id}',
                         {'fields': 'properties.timeZone,sheets.properties(sheetId,title)'}).json()
        gid = next((s['properties']['sheetId'] for s in meta.get('sheets', [])
                    if s.get('properties', {}).get('title') == tab), None)
        if gid is None:
            raise GoogleError('NOT_FOUND', f'The sheet has no tab named "{tab}".')
        quoted = "'" + tab.replace("'", "''") + "'"
        values = self._get(f'{SHEETS_URL}/{sheet_id}/values/{quoted}',
                           {'valueRenderOption': 'FORMATTED_VALUE', 'majorDimension': 'ROWS'}).json()
        return {'time_zone': (meta.get('properties') or {}).get('timeZone'), 'gid': gid,
                'values': [[str(cell) for cell in row] for row in values.get('values', [])]}

    # -- the photos ----------------------------------------------------------

    def file_metadata(self, file_id):
        if not valid_file_id(file_id):
            raise GoogleError('BAD_FILE_ID', 'That is not a Google Drive file id.')
        data = self._get(f'{DRIVE_URL}/{file_id}', {'fields': _FILE_FIELDS, 'supportsAllDrives': 'true'}).json()
        size = data.get('size')
        return {'id': data.get('id'), 'name': data.get('name'), 'mime_type': data.get('mimeType'),
                'size': int(size) if str(size or '').isdigit() else None,
                'parents': list(data.get('parents') or []), 'trashed': bool(data.get('trashed')),
                'web_view_link': data.get('webViewLink')}

    def open_media(self, file_id, max_bytes=MAX_PHOTO_BYTES):
        """The file's bytes as a stream: returns an iterator of chunks. Nothing is kept.

        Raises before the first chunk when Google refuses; stops with TOO_LARGE
        if the file turns out bigger than `max_bytes`.
        """
        if not valid_file_id(file_id):
            raise GoogleError('BAD_FILE_ID', 'That is not a Google Drive file id.')
        request = self.http.build_request('GET', f'{DRIVE_URL}/{file_id}',
                                          params={'alt': 'media', 'supportsAllDrives': 'true'},
                                          headers={'Authorization': f'Bearer {self._token()}'})
        try:
            response = self.http.send(request, stream=True)
        except httpx.HTTPError as exc:
            raise GoogleError('UNAVAILABLE', f'Could not reach Google Drive ({type(exc).__name__}).') from None
        if response.status_code != 200:
            response.close()
            self._check(response)

        def chunks():
            sent = 0
            try:
                for chunk in response.iter_bytes():
                    sent += len(chunk)
                    if sent > max_bytes:
                        raise GoogleError('TOO_LARGE', 'The photo is larger than 15 MB.')
                    yield chunk
            except httpx.HTTPError as exc:
                # Drive answered and then timed out or dropped the connection part-way through.
                raise GoogleError('UNAVAILABLE', f'Google Drive stopped sending the photo ({type(exc).__name__}).') from None
            finally:
                response.close()
        return chunks()

    def download(self, file_id, max_bytes=MAX_PHOTO_BYTES):
        """The whole file in memory (for the extraction model). Never written to disk."""
        return b''.join(self.open_media(file_id, max_bytes))


def sheet_row_url(sheet_id, gid, row_number):
    return f'https://docs.google.com/spreadsheets/d/{sheet_id}/edit#gid={gid}&range=A{row_number}'


def drive_file_url(file_id):
    return f'https://drive.google.com/file/d/{file_id}/view'
