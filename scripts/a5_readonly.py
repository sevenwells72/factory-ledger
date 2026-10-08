"""Credential-safe, transaction-scoped read-only connections for A5 preflights."""
from contextlib import contextmanager, closing
import os
from urllib.parse import urlsplit

import psycopg2
from psycopg2.extras import RealDictCursor

from scripts.seed_staging import secret_file
from staging_safety import database_host


@contextmanager
def readonly_cursor(database_url_file):
    uri = secret_file(database_url_file)
    host = database_host(uri)  # rejects libpq URI routing overrides
    if any(os.getenv(key) for key in ('PGHOSTADDR', 'PGSERVICE', 'PGSERVICEFILE')):
        raise RuntimeError('Read-only preflight refuses libpq routing overrides')
    if host not in ('localhost', '127.0.0.1', '::1') and (urlsplit(uri).port or 5432) != 5432:
        raise RuntimeError('Use the session pooler on port 5432 for remote preflights')
    with closing(psycopg2.connect(uri, connect_timeout=10,
                                 application_name='a5_readonly_preflight')) as conn:
        conn.autocommit = True
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute('BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
            try:
                cur.execute("SET LOCAL statement_timeout='30s'")
                cur.execute('SET LOCAL search_path=public')
                yield cur
            finally:
                cur.execute('ROLLBACK')
