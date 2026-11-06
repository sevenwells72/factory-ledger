#!/usr/bin/env python3
"""O-5 read-only settings gate. See docs/O5-PROD-SIGNIN.md.

Only fixed PASS/FAIL labels leave this process. Railway variables are obtained
with explicit targeting and --no-local, never dumped, saved, or put in argv.
No application imports, session requests, database writes, or hosting mutations.
Requires authenticated railway/netlify CLIs and psycopg2 in this Python runtime.
"""
import argparse
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.error import HTTPError
from urllib.parse import parse_qsl, unquote, urlsplit
from urllib.request import Request, build_opener, HTTPRedirectHandler

PROJECT = '2206e070-d160-4528-a4f1-86a587ad88c3'
RAILWAY_ENV = 'production'  # Staging is a SERVICE in this same environment.
SERVICES = {'production': 'FastAPI', 'staging': 'FastAPI-staging'}
BASES = {
    'production': 'https://fastapi-production-b73a.up.railway.app',
    'staging': 'https://fastapi-staging-production-dd7b.up.railway.app',
}
SITE = 'https://cns-factory-ledger.netlify.app'
PREVIEW = 'https://deploy-preview-122--cns-factory-ledger.netlify.app'
PROD_REF = 'vrafvwcdpcijvxdvefpr'
PROD_POOLER = 'aws-1-us-east-1.pooler.supabase.com'
LIMIT = 2 * 1024 * 1024
ENV_LABELS = (
    'RAILWAY_TARGET', 'ENVIRONMENT', 'FORWARDED_ALLOW_IPS',
    'KEY_SESSIONS_ENABLED', 'PIN_LOGIN_DISABLED', 'ASSISTANT_ENABLED',
    'DATABASE_URL_PRESENT', 'DATABASE_TARGET', 'API_KEY_PRESENT',
    'DASHBOARD_API_KEY_PRESENT', 'SHARED_KEYS_DISTINCT', 'OPENAI_API_KEY_PRESENT',
    'ASSISTANT_MODEL', 'ASSISTANT_TRANSCRIBE_MODEL', 'SUPABASE_URL',
    'SUPABASE_SERVICE_ROLE_KEY_POLICY', 'EVIDENCE_STORAGE_DIR_POLICY',
    'PACKING_SLIP_SIGNING_KEY_POLICY', 'ORDER_LEGACY_CLOSE_CUTOFF_FORMAT', 'DATABASE_READ_ONLY',
    'MIGRATION_077', 'SESSION_TABLES_PRIVATE', 'SESSION_TABLE_ACCESS', 'SESSION_TABLE_COLUMNS',
    'NOTES_AUDIT_CONSTRAINTS', 'ASSISTANT_SCHEMA', 'ACTIVE_STAFF_ROLES',
    'DASHBOARD_ACTOR_KEYS_DISTINCT', 'READONLY_ACTOR_KEYS_DISTINCT',
    'HEALTH', 'DEPLOYED_ASSISTANT_FLAG', 'DEPLOYED_SESSION_FLAG', 'SESSION_IDLE_TIMEOUT',
    'SESSION_CONFIG_NO_STORE', 'NO_SESSION_CONFIG_COOKIE',
    'DASHBOARD_KEY_DELIVERY_MATCH', 'DASHBOARD_KEY_NO_STORE',
    'CORS_CONFIG_READ', 'CORS_SESSION_HEADERS',
)


class Results:
    def __init__(self):
        self.items = {}

    def check(self, name, condition):
        self.items[name] = bool(condition)

    def emit(self):
        for name, ok in self.items.items():
            print(('PASS ' if ok else 'FAIL ') + name)
        return 0 if self.items and all(self.items.values()) else 1


class StatusParser(argparse.ArgumentParser):
    def error(self, message):
        # Do not echo an accidental credential supplied as an argument.
        print('FAIL ARGUMENTS')
        raise SystemExit(1)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def request(url, method='GET', headers=None):
    # Deliberately no auth headers, cookies, redirects, proxy env, or POSTs.
    from urllib.request import ProxyHandler
    if method not in ('GET', 'OPTIONS'):
        raise ValueError
    try:
        with build_opener(ProxyHandler({}), NoRedirect()).open(
            Request(url, method=method, headers=headers or {}), timeout=12
        ) as response:
            body = response.read(LIMIT + 1)
            if len(body) > LIMIT:
                raise ValueError
            return response.status, response.headers, body
    except HTTPError as error:
        return error.code, error.headers, b''


def safe_request(url, method='GET', headers=None):
    try:
        return request(url, method, headers)
    except Exception:
        return 0, {}, b''


def decoded(body):
    try:
        return json.loads(body)
    except Exception:
        return {}


def no_store(headers):
    return 'no-store' in headers.get('Cache-Control', '').lower()


def clean_cli_environment():
    # Prevent an absent Railway variable from silently inheriting a local app
    # value. Preserve only CLI login/runtime inputs, not generic app/PG vars.
    allowed = ('PATH', 'HOME', 'USER', 'LOGNAME', 'TMPDIR', 'LANG', 'LC_ALL',
               'SYSTEMROOT', 'RAILWAY_TOKEN', 'RAILWAY_API_TOKEN',
               'NETLIFY_AUTH_TOKEN', 'XDG_CONFIG_HOME', 'SSL_CERT_FILE')
    return {key: os.environ[key] for key in allowed if key in os.environ}


def run_cli(argv, timeout=90):
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                          env=clean_cli_environment())


def database_target(env, target):
    """Match the existing staging guard; reject libpq routing overrides."""
    raw = env.get('DATABASE_URL', '')
    try:
        parts = urlsplit(raw)
        host = unquote(parts.hostname or '').lower().rstrip('.')
        user = unquote(parts.username or '').lower()
        if (parts.scheme not in ('postgres', 'postgresql') or not host
                or any(c in host for c in ',/ \\') or not user
                or not parts.path.lstrip('/') or parts.fragment):
            return False
        if any(k not in {'sslmode', 'connect_timeout', 'application_name'}
               for k, _ in parse_qsl(parts.query, keep_blank_values=True)):
            return False
        _ = parts.port
        if any(env.get(k) for k in ('PGHOSTADDR', 'PGSERVICE', 'PGSERVICEFILE')):
            return False
        production = (host == f'db.{PROD_REF}.supabase.co'
                      or (host == PROD_POOLER and user == f'postgres.{PROD_REF}'))
        if target == 'production':
            return production
        configured = env.get('PRODUCTION_DATABASE_HOST', '').strip().lower().rstrip('.')
        expected = env.get('STAGING_DATABASE_HOST', '').strip().lower().rstrip('.')
        ref = env.get('STAGING_DATABASE_PROJECT_REF', '').strip().lower()
        return bool(configured and host not in {configured, PROD_POOLER,
                    f'db.{PROD_REF}.supabase.co'}
                    and PROD_REF not in unquote(raw).lower()
                    and (not expected or expected == host)
                    and (not ref or host == f'db.{ref}.supabase.co'
                         or user == f'postgres.{ref}'))
    except Exception:
        return False


def database_checks(result, env, target_ok):
    labels = ENV_LABELS[ENV_LABELS.index('DATABASE_READ_ONLY'):ENV_LABELS.index('HEALTH')]
    for label in labels:
        result.check(label, False)
    if not target_ok:
        return
    conn = None
    try:
        import psycopg2
        conn = psycopg2.connect(env['DATABASE_URL'], connect_timeout=5)
        with conn.cursor() as cur:
            # Transaction-local ONLY: never poison Supabase's shared pooler.
            cur.execute('SET TRANSACTION READ ONLY')
            cur.execute("SET LOCAL statement_timeout = '5s'")
            cur.execute("SET LOCAL lock_timeout = '2s'")
            cur.execute('SHOW transaction_read_only')
            result.check('DATABASE_READ_ONLY', cur.fetchone()[0] == 'on')
            cur.execute("SELECT EXISTS (SELECT 1 FROM public.migration_markers WHERE name='077_key_sessions')")
            result.check('MIGRATION_077', cur.fetchone()[0])
            cur.execute("""SELECT count(*)=2 AND bool_and(c.relrowsecurity)
                AND bool_and(NOT EXISTS (SELECT 1 FROM pg_policy p WHERE p.polrelid=c.oid))
                AND bool_and(NOT EXISTS (SELECT 1 FROM aclexplode(coalesce(c.relacl,acldefault('r',c.relowner))) a
                    LEFT JOIN pg_roles r ON r.oid=a.grantee
                    WHERE a.grantee=0 OR r.rolname IN ('anon','authenticated')))
                FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
                WHERE n.nspname='public' AND c.relname IN ('key_actor_sessions','key_session_attempts')""")
            result.check('SESSION_TABLES_PRIVATE', cur.fetchone()[0])
            cur.execute("""SELECT
                (SELECT bool_and(has_table_privilege(current_user,'public.key_actor_sessions',p))
                    FROM unnest(ARRAY['SELECT','INSERT','UPDATE']) AS p)
                AND (SELECT bool_and(has_table_privilege(current_user,'public.key_session_attempts',p))
                    FROM unnest(ARRAY['SELECT','INSERT']) AS p)
                AND has_sequence_privilege(current_user,'public.key_actor_sessions_id_seq','USAGE')
                AND has_sequence_privilege(current_user,'public.key_session_attempts_id_seq','USAGE')
                AND NOT row_security_active('public.key_actor_sessions'::regclass)
                AND NOT row_security_active('public.key_session_attempts'::regclass)""")
            result.check('SESSION_TABLE_ACCESS', cur.fetchone()[0])
            required = {
                'key_actor_sessions': {'id','token_hash','actor_id','actor_key_hash','created_at',
                    'last_seen_at','expires_at','ended_at','ended_reason'},
                'key_session_attempts': {'id','source_hash','ok','attempted_at'},
            }
            cur.execute('SELECT table_name,column_name FROM information_schema.columns '
                        "WHERE table_schema='public' AND table_name=ANY(%s)", (list(required),))
            columns = cur.fetchall()
            result.check('SESSION_TABLE_COLUMNS', all(names <= {c for t,c in columns if t == table}
                                                     for table,names in required.items()))
            cur.execute("""SELECT count(*)=2 AND bool_and(CASE
                WHEN conname='actor_write_audit_target_table_check' THEN position('notes' IN pg_get_constraintdef(oid))>0
                ELSE position('PUT' IN pg_get_constraintdef(oid))>0 AND position('DELETE' IN pg_get_constraintdef(oid))>0 END)
                FROM pg_constraint WHERE conrelid='public.actor_write_audit'::regclass
                AND conname IN ('actor_write_audit_target_table_check','actor_write_audit_method_check')""")
            result.check('NOTES_AUDIT_CONSTRAINTS', cur.fetchone()[0])
            cur.execute("""SELECT EXISTS (SELECT 1 FROM public.migration_markers WHERE name='068_fl_assistant')
                AND to_regclass('public.assistant_sessions') IS NOT NULL
                AND to_regclass('public.assistant_turns') IS NOT NULL
                AND to_regclass('public.assistant_drafts') IS NOT NULL
                AND to_regclass('public.assistant_attachment_links') IS NOT NULL
                AND to_regclass('public.assistant_unmatched_words') IS NOT NULL""")
            result.check('ASSISTANT_SCHEMA', cur.fetchone()[0])
            shared = [env.get(k, '').strip() for k in ('API_KEY','DASHBOARD_API_KEY','READONLY_API_KEY')]
            hashes = [hashlib.sha256(k.encode()).hexdigest() for k in shared if k]
            cur.execute("""SELECT count(DISTINCT role)=3 FROM public.actors WHERE active
                AND role IN ('owner','office','floor') AND length(trim(name))>0
                AND key_hash ~ '^[a-f0-9]{64}$' AND NOT (key_hash=ANY(%s))""", (hashes,))
            result.check('ACTIVE_STAFF_ROLES', cur.fetchone()[0])
            cur.execute('SELECT NOT EXISTS (SELECT 1 FROM public.actors WHERE key_hash=%s)',
                        (hashlib.sha256(shared[1].encode()).hexdigest(),))
            result.check('DASHBOARD_ACTOR_KEYS_DISTINCT', bool(shared[1]) and cur.fetchone()[0])
            cur.execute('SELECT NOT EXISTS (SELECT 1 FROM public.actors WHERE active AND key_hash=%s)',
                        (hashlib.sha256(shared[2].encode()).hexdigest(),))
            result.check('READONLY_ACTOR_KEYS_DISTINCT', cur.fetchone()[0])
    except Exception:
        # Exception strings can include DSNs, query parameters and credentials.
        # Incomplete results retain FAIL; never print/log/chain the exception.
        pass
    finally:
        if conn is not None:
            try:
                conn.rollback()
                conn.close()
            except Exception:
                result.check('DATABASE_READ_ONLY', False)


def railway_child(args):
    result = Results()
    env = os.environ
    value = lambda k: env.get(k, '').strip()
    result.check('RAILWAY_TARGET', env.get('RAILWAY_PROJECT_ID') == PROJECT
                 and env.get('RAILWAY_ENVIRONMENT_NAME') == RAILWAY_ENV
                 and env.get('RAILWAY_SERVICE_NAME') == SERVICES[args.environment])
    result.check('ENVIRONMENT', value('ENVIRONMENT') == args.environment)
    result.check('FORWARDED_ALLOW_IPS', env.get('FORWARDED_ALLOW_IPS') == '*')
    result.check('KEY_SESSIONS_ENABLED', env.get('KEY_SESSIONS_ENABLED') == '1')
    result.check('PIN_LOGIN_DISABLED', env.get('PIN_LOGIN_ENABLED', '0') == '0')
    result.check('ASSISTANT_ENABLED', env.get('ASSISTANT_ENABLED') == '1')
    result.check('DATABASE_URL_PRESENT', value('DATABASE_URL'))
    target_ok = database_target(env, args.environment) and result.items['RAILWAY_TARGET']
    result.check('DATABASE_TARGET', target_ok)
    result.check('API_KEY_PRESENT', value('API_KEY'))
    result.check('DASHBOARD_API_KEY_PRESENT', value('DASHBOARD_API_KEY'))
    shared = [value(k) for k in ('API_KEY','DASHBOARD_API_KEY','READONLY_API_KEY') if value(k)]
    result.check('SHARED_KEYS_DISTINCT', len(shared) == len(set(shared)))
    result.check('OPENAI_API_KEY_PRESENT', value('OPENAI_API_KEY'))
    result.check('ASSISTANT_MODEL', env.get('ASSISTANT_MODEL', 'gpt-6.1-sol') == 'gpt-6.1-sol')
    result.check('ASSISTANT_TRANSCRIBE_MODEL', env.get('ASSISTANT_TRANSCRIBE_MODEL', 'gpt-transcribe') == 'gpt-transcribe')
    storage = urlsplit(value('SUPABASE_URL'))
    staging_volume = args.environment == 'staging' and value('EVIDENCE_STORAGE_DIR').startswith('/')
    result.check('SUPABASE_URL', staging_volume or (storage.scheme == 'https' and bool(storage.hostname)
                 and not storage.username and not storage.password
                 and storage.port in (None, 443) and storage.path in ('', '/')
                 and not storage.query and not storage.fragment
                 and (args.environment != 'production' or storage.hostname == f'{PROD_REF}.supabase.co')))
    result.check('SUPABASE_SERVICE_ROLE_KEY_POLICY', staging_volume or value('SUPABASE_SERVICE_ROLE_KEY'))
    result.check('EVIDENCE_STORAGE_DIR_POLICY', staging_volume or not env.get('EVIDENCE_STORAGE_DIR'))
    result.check('PACKING_SLIP_SIGNING_KEY_POLICY', value('PACKING_SLIP_SIGNING_KEY')
                 or (args.environment == 'staging' and value('API_KEY')))
    cutoff = value('ORDER_LEGACY_CLOSE_CUTOFF')
    try:
        result.check('ORDER_LEGACY_CLOSE_CUTOFF_FORMAT', not cutoff or date.fromisoformat(cutoff).isoformat() == cutoff)
    except ValueError:
        result.check('ORDER_LEGACY_CLOSE_CUTOFF_FORMAT', False)
    database_checks(result, env, target_ok)
    base = BASES[args.environment]
    status, _, _ = safe_request(base + '/health')
    result.check('HEALTH', status == 200)
    status, headers, body = safe_request(base + '/dash/fl-assistant')
    result.check('DEPLOYED_ASSISTANT_FLAG', status == 200 and no_store(headers)
                 and b'key-session.js' in body)
    origin = SITE if args.environment == 'production' else args.preview_url
    status, headers, body = safe_request(base + '/auth/session/config', headers={'Origin': origin})
    config = decoded(body)
    result.check('DEPLOYED_SESSION_FLAG', status == 200 and config.get('key_sessions_enabled') is True)
    result.check('SESSION_IDLE_TIMEOUT', status == 200 and config.get('idle_minutes') == 180)
    result.check('SESSION_CONFIG_NO_STORE', status == 200 and no_store(headers))
    result.check('NO_SESSION_CONFIG_COOKIE', status == 200 and not headers.get('Set-Cookie'))
    result.check('CORS_CONFIG_READ', status == 200 and headers.get('Access-Control-Allow-Origin') in ('*', origin))
    status, headers, body = safe_request(base + '/auth/dashboard-key')
    result.check('DASHBOARD_KEY_DELIVERY_MATCH', status == 200 and bool(value('DASHBOARD_API_KEY'))
                 and decoded(body).get('dashboard_key') == value('DASHBOARD_API_KEY'))
    result.check('DASHBOARD_KEY_NO_STORE', status == 200 and no_store(headers))
    status, headers, _ = safe_request(base + '/auth/session', method='OPTIONS', headers={
        'Origin': origin, 'Access-Control-Request-Method': 'POST',
        'Access-Control-Request-Headers': 'content-type,x-api-key,x-fl-background'})
    allowed = {v.strip().lower() for v in headers.get('Access-Control-Allow-Headers', '').split(',')}
    result.check('CORS_SESSION_HEADERS', status == 200
                 and headers.get('Access-Control-Allow-Origin') == origin
                 and 'POST' in headers.get('Access-Control-Allow-Methods', '').split(', ')
                 and {'content-type','x-api-key','x-fl-background'} <= allowed)
    # Fixed allowlist controls output order and makes omissions fail closed.
    clean = Results()
    for label in ENV_LABELS:
        clean.check(label, result.items.get(label, False))
    return clean.emit()


def hosted_dashboard(result, prefix, origin, path=''):
    status, headers, body = safe_request(origin + path + '/index.html')
    html = body.decode('utf-8', errors='replace')
    runtime = html.find('src="dashboard-runtime.js')
    session = html.find('src="key-session.js')
    result.check(prefix + '_HTTPS', status == 200)
    result.check(prefix + '_RUNTIME_FIRST', status == 200 and 0 <= runtime < session)
    if prefix == 'NETLIFY_PRODUCTION':
        result.check(prefix + '_NO_STORE', status == 200 and no_store(headers))
    status, _, body = safe_request(origin + path + '/dashboard-runtime.js')
    code = body.decode('utf-8', errors='replace')
    result.check(prefix + '_O7_RUNTIME', status == 200 and "base+'/auth/dashboard-key'" in code
                 and "cache:'no-store'" in code and 'KEY_TTL_MS=60000' in code)
    result.check(prefix + '_API_ROUTING', status == 200
                 and "PRODUCTION_SITE='cns-factory-ledger.netlify.app'" in code
                 and "if(host.endsWith('.netlify.app')&&host!==PRODUCTION_SITE)return STAGING;" in code
                 and "if(host===PRODUCTION_SITE||!location||location.protocol==='file:')return PRODUCTION;" in code
                 and 'return location.origin;' in code
                 and f"PRODUCTION='{BASES['production']}'" in code
                 and f"STAGING='{BASES['staging']}'" in code
                 and 'apiBase(root.location,root.FL_SESSION_API_BASE)' in code)
    # Existing canonical pages must not override the verified routing in HTML.
    result.check(prefix + '_NO_HTML_OVERRIDE', status == 200 and 'FL_SESSION_API_BASE' not in html)
    status, _, body = safe_request(origin + path + '/key-session.js')
    code = body.decode('utf-8', errors='replace')
    result.check(prefix + '_TOKEN_STORAGE', status == 200 and 'sessionStorage' in code
                 and 'X-API-Key' in code and 'document.cookie' not in code
                 and not re.search(r'credentials\s*:\s*[\'"]include', code))


def netlify_checks(result):
    labels = ('NETLIFY_CONFIG_READ', 'NETLIFY_REPOSITORY_BRANCH', 'NETLIFY_STATIC_BUILD',
              'NETLIFY_PREVIEWS', 'NETLIFY_PUBLISHED', 'NETLIFY_PUBLIC_ACCESS',
              'P1_28_NO_CUSTOM_DOMAIN')
    for label in labels:
        result.check(label, False)
    try:
        raw = run_cli(['netlify','api','getSite','--data',json.dumps({'site_id': urlsplit(SITE).hostname})], timeout=60)
        if raw.returncode:
            return
        site = json.loads(raw.stdout)
        build = site.get('build_settings') or {}
        deploy = site.get('published_deploy') or {}
        result.check('NETLIFY_CONFIG_READ', site.get('name') == 'cns-factory-ledger')
        result.check('NETLIFY_REPOSITORY_BRANCH', build.get('repo_path') == 'sevenwells72/factory-ledger'
                     and build.get('repo_branch') == 'main')
        # Empty dashboard/base UI is compatible with authoritative netlify.toml.
        result.check('NETLIFY_STATIC_BUILD', build.get('base') in (None,'','dashboard')
                     and str(build.get('dir','')).strip('/') in ('dashboard','.')
                     and not build.get('cmd'))
        result.check('NETLIFY_PREVIEWS', 'skip_prs' in build and build['skip_prs'] in (None, False)
                     and build.get('stop_builds') is False
                     and 'skip_automatic_builds' in build and build['skip_automatic_builds'] in (None, False))
        result.check('NETLIFY_PUBLISHED', deploy.get('state') == 'ready'
                     and deploy.get('context') == 'production' and deploy.get('branch') == 'main'
                     and not deploy.get('locked'))
        result.check('NETLIFY_PUBLIC_ACCESS', site.get('has_password') is False
                     and not site.get('disabled') and not site.get('identity_instance_id'))
        result.check('P1_28_NO_CUSTOM_DOMAIN', not site.get('custom_domain') and not site.get('domain_aliases'))
    except Exception:
        pass


def main(argv=None):
    parser = StatusParser(description=__doc__)
    parser.add_argument('environment', choices=tuple(SERVICES))
    parser.add_argument('--preview-url', default=PREVIEW,
                        help='Reviewed deploy preview origin; defaults to PR 122')
    parser.add_argument('--proxy-trust-reviewed', action='store_true', help=(
        'Attest proxy headers are enabled, Railway edge sanitizes forwarded headers, '
        'and no untrusted direct/private/TCP caller can reach this service.'))
    parser.add_argument('--in-railway', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not re.fullmatch(r'https://deploy-preview-\d+--cns-factory-ledger\.netlify\.app', args.preview_url):
        print('FAIL PREVIEW_URL')
        return 1
    if args.in_railway:
        return railway_child(args)
    result = Results()
    result.check('PROXY_TRUST_REVIEWED', args.proxy_trust_reviewed)
    try:
        run = run_cli(['railway','run','--project',PROJECT,'--environment',RAILWAY_ENV,
                       '--service',SERVICES[args.environment],'--no-local',sys.executable,
                       str(Path(__file__).resolve()),args.environment,'--preview-url',args.preview_url,
                       '--in-railway'], timeout=180)
        # Never replay raw CLI stdout/stderr: print only recognized fixed labels.
        lines = run.stdout.splitlines()
        valid = (run.returncode in (0,1) and len(lines) == len(ENV_LABELS)
                 and all(line in {'PASS '+label,'FAIL '+label}
                         for line,label in zip(lines,ENV_LABELS)))
        result.check('RAILWAY_READ_COMPLETED', valid)
        for label, line in zip(ENV_LABELS, lines if valid else ['']*len(ENV_LABELS)):
            result.check(label, line == 'PASS '+label)
    except Exception:
        result.check('RAILWAY_READ_COMPLETED', False)
        for label in ENV_LABELS:
            result.check(label, False)
    netlify_checks(result)
    hosted_dashboard(result, 'NETLIFY_PRODUCTION', SITE)
    status, headers, _ = safe_request(SITE.replace('https:', 'http:') + '/')
    result.check('NETLIFY_HTTPS_REDIRECT', status in (301,302,307,308)
                 and headers.get('Location') == SITE + '/')
    for page in ('eula', 'privacy'):
        status, _, body = safe_request(SITE + '/legal/' + page + '.html')
        result.check('NETLIFY_LEGAL_' + page.upper(), status == 200 and b'<html' in body.lower())
    if args.environment == 'staging':
        hosted_dashboard(result, 'NETLIFY_PREVIEW', args.preview_url)
    hosted_dashboard(result, 'RAILWAY_DASHBOARD', BASES[args.environment], '/dashboard')
    return result.emit()


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (Exception, KeyboardInterrupt):
        # Never expose unexpected credential-bearing errors.
        print('FAIL CHECK_INCOMPLETE')
        sys.exit(1)
