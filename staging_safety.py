"""Database isolation checks. No connections and no credential-bearing errors."""
import os
from urllib.parse import parse_qsl, unquote, urlsplit

# Non-secret identities already recorded in tests/conftest.py and deployment config.
PRODUCTION_PROJECT_REF = "vrafvwcdpcijvxdvefpr"
PRODUCTION_DATABASE_HOST = "aws-1-us-east-1.pooler.supabase.com"


def database_host(database_url):
    try:
        parts = urlsplit(database_url)
        host = unquote(parts.hostname or "").lower().rstrip(".")
        if (
            parts.scheme not in ("postgres", "postgresql")
            or not host or any(c in host for c in ",/ \\")
            or not parts.username or not parts.path.lstrip("/")
            or parts.fragment
        ):
            raise ValueError
        # libpq allows query-string routing overrides. Do not validate one host
        # and then connect to a different host, service, user or database.
        if any(k not in {"sslmode", "connect_timeout", "application_name"}
               for k, _ in parse_qsl(parts.query, keep_blank_values=True)):
            raise ValueError
        _ = parts.port
        return host
    except (TypeError, ValueError):
        raise RuntimeError("Invalid staging database URI or routing override") from None


def assert_staging_database(database_url, environment=None, production_host=None):
    environment = os.getenv("ENVIRONMENT", "") if environment is None else environment
    if environment.strip().lower() != "staging":
        return
    host = database_host(database_url)
    configured = (production_host if production_host is not None
                  else os.getenv("PRODUCTION_DATABASE_HOST", "")).strip().lower().rstrip(".")
    if not configured:
        raise RuntimeError("Staging requires PRODUCTION_DATABASE_HOST")
    forbidden = {configured, PRODUCTION_DATABASE_HOST,
                 f"db.{PRODUCTION_PROJECT_REF}.supabase.co"}
    if host in forbidden or PRODUCTION_PROJECT_REF in unquote(database_url).lower():
        raise RuntimeError("Staging database matches production; refusing startup and data sweeps")
    if os.getenv("PGHOSTADDR") or os.getenv("PGSERVICE") or os.getenv("PGSERVICEFILE"):
        raise RuntimeError("Staging refuses libpq routing overrides")
    expected = os.getenv("STAGING_DATABASE_HOST", "").strip().lower().rstrip(".")
    if expected and host != expected:
        raise RuntimeError("Staging database does not match STAGING_DATABASE_HOST")
    expected_ref = os.getenv("STAGING_DATABASE_PROJECT_REF", "").strip().lower()
    if expected_ref:
        parts = urlsplit(database_url)
        user = unquote(parts.username or "").lower()
        if host != f"db.{expected_ref}.supabase.co" and user != f"postgres.{expected_ref}":
            raise RuntimeError("Staging database does not match STAGING_DATABASE_PROJECT_REF")
