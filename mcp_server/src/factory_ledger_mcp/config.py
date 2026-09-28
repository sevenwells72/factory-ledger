"""Phase 1 configuration: locked by default, loopback-only development access."""

import os
from dataclasses import dataclass, field
from urllib.parse import urlsplit


@dataclass(frozen=True)
class Settings:
    environment: str = "local"
    auth_mode: str = "locked"
    dev_token: str = field(default="", repr=False)
    test_api_key: str = field(default="", repr=False)
    dev_role: str = "reader"
    ledger_url: str = "http://127.0.0.1:8100"

    def __post_init__(self):
        if self.environment not in {"local", "test", "production"}:
            raise ValueError("MCP_ENV must be local, test, or production")
        if self.auth_mode not in {"locked", "local_stub"}:
            raise ValueError("Google OAuth is TODO; supported modes are locked and local_stub")
        if self.dev_role not in {"admin", "floor", "reader"}:
            raise ValueError("MCP_DEV_ROLE must be admin, floor, or reader")
        if self.auth_mode == "local_stub":
            if self.environment not in {"local", "test"} or os.getenv("RAILWAY_ENVIRONMENT_ID"):
                raise ValueError("Local auth stub cannot run in production or on Railway")
            if len(self.dev_token) < 16:
                raise ValueError(
                    "Set MCP_DEV_TOKEN to a local-only token of at least 16 characters"
                )
        url = urlsplit(self.ledger_url)
        if (
            url.scheme != "http"
            or url.hostname not in {"127.0.0.1", "::1"}
            or url.username
            or url.password
            or url.path not in {"", "/"}
            or url.query
            or url.fragment
        ):
            raise ValueError(
                "Phase 1 requires a numeric loopback HTTP ledger URL; remote APIs blocked"
            )

    @classmethod
    def from_env(cls):
        return cls(
            environment=os.getenv("MCP_ENV", "local"),
            auth_mode=os.getenv("MCP_AUTH_MODE", "locked"),
            dev_token=os.getenv("MCP_DEV_TOKEN", ""),
            test_api_key=os.getenv("MCP_TEST_API_KEY", ""),
            dev_role=os.getenv("MCP_DEV_ROLE", "reader"),
            ledger_url=os.getenv("MCP_LEDGER_API_URL", "http://127.0.0.1:8100"),
        )
