FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.11.18 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable
ENV MCP_ENV=staging MCP_AUTH_MODE=locked MCP_WRITES_ENABLED=false
ENV MCP_LEDGER_API_URL=https://fastapi-staging-production-dd7b.up.railway.app PYTHONUNBUFFERED=1
RUN useradd --create-home appuser
USER appuser
CMD ["/app/.venv/bin/factory-ledger-mcp"]
