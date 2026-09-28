# The application only. Ollama runs as its own service -- see compose.yaml.
#
# v0.1 installed Ollama into this image with `curl | sh` and started both
# processes from one entrypoint. Splitting them means the model cache
# survives an image rebuild, and the app can equally point at an Ollama
# running on the host or on another machine.

FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.7.3 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# pyproject references both of these, so they must be present to build.
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src

# --no-editable matters: the default editable install would leave the
# venv pointing at /app/src, which does not exist in the runtime stage.
RUN uv sync --frozen --no-dev --no-editable


FROM python:3.12-slim AS runtime

RUN useradd --create-home --uid 1000 playground

WORKDIR /app
COPY --from=builder --chown=playground:playground /app/.venv /app/.venv
COPY --chown=playground:playground frontend ./frontend
COPY --chown=playground:playground examples ./examples

# PLAYGROUND_FRONTEND_DIR is required: the package lives in
# site-packages, so the interface cannot be located relative to it.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PLAYGROUND_FRONTEND_DIR=/app/frontend \
    PLAYGROUND_PORT=5000 \
    OLLAMA_HOST=http://ollama:11434

USER playground
EXPOSE 5000

# /api/health answers 503 while Ollama is unreachable, so the container is
# reported unhealthy until it can actually serve a request.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request as r; r.urlopen('http://localhost:5000/api/health')"

CMD ["python", "-m", "playground.web.app"]
