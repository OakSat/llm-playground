# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An educational playground for local small language models, backed by Ollama. It has two surfaces sharing one client: a Flask web app (`src/playground/web/`) and a `slm` CLI (`src/playground/cli/`). The point of the project is pedagogical — design choices favour making model behaviour visible over the shortest implementation.

## Commands

```bash
uv sync --all-groups                 # install, including dev tools
uv run python -m playground.web.app  # serve at http://localhost:5000
uv run slm models|extract|eval       # the CLI

uv run ruff check . && uv run ruff format --check .
uv run mypy                          # strict, src only
uv run pytest -m "not integration"   # mocked; needs no Ollama. This is what CI runs
uv run pytest -m integration         # real model; skips if Ollama is unreachable
uv run pytest tests/test_cli.py::test_valid_extraction   # a single test
```

**On macOS, port 5000 is held by the AirPlay Receiver** and answers every request with 403. Use `PLAYGROUND_PORT=5055` when running or testing locally.

## Architecture

`src/playground/ollama_client.py` is the only module that talks to Ollama; both the web app and the CLI go through it. Change API behaviour there, not in a route or a command.

- **Streaming is NDJSON inbound, SSE outbound.** Ollama emits one JSON object per line with deltas in `message.content`; only the final line has `done: true` and the stats. `routes.py` reframes this as SSE because hosted LLM APIs use SSE. Once stream headers are sent the status can't change, so mid-stream failures arrive as an `error` event carrying the intended status.
- **Durations from Ollama are nanoseconds.** `Stats` converts; never use the raw fields directly.
- **The server is stateless.** Conversation history lives in the browser and is re-sent each turn. That is deliberate — it makes context growth visible and lets comparison lanes diverge without server-side session state.
- **Capabilities are queried, not assumed.** `POST /api/show` returns a `capabilities` list. Sending `think` to a model without `thinking` is a hard 400 from Ollama, so the client checks first and raises `CapabilityError`. The UI only offers the reasoning toggle for capable models.
- **Errors carry `http_status`** (`errors.py`), so routes map failures without re-deriving the mapping.

## Feature flags

Seven `FEATURE_*` flags in `config.py`, documented in `.env.TEMPLATE`. They are **enforced server-side** (409), not merely hidden in the UI — that distinction is a teaching point, so don't "simplify" it into client-only gating. Two behave differently on purpose: `stats` and `reasoning` strip fields from a successful response rather than refusing it, and disabling `model_switcher` narrows `/api/models` to the default model instead of erroring.

When adding an ability, add its flag, enforce it in the route, add a `data-feature` attribute to its control, and extend the parametrized matrix in `tests/test_features.py`.

## Conventions

- Model names are matched tolerantly — config says `llama3.2`, Ollama reports `llama3.2:latest`. Use `_matches` (server) / `sameModel` (client); exact comparison silently selects the wrong model.
- Frontend is vanilla ES modules with no build step, and should stay that way. All model output goes through `textContent`, never `innerHTML`.
- CSS sets `display` on layout classes, so a global `[hidden]` rule keeps the attribute working. Adding a rule that sets `display` on a hideable element will silently break feature and capability gating.
- Tests mock the HTTP layer with `respx` using payloads recorded from a real Ollama (`tests/payloads.py`). If you change what the client parses, re-record rather than hand-editing fixtures, and check the integration tests still pass.
- `slm eval` prints a majority-class baseline, and warms each model before timing. Neither is decoration: without the baseline an accuracy figure is unreadable, and without the warm-up the first row measures disk.

## Cautions

- `llama4` is 67 GB — never put it in a default sweep or example command.
- Reasoning models count thinking tokens in `eval_count`, so their tokens/sec is not comparable with a non-reasoning model's.
