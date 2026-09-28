# 🧠 LLM Playground

An educational playground for running small language models locally with
[Ollama](https://ollama.com/). It exists to make the behaviour of a local model
*visible*: what temperature actually does, what a context window costs, what a
reasoning model spends its tokens on, and whether a bigger model is really
better for your task.

It has two surfaces, deliberately:

- **A web playground** — chat with streaming, conversation memory, sampling
  controls, schema-forced output, a reasoning view, and side-by-side model
  comparison with live token statistics.
- **A command-line tool (`slm`)** — the same model used as *infrastructure*
  rather than a chat: batch extraction into a JSON schema, and an evaluation
  harness that scores models against gold labels.

Every ability can be switched off individually, so the project can be
introduced one concept at a time.

---

## Requirements

- [Ollama](https://ollama.com/download) with at least one model pulled
  (`ollama pull llama3.2`)
- Python 3.11+ and [uv](https://docs.astral.sh/uv/), *or* Docker

---

## Run it

### Locally

```bash
uv sync
uv run python -m playground.web.app
```

Then open **<http://localhost:5000>**.

> The page must be served by the app — opening `frontend/index.html` directly
> from the filesystem will not work, because it calls the API at a relative
> path and uses ES modules.

> **On macOS**, port 5000 is taken by the AirPlay Receiver, which answers every
> request with `403`. Use another port:
> `PLAYGROUND_PORT=5055 uv run python -m playground.web.app`

### With Docker

```bash
docker compose up
```

This starts Ollama and the app as separate services and pulls the default model
on first run. Models are kept in a named volume, so they survive a rebuild.
Open <http://localhost:5000> (or set `PLAYGROUND_HOST_PORT=5055` on macOS).

---

## The CLI

```bash
# What is installed, and what can it do?
uv run slm models

# Extract structured data from a batch, validating every row.
uv run slm extract \
  --input examples/support_tickets.jsonl \
  --schema examples/schemas/ticket.json \
  --model llama3.2 --out results.jsonl

# Score models against the gold labels in the dataset.
uv run slm eval \
  --input examples/support_tickets.jsonl \
  --schema examples/schemas/ticket.json \
  --models llama3.2,deepseek-r1 --temperatures 0,1.0
```

`eval` prints valid-JSON rate, per-field accuracy, latency and tokens/sec,
alongside a **majority-class baseline** — the score you would get by always
guessing the commonest label. A model that cannot beat the baseline has not
learned the task.

Measured on the 30 sample tickets with `llama3.2`: **73%** accuracy at
temperature 0, **68%** at 1.0, against a **37%** baseline. On the first eight
rows, `llama3.2` scored **88%** at 0.61s per row while `deepseek-r1` scored
**79%** at 27.65s — bigger and reasoning is not automatically better, and
here the cost gap was 45×.

---

## Configuration

Copy `.env.TEMPLATE` to `.env`. Beyond the host, model, port and timeout, each
ability has its own flag:

| Flag | Ability |
| --- | --- |
| `FEATURE_STREAMING` | Tokens appear as they are generated |
| `FEATURE_MULTI_TURN` | Earlier turns are remembered |
| `FEATURE_MODEL_SWITCHER` | Models can be chosen in the interface |
| `FEATURE_COMPARE` | Two models run side by side |
| `FEATURE_STRUCTURED_OUTPUT` | Replies can be constrained to a JSON schema |
| `FEATURE_REASONING` | A reasoning model's scratchpad is shown |
| `FEATURE_STATS` | Token counts, speed and context use are reported |

Flags are **enforced by the server**, not just hidden in the interface — a
disabled ability answers `409` even when called directly with curl. Seeing that
difference is one of the exercises.

---

## Learn from it

- **[docs/exercises.md](docs/exercises.md)** — guided experiments: what
  temperature really changes, what happens when the context window fills,
  whether a schema improves accuracy, and what reasoning costs.
- **[docs/architecture.md](docs/architecture.md)** — how a request flows, why
  the server is stateless, and why NDJSON becomes SSE at the boundary.

---

## Development

```bash
uv sync --all-groups
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest -m "not integration"   # mocked; needs no Ollama
uv run pytest -m integration         # runs against a real model
```

---

## License

MIT — see [LICENSE](LICENSE).
