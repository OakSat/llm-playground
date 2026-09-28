# Architecture

Two surfaces, one client. The web playground and the `slm` CLI both talk to
Ollama through the same `OllamaClient`; neither knows anything about the API
that the other does not.

```
  browser ──HTTP/SSE──▶ Flask (playground.web) ──┐
                                                 ├──▶ OllamaClient ──HTTP/NDJSON──▶ Ollama
  terminal ───────────▶ slm  (playground.cli) ───┘
```

| Module | Responsibility |
| --- | --- |
| `playground.config` | Settings and the feature flags, read from the environment |
| `playground.ollama_client` | The only code that speaks to Ollama |
| `playground.errors` | Typed failures, each carrying an `http_status` |
| `playground.web` | Flask app factory and the `/api` blueprint |
| `playground.cli` | `models`, `extract` and `eval` |

---

## Why not just run `ollama`?

v0.1 called `subprocess.run(["ollama", "run", model, prompt])` per request.
That one decision blocked everything: a CLI invocation returns only when it is
finished (no streaming), takes a single prompt (no conversation), exposes no
sampling parameters, cannot be given a schema, and reports no token counts. It
also discarded `stderr` and the exit code, so a failed generation reached the
browser as `HTTP 200` with an empty body.

Talking to the HTTP API directly is what makes every current feature possible.

---

## NDJSON in, SSE out

Ollama streams **NDJSON**: one JSON object per line, text deltas in
`message.content`, and only the final object carrying `done: true` with the
timing statistics.

The browser is served **Server-Sent Events** instead. The translation costs a
few lines in `routes.py` and is worth it because every major hosted LLM API
(OpenAI, Anthropic) streams SSE — so what you learn here transfers.

Two consequences worth knowing:

- `EventSource` can only issue `GET`, and a chat turn must `POST` its messages.
  The client therefore reads the stream with `fetch()` and
  `response.body.getReader()`, reassembling frames that may straddle chunks.
- Once a stream's headers are sent, the status code can no longer change. A
  mid-stream failure is delivered as an `error` **event** with the intended
  status inside the payload, not as an HTTP status.

---

## The server is stateless

Conversation history lives in the browser and is sent in full with every
request. Nothing is stored between turns.

This is a deliberate teaching choice as much as a design one:

- The context window visibly grows, because you are the one re-sending it.
- Two comparison lanes can diverge, since each keeps its own history.
- A comparison is just two concurrent requests. No session, no lane state, no
  coordination on the server.

The cost is that the client must re-send everything each turn — which is
exactly the cost real applications pay, and exactly what the context counter in
the statistics strip is showing you.

---

## Durations are nanoseconds

Ollama reports `total_duration`, `load_duration` and `eval_duration` in
nanoseconds. `Stats` converts before computing anything, so
`tokens_per_second` is `eval_count / (eval_duration / 1e9)`.

`load_duration` is worth watching on its own: loading a cold model takes
seconds, while generating the reply may take milliseconds. That is why the
interface shows a spinner before the first token, and why the evaluation
harness warms each model before timing it.

---

## Capabilities are asked for, not assumed

`POST /api/show` returns a `capabilities` list per model. `deepseek-r1` reports
`["tools", "thinking", "completion"]`; `llama3.2` reports
`["completion", "tools"]`.

This matters because asking a model to think when it cannot is a hard `400`
from Ollama, not a silent no-op. So:

- the client checks capabilities before sending `think`, and raises a typed
  `CapabilityError` without spending a round trip;
- the interface only offers the reasoning toggle for models that declare it;
- in a comparison, each lane decides independently, so pairing a reasoning
  model with a plain one cannot fail.

---

## Feature flags are enforced, not hidden

Each ability has a `FEATURE_*` flag. Two separate things happen when one is off:

1. `GET /api/config` reports it, and the interface hides the control.
2. The route **refuses the request** with `409`.

The second is the real one. Hiding a control is presentation; anyone can open
a terminal. Try it:

```bash
FEATURE_STREAMING=0 uv run python -m playground.web.app
curl -X POST localhost:5000/api/chat/stream \
     -H 'Content-Type: application/json' \
     -d '{"messages":[{"role":"user","content":"hi"}]}'
# {"error":"The 'streaming' feature is disabled. ...","type":"FeatureDisabledError"}
```

Two flags behave differently, because they govern display rather than
capability: with `FEATURE_STATS` or `FEATURE_REASONING` off, the request
succeeds and those fields are stripped from the response. And disabling
`FEATURE_MODEL_SWITCHER` narrows `/api/models` to the default model instead of
failing — the gated ability is *switching*, not *knowing* what the current
model can do.

---

## Testing

The HTTP layer is mocked with `respx` using payloads recorded from a real
Ollama, so the whole suite runs with nothing installed — that is what keeps CI
green on a plain runner.

A handful of tests are marked `integration` and exercise a real model. They
skip when Ollama is unreachable and are excluded from CI. Their job is to catch
the recorded fixtures drifting away from the real wire format.
