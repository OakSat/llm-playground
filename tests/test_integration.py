"""End-to-end checks against a real Ollama.

These are excluded from CI (`pytest -m "not integration"`) and skip
automatically when Ollama is unreachable, so they never block a run. Their
job is to catch the mocked suite drifting away from the real wire format.

Run them with:  uv run pytest -m integration
"""

import os

import pytest

from playground.config import DEFAULT_HOST, normalise_host
from playground.errors import CapabilityError
from playground.ollama_client import OllamaClient

pytestmark = pytest.mark.integration

MODEL = "llama3.2"


@pytest.fixture(scope="module")
def live() -> OllamaClient:
    host = normalise_host(os.environ.get("OLLAMA_HOST", DEFAULT_HOST))
    with OllamaClient(host, timeout=120.0) as client:
        if not client.health().reachable:
            pytest.skip(f"No Ollama at {host}")
        if not any(m.name.startswith(MODEL) for m in client.list_models()):
            pytest.skip(f"Model {MODEL} is not installed")
        yield client


def test_health_and_models(live: OllamaClient) -> None:
    assert live.health().version
    assert live.list_models()


def test_real_round_trip_reports_plausible_stats(live: OllamaClient) -> None:
    result = live.chat(
        MODEL,
        [{"role": "user", "content": "Reply with exactly: pong"}],
        options={"temperature": 0},
    )
    assert result.content.strip()
    assert result.stats.context_used and result.stats.context_used > 0
    speed = result.stats.tokens_per_second
    # A laptop-class model lands in the tens-to-hundreds range; anything
    # outside it means the nanosecond conversion has gone wrong.
    assert speed is not None and 0.1 < speed < 10_000


def test_real_streaming_reassembles(live: OllamaClient) -> None:
    chunks = list(
        live.stream_chat(
            MODEL,
            [{"role": "user", "content": "Count: 1 2 3"}],
            options={"temperature": 0},
        )
    )
    assert len(chunks) > 1, "expected several NDJSON lines"
    assert "".join(c.content for c in chunks).strip()
    assert chunks[-1].done is True
    assert chunks[-1].stats is not None
    assert all(not c.done for c in chunks[:-1])


def test_real_structured_output_obeys_schema(live: OllamaClient) -> None:
    import json

    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
    }
    result = live.chat(
        MODEL,
        [{"role": "user", "content": "Alon is 30 years old."}],
        options={"temperature": 0},
        response_format=schema,
    )
    parsed = json.loads(result.content)
    assert set(parsed) == {"name", "age"}
    assert isinstance(parsed["age"], int)


def test_capability_gate_matches_reality(live: OllamaClient) -> None:
    """llama3.2 genuinely lacks `thinking`; Ollama would answer 400."""
    assert "thinking" not in live.capabilities(MODEL)
    with pytest.raises(CapabilityError):
        live.chat(MODEL, [{"role": "user", "content": "hi"}], think=True)
