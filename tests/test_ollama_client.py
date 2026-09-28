"""The shared Ollama client.

Every test mocks the HTTP layer, so none of this needs Ollama running.
"""

import json

import httpx
import pytest
import respx

from playground.errors import (
    CapabilityError,
    OllamaResponseError,
    OllamaUnavailableError,
)
from playground.ollama_client import OllamaClient, Stats

from .payloads import (
    CAPABILITIES,
    CHAT_RESPONSE,
    HOST,
    STREAM_BODY,
    TAGS_RESPONSE,
    THINKING_RESPONSE,
    ndjson,
)


def _show_route(respx_mock: respx.MockRouter) -> None:
    """Answer /api/show using the recorded capability table."""

    def responder(request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        return httpx.Response(200, json={"capabilities": CAPABILITIES.get(model, [])})

    respx_mock.post(f"{HOST}/api/show").mock(side_effect=responder)


# -- statistics ------------------------------------------------------------


def test_tokens_per_second_converts_nanoseconds() -> None:
    """Ollama reports durations in nanoseconds; a naive read is 1e9x wrong."""
    stats = Stats(eval_count=14, eval_duration_ns=258_452_000)
    assert stats.tokens_per_second == pytest.approx(54.17, abs=0.01)


def test_context_used_is_prompt_plus_completion() -> None:
    assert Stats(prompt_eval_count=31, eval_count=14).context_used == 45


def test_stats_degrade_gracefully_when_absent() -> None:
    """A cached or empty response may omit timings; that is not an error."""
    empty = Stats()
    assert empty.tokens_per_second is None
    assert empty.context_used is None
    assert empty.load_seconds is None


def test_zero_duration_does_not_divide_by_zero() -> None:
    assert Stats(eval_count=5, eval_duration_ns=0).tokens_per_second is None


# -- introspection ---------------------------------------------------------


@respx.mock
def test_health_reports_version(client: OllamaClient) -> None:
    respx.get(f"{HOST}/api/version").mock(httpx.Response(200, json={"version": "0.34.1"}))
    health = client.health()
    assert health.reachable is True
    assert health.version == "0.34.1"


@respx.mock
def test_health_reports_unreachable_without_raising(client: OllamaClient) -> None:
    """The UI needs to render a banner, so health must not explode."""
    respx.get(f"{HOST}/api/version").mock(side_effect=httpx.ConnectError("refused"))
    health = client.health()
    assert health.reachable is False
    assert "ollama serve" in (health.detail or "")


@respx.mock
def test_list_models_parses_details(client: OllamaClient) -> None:
    respx.get(f"{HOST}/api/tags").mock(httpx.Response(200, json=TAGS_RESPONSE))
    models = client.list_models()
    assert [m.name for m in models] == ["llama3.2:latest", "deepseek-r1:latest"]
    assert models[0].parameter_size == "3.2B"
    assert models[0].quantization == "Q4_K_M"
    assert models[0].context_length == 131072


@respx.mock
def test_list_models_can_enrich_with_capabilities(client: OllamaClient) -> None:
    respx.get(f"{HOST}/api/tags").mock(httpx.Response(200, json=TAGS_RESPONSE))
    _show_route(respx.mock)
    models = {m.name: m for m in client.list_models(with_capabilities=True)}
    assert models["deepseek-r1:latest"].can_think is True
    assert models["llama3.2:latest"].can_think is False


@respx.mock
def test_capabilities_are_cached(client: OllamaClient) -> None:
    """The UI asks on every model switch; one lookup per model is enough."""
    route = respx.post(f"{HOST}/api/show").mock(
        httpx.Response(200, json={"capabilities": ["completion"]})
    )
    client.capabilities("llama3.2")
    client.capabilities("llama3.2")
    assert route.call_count == 1


# -- chat ------------------------------------------------------------------


@respx.mock
def test_chat_returns_content_and_stats(client: OllamaClient) -> None:
    respx.post(f"{HOST}/api/chat").mock(httpx.Response(200, json=CHAT_RESPONSE))
    result = client.chat("llama3.2", [{"role": "user", "content": "hi"}])
    assert result.content == "Hello there!"
    assert result.done_reason == "stop"
    assert result.stats.context_used == 35
    assert result.stats.tokens_per_second == pytest.approx(72.3, abs=0.1)


@respx.mock
def test_chat_sends_options_and_schema(client: OllamaClient) -> None:
    route = respx.post(f"{HOST}/api/chat").mock(httpx.Response(200, json=CHAT_RESPONSE))
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    client.chat(
        "llama3.2",
        [{"role": "user", "content": "hi"}],
        options={"temperature": 0, "seed": 7},
        response_format=schema,
    )
    sent = json.loads(route.calls[0].request.content)
    assert sent["options"] == {"temperature": 0, "seed": 7}
    assert sent["format"] == schema
    assert sent["stream"] is False


@respx.mock
def test_thinking_is_separate_from_the_answer(client: OllamaClient) -> None:
    _show_route(respx.mock)
    respx.post(f"{HOST}/api/chat").mock(httpx.Response(200, json=THINKING_RESPONSE))
    result = client.chat("deepseek-r1:latest", [{"role": "user", "content": "2+2"}], think=True)
    assert result.content.strip() == "4"
    assert "2+2" in result.thinking


# -- capability gating -----------------------------------------------------


@respx.mock
def test_thinking_on_unsupported_model_fails_before_generating(
    client: OllamaClient,
) -> None:
    """Ollama answers this with a raw 400; we catch it first and never call chat."""
    _show_route(respx.mock)
    chat = respx.post(f"{HOST}/api/chat").mock(httpx.Response(200, json=CHAT_RESPONSE))
    with pytest.raises(CapabilityError) as exc:
        client.chat("llama3.2:latest", [{"role": "user", "content": "hi"}], think=True)
    assert exc.value.http_status == 400
    assert chat.call_count == 0


# -- streaming -------------------------------------------------------------


@respx.mock
def test_stream_yields_deltas_then_final_stats(client: OllamaClient) -> None:
    respx.post(f"{HOST}/api/chat").mock(httpx.Response(200, content=STREAM_BODY))
    chunks = list(client.stream_chat("llama3.2", [{"role": "user", "content": "count"}]))

    assert "".join(c.content for c in chunks) == "1, 2"
    assert [c.done for c in chunks] == [False, False, False, True]

    final = chunks[-1]
    assert final.stats is not None
    assert final.stats.eval_count == 14
    assert final.done_reason == "stop"
    # Only the terminating line carries statistics.
    assert all(c.stats is None for c in chunks[:-1])


@respx.mock
def test_stream_skips_blank_keepalive_lines(client: OllamaClient) -> None:
    body = '\n{"message":{"content":"a"},"done":false}\n\n{"message":{"content":""},"done":true}\n'
    respx.post(f"{HOST}/api/chat").mock(httpx.Response(200, content=body))
    chunks = list(client.stream_chat("llama3.2", [{"role": "user", "content": "hi"}]))
    assert len(chunks) == 2


@respx.mock
def test_error_inside_a_stream_raises(client: OllamaClient) -> None:
    body = ndjson(
        {"message": {"content": "partial"}, "done": False},
        {"error": "model runner crashed"},
    )
    respx.post(f"{HOST}/api/chat").mock(httpx.Response(200, content=body))
    stream = client.stream_chat("llama3.2", [{"role": "user", "content": "hi"}])
    assert next(stream).content == "partial"
    with pytest.raises(OllamaResponseError, match="model runner crashed"):
        next(stream)


@respx.mock
def test_malformed_stream_line_raises(client: OllamaClient) -> None:
    respx.post(f"{HOST}/api/chat").mock(httpx.Response(200, content="not json\n"))
    with pytest.raises(OllamaResponseError, match="Malformed NDJSON"):
        list(client.stream_chat("llama3.2", [{"role": "user", "content": "hi"}]))


@respx.mock
def test_streaming_http_error_is_typed(client: OllamaClient) -> None:
    respx.post(f"{HOST}/api/chat").mock(
        httpx.Response(400, json={"error": '"llama3.2" does not support thinking'})
    )
    with pytest.raises(OllamaResponseError, match="does not support thinking"):
        list(client.stream_chat("llama3.2", [{"role": "user", "content": "hi"}]))


# -- failure mapping -------------------------------------------------------


@respx.mock
def test_unreachable_ollama_is_typed(client: OllamaClient) -> None:
    respx.post(f"{HOST}/api/chat").mock(side_effect=httpx.ConnectError("refused"))
    with pytest.raises(OllamaUnavailableError) as exc:
        client.chat("llama3.2", [{"role": "user", "content": "hi"}])
    assert exc.value.http_status == 503


@respx.mock
def test_missing_model_surfaces_as_404(client: OllamaClient) -> None:
    respx.post(f"{HOST}/api/chat").mock(
        httpx.Response(404, json={"error": "model 'nope' not found"})
    )
    with pytest.raises(OllamaResponseError) as exc:
        client.chat("nope", [{"role": "user", "content": "hi"}])
    assert exc.value.http_status == 404
    assert "not found" in exc.value.message


@respx.mock
def test_upstream_failure_is_never_reported_as_success(client: OllamaClient) -> None:
    """v0.1 swallowed stderr and returned HTTP 200 with an empty body."""
    respx.post(f"{HOST}/api/chat").mock(httpx.Response(500, text="boom"))
    with pytest.raises(OllamaResponseError) as exc:
        client.chat("llama3.2", [{"role": "user", "content": "hi"}])
    assert exc.value.http_status == 502
