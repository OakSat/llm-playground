"""The /api surface."""

import json
from collections.abc import Iterator
from contextlib import contextmanager

import httpx
import pytest
import respx
from flask.testing import FlaskClient

from playground.config import Features, Settings
from playground.ollama_client import OllamaClient
from playground.web.app import create_app

from .payloads import CHAT_RESPONSE, HOST, STREAM_BODY, TAGS_RESPONSE, THINKING_RESPONSE


@contextmanager
def make_client(**features: bool) -> Iterator[FlaskClient]:
    """Build a test client with a specific set of abilities switched on."""
    settings = Settings(
        ollama_host=HOST,
        default_model="llama3.2",
        features=Features(**features),
    )
    with OllamaClient(HOST, timeout=5.0) as ollama:
        app = create_app(settings, ollama)
        app.config.update(TESTING=True)
        with app.test_client() as test_client:
            yield test_client


@pytest.fixture
def api() -> Iterator[FlaskClient]:
    with make_client() as test_client:
        yield test_client


def show_route(capabilities: list[str] | None = None) -> None:
    respx.post(f"{HOST}/api/show").mock(
        httpx.Response(200, json={"capabilities": capabilities or ["completion", "tools"]})
    )


def sse_events(body: bytes) -> list[tuple[str, dict]]:
    """Parse an SSE response into (event, data) pairs."""
    events = []
    for block in body.decode().strip().split("\n\n"):
        if not block.strip():
            continue
        name = ""
        payload = ""
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line.removeprefix("event: ")
            elif line.startswith("data: "):
                payload = line.removeprefix("data: ")
        events.append((name, json.loads(payload)))
    return events


# -- health and config -----------------------------------------------------


@respx.mock
def test_health_ok(api: FlaskClient) -> None:
    respx.get(f"{HOST}/api/version").mock(httpx.Response(200, json={"version": "0.34.1"}))
    respx.get(f"{HOST}/api/tags").mock(httpx.Response(200, json=TAGS_RESPONSE))
    response = api.get("/api/health")
    assert response.status_code == 200
    assert response.json["ok"] is True
    assert response.json["models"] == 2


@respx.mock
def test_health_is_503_when_ollama_is_down(api: FlaskClient) -> None:
    """The Docker healthcheck depends on this status code."""
    respx.get(f"{HOST}/api/version").mock(side_effect=httpx.ConnectError("refused"))
    response = api.get("/api/health")
    assert response.status_code == 503
    assert response.json["ok"] is False


@respx.mock
def test_health_survives_a_failing_model_list(api: FlaskClient) -> None:
    respx.get(f"{HOST}/api/version").mock(httpx.Response(200, json={"version": "0.34.1"}))
    respx.get(f"{HOST}/api/tags").mock(httpx.Response(500, text="boom"))
    response = api.get("/api/health")
    assert response.status_code == 200
    assert response.json["models"] is None


def test_config_lists_every_feature(api: FlaskClient) -> None:
    response = api.get("/api/config")
    assert set(response.json["features"]) == set(Features.names())
    assert response.json["defaults"]["model"] == "llama3.2"


# -- models ----------------------------------------------------------------


@respx.mock
def test_models_are_enriched_with_capabilities(api: FlaskClient) -> None:
    respx.get(f"{HOST}/api/tags").mock(httpx.Response(200, json=TAGS_RESPONSE))
    show_route(["completion", "thinking"])
    models = api.get("/api/models").json["models"]
    assert len(models) == 2
    assert "thinking" in models[0]["capabilities"]
    assert models[0]["parameter_size"] == "3.2B"


@respx.mock
def test_models_narrows_to_the_default_when_switching_is_off() -> None:
    """Switching is the gated ability; knowing the current model is not."""
    respx.get(f"{HOST}/api/tags").mock(httpx.Response(200, json=TAGS_RESPONSE))
    show_route()
    with make_client(model_switcher=False) as client:
        models = client.get("/api/models").json["models"]
    assert [m["name"] for m in models] == ["llama3.2:latest"]


# -- chat ------------------------------------------------------------------


@respx.mock
def test_chat_returns_content_and_stats(api: FlaskClient) -> None:
    respx.post(f"{HOST}/api/chat").mock(httpx.Response(200, json=CHAT_RESPONSE))
    response = api.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert response.status_code == 200
    assert response.json["content"] == "Hello there!"
    assert response.json["stats"]["context_used"] == 35


@respx.mock
def test_chat_forwards_options_and_schema(api: FlaskClient) -> None:
    route = respx.post(f"{HOST}/api/chat").mock(httpx.Response(200, json=CHAT_RESPONSE))
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    api.post(
        "/api/chat",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "options": {"temperature": 0.2, "seed": 3},
            "format": schema,
        },
    )
    sent = json.loads(route.calls[0].request.content)
    assert sent["options"] == {"temperature": 0.2, "seed": 3}
    assert sent["format"] == schema


@respx.mock
def test_upstream_failure_is_not_reported_as_success(api: FlaskClient) -> None:
    """The v0.1 regression guard: a failed generation must not return 200."""
    respx.post(f"{HOST}/api/chat").mock(httpx.Response(500, json={"error": "runner died"}))
    response = api.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert response.status_code == 502
    assert "runner died" in response.json["error"]


@respx.mock
def test_unreachable_ollama_is_503(api: FlaskClient) -> None:
    respx.post(f"{HOST}/api/chat").mock(side_effect=httpx.ConnectError("refused"))
    response = api.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert response.status_code == 503


@respx.mock
def test_missing_model_is_404(api: FlaskClient) -> None:
    respx.post(f"{HOST}/api/chat").mock(httpx.Response(404, json={"error": "not found"}))
    response = api.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert response.status_code == 404


@respx.mock
def test_thinking_is_returned_separately(api: FlaskClient) -> None:
    show_route(["completion", "thinking"])
    respx.post(f"{HOST}/api/chat").mock(httpx.Response(200, json=THINKING_RESPONSE))
    response = api.post(
        "/api/chat",
        json={"messages": [{"role": "user", "content": "2+2"}], "think": True},
    )
    assert response.json["content"].strip() == "4"
    assert response.json["thinking"].strip()


@respx.mock
def test_thinking_on_an_incapable_model_is_400(api: FlaskClient) -> None:
    show_route(["completion"])
    chat = respx.post(f"{HOST}/api/chat").mock(httpx.Response(200, json=CHAT_RESPONSE))
    response = api.post(
        "/api/chat",
        json={"messages": [{"role": "user", "content": "hi"}], "think": True},
    )
    assert response.status_code == 400
    assert chat.call_count == 0


# -- request validation ----------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"messages": []},
        {"messages": "hi"},
        {"messages": [{"role": "wizard", "content": "hi"}]},
        {"messages": [{"role": "user"}]},
        {"messages": [{"role": "user", "content": 42}]},
        {"messages": [{"role": "user", "content": "hi"}], "options": {"temprature": 1}},
        {"messages": [{"role": "user", "content": "hi"}], "options": "hot"},
    ],
)
def test_malformed_bodies_are_rejected(api: FlaskClient, body: dict) -> None:
    assert api.post("/api/chat", json=body).status_code == 400


def test_misspelled_option_names_the_allowed_set(api: FlaskClient) -> None:
    response = api.post(
        "/api/chat",
        json={"messages": [{"role": "user", "content": "hi"}], "options": {"temprature": 1}},
    )
    assert "temprature" in response.json["error"]
    assert "temperature" in response.json["error"]


# -- streaming -------------------------------------------------------------


@respx.mock
def test_stream_emits_sse_deltas_then_done(api: FlaskClient) -> None:
    respx.post(f"{HOST}/api/chat").mock(httpx.Response(200, content=STREAM_BODY))
    response = api.post(
        "/api/chat/stream", json={"messages": [{"role": "user", "content": "count"}]}
    )
    assert response.status_code == 200
    assert response.mimetype == "text/event-stream"

    events = sse_events(response.data)
    assert [name for name, _ in events] == ["delta", "delta", "delta", "done"]
    assert "".join(data["content"] for _, data in events) == "1, 2"
    assert events[-1][1]["stats"]["completion_tokens"] == 14


@respx.mock
def test_stream_reports_failure_as_an_sse_error_event(api: FlaskClient) -> None:
    """Headers are already sent, so the status code cannot carry the failure."""
    respx.post(f"{HOST}/api/chat").mock(side_effect=httpx.ConnectError("refused"))
    response = api.post("/api/chat/stream", json={"messages": [{"role": "user", "content": "x"}]})
    events = sse_events(response.data)
    assert events[-1][0] == "error"
    assert events[-1][1]["status"] == 503


@respx.mock
def test_stream_sets_headers_that_stop_proxy_buffering(api: FlaskClient) -> None:
    respx.post(f"{HOST}/api/chat").mock(httpx.Response(200, content=STREAM_BODY))
    response = api.post("/api/chat/stream", json={"messages": [{"role": "user", "content": "x"}]})
    assert response.headers["Cache-Control"] == "no-cache"
    assert response.headers["X-Accel-Buffering"] == "no"


# -- static assets ---------------------------------------------------------


def test_serves_the_interface_at_the_root(api: FlaskClient) -> None:
    response = api.get("/")
    assert response.status_code == 200
    assert b"LLM Playground" in response.data


@pytest.mark.parametrize(
    "path",
    ["/style.css", "/js/main.js", "/js/api.js", "/js/state.js", "/js/ui.js"],
)
def test_serves_the_split_frontend_assets(api: FlaskClient, path: str) -> None:
    """The page is no longer a single file, so each part must be reachable."""
    assert api.get(path).status_code == 200
