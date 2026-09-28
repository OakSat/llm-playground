"""Feature flags are enforced by the server, not just hidden in the UI.

Every ability must be refusable from outside the browser: a request made
with curl against a disabled ability has to come back 409, otherwise the
flag is decoration rather than configuration.
"""

import httpx
import pytest
import respx

from playground.config import Features

from .payloads import CHAT_RESPONSE, HOST, STREAM_BODY, TAGS_RESPONSE
from .test_routes import make_client, show_route, sse_events

CONVERSATION = [
    {"role": "user", "content": "first"},
    {"role": "assistant", "content": "reply"},
    {"role": "user", "content": "second"},
]


def stub_ollama() -> None:
    respx.post(f"{HOST}/api/chat").mock(httpx.Response(200, json=CHAT_RESPONSE))
    respx.get(f"{HOST}/api/tags").mock(httpx.Response(200, json=TAGS_RESPONSE))
    show_route(["completion", "thinking"])


@pytest.mark.parametrize("name", Features.names())
def test_config_reports_each_flag_as_disabled(name: str) -> None:
    with make_client(**{name: False}) as client:
        assert client.get("/api/config").json["features"][name] is False


# -- each ability, refused at the wire when switched off -------------------


@respx.mock
def test_streaming_disabled_refuses_the_stream_endpoint() -> None:
    stub_ollama()
    with make_client(streaming=False) as client:
        response = client.post(
            "/api/chat/stream", json={"messages": [{"role": "user", "content": "hi"}]}
        )
    assert response.status_code == 409
    assert "FEATURE_STREAMING=1" in response.json["error"]


@respx.mock
def test_multi_turn_disabled_refuses_a_conversation() -> None:
    stub_ollama()
    with make_client(multi_turn=False) as client:
        response = client.post("/api/chat", json={"messages": CONVERSATION})
    assert response.status_code == 409


@respx.mock
def test_multi_turn_disabled_still_allows_a_single_turn() -> None:
    """Disabling memory must not disable the playground."""
    stub_ollama()
    with make_client(multi_turn=False) as client:
        response = client.post(
            "/api/chat",
            json={
                "messages": [
                    {"role": "system", "content": "be brief"},
                    {"role": "user", "content": "hi"},
                ]
            },
        )
    assert response.status_code == 200


@respx.mock
def test_model_switcher_disabled_refuses_another_model() -> None:
    stub_ollama()
    with make_client(model_switcher=False) as client:
        response = client.post(
            "/api/chat",
            json={"model": "deepseek-r1", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 409


@respx.mock
def test_model_switcher_disabled_still_allows_the_default_model() -> None:
    stub_ollama()
    with make_client(model_switcher=False) as client:
        response = client.post(
            "/api/chat",
            json={"model": "llama3.2", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 200


@respx.mock
def test_compare_disabled_refuses_a_comparison_request() -> None:
    """A lane declares itself with `compare`, so the server can refuse it."""
    stub_ollama()
    with make_client(compare=False) as client:
        response = client.post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": "hi"}], "compare": True},
        )
    assert response.status_code == 409


@respx.mock
def test_structured_output_disabled_refuses_a_schema() -> None:
    stub_ollama()
    with make_client(structured_output=False) as client:
        response = client.post(
            "/api/chat",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "format": {"type": "object"},
            },
        )
    assert response.status_code == 409


@respx.mock
def test_reasoning_disabled_refuses_thinking() -> None:
    stub_ollama()
    with make_client(reasoning=False) as client:
        response = client.post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": "hi"}], "think": True},
        )
    assert response.status_code == 409


@respx.mock
def test_reasoning_disabled_withholds_the_scratchpad() -> None:
    stub_ollama()
    with make_client(reasoning=False) as client:
        response = client.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert "thinking" not in response.json


@respx.mock
def test_stats_disabled_withholds_statistics() -> None:
    """Statistics are displayed, not requested, so the flag strips them."""
    stub_ollama()
    with make_client(stats=False) as client:
        response = client.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert response.status_code == 200
    assert "stats" not in response.json


@respx.mock
def test_stats_disabled_withholds_statistics_when_streaming() -> None:
    respx.post(f"{HOST}/api/chat").mock(httpx.Response(200, content=STREAM_BODY))
    with make_client(stats=False) as client:
        response = client.post(
            "/api/chat/stream", json={"messages": [{"role": "user", "content": "hi"}]}
        )
    events = sse_events(response.data)
    assert events[-1][0] == "done"
    assert "stats" not in events[-1][1]


# -- flags do not interfere with one another -------------------------------


@respx.mock
def test_disabling_one_ability_leaves_the_others_working() -> None:
    stub_ollama()
    with make_client(compare=False) as client:
        assert client.post("/api/chat", json={"messages": CONVERSATION}).status_code == 200
        assert client.get("/api/models").status_code == 200


@respx.mock
def test_everything_off_still_serves_a_single_prompt() -> None:
    """The floor of the product: one prompt, one reply, no extras."""
    stub_ollama()
    all_off = dict.fromkeys(Features.names(), False)
    with make_client(**all_off) as client:
        response = client.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert response.status_code == 200
    assert response.json["content"] == "Hello there!"
