"""The /api blueprint.

Feature flags are enforced *here*, not merely reflected in the interface.
A disabled ability is refused with 409 even when called directly with
curl, which is the point: hiding a control is presentation, refusing the
request is enforcement.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from typing import Any

from flask import Blueprint, Response, current_app, jsonify, request, stream_with_context

from ..config import Settings
from ..errors import (
    FeatureDisabledError,
    InvalidRequestError,
    OllamaError,
    PlaygroundError,
)
from ..ollama_client import ChatChunk, OllamaClient

api = Blueprint("api", __name__, url_prefix="/api")

VALID_ROLES = frozenset({"system", "user", "assistant"})

# Sampling parameters the playground exposes. Anything else is rejected so
# a typo becomes an error rather than a silently ignored setting.
ALLOWED_OPTIONS = frozenset({"temperature", "top_p", "top_k", "seed", "num_ctx", "num_predict"})


def settings() -> Settings:
    value: Settings = current_app.config["SETTINGS"]
    return value


def client() -> OllamaClient:
    value: OllamaClient = current_app.config["OLLAMA_CLIENT"]
    return value


def require_feature(name: str) -> None:
    if not settings().features.is_enabled(name):
        raise FeatureDisabledError(name)


@api.get("/health")
def health() -> tuple[Response, int]:
    """Report whether Ollama is reachable. Used by the Docker healthcheck."""
    status = client().health()
    body: dict[str, Any] = {
        "ok": status.reachable,
        "ollama": {
            "host": settings().ollama_host,
            "reachable": status.reachable,
            "version": status.version,
            "detail": status.detail,
        },
    }
    if status.reachable:
        try:
            body["models"] = len(client().list_models())
        except OllamaError:
            # Reachability is the question being asked; a failure to also
            # list models should not turn a healthy answer into an error.
            body["models"] = None
    return jsonify(body), 200 if status.reachable else 503


@api.get("/config")
def config() -> Response:
    """Tell the interface which abilities to render, and the defaults to use."""
    current = settings()
    return jsonify(
        {
            "features": current.features.as_dict(),
            "defaults": {
                "model": current.default_model,
                "request_timeout": current.request_timeout,
            },
        }
    )


@api.get("/models")
def models() -> Response:
    """List installed models, enriched with their capabilities.

    With the model switcher disabled only the default model is returned:
    the ability being gated is *switching*, not knowing what the current
    model can do. The interface still needs capabilities to decide whether
    to offer the reasoning toggle.
    """
    installed = client().list_models(with_capabilities=True)
    current = settings()
    if not current.features.is_enabled("model_switcher"):
        installed = [m for m in installed if _matches(m.name, current.default_model)]
    return jsonify({"models": [m.as_dict() for m in installed]})


@api.post("/chat")
def chat() -> Response:
    """Generate a complete reply in one response."""
    payload = _parse_request(request.get_json(silent=True))
    result = client().chat(
        payload.model,
        payload.messages,
        options=payload.options,
        response_format=payload.response_format,
        think=payload.think,
    )
    body: dict[str, Any] = {
        "model": payload.model,
        "content": result.content,
        "done_reason": result.done_reason,
    }
    if settings().features.is_enabled("reasoning"):
        body["thinking"] = result.thinking
    if settings().features.is_enabled("stats"):
        body["stats"] = result.stats.as_dict()
    return jsonify(body)


@api.post("/chat/stream")
def chat_stream() -> Response:
    """Stream a reply as Server-Sent Events.

    Ollama speaks NDJSON, but every major hosted LLM API streams SSE, so
    the translation happens here: it costs a few lines and keeps the
    browser side conventional. Note that EventSource cannot issue a POST,
    so the client reads this with fetch() and a stream reader.
    """
    require_feature("streaming")
    payload = _parse_request(request.get_json(silent=True))
    show_stats = settings().features.is_enabled("stats")
    show_thinking = settings().features.is_enabled("reasoning")

    def events() -> Iterator[str]:
        try:
            stream = client().stream_chat(
                payload.model,
                payload.messages,
                options=payload.options,
                response_format=payload.response_format,
                think=payload.think,
            )
            for chunk in stream:
                event = "done" if chunk.done else "delta"
                yield _sse(event, _chunk_body(chunk, show_stats, show_thinking))
        except PlaygroundError as exc:
            # Headers are already sent, so the status code can no longer be
            # changed. The failure is reported as an SSE event instead, and
            # the intended status travels in the payload.
            yield _sse("error", {"error": str(exc), "status": exc.http_status})

    return Response(
        stream_with_context(events()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # stop proxies buffering the stream
        },
    )


def _chunk_body(chunk: ChatChunk, show_stats: bool, show_thinking: bool) -> dict[str, Any]:
    body: dict[str, Any] = {"content": chunk.content}
    if show_thinking and chunk.thinking:
        body["thinking"] = chunk.thinking
    if chunk.done:
        body["done_reason"] = chunk.done_reason
        if show_stats and chunk.stats is not None:
            body["stats"] = chunk.stats.as_dict()
    return body


def _sse(event: str, data: Mapping[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


class _ChatRequest:
    """A validated chat request."""

    __slots__ = ("messages", "model", "options", "response_format", "think")

    def __init__(
        self,
        model: str,
        messages: list[dict[str, str]],
        options: dict[str, Any] | None,
        response_format: Any,
        think: bool,
    ) -> None:
        self.model = model
        self.messages = messages
        self.options = options
        self.response_format = response_format
        self.think = think


def _parse_request(raw: Any) -> _ChatRequest:
    """Validate a chat body and enforce the feature flags it touches."""
    if not isinstance(raw, dict):
        raise InvalidRequestError("Expected a JSON object body.")

    current = settings()
    features = current.features

    model = raw.get("model") or current.default_model
    if not isinstance(model, str):
        raise InvalidRequestError("'model' must be a string.")
    if not features.is_enabled("model_switcher") and not _matches(model, current.default_model):
        raise FeatureDisabledError("model_switcher")

    # A request may declare itself part of a side-by-side comparison so the
    # server can refuse it when comparison is switched off.
    if raw.get("compare") and not features.is_enabled("compare"):
        raise FeatureDisabledError("compare")

    messages = _parse_messages(raw.get("messages"))
    if not features.is_enabled("multi_turn"):
        conversational = [m for m in messages if m["role"] != "system"]
        if len(conversational) > 1:
            raise FeatureDisabledError("multi_turn")

    response_format = raw.get("format")
    if response_format is not None and not features.is_enabled("structured_output"):
        raise FeatureDisabledError("structured_output")

    think = bool(raw.get("think"))
    if think and not features.is_enabled("reasoning"):
        raise FeatureDisabledError("reasoning")

    return _ChatRequest(model, messages, _parse_options(raw.get("options")), response_format, think)


def _parse_messages(raw: Any) -> list[dict[str, str]]:
    if not isinstance(raw, list) or not raw:
        raise InvalidRequestError("'messages' must be a non-empty list.")
    messages: list[dict[str, str]] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise InvalidRequestError(f"messages[{index}] must be an object.")
        role = entry.get("role")
        content = entry.get("content")
        if role not in VALID_ROLES:
            allowed = ", ".join(sorted(VALID_ROLES))
            raise InvalidRequestError(f"messages[{index}].role must be one of: {allowed}.")
        if not isinstance(content, str):
            raise InvalidRequestError(f"messages[{index}].content must be a string.")
        messages.append({"role": role, "content": content})
    return messages


def _parse_options(raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise InvalidRequestError("'options' must be an object.")
    unknown = set(raw) - ALLOWED_OPTIONS
    if unknown:
        allowed = ", ".join(sorted(ALLOWED_OPTIONS))
        raise InvalidRequestError(
            f"Unsupported option(s): {', '.join(sorted(unknown))}. Allowed: {allowed}."
        )
    return dict(raw)


def _matches(model_name: str, wanted: str) -> bool:
    """Compare model names, tolerating an implicit ``:latest`` tag."""
    return model_name == wanted or model_name.split(":")[0] == wanted.split(":")[0]
