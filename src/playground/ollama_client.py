"""The single point of contact with Ollama.

Both surfaces of this project -- the web playground and the ``slm`` CLI --
go through this module. v0.1 instead shelled out to ``ollama run`` via
``subprocess``, which made streaming, multi-turn conversation, sampling
parameters, structured output and token accounting all impossible. Talking
to the HTTP API directly is what unlocks them.

Two behaviours of the API are worth knowing, because they are easy to guess
wrong (both verified against Ollama 0.34.1):

* Streaming responses are **NDJSON**, not SSE: one JSON object per line,
  with text deltas in ``message.content``. Only the final object carries
  ``done: true`` and the timing statistics.
* Durations are reported in **nanoseconds**.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import httpx

from .config import DEFAULT_TIMEOUT, normalise_host
from .errors import (
    CapabilityError,
    OllamaResponseError,
    OllamaUnavailableError,
)

NANOSECONDS = 1_000_000_000

# Chat messages as sent to the API: {"role": ..., "content": ...}
Message = Mapping[str, str]
JsonSchema = Mapping[str, Any]


def _seconds(nanoseconds: int | None) -> float | None:
    return None if nanoseconds is None else nanoseconds / NANOSECONDS


@dataclass(frozen=True, slots=True)
class Stats:
    """Timing and token counts reported by Ollama when a response completes."""

    total_duration_ns: int | None = None
    load_duration_ns: int | None = None
    prompt_eval_count: int | None = None
    prompt_eval_duration_ns: int | None = None
    eval_count: int | None = None
    eval_duration_ns: int | None = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Stats:
        return cls(
            total_duration_ns=payload.get("total_duration"),
            load_duration_ns=payload.get("load_duration"),
            prompt_eval_count=payload.get("prompt_eval_count"),
            prompt_eval_duration_ns=payload.get("prompt_eval_duration"),
            eval_count=payload.get("eval_count"),
            eval_duration_ns=payload.get("eval_duration"),
        )

    @property
    def tokens_per_second(self) -> float | None:
        """Generation speed, or None when Ollama reported no usable timing."""
        if not self.eval_count or not self.eval_duration_ns:
            return None
        return self.eval_count / (self.eval_duration_ns / NANOSECONDS)

    @property
    def context_used(self) -> int | None:
        """Tokens occupying the context window: prompt plus generated."""
        if self.prompt_eval_count is None and self.eval_count is None:
            return None
        return (self.prompt_eval_count or 0) + (self.eval_count or 0)

    @property
    def load_seconds(self) -> float | None:
        """Time spent loading the model. Dominates first-token latency when cold."""
        return _seconds(self.load_duration_ns)

    @property
    def total_seconds(self) -> float | None:
        return _seconds(self.total_duration_ns)

    def as_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_eval_count,
            "completion_tokens": self.eval_count,
            "context_used": self.context_used,
            "tokens_per_second": self.tokens_per_second,
            "load_seconds": self.load_seconds,
            "total_seconds": self.total_seconds,
        }


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """One installed model, as listed by ``/api/tags``."""

    name: str
    size_bytes: int | None = None
    parameter_size: str | None = None
    quantization: str | None = None
    context_length: int | None = None
    capabilities: frozenset[str] = frozenset()

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ModelInfo:
        details = payload.get("details") or {}
        return cls(
            name=payload.get("name", ""),
            size_bytes=payload.get("size"),
            parameter_size=details.get("parameter_size"),
            quantization=details.get("quantization_level"),
            context_length=details.get("context_length"),
        )

    def with_capabilities(self, capabilities: frozenset[str]) -> ModelInfo:
        return ModelInfo(
            name=self.name,
            size_bytes=self.size_bytes,
            parameter_size=self.parameter_size,
            quantization=self.quantization,
            context_length=self.context_length,
            capabilities=capabilities,
        )

    @property
    def can_think(self) -> bool:
        return "thinking" in self.capabilities

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "size_bytes": self.size_bytes,
            "parameter_size": self.parameter_size,
            "quantization": self.quantization,
            "context_length": self.context_length,
            "capabilities": sorted(self.capabilities),
        }


@dataclass(frozen=True, slots=True)
class ChatChunk:
    """One NDJSON line of a streaming response."""

    content: str = ""
    thinking: str = ""
    done: bool = False
    done_reason: str | None = None
    stats: Stats | None = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ChatChunk:
        message = payload.get("message") or {}
        done = bool(payload.get("done"))
        return cls(
            content=message.get("content") or "",
            # Reasoning models report their scratchpad separately from the
            # answer, so the UI can present it as a distinct block.
            thinking=message.get("thinking") or "",
            done=done,
            done_reason=payload.get("done_reason"),
            stats=Stats.from_payload(payload) if done else None,
        )


@dataclass(frozen=True, slots=True)
class ChatResult:
    """A complete, non-streamed response."""

    content: str
    thinking: str = ""
    done_reason: str | None = None
    stats: Stats = Stats()

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ChatResult:
        message = payload.get("message") or {}
        return cls(
            content=message.get("content") or "",
            thinking=message.get("thinking") or "",
            done_reason=payload.get("done_reason"),
            stats=Stats.from_payload(payload),
        )


@dataclass(frozen=True, slots=True)
class Health:
    reachable: bool
    version: str | None = None
    detail: str | None = None


class OllamaClient:
    """A small, typed wrapper over the Ollama HTTP API."""

    def __init__(
        self,
        host: str,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        client: httpx.Client | None = None,
    ) -> None:
        self.host = normalise_host(host)
        self.timeout = timeout
        self._client = client or httpx.Client(base_url=self.host, timeout=timeout)
        self._owns_client = client is None
        self._capability_cache: dict[str, frozenset[str]] = {}

    def __enter__(self) -> OllamaClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    # -- transport ---------------------------------------------------------

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        """Perform a request, translating failures into typed errors."""
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.RequestError as exc:
            raise OllamaUnavailableError(
                f"Could not reach Ollama at {self.host}: {exc}. Is `ollama serve` running?"
            ) from exc
        if response.is_error:
            raise OllamaResponseError(response.status_code, _error_message(response))
        return response

    # -- introspection -----------------------------------------------------

    def health(self) -> Health:
        """Report whether Ollama is reachable. Never raises."""
        try:
            response = self._request("GET", "/api/version")
        except OllamaUnavailableError as exc:
            return Health(reachable=False, detail=str(exc))
        except OllamaResponseError as exc:
            return Health(reachable=False, detail=exc.message)
        return Health(reachable=True, version=response.json().get("version"))

    def list_models(self, *, with_capabilities: bool = False) -> list[ModelInfo]:
        payload = self._request("GET", "/api/tags").json()
        models = [ModelInfo.from_payload(entry) for entry in payload.get("models", [])]
        if not with_capabilities:
            return models
        return [model.with_capabilities(self.capabilities(model.name)) for model in models]

    def capabilities(self, model: str) -> frozenset[str]:
        """Return a model's capabilities, e.g. ``{"completion", "tools", "thinking"}``.

        Results are cached: a model's capabilities do not change while it is
        installed, and the UI asks for them on every model switch.
        """
        cached = self._capability_cache.get(model)
        if cached is not None:
            return cached
        payload = self._request("POST", "/api/show", json={"model": model}).json()
        capabilities = frozenset(payload.get("capabilities") or ())
        self._capability_cache[model] = capabilities
        return capabilities

    def supports(self, model: str, capability: str) -> bool:
        return capability in self.capabilities(model)

    # -- generation --------------------------------------------------------

    def chat(
        self,
        model: str,
        messages: Sequence[Message],
        *,
        options: Mapping[str, Any] | None = None,
        response_format: JsonSchema | str | None = None,
        think: bool = False,
    ) -> ChatResult:
        body = self._build_body(
            model, messages, options=options, response_format=response_format, think=think
        )
        body["stream"] = False
        payload = self._request("POST", "/api/chat", json=body).json()
        return ChatResult.from_payload(payload)

    def stream_chat(
        self,
        model: str,
        messages: Sequence[Message],
        *,
        options: Mapping[str, Any] | None = None,
        response_format: JsonSchema | str | None = None,
        think: bool = False,
    ) -> Iterator[ChatChunk]:
        body = self._build_body(
            model, messages, options=options, response_format=response_format, think=think
        )
        body["stream"] = True

        try:
            with self._client.stream("POST", "/api/chat", json=body) as response:
                if response.is_error:
                    response.read()
                    raise OllamaResponseError(response.status_code, _error_message(response))
                for line in response.iter_lines():
                    chunk = _parse_stream_line(line)
                    if chunk is not None:
                        yield chunk
        except httpx.RequestError as exc:
            raise OllamaUnavailableError(
                f"Lost connection to Ollama at {self.host}: {exc}."
            ) from exc

    def _build_body(
        self,
        model: str,
        messages: Sequence[Message],
        *,
        options: Mapping[str, Any] | None,
        response_format: JsonSchema | str | None,
        think: bool,
    ) -> dict[str, Any]:
        if think:
            # Ollama rejects `think` with a 400 on models that cannot reason.
            # Checking first turns that into a clear, typed error and saves a
            # round trip.
            self._ensure_supports(model, "thinking")

        body: dict[str, Any] = {"model": model, "messages": list(messages)}
        if options:
            body["options"] = dict(options)
        if response_format is not None:
            body["format"] = response_format
        if think:
            body["think"] = True
        return body

    def _ensure_supports(self, model: str, capability: str) -> None:
        if not self.supports(model, capability):
            raise CapabilityError(model, capability)


def _parse_stream_line(line: str) -> ChatChunk | None:
    """Turn one NDJSON line into a chunk, skipping keep-alive blanks."""
    line = line.strip()
    if not line:
        return None
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise OllamaResponseError(200, f"Malformed NDJSON from Ollama: {line!r}") from exc
    if "error" in payload:
        raise OllamaResponseError(200, str(payload["error"]))
    return ChatChunk.from_payload(payload)


def _error_message(response: httpx.Response) -> str:
    """Pull Ollama's ``{"error": ...}`` message out of a failed response."""
    try:
        payload = response.json()
    except ValueError:
        return response.text.strip() or response.reason_phrase
    if isinstance(payload, dict) and "error" in payload:
        return str(payload["error"])
    return response.text.strip()
