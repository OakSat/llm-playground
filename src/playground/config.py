"""Environment-driven configuration and the feature-flag layer.

Every ability of the playground can be switched on or off independently so
the project can be introduced one concept at a time. Flags are read from
``FEATURE_<NAME>`` environment variables; see ``.env.TEMPLATE``.

Flags are advisory to the UI but *binding* on the server: the web layer
rejects a disabled ability outright rather than relying on a hidden
control. See ``docs/exercises.md`` (phase 5) for why that distinction
matters.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field, fields

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSY = frozenset({"0", "false", "no", "off"})

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"
DEFAULT_TIMEOUT = 120.0
DEFAULT_PORT = 5000


def _parse_bool(raw: str, *, variable: str) -> bool:
    value = raw.strip().lower()
    if value in _TRUTHY:
        return True
    if value in _FALSY:
        return False
    accepted = ", ".join(sorted(_TRUTHY | _FALSY))
    raise ValueError(f"{variable}={raw!r} is not a boolean. Use one of: {accepted}.")


def normalise_host(raw: str) -> str:
    """Return a usable base URL for Ollama.

    Ollama's own convention allows a bare ``host:port`` (its CLI documents
    ``OLLAMA_HOST`` defaulting to ``127.0.0.1:11434``), so a missing scheme
    is accepted rather than producing a confusing request failure later.
    """
    host = raw.strip().rstrip("/")
    if not host:
        return DEFAULT_HOST
    if "://" not in host:
        host = f"http://{host}"
    return host


@dataclass(frozen=True, slots=True)
class Features:
    """Which abilities are switched on."""

    streaming: bool = True
    multi_turn: bool = True
    model_switcher: bool = True
    compare: bool = True
    structured_output: bool = True
    reasoning: bool = True
    stats: bool = True

    @classmethod
    def names(cls) -> tuple[str, ...]:
        return tuple(f.name for f in fields(cls))

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Features:
        source = os.environ if env is None else env
        overrides: dict[str, bool] = {}
        for name in cls.names():
            variable = f"FEATURE_{name.upper()}"
            raw = source.get(variable)
            if raw is not None:
                overrides[name] = _parse_bool(raw, variable=variable)
        return cls(**overrides)

    def is_enabled(self, name: str) -> bool:
        """Look a flag up by name, rejecting unknown names.

        Routes gate on strings, so a typo must fail loudly rather than
        silently reporting a non-existent ability as disabled.
        """
        if name not in self.names():
            known = ", ".join(self.names())
            raise KeyError(f"Unknown feature {name!r}. Known features: {known}.")
        return bool(getattr(self, name))

    def as_dict(self) -> dict[str, bool]:
        """Serialisable form, sent to the browser by ``GET /api/config``."""
        return {name: bool(getattr(self, name)) for name in self.names()}


@dataclass(frozen=True, slots=True)
class Settings:
    """Everything the app needs to know at startup."""

    ollama_host: str = DEFAULT_HOST
    default_model: str = DEFAULT_MODEL
    request_timeout: float = DEFAULT_TIMEOUT
    port: int = DEFAULT_PORT
    features: Features = field(default_factory=Features)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        source = os.environ if env is None else env

        # OLLAMA_MODEL is the v0.1 spelling. It is still honoured so an
        # existing .env keeps working after upgrading.
        model = source.get("PLAYGROUND_MODEL") or source.get("OLLAMA_MODEL") or DEFAULT_MODEL

        raw_timeout = source.get("PLAYGROUND_REQUEST_TIMEOUT")
        if raw_timeout is None:
            timeout = DEFAULT_TIMEOUT
        else:
            try:
                timeout = float(raw_timeout)
            except ValueError as exc:
                raise ValueError(
                    f"PLAYGROUND_REQUEST_TIMEOUT={raw_timeout!r} is not a number."
                ) from exc
            if timeout <= 0:
                raise ValueError("PLAYGROUND_REQUEST_TIMEOUT must be greater than zero.")

        raw_port = source.get("PLAYGROUND_PORT")
        try:
            port = DEFAULT_PORT if raw_port is None else int(raw_port)
        except ValueError as exc:
            raise ValueError(f"PLAYGROUND_PORT={raw_port!r} is not an integer.") from exc
        if not 1 <= port <= 65535:
            raise ValueError("PLAYGROUND_PORT must be between 1 and 65535.")

        return cls(
            ollama_host=normalise_host(source.get("OLLAMA_HOST", DEFAULT_HOST)),
            default_model=model.strip(),
            request_timeout=timeout,
            port=port,
            features=Features.from_env(source),
        )
