"""Typed errors for the playground.

Every error carries an ``http_status`` so the web layer can translate a
failure into an honest response code without re-deriving the mapping. This
is the direct fix for the v0.1 behaviour, where a failed generation was
reported to the browser as HTTP 200 with an empty body.
"""


class PlaygroundError(Exception):
    """Base class for every error this project raises deliberately."""

    http_status = 500


class FeatureDisabledError(PlaygroundError):
    """A capability was requested while switched off in configuration."""

    http_status = 409

    def __init__(self, feature: str) -> None:
        self.feature = feature
        super().__init__(
            f"The {feature!r} feature is disabled. "
            f"Set FEATURE_{feature.upper()}=1 and restart to enable it."
        )


class InvalidRequestError(PlaygroundError):
    """The caller sent a malformed request body."""

    http_status = 400


class CapabilityError(PlaygroundError):
    """The requested model cannot do what was asked of it.

    Raised before contacting Ollama, so the caller gets a clear message
    instead of a raw upstream 400.
    """

    http_status = 400

    def __init__(self, model: str, capability: str) -> None:
        self.model = model
        self.capability = capability
        super().__init__(f"Model {model!r} does not support {capability!r}.")


class OllamaError(PlaygroundError):
    """Base class for problems reaching or talking to Ollama."""

    http_status = 502


class OllamaUnavailableError(OllamaError):
    """Ollama could not be reached at all (not running, wrong host, timeout)."""

    http_status = 503


class OllamaResponseError(OllamaError):
    """Ollama replied, but with an error status."""

    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        self.message = message
        # A model that is simply not pulled yet is the user's problem to fix,
        # not an upstream fault, so it is surfaced as 404 rather than 502.
        self.http_status = 404 if status_code == 404 else 502
        super().__init__(f"Ollama returned {status_code}: {message}")
