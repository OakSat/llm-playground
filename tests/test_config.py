"""Configuration and feature-flag parsing."""

import pytest

from playground.config import DEFAULT_HOST, Features, Settings, normalise_host


def test_every_ability_defaults_to_enabled() -> None:
    features = Features()
    assert all(features.as_dict().values())
    assert len(Features.names()) == 7


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "on", " True "])
def test_truthy_spellings_enable(raw: str) -> None:
    assert Features.from_env({"FEATURE_COMPARE": raw}).compare is True


@pytest.mark.parametrize("raw", ["0", "false", "FALSE", "no", "off", " 0 "])
def test_falsy_spellings_disable(raw: str) -> None:
    assert Features.from_env({"FEATURE_COMPARE": raw}).compare is False


def test_flags_are_independent() -> None:
    """Switching one ability off must not disturb the others."""
    features = Features.from_env({"FEATURE_REASONING": "0"})
    assert features.reasoning is False
    assert features.streaming is True
    assert features.compare is True


def test_every_flag_is_reachable_from_the_environment() -> None:
    """Guards against a flag existing in code but having no way to set it."""
    for name in Features.names():
        features = Features.from_env({f"FEATURE_{name.upper()}": "0"})
        assert features.is_enabled(name) is False


def test_nonsense_flag_value_is_rejected_loudly() -> None:
    with pytest.raises(ValueError, match="FEATURE_STREAMING"):
        Features.from_env({"FEATURE_STREAMING": "maybe"})


def test_unknown_feature_name_raises() -> None:
    """A typo in a route's gate must fail, not silently read as disabled."""
    with pytest.raises(KeyError, match="stremaing"):
        Features().is_enabled("stremaing")


def test_settings_defaults() -> None:
    settings = Settings.from_env({})
    assert settings.ollama_host == DEFAULT_HOST
    assert settings.default_model == "llama3.2"
    assert settings.request_timeout == 120.0


def test_timeout_is_no_longer_the_v01_thirty_seconds() -> None:
    """Reasoning models routinely exceed the old hard-coded 30s cap."""
    assert Settings.from_env({}).request_timeout > 30


def test_legacy_ollama_model_variable_still_works() -> None:
    """An existing v0.1 .env must keep working after upgrading."""
    assert Settings.from_env({"OLLAMA_MODEL": "mistral"}).default_model == "mistral"


def test_playground_model_wins_over_legacy_name() -> None:
    settings = Settings.from_env({"PLAYGROUND_MODEL": "new", "OLLAMA_MODEL": "old"})
    assert settings.default_model == "new"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("http://localhost:11434", "http://localhost:11434"),
        ("http://localhost:11434/", "http://localhost:11434"),
        # Ollama's own convention allows a bare host:port.
        ("127.0.0.1:11434", "http://127.0.0.1:11434"),
        ("ollama:11434", "http://ollama:11434"),
        ("https://remote.example", "https://remote.example"),
        ("", DEFAULT_HOST),
    ],
)
def test_host_normalisation(raw: str, expected: str) -> None:
    assert normalise_host(raw) == expected


@pytest.mark.parametrize("raw", ["soon", "-1", "0"])
def test_bad_timeout_is_rejected(raw: str) -> None:
    with pytest.raises(ValueError, match="PLAYGROUND_REQUEST_TIMEOUT|greater than zero"):
        Settings.from_env({"PLAYGROUND_REQUEST_TIMEOUT": raw})


def test_settings_carries_feature_flags() -> None:
    settings = Settings.from_env({"OLLAMA_HOST": "box:11434", "FEATURE_STATS": "0"})
    assert settings.ollama_host == "http://box:11434"
    assert settings.features.stats is False


def test_port_defaults_to_five_thousand() -> None:
    assert Settings.from_env({}).port == 5000


def test_port_is_configurable() -> None:
    """macOS AirPlay Receiver occupies 5000, so this must be overridable."""
    assert Settings.from_env({"PLAYGROUND_PORT": "5050"}).port == 5050


@pytest.mark.parametrize("raw", ["http", "0", "70000", "-1"])
def test_bad_port_is_rejected(raw: str) -> None:
    with pytest.raises(ValueError, match="PLAYGROUND_PORT"):
        Settings.from_env({"PLAYGROUND_PORT": raw})
