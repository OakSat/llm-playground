"""Shared fixtures.

The Ollama HTTP layer is mocked throughout, so the whole suite runs with
nothing installed and no model loaded -- which is what lets CI stay green
on a plain runner. Recorded payloads live in ``payloads.py``.
"""

from collections.abc import Iterator

import pytest

from playground.ollama_client import OllamaClient

from .payloads import HOST


@pytest.fixture
def host() -> str:
    return HOST


@pytest.fixture
def client() -> Iterator[OllamaClient]:
    with OllamaClient(HOST, timeout=5.0) as instance:
        yield instance
