"""Phase 0 guards: the package imports, and its metadata stays in sync."""

import importlib.metadata
import tomllib
from pathlib import Path

import pytest

import playground
from playground.cli.__main__ import main

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def test_package_imports_and_exposes_a_version() -> None:
    assert playground.__version__


def test_version_matches_pyproject() -> None:
    """__init__ and pyproject must not drift apart."""
    declared = tomllib.loads(PYPROJECT.read_text())["project"]["version"]
    assert playground.__version__ == declared


def test_console_script_entry_point_resolves() -> None:
    """Guards the `slm = playground.cli.__main__:main` wiring in pyproject."""
    (entry,) = [
        ep for ep in importlib.metadata.entry_points(group="console_scripts") if ep.name == "slm"
    ]
    assert entry.load() is main


def test_cli_requires_a_subcommand() -> None:
    """argparse exits 2 on a missing subcommand; behaviour lives in test_cli."""
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2
