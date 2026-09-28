"""Flask application factory.

The server is deliberately stateless: conversation history lives in the
browser and is sent with every request. That is what makes the growth of
the context window visible, and what lets two comparison lanes run
concurrently without interfering with one another.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, send_from_directory

from ..config import Settings
from ..errors import PlaygroundError
from ..ollama_client import OllamaClient
from .routes import api

# src/playground/web/app.py -> repository root
DEFAULT_FRONTEND_DIR = Path(__file__).resolve().parents[3] / "frontend"


def frontend_dir() -> Path:
    override = os.environ.get("PLAYGROUND_FRONTEND_DIR")
    return Path(override) if override else DEFAULT_FRONTEND_DIR


def create_app(
    settings: Settings | None = None,
    client: OllamaClient | None = None,
) -> Flask:
    """Build the application. Both dependencies are injectable for tests."""
    resolved = settings or Settings.from_env()
    app = Flask(__name__, static_folder=str(frontend_dir()), static_url_path="")

    app.config["SETTINGS"] = resolved
    app.config["OLLAMA_CLIENT"] = client or OllamaClient(
        resolved.ollama_host, timeout=resolved.request_timeout
    )

    app.register_blueprint(api)

    @app.get("/")
    def index() -> Response:
        # Flask serves the interface itself, so the page and the API share
        # an origin and the frontend can use relative URLs.
        return send_from_directory(str(frontend_dir()), "index.html")

    @app.errorhandler(PlaygroundError)
    def handle_known_error(exc: PlaygroundError) -> tuple[Response, int]:
        body: dict[str, Any] = {"error": str(exc), "type": type(exc).__name__}
        return jsonify(body), exc.http_status

    return app


def main() -> None:
    """Run the development server."""
    settings = Settings.from_env()
    create_app(settings).run(host="0.0.0.0", port=settings.port)


if __name__ == "__main__":
    main()
