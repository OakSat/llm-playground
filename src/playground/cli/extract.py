"""Structured extraction: the model as a text-to-struct function.

No conversation, no chat window. A batch of text goes in, a schema
constrains the shape of what comes out, and every row is validated. This
is the other way to use a small local model, and the one that scales.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from ..errors import InvalidRequestError, PlaygroundError
from ..ollama_client import JsonSchema, OllamaClient

DEFAULT_INSTRUCTION = (
    "Classify the support ticket below. Answer only with the JSON object described by the schema."
)


@dataclass(frozen=True, slots=True)
class Row:
    """One input record, optionally carrying gold labels for scoring."""

    id: str
    text: str
    labels: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class RowResult:
    row: Row
    raw: str
    parsed: Mapping[str, Any] | None
    valid: bool
    error: str | None
    seconds: float
    eval_count: int | None
    tokens_per_second: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.row.id,
            "text": self.row.text,
            "labels": dict(self.row.labels),
            "prediction": dict(self.parsed) if self.parsed else None,
            "raw": self.raw,
            "valid": self.valid,
            "error": self.error,
            "seconds": round(self.seconds, 3),
            "tokens_per_second": self.tokens_per_second,
        }


def load_rows(path: Path, limit: int | None = None) -> list[Row]:
    """Read a JSONL batch. Each line needs `text`; `id` and `labels` are optional."""
    rows: list[Row] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise InvalidRequestError(f"{path}:{number} is not valid JSON: {exc}") from exc
            text = payload.get("text")
            if not isinstance(text, str) or not text.strip():
                raise InvalidRequestError(f"{path}:{number} has no 'text' field.")
            rows.append(
                Row(
                    id=str(payload.get("id") or number),
                    text=text,
                    labels=payload.get("labels") or {},
                )
            )
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise InvalidRequestError(f"{path} contains no rows.")
    return rows


def load_schema(path: Path) -> dict[str, Any]:
    schema = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(schema, dict):
        raise InvalidRequestError(f"{path} must contain a JSON object.")
    # Fail now on a malformed schema rather than after a long sweep.
    Draft202012Validator.check_schema(schema)
    return schema


def schema_fields(schema: Mapping[str, Any]) -> list[str]:
    """The fields a prediction is scored on, in schema order."""
    properties = schema.get("properties")
    if isinstance(properties, dict):
        return list(properties)
    return []


def extract_row(
    client: OllamaClient,
    model: str,
    row: Row,
    schema: JsonSchema,
    *,
    options: Mapping[str, Any] | None = None,
    instruction: str = DEFAULT_INSTRUCTION,
) -> RowResult:
    """Run one row and validate the reply against the schema."""
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": row.text},
    ]

    started = time.perf_counter()
    try:
        result = client.chat(model, messages, options=options, response_format=schema)
    except PlaygroundError as exc:
        # A failed row is data, not a crash: the sweep continues and the
        # failure rate becomes part of the result.
        return RowResult(
            row=row,
            raw="",
            parsed=None,
            valid=False,
            error=str(exc),
            seconds=time.perf_counter() - started,
            eval_count=None,
            tokens_per_second=None,
        )
    seconds = time.perf_counter() - started

    parsed, error = _parse_and_validate(result.content, schema)
    return RowResult(
        row=row,
        raw=result.content,
        parsed=parsed,
        valid=error is None,
        error=error,
        seconds=seconds,
        eval_count=result.stats.eval_count,
        tokens_per_second=result.stats.tokens_per_second,
    )


def _parse_and_validate(
    content: str, schema: Mapping[str, Any]
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        return None, f"not valid JSON: {exc}"
    if not isinstance(parsed, dict):
        return None, "reply was not a JSON object"

    errors = sorted(Draft202012Validator(schema).iter_errors(parsed), key=lambda e: list(e.path))
    if errors:
        first = errors[0]
        location = ".".join(str(part) for part in first.path) or "(root)"
        return parsed, f"schema violation at {location}: {first.message}"
    return parsed, None


def extract_all(
    client: OllamaClient,
    model: str,
    rows: Sequence[Row],
    schema: JsonSchema,
    *,
    options: Mapping[str, Any] | None = None,
    instruction: str = DEFAULT_INSTRUCTION,
) -> Iterator[RowResult]:
    for row in rows:
        yield extract_row(client, model, row, schema, options=options, instruction=instruction)


def write_jsonl(path: Path, results: Sequence[RowResult]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result.as_dict()) + "\n")
