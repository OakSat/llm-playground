"""The `slm` CLI: extraction and the evaluation harness."""

import json
from pathlib import Path

import httpx
import pytest
import respx

from playground.cli.__main__ import main
from playground.cli.evaluate import majority_baseline, run_sweep, tally, write_csv
from playground.cli.extract import (
    Row,
    extract_row,
    load_rows,
    load_schema,
    schema_fields,
)
from playground.errors import InvalidRequestError
from playground.ollama_client import OllamaClient

from .payloads import HOST, TAGS_RESPONSE

SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": ["billing", "bug"]},
        "priority": {"type": "string", "enum": ["low", "high"]},
    },
    "required": ["category", "priority"],
}

ROWS = [
    Row(id="1", text="charged twice", labels={"category": "billing", "priority": "high"}),
    Row(id="2", text="app crashes", labels={"category": "bug", "priority": "high"}),
]


def reply(
    content: str, *, eval_count: int = 10, eval_duration: int = 500_000_000
) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "message": {"role": "assistant", "content": content},
            "done": True,
            "done_reason": "stop",
            "eval_count": eval_count,
            "eval_duration": eval_duration,
        },
    )


@pytest.fixture
def client() -> OllamaClient:
    with OllamaClient(HOST, timeout=5.0) as instance:
        yield instance


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    path = tmp_path / "rows.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"id": row.id, "text": row.text, "labels": dict(row.labels)}) for row in ROWS
        )
        + "\n"
    )
    return path


@pytest.fixture
def schema_file(tmp_path: Path) -> Path:
    path = tmp_path / "schema.json"
    path.write_text(json.dumps(SCHEMA))
    return path


# -- loading ---------------------------------------------------------------


def test_load_rows(dataset: Path) -> None:
    rows = load_rows(dataset)
    assert [row.id for row in rows] == ["1", "2"]
    assert rows[0].labels["category"] == "billing"


def test_load_rows_honours_limit(dataset: Path) -> None:
    assert len(load_rows(dataset, limit=1)) == 1


def test_load_rows_rejects_a_row_without_text(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id": "1"}\n')
    with pytest.raises(InvalidRequestError, match="no 'text' field"):
        load_rows(path)


def test_load_rows_rejects_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("{nope}\n")
    with pytest.raises(InvalidRequestError, match="not valid JSON"):
        load_rows(path)


def test_load_rows_rejects_an_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("\n\n")
    with pytest.raises(InvalidRequestError, match="no rows"):
        load_rows(path)


def test_load_schema_rejects_an_invalid_schema(tmp_path: Path) -> None:
    """Catching this now avoids discovering it after a long sweep."""
    path = tmp_path / "schema.json"
    path.write_text('{"type": "not-a-type"}')
    with pytest.raises(Exception, match="not-a-type|is not valid"):
        load_schema(path)


def test_schema_fields_follows_schema_order() -> None:
    assert schema_fields(SCHEMA) == ["category", "priority"]


# -- extraction ------------------------------------------------------------


@respx.mock
def test_valid_extraction(client: OllamaClient) -> None:
    respx.post(f"{HOST}/api/chat").mock(
        reply(json.dumps({"category": "billing", "priority": "high"}))
    )
    result = extract_row(client, "llama3.2", ROWS[0], SCHEMA)
    assert result.valid is True
    assert result.parsed == {"category": "billing", "priority": "high"}
    assert result.tokens_per_second == pytest.approx(20.0)


@respx.mock
def test_unparseable_reply_is_recorded_not_raised(client: OllamaClient) -> None:
    respx.post(f"{HOST}/api/chat").mock(reply("sorry, I cannot do that"))
    result = extract_row(client, "llama3.2", ROWS[0], SCHEMA)
    assert result.valid is False
    assert "not valid JSON" in (result.error or "")


@respx.mock
def test_schema_violation_is_reported_with_its_location(client: OllamaClient) -> None:
    respx.post(f"{HOST}/api/chat").mock(
        reply(json.dumps({"category": "billing", "priority": "urgent"}))
    )
    result = extract_row(client, "llama3.2", ROWS[0], SCHEMA)
    assert result.valid is False
    assert "priority" in (result.error or "")


@respx.mock
def test_a_failed_row_does_not_abort_the_batch(client: OllamaClient) -> None:
    """A sweep must survive one bad row; the failure becomes a data point."""
    respx.post(f"{HOST}/api/chat").mock(side_effect=httpx.ConnectError("refused"))
    result = extract_row(client, "llama3.2", ROWS[0], SCHEMA)
    assert result.valid is False
    assert result.parsed is None
    assert "Could not reach Ollama" in (result.error or "")


# -- scoring ---------------------------------------------------------------


@respx.mock
def test_accuracy_counts_only_labelled_fields(client: OllamaClient) -> None:
    respx.post(f"{HOST}/api/chat").mock(
        reply(json.dumps({"category": "billing", "priority": "high"}))
    )
    results = [extract_row(client, "m", row, SCHEMA) for row in ROWS]
    counts = tally(results, ["category", "priority"])

    # Both rows predicted billing/high: category right once, priority twice.
    assert counts.field_correct["category"] == 1
    assert counts.field_correct["priority"] == 2
    assert counts.scored == {"category": 2, "priority": 2}


def test_majority_baseline() -> None:
    rows = [
        Row(id="1", text="a", labels={"category": "billing"}),
        Row(id="2", text="b", labels={"category": "billing"}),
        Row(id="3", text="c", labels={"category": "bug"}),
    ]
    assert majority_baseline(rows, ["category"]) == pytest.approx({"category": 2 / 3})


def test_unlabelled_rows_are_extracted_but_not_scored() -> None:
    results = tally([], ["category"])
    assert results.scored == {}


@respx.mock
def test_sweep_covers_every_combination(client: OllamaClient) -> None:
    respx.post(f"{HOST}/api/chat").mock(
        reply(json.dumps({"category": "billing", "priority": "high"}))
    )
    scores = run_sweep(
        client,
        ["m1", "m2"],
        [0.0, 0.7],
        ROWS,
        SCHEMA,
        ["category", "priority"],
        repeat=2,
    )
    assert len(scores) == 4
    assert {(s.model, s.temperature) for s in scores} == {
        ("m1", 0.0),
        ("m1", 0.7),
        ("m2", 0.0),
        ("m2", 0.7),
    }
    # Two rows, repeated twice.
    assert all(score.rows == 4 for score in scores)


@respx.mock
def test_csv_has_a_row_per_combination(client: OllamaClient, tmp_path: Path) -> None:
    respx.post(f"{HOST}/api/chat").mock(
        reply(json.dumps({"category": "billing", "priority": "high"}))
    )
    scores = run_sweep(client, ["m1"], [0.0], ROWS, SCHEMA, ["category"])
    out = tmp_path / "scores.csv"
    write_csv(out, scores, ["category"])
    lines = out.read_text().strip().splitlines()
    assert lines[0].startswith("model,temperature,rows,valid_rate")
    assert len(lines) == 2


# -- end to end ------------------------------------------------------------


@respx.mock
def test_models_command(capsys: pytest.CaptureFixture[str]) -> None:
    respx.get(f"{HOST}/api/tags").mock(httpx.Response(200, json=TAGS_RESPONSE))
    respx.post(f"{HOST}/api/show").mock(
        httpx.Response(200, json={"capabilities": ["completion", "thinking"]})
    )
    assert main(["--host", HOST, "models"]) == 0
    assert "llama3.2" in capsys.readouterr().out


@respx.mock
def test_extract_command_writes_results(dataset: Path, schema_file: Path, tmp_path: Path) -> None:
    respx.post(f"{HOST}/api/chat").mock(
        reply(json.dumps({"category": "billing", "priority": "high"}))
    )
    out = tmp_path / "out.jsonl"
    code = main(
        [
            "--host",
            HOST,
            "extract",
            "--input",
            str(dataset),
            "--schema",
            str(schema_file),
            "--out",
            str(out),
        ]
    )
    assert code == 0
    written = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(written) == 2
    assert written[0]["prediction"] == {"category": "billing", "priority": "high"}


@respx.mock
def test_extract_command_exits_nonzero_when_a_row_fails(dataset: Path, schema_file: Path) -> None:
    """A batch that did not fully validate must not look successful."""
    respx.post(f"{HOST}/api/chat").mock(reply("not json"))
    code = main(["--host", HOST, "extract", "--input", str(dataset), "--schema", str(schema_file)])
    assert code == 1


@respx.mock
def test_eval_command_prints_a_table_with_a_baseline(
    dataset: Path, schema_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    respx.post(f"{HOST}/api/chat").mock(
        reply(json.dumps({"category": "billing", "priority": "high"}))
    )
    out = tmp_path / "scores.csv"
    code = main(
        [
            "--host",
            HOST,
            "eval",
            "--input",
            str(dataset),
            "--schema",
            str(schema_file),
            "--models",
            "m1,m2",
            "--temperatures",
            "0,0.7",
            "--out-csv",
            str(out),
        ]
    )
    assert code == 0
    printed = capsys.readouterr().out
    assert "Extraction accuracy" in printed
    assert "baseline" in printed
    assert out.exists()
