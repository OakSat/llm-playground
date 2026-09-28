"""Measuring an extraction task across models and settings.

Turns "which local model should I use?" into an answer with numbers
behind it: how often the output is schema-valid, how often it is
*correct*, and what it costs in latency and tokens per second.
"""

from __future__ import annotations

import contextlib
import csv
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..ollama_client import JsonSchema, OllamaClient
from .extract import DEFAULT_INSTRUCTION, Row, RowResult, extract_all

WARMUP_PROMPT = "ok"


@dataclass(frozen=True, slots=True)
class Score:
    """The outcome of one (model, temperature) run over the whole batch."""

    model: str
    temperature: float
    rows: int
    valid: int
    field_correct: Mapping[str, int]
    scored: Mapping[str, int]
    latencies: Sequence[float]
    speeds: Sequence[float]

    @property
    def valid_rate(self) -> float:
        return self.valid / self.rows if self.rows else 0.0

    def accuracy(self, field_name: str) -> float | None:
        total = self.scored.get(field_name, 0)
        if not total:
            return None
        return self.field_correct.get(field_name, 0) / total

    @property
    def overall_accuracy(self) -> float | None:
        total = sum(self.scored.values())
        if not total:
            return None
        return sum(self.field_correct.values()) / total

    @property
    def mean_latency(self) -> float | None:
        return statistics.fmean(self.latencies) if self.latencies else None

    @property
    def mean_speed(self) -> float | None:
        return statistics.fmean(self.speeds) if self.speeds else None


@dataclass
class _Tally:
    rows: int = 0
    valid: int = 0
    field_correct: dict[str, int] = field(default_factory=dict)
    scored: dict[str, int] = field(default_factory=dict)
    latencies: list[float] = field(default_factory=list)
    speeds: list[float] = field(default_factory=list)


def tally(results: Sequence[RowResult], fields: Sequence[str]) -> _Tally:
    counts = _Tally()
    for result in results:
        counts.rows += 1
        counts.latencies.append(result.seconds)
        if result.tokens_per_second is not None:
            counts.speeds.append(result.tokens_per_second)
        if result.valid:
            counts.valid += 1

        for name in fields:
            gold = result.row.labels.get(name)
            if gold is None:
                continue  # unlabelled rows are extracted but not scored
            counts.scored[name] = counts.scored.get(name, 0) + 1
            predicted = (result.parsed or {}).get(name)
            if _same(predicted, gold):
                counts.field_correct[name] = counts.field_correct.get(name, 0) + 1
    return counts


def _same(predicted: Any, gold: str) -> bool:
    return isinstance(predicted, str) and predicted.strip().lower() == gold.strip().lower()


def majority_baseline(rows: Sequence[Row], fields: Sequence[str]) -> dict[str, float]:
    """Accuracy of always guessing each field's most common label.

    Without this, a number like 47% looks like skill when the dataset is
    simply unbalanced. Any model worth using must beat it.
    """
    baseline: dict[str, float] = {}
    for name in fields:
        labels = [row.labels[name] for row in rows if name in row.labels]
        if not labels:
            continue
        most_common = max(set(labels), key=labels.count)
        baseline[name] = labels.count(most_common) / len(labels)
    return baseline


def warm_up(client: OllamaClient, model: str) -> None:
    """Load the model before timing it.

    Loading dominates the first call -- seconds, against tens of
    milliseconds of generation -- so without this the first row of every
    sweep is measuring disk, not the model.
    """
    # A cold-start failure is ignored here; it resurfaces per row, where
    # it belongs in the results rather than aborting the sweep.
    with contextlib.suppress(Exception):
        client.chat(
            model,
            [{"role": "user", "content": WARMUP_PROMPT}],
            options={"num_predict": 1},
        )


def run_sweep(
    client: OllamaClient,
    models: Sequence[str],
    temperatures: Sequence[float],
    rows: Sequence[Row],
    schema: JsonSchema,
    fields: Sequence[str],
    *,
    repeat: int = 1,
    instruction: str = DEFAULT_INSTRUCTION,
    on_run: Callable[[str, float], None] | None = None,
) -> list[Score]:
    """Run every (model, temperature) combination and score each."""
    scores: list[Score] = []
    for model in models:
        warm_up(client, model)
        for temperature in temperatures:
            combined = _Tally()
            for _ in range(repeat):
                results = list(
                    extract_all(
                        client,
                        model,
                        rows,
                        schema,
                        options={"temperature": temperature},
                        instruction=instruction,
                    )
                )
                _merge(combined, tally(results, fields))
                if on_run is not None:
                    on_run(model, temperature)
            scores.append(
                Score(
                    model=model,
                    temperature=temperature,
                    rows=combined.rows,
                    valid=combined.valid,
                    field_correct=dict(combined.field_correct),
                    scored=dict(combined.scored),
                    latencies=combined.latencies,
                    speeds=combined.speeds,
                )
            )
    return scores


def _merge(into: _Tally, other: _Tally) -> None:
    into.rows += other.rows
    into.valid += other.valid
    into.latencies.extend(other.latencies)
    into.speeds.extend(other.speeds)
    for name, count in other.field_correct.items():
        into.field_correct[name] = into.field_correct.get(name, 0) + count
    for name, count in other.scored.items():
        into.scored[name] = into.scored.get(name, 0) + count


def write_csv(path: Path, scores: Sequence[Score], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["model", "temperature", "rows", "valid_rate", "overall_accuracy"]
            + [f"accuracy_{name}" for name in fields]
            + ["mean_latency_s", "mean_tokens_per_second"]
        )
        for score in scores:
            writer.writerow(
                [
                    score.model,
                    score.temperature,
                    score.rows,
                    _round(score.valid_rate),
                    _round(score.overall_accuracy),
                    *[_round(score.accuracy(name)) for name in fields],
                    _round(score.mean_latency),
                    _round(score.mean_speed),
                ]
            )


def _round(value: float | None, digits: int = 4) -> str:
    return "" if value is None else str(round(value, digits))
