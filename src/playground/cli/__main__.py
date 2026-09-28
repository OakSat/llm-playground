"""The `slm` command: a local model used as infrastructure, not a chat.

    slm models
    slm extract --input tickets.jsonl --schema ticket.json --out results.jsonl
    slm eval    --input tickets.jsonl --schema ticket.json \
                --models llama3.2,deepseek-r1 --temperatures 0,0.7
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .. import __version__
from ..config import Settings
from ..errors import PlaygroundError
from ..ollama_client import OllamaClient
from .evaluate import Score, majority_baseline, run_sweep, write_csv
from .extract import extract_all, load_rows, load_schema, schema_fields, write_jsonl


def build_parser(settings: Settings) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="slm", description=__doc__.splitlines()[0])
    parser.add_argument("-V", "--version", action="version", version=f"slm {__version__}")
    parser.add_argument("--host", default=settings.ollama_host, help="Ollama base URL.")
    parser.add_argument("--timeout", type=float, default=settings.request_timeout)

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("models", help="List installed models and their capabilities.")

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--input", type=Path, required=True, help="JSONL batch of records.")
    shared.add_argument("--schema", type=Path, required=True, help="JSON Schema for the output.")
    shared.add_argument("--limit", type=int, help="Only use the first N rows.")

    extract = sub.add_parser("extract", parents=[shared], help="Extract structured data.")
    extract.add_argument("--model", default=settings.default_model)
    extract.add_argument("--temperature", type=float, default=0.0)
    extract.add_argument("--out", type=Path, help="Write results as JSONL.")

    evaluate = sub.add_parser("eval", parents=[shared], help="Score models against gold labels.")
    evaluate.add_argument(
        "--models",
        default=settings.default_model,
        help="Comma-separated model names to compare.",
    )
    evaluate.add_argument(
        "--temperatures",
        default="0",
        help="Comma-separated temperatures to sweep.",
    )
    evaluate.add_argument("--repeat", type=int, default=1, help="Runs per combination.")
    evaluate.add_argument("--out-csv", type=Path, help="Write the table as CSV.")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    settings = Settings.from_env()
    args = build_parser(settings).parse_args(argv)
    console = Console()

    try:
        with OllamaClient(args.host, timeout=args.timeout) as client:
            if args.command == "models":
                return command_models(client, console)
            if args.command == "extract":
                return command_extract(client, console, args)
            return command_eval(client, console, args)
    except PlaygroundError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1
    except KeyboardInterrupt:
        console.print("[yellow]Interrupted.[/yellow]")
        return 130


def command_models(client: OllamaClient, console: Console) -> int:
    table = Table(title="Installed models")
    table.add_column("Name")
    table.add_column("Params", justify="right")
    table.add_column("Quant")
    table.add_column("Context", justify="right")
    table.add_column("Capabilities")

    for model in client.list_models(with_capabilities=True):
        table.add_row(
            model.name,
            model.parameter_size or "-",
            model.quantization or "-",
            f"{model.context_length:,}" if model.context_length else "-",
            ", ".join(sorted(model.capabilities)) or "-",
        )
    console.print(table)
    return 0


def command_extract(client: OllamaClient, console: Console, args: argparse.Namespace) -> int:
    rows = load_rows(args.input, args.limit)
    schema = load_schema(args.schema)

    results = []
    with console.status(f"Extracting {len(rows)} rows with {args.model}…"):
        for result in extract_all(
            client, args.model, rows, schema, options={"temperature": args.temperature}
        ):
            results.append(result)

    valid = sum(1 for r in results if r.valid)
    for result in results:
        mark = "[green]ok[/green]" if result.valid else "[red]invalid[/red]"
        detail = result.parsed if result.valid else result.error
        console.print(f"{result.row.id:>5}  {mark}  {detail}")

    console.print(f"\n{valid}/{len(results)} rows matched the schema.")
    if args.out:
        write_jsonl(args.out, results)
        console.print(f"Wrote {args.out}")
    return 0 if valid == len(results) else 1


def command_eval(client: OllamaClient, console: Console, args: argparse.Namespace) -> int:
    rows = load_rows(args.input, args.limit)
    schema = load_schema(args.schema)
    fields = schema_fields(schema)

    models = [name.strip() for name in args.models.split(",") if name.strip()]
    temperatures = [float(value) for value in args.temperatures.split(",") if value.strip()]
    total = len(models) * len(temperatures) * args.repeat

    console.print(
        f"Sweeping {len(models)} model(s) x {len(temperatures)} temperature(s) "
        f"x {args.repeat} run(s) over {len(rows)} rows."
    )

    done = 0
    with console.status("Warming up…") as status:

        def progress(model: str, temperature: float) -> None:
            nonlocal done
            done += 1
            status.update(f"[{done}/{total}] {model} @ temperature {temperature}")

        scores = run_sweep(
            client,
            models,
            temperatures,
            rows,
            schema,
            fields,
            repeat=args.repeat,
            on_run=progress,
        )

    render_table(console, scores, majority_baseline(rows, fields), fields)

    if args.out_csv:
        write_csv(args.out_csv, scores, fields)
        console.print(f"Wrote {args.out_csv}")
    return 0


def render_table(
    console: Console,
    scores: Sequence[Score],
    baseline: dict[str, float],
    fields: Sequence[str],
) -> None:
    table = Table(title="Extraction accuracy")
    table.add_column("Model")
    table.add_column("Temp", justify="right")
    table.add_column("Valid JSON", justify="right")
    table.add_column("Accuracy", justify="right")
    for name in fields:
        table.add_column(name, justify="right")
    table.add_column("Latency", justify="right")
    table.add_column("Speed", justify="right")

    for score in scores:
        table.add_row(
            score.model,
            f"{score.temperature:g}",
            _percent(score.valid_rate),
            _percent(score.overall_accuracy),
            *[_percent(score.accuracy(name)) for name in fields],
            _seconds(score.mean_latency),
            _speed(score.mean_speed),
        )

    # The reference point: always guessing each field's commonest label.
    # A model that cannot beat this has not learned the task.
    if baseline:
        overall = sum(baseline.values()) / len(baseline)
        table.add_section()
        table.add_row(
            "[dim]baseline[/dim]",
            "[dim]-[/dim]",
            "[dim]-[/dim]",
            f"[dim]{_percent(overall)}[/dim]",
            *[f"[dim]{_percent(baseline.get(name))}[/dim]" for name in fields],
            "[dim]-[/dim]",
            "[dim]-[/dim]",
        )

    console.print(table)
    console.print(
        "[dim]Accuracy is per field against the gold labels. A reasoning model spends "
        "its thinking tokens inside the same counters, so its speed and latency are not "
        "directly comparable with a non-reasoning one.[/dim]"
    )


def _percent(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.0f}%"


def _seconds(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}s"


def _speed(value: float | None) -> str:
    return "-" if value is None else f"{value:.0f} t/s"


if __name__ == "__main__":
    sys.exit(main())
