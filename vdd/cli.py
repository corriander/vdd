"""vdd command-line interface."""
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Value-Driven Design tools.")
requirements_app = typer.Typer(help="Requirements weighting tools.")
app.add_typer(requirements_app, name="requirements")

console = Console()


def _force_utf8_stdio() -> None:
    """Make stdout/stderr capable of encoding what we render.

    Tables and bars use box-drawing and block characters. On Windows a
    redirected or piped stream defaults to the locale encoding (cp1252),
    which can't represent them, so `vdd coda model.xlsx > out.txt` dies
    with UnicodeEncodeError. An interactive console is already UTF-8, so
    this only bites when the output is captured.
    """
    for stream in (sys.stdout, sys.stderr):
        encoding = getattr(stream, 'encoding', None) or ''
        if encoding.lower().replace('-', '') == 'utf8':
            continue
        try:
            stream.reconfigure(encoding='utf-8')
        except (AttributeError, OSError):
            # Not a reconfigurable TextIOWrapper (e.g. replaced by a
            # test harness). Leave it alone rather than fail here.
            pass


@app.callback()
def main() -> None:
    """Value-Driven Design tools."""
    _force_utf8_stdio()


def _bar(value: float, width: int = 10) -> str:
    filled = round(value * width)
    return "█" * filled + "░" * (width - filled) + f" {value:.2f}"


@app.command()
def coda(
    file: Annotated[Path, typer.Argument(help="Excel model file")],
    parser: Annotated[str, typer.Option(help="Parser: compact or full")] = "compact",
) -> None:
    """Load a CODA model from an Excel file and display results."""
    from vdd.coda import CODA
    from vdd.coda.io import CompactExcelParser, ExcelParser

    parser_cls = CompactExcelParser if parser == "compact" else ExcelParser
    model = CODA.from_excel(file, parser_class=parser_cls)

    # Default unset characteristic values to the midpoint of their bounds.
    for char in model.characteristics:
        try:
            char.value  # noqa: B018
        except AttributeError:
            lo, hi = char.limits
            char.value = ((lo or 0.0) + (hi or 1.0)) / 2

    try:
        merit = model.merit
        satisfactions = model.satisfaction[:, 0]
    except ValueError as err:
        console.print(f"[red]Error:[/red] {err}")
        raise typer.Exit(code=1)

    console.print(f"Merit: {merit:.2f}")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Requirement")
    table.add_column("Weight", justify="right")
    table.add_column("Satisfaction")

    weights = model.weight[:, 0]

    for req, w, s in zip(model.requirements, weights, satisfactions):
        table.add_row(req.name, f"{w:.2f}", _bar(s))

    console.print(table)


@requirements_app.command("weight")
def weight(
    reqs: Annotated[list[str], typer.Argument(help="Requirement names")],
    shuffle: Annotated[bool, typer.Option(help="Shuffle question order")] = True,
) -> None:
    """Interactively weight requirements using pairwise comparison."""
    from vdd.requirements.models import BinWM

    bwm = BinWM(*reqs)
    bwm.prompt(shuffle=shuffle)

    table = Table(show_header=True, header_style="bold")
    table.add_column("Requirement")
    table.add_column("Score")

    for req, score in zip(bwm.requirements, bwm.score):
        table.add_row(req, _bar(score))

    console.print(table)
