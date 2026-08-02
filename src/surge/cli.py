"""Minimal command-line interface for the core ``surge-grid`` package."""

from __future__ import annotations

import typer

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command()
def load(ba: str, start: str, end: str) -> None:
    """Fetch hourly BA load from EIA-930 and print the first 20 rows."""
    import surge

    df = surge.load(ba=ba, start=start, end=end)
    typer.echo(df.head(20))


if __name__ == "__main__":
    app()
