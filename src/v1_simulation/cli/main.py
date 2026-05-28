from __future__ import annotations

import typer

from v1_simulation.cli.analyze import analyze_command
from v1_simulation.cli.run import run_command
from v1_simulation.cli.sweep import sweep_command
from v1_simulation.cli.train import train_command


app = typer.Typer(
    no_args_is_help=True,
    help="CLI for V1 simulation training, simulation, analysis, and parameter sweeps.",
)

app.command("run")(run_command)
app.command("train")(train_command)
app.command("analyze")(analyze_command)
app.command("sweep")(sweep_command)


def main() -> None:
    app()


if __name__ == "__main__":
    main()


__all__ = ["app", "main"]
