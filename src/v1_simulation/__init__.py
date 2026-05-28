from v1_simulation.simulation import run_bcm_training, run_simulation


def main() -> None:
    from v1_simulation.cli.main import main as cli_main

    cli_main()


__all__ = ["main", "run_bcm_training", "run_simulation"]
