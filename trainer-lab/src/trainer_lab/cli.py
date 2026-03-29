import json

import typer

from trainer_lab.training.mini_bench import run_mini_benchmark


app = typer.Typer(help="trainer-lab scaffold CLI")


@app.command()
def info() -> None:
    payload = {
        "status": "placeholder",
        "service": "trainer-lab",
        "phase": "scaffolding",
    }
    typer.echo(json.dumps(payload))


@app.command("mini-bench")
def mini_bench(
    steps: int = 6,
    batch_size: int = 64,
    board_size: int = 15,
    device: str | None = None,
) -> None:
    """Run a short synthetic training benchmark and print JSON metrics."""
    payload = run_mini_benchmark(
        steps=steps,
        batch_size=batch_size,
        board_size=board_size,
        device=device,
    )
    typer.echo(json.dumps(payload))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
