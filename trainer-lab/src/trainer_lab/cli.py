import json

import typer


app = typer.Typer(help="trainer-lab scaffold CLI")


@app.command()
def info() -> None:
    payload = {
        "status": "placeholder",
        "service": "trainer-lab",
        "phase": "scaffolding",
    }
    typer.echo(json.dumps(payload))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
