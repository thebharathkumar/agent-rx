"""cli.py - command-line entry point for agent-rx."""

from __future__ import annotations

from pathlib import Path

import click

from agent_rx.environment import simulate
from agent_rx.loop import run_loop
from agent_rx.reporter import render_report
from agent_rx.schema import RunConfig, write_ndjson


@click.group()
@click.version_option()
def main() -> None:
    """agent-rx: close the loop on agent-triage."""


@main.command()
@click.option("--max-iters", default=8, show_default=True, help="Max remediation attempts.")
@click.option("--llm/--no-llm", default=False, help="Use Claude to propose fixes if available.")
@click.option("--output", type=click.Path(), default=None, help="Write the report to a file.")
def run(max_iters: int, llm: bool, output: str | None) -> None:
    """Run the remediation loop and print a report."""
    result = run_loop(max_iters=max_iters, prefer_llm=llm)
    report = render_report(result)
    if output:
        Path(output).write_text(report, encoding="utf-8")
        click.echo(f"Wrote report to {output}")
    else:
        click.echo(report)


@main.command()
def demo() -> None:
    """Run the offline, deterministic end-to-end demo (no API key needed)."""
    result = run_loop()
    click.echo(render_report(result))


@main.command()
@click.option("--episodes", default=60, show_default=True, help="Remediation episodes to simulate.")
@click.option("--test-frac", default=0.3, show_default=True)
def train(episodes: int, test_frac: float) -> None:
    """Pretrain the learned prioritizer offline and print held-out metrics."""
    from agent_rx.training import generate_dataset, train_and_evaluate

    ds = generate_dataset(n_episodes=episodes)
    _, report = train_and_evaluate(ds, test_frac=test_frac)
    click.echo(report.summary())


@main.command()
@click.option("--output", type=click.Path(), required=True, help="NDJSON output path.")
@click.option("--runs", default=12, show_default=True)
@click.option("--turns", default=20, show_default=True)
@click.option("--seed", default=42, show_default=True)
def trace(output: str, runs: int, turns: int, seed: int) -> None:
    """Emit agent-triage-compatible NDJSON traces from the baseline system."""
    events = simulate(RunConfig(), num_runs=runs, num_turns=turns, seed=seed)
    write_ndjson(events, Path(output))
    click.echo(f"Wrote {len(events)} events to {output} (load with `triage report`).")


if __name__ == "__main__":
    main()
