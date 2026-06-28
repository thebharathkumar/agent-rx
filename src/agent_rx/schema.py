"""schema.py - trace event schema, shared across the pipeline.

The schema is intentionally a strict subset of the agent-triage NDJSON
schema (https://github.com/thebharathkumar/agent-triage). Traces emitted
by this package's environment load directly into agent-triage, and traces
produced by agent-triage-compatible systems load here. Dataclasses are used
instead of pydantic so the core runs on the standard library alone.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# The four canonical failure classifications, ordered by operational cost.
# Mirrors agent-triage so reports line up across both tools.
CLASSIFICATIONS = (
    "coordination_failure",
    "agent_error",
    "information_lag",
    "environment_constraint",
)


@dataclass(frozen=True)
class TraceEvent:
    """One action taken by one agent on one turn of one run."""

    event_id: str
    run_id: str
    turn: int
    agent_id: str
    tool_name: str
    action_succeeded: bool
    failure_classification: str | None = None
    divergence_fields: tuple[str, ...] = ()
    latency_ms: int = 0

    def to_json_line(self) -> str:
        """Serialize to an agent-triage-compatible NDJSON line."""
        payload: dict[str, Any] = {
            "event_id": self.event_id,
            "run_id": self.run_id,
            "turn": self.turn,
            "agent_id": self.agent_id,
            "action_taken": {"tool_name": self.tool_name, "tool_input": {}},
            "action_succeeded": self.action_succeeded,
            "divergence_fields": list(self.divergence_fields),
            "failure_classification": self.failure_classification,
            "latency_ms": {"total": self.latency_ms},
        }
        return json.dumps(payload)

    @classmethod
    def from_obj(cls, obj: dict[str, Any]) -> TraceEvent:
        """Parse one decoded NDJSON object (agent-triage shape)."""
        action = obj.get("action_taken", {}) or {}
        latency = obj.get("latency_ms", {})
        if isinstance(latency, dict):
            latency_total = int(latency.get("total", 0))
        else:
            latency_total = int(latency or 0)
        raw_class = obj.get("failure_classification")
        classification = None if raw_class in (None, "null", "") else str(raw_class)
        return cls(
            event_id=str(obj.get("event_id", "")),
            run_id=str(obj["run_id"]),
            turn=int(obj["turn"]),
            agent_id=str(obj["agent_id"]),
            tool_name=str(action.get("tool_name", obj.get("tool_name", "unknown"))),
            action_succeeded=bool(obj["action_succeeded"]),
            failure_classification=classification,
            divergence_fields=tuple(obj.get("divergence_fields", []) or ()),
            latency_ms=latency_total,
        )


def write_ndjson(events: list[TraceEvent], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for ev in events:
            fh.write(ev.to_json_line() + "\n")


def read_ndjson(path: Path) -> tuple[list[TraceEvent], list[str]]:
    """Read an NDJSON file. Returns (events, parse_errors)."""
    events: list[TraceEvent] = []
    errors: list[str] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(TraceEvent.from_obj(json.loads(line)))
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                errors.append(f"{path}:{lineno}: {exc}")
    return events, errors


@dataclass
class RunConfig:
    """Tunable knobs of the multi-agent system under test.

    Every field is a lever the remediation agent is allowed to move. The
    action space (actions.py) is defined entirely in terms of these fields,
    which keeps proposed fixes safe, typed, and reproducible: a "fix" is a
    diff against this config, never arbitrary code.
    """

    # Probability the planner emits a target consistent with the navigator's
    # belief. Low values -> coordination_failure. Ground-truth fix: raise it.
    planner_consistency: float = 0.55
    # Retries the navigator attempts on a transient tool error. 0 -> the
    # error surfaces as an unrecovered agent_error. Ground-truth fix: >= 1.
    navigator_retries: int = 0
    # Per-action timeout budget (ms). Too low -> information_lag as the agent
    # acts on stale belief state. Ground-truth fix: raise it.
    timeout_ms: int = 800
    # Base rate of environment friction (wall bumps). These self-resolve and
    # should NOT be "fixed"; included so the agent can learn to leave them be.
    friction_rate: float = 0.20

    def with_changes(self, **changes: Any) -> RunConfig:
        data = asdict(self)
        data.update(changes)
        return RunConfig(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
