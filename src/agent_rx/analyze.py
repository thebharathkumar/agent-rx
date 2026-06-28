"""analyze.py - run agent-rx over real trace files (read-only advisory mode).

The full loop needs a re-runnable system to A/B-test a fix. Against a batch of
static trace files (e.g. real dungeon-traces output) you cannot re-run, so this
module exposes the half of the loop that does work on static data: ingest real
NDJSON, score incidents, rank them with the (optionally pretrained) learned
prioritizer, and recommend a concrete patch for each. The output is a
prioritized remediation plan, not a verified fix.

The trace schema is identical to agent-triage, so anything triage can read,
agent-rx can analyze.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agent_rx.prioritizer import Prioritizer
from agent_rx.proposer import HeuristicProposer, Proposal
from agent_rx.schema import TraceEvent, read_ndjson
from agent_rx.severity import ScoredIncident, failure_rate, score_events


def load_paths(paths: list[str | Path]) -> tuple[list[TraceEvent], list[str]]:
    """Load NDJSON from files, directories, or a mix. Returns (events, errors).

    A directory loads every ``*.ndjson`` / ``*.jsonl`` inside it (non-recursive),
    matching agent-triage's batch behavior.
    """
    events: list[TraceEvent] = []
    errors: list[str] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            files = sorted(
                [*p.glob("*.ndjson"), *p.glob("*.jsonl")]
            )
        else:
            files = [p]
        for f in files:
            evs, errs = read_ndjson(f)
            events.extend(evs)
            errors.extend(errs)
    return events, errors


@dataclass
class Recommendation:
    rank: int
    incident: ScoredIncident
    priority: float          # prioritizer score in [0, 1]
    proposal: Proposal | None


@dataclass
class AnalysisReport:
    n_events: int
    n_incidents: int
    failure_rate: float
    regime: str
    recommendations: list[Recommendation]


def analyze_events(
    events: list[TraceEvent],
    *,
    prioritizer: Prioritizer | None = None,
    top: int = 5,
) -> AnalysisReport:
    """Score, prioritize, and recommend fixes for a batch of real traces."""
    incidents = score_events(events)
    prio = prioritizer or Prioritizer()
    proposer = HeuristicProposer()

    ranked = prio.rank(incidents)[:top]
    recs: list[Recommendation] = []
    for i, inc in enumerate(ranked, start=1):
        proposals = proposer.propose(inc)
        # Drop pure no-op acknowledgements (e.g. environment_constraint) so the
        # recommendation reads as "no action" rather than a fake fix.
        actionable = [p for p in proposals if p.patch.action_id != "acknowledge_environment"]
        recs.append(
            Recommendation(
                rank=i,
                incident=inc,
                priority=prio.score(inc),
                proposal=actionable[0] if actionable else None,
            )
        )

    return AnalysisReport(
        n_events=len(events),
        n_incidents=len(incidents),
        failure_rate=failure_rate(events),
        regime=prio.regime,
        recommendations=recs,
    )
