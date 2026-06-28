"""severity.py - group trace events into incidents and score them.

This is a compact, dependency-free reimplementation of the agent-triage
grouping + scoring contract, kept in-tree so agent-rx runs standalone. The
scoring model (weights, recovery window, frequency/severity split) is
deliberately identical to agent-triage so a remediation report and a triage
report agree on what "the top incident" is. If agent-triage is installed,
``score_events`` can be swapped for ``triage.scorer.score_patterns`` without
changing any downstream code.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import median

from agent_rx.schema import TraceEvent

CLASSIFICATION_WEIGHTS: dict[str, float] = {
    "coordination_failure": 1.0,
    "agent_error": 0.7,
    "information_lag": 0.5,
    "environment_constraint": 0.2,
    "unclassified": 0.3,
}
NO_RECOVERY_MULTIPLIER = 1.5
RECOVERY_WINDOW = 3
FREQUENCY_WEIGHT = 0.4
SEVERITY_WEIGHT = 0.6
CONFIDENCE_THRESHOLD = 5


@dataclass
class ScoredIncident:
    pattern_id: str
    agent_id: str
    tool_name: str
    classification: str
    divergence_fields: tuple[str, ...]
    frequency: int
    frequency_score: float
    severity_score: float
    recovery_rate: float
    final_score: float
    confidence: float
    median_recovery_latency: float | None
    runs_seen_in: int
    runs_total: int

    @property
    def confidence_label(self) -> str:
        if self.confidence >= 0.8:
            return "high"
        if self.confidence >= 0.4:
            return "medium"
        return "low"

    def display_name(self) -> str:
        div = "+".join(self.divergence_fields) if self.divergence_fields else "no-divergence"
        return f"[{self.agent_id}] {self.tool_name} / {self.classification} / {div}"


def _is_failure(ev: TraceEvent) -> bool:
    return (not ev.action_succeeded) or (ev.failure_classification is not None)


def _build_timeline(events: list[TraceEvent]) -> dict[tuple[str, str], list[tuple[int, bool]]]:
    timeline: dict[tuple[str, str], list[tuple[int, bool]]] = defaultdict(list)
    for ev in events:
        timeline[(ev.run_id, ev.agent_id)].append((ev.turn, ev.action_succeeded))
    for key in timeline:
        timeline[key].sort()
    return timeline


def score_events(events: list[TraceEvent]) -> list[ScoredIncident]:
    """Group failures into incident patterns and score them, highest first."""
    timeline = _build_timeline(events)
    total_runs = len({ev.run_id for ev in events})

    buckets: dict[tuple, list[TraceEvent]] = defaultdict(list)
    order: list[tuple] = []
    for ev in events:
        if not _is_failure(ev):
            continue
        classification = ev.failure_classification or "unclassified"
        key = (ev.agent_id, ev.tool_name, classification, ev.divergence_fields)
        if key not in buckets:
            order.append(key)
        buckets[key].append(ev)

    max_freq = max((len(buckets[k]) for k in order), default=1)
    scored: list[ScoredIncident] = []
    for key in order:
        agent_id, tool_name, classification, divergence = key
        incidents = buckets[key]
        freq = len(incidents)

        recovered, latencies = 0, []
        for inc in incidents:
            offset = _first_success_offset(inc, timeline)
            if offset is not None and offset <= RECOVERY_WINDOW:
                recovered += 1
                latencies.append(offset)
        recovery_rate = recovered / freq

        freq_score = (freq / max_freq) * 10.0
        base = CLASSIFICATION_WEIGHTS.get(classification, 0.3)
        mult = 1.0 if recovery_rate > 0 else NO_RECOVERY_MULTIPLIER
        sev_score = base * 10.0 * mult
        final = freq_score * FREQUENCY_WEIGHT + sev_score * SEVERITY_WEIGHT

        div = "+".join(sorted(divergence)) if divergence else "none"
        scored.append(
            ScoredIncident(
                pattern_id=f"{agent_id}-{tool_name}-{classification}-{div}",
                agent_id=agent_id,
                tool_name=tool_name,
                classification=classification,
                divergence_fields=divergence,
                frequency=freq,
                frequency_score=freq_score,
                severity_score=sev_score,
                recovery_rate=recovery_rate,
                final_score=final,
                confidence=min(1.0, freq / CONFIDENCE_THRESHOLD),
                median_recovery_latency=float(median(latencies)) if latencies else None,
                runs_seen_in=len({inc.run_id for inc in incidents}),
                runs_total=total_runs,
            )
        )

    scored.sort(key=lambda s: s.final_score, reverse=True)
    return scored


def _first_success_offset(
    incident: TraceEvent,
    timeline: dict[tuple[str, str], list[tuple[int, bool]]],
) -> int | None:
    agent_turns = timeline.get((incident.run_id, incident.agent_id), [])
    for turn, ok in agent_turns:
        if turn > incident.turn and ok:
            return turn - incident.turn
    return None


def failure_rate(events: list[TraceEvent]) -> float:
    """Fraction of all actions that failed. The loop's headline metric."""
    if not events:
        return 0.0
    return sum(1 for ev in events if _is_failure(ev)) / len(events)
