"""actions.py - the constrained action space of the remediation agent.

A "fix" is never arbitrary code. It is one entry from this registry applied
as a typed diff against RunConfig. Constraining the action space is what
makes the loop safe (no code execution), reproducible (a patch is data), and
measurable (we can attribute an improvement to exactly one lever).

Each action declares the classification it targets, so the proposer can map a
diagnosed incident to candidate patches, and evaluation can check whether the
chosen lever matches the environment's ground truth.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from agent_rx.schema import RunConfig


@dataclass(frozen=True)
class Patch:
    action_id: str
    target_classification: str
    field: str
    description: str
    _apply: Callable[[RunConfig], RunConfig]

    def apply(self, cfg: RunConfig) -> RunConfig:
        return self._apply(cfg)

    def changed(self, cfg: RunConfig) -> bool:
        """True if applying this patch would actually move the config."""
        before = getattr(cfg, self.field)
        after = getattr(self.apply(cfg), self.field)
        return before != after


def _raise_planner_consistency() -> Patch:
    return Patch(
        action_id="raise_planner_consistency",
        target_classification="coordination_failure",
        field="planner_consistency",
        description="Raise planner_consistency by +0.35 (cap 0.95) to reduce target desync.",
        _apply=lambda c: c.with_changes(
            planner_consistency=round(min(0.95, c.planner_consistency + 0.35), 3)
        ),
    )


def _add_navigator_retry() -> Patch:
    return Patch(
        action_id="add_navigator_retry",
        target_classification="agent_error",
        field="navigator_retries",
        description="Add one navigator retry so transient tool faults recover in-window.",
        _apply=lambda c: c.with_changes(navigator_retries=max(1, c.navigator_retries + 1)),
    )


def _raise_timeout() -> Patch:
    return Patch(
        action_id="raise_timeout",
        target_classification="information_lag",
        field="timeout_ms",
        description="Double the timeout budget (cap 3000ms) so belief state refreshes in time.",
        _apply=lambda c: c.with_changes(timeout_ms=min(3000, c.timeout_ms * 2)),
    )


def _acknowledge_environment() -> Patch:
    return Patch(
        action_id="acknowledge_environment",
        target_classification="environment_constraint",
        field="friction_rate",
        description="No-op: environment friction self-resolves; acknowledge and do not patch.",
        _apply=lambda c: c,  # intentionally inert
    )


# Registry keyed by action_id. Order matters only for deterministic listing.
_BUILDERS: dict[str, Callable[[], Patch]] = {
    "raise_planner_consistency": _raise_planner_consistency,
    "add_navigator_retry": _add_navigator_retry,
    "raise_timeout": _raise_timeout,
    "acknowledge_environment": _acknowledge_environment,
}


def all_actions() -> list[Patch]:
    return [build() for build in _BUILDERS.values()]


def action_by_id(action_id: str) -> Patch | None:
    builder = _BUILDERS.get(action_id)
    return builder() if builder else None


def candidate_patches_for(classification: str) -> list[Patch]:
    """Patches whose declared target matches a diagnosed classification."""
    return [p for p in all_actions() if p.target_classification == classification]


def action_schema() -> list[dict[str, str]]:
    """Machine-readable action catalog handed to the LLM proposer."""
    return [
        {
            "action_id": p.action_id,
            "targets": p.target_classification,
            "description": p.description,
        }
        for p in all_actions()
    ]
