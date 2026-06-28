"""loop.py - the closed remediation loop.

Pipeline per iteration:

  1. simulate the system under the current config and score incidents
  2. the learned prioritizer ranks incidents by expected fix value
  3. the proposer diagnoses the top incident and proposes a patch
  4. run a matched A/B (control = current config, treatment = patched) on a
     held-out seed range, then a two-proportion z-test on the failure rate
  5. accept the patch only if the improvement is significant
  6. log the outcome as a training example and retrain the prioritizer

The held-out evaluation seeds are disjoint from the seeds used to surface
incidents, so a fix has to generalize, not just fit the batch it was found in.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent_rx.actions import Patch
from agent_rx.environment import GROUND_TRUTH_FIX, simulate
from agent_rx.prioritizer import Prioritizer
from agent_rx.proposer import HeuristicProposer, LLMProposer, Proposal
from agent_rx.schema import RunConfig
from agent_rx.severity import ScoredIncident, score_events
from agent_rx.stats import SignificanceResult, two_proportion_test


@dataclass
class IterationRecord:
    index: int
    incident: str
    classification: str
    action_id: str
    rationale: str
    source: str
    regime: str             # prioritizer regime at decision time
    result: SignificanceResult
    accepted: bool
    ground_truth_correct: bool


@dataclass
class LoopResult:
    baseline_config: RunConfig
    final_config: RunConfig
    overall: SignificanceResult
    iterations: list[IterationRecord]
    accepted: list[str]
    prioritizer_metrics: dict[str, float]
    final_regime: str

    @property
    def precision(self) -> float:
        """Fraction of accepted fixes that targeted the right ground-truth lever."""
        accepted = [it for it in self.iterations if it.accepted]
        if not accepted:
            return 0.0
        correct = sum(1 for it in accepted if it.ground_truth_correct)
        return correct / len(accepted)


def _failure_counts(cfg: RunConfig, runs: int, turns: int, seed: int) -> tuple[int, int]:
    events = simulate(cfg, num_runs=runs, num_turns=turns, seed=seed)
    failures = sum(
        1 for e in events if (not e.action_succeeded) or e.failure_classification is not None
    )
    return failures, len(events)


def _cfg_sig(cfg: RunConfig) -> tuple:
    return tuple(sorted(cfg.to_dict().items()))


def run_loop(
    baseline: RunConfig | None = None,
    *,
    max_iters: int = 8,
    discover_runs: int = 12,
    discover_turns: int = 20,
    discover_seed: int = 42,
    eval_runs: int = 40,
    eval_turns: int = 20,
    eval_seed: int = 1000,
    alpha: float = 0.05,
    prefer_llm: bool = False,
    prioritizer: Prioritizer | None = None,
) -> LoopResult:
    cfg = baseline or RunConfig()
    proposer = LLMProposer() if (prefer_llm and LLMProposer().available()) else HeuristicProposer()
    prioritizer = prioritizer or Prioritizer()

    iterations: list[IterationRecord] = []
    accepted: list[str] = []
    tried: set[tuple] = set()

    for i in range(max_iters):
        events = simulate(cfg, num_runs=discover_runs, num_turns=discover_turns, seed=discover_seed)
        incidents = score_events(events)
        if not incidents:
            break

        choice = _select(incidents, proposer, prioritizer, cfg, tried)
        if choice is None:
            break  # nothing left to try -> converged
        inc, proposal = choice
        patch = proposal.patch
        tried.add((_cfg_sig(cfg), patch.action_id))

        candidate = patch.apply(cfg)
        c_fail, c_tot = _failure_counts(cfg, eval_runs, eval_turns, eval_seed)
        t_fail, t_tot = _failure_counts(candidate, eval_runs, eval_turns, eval_seed)
        result = two_proportion_test(c_fail, c_tot, t_fail, t_tot, alpha=alpha)

        accepted_now = result.significant
        prioritizer.add_outcome(inc, accepted_now)
        gt_correct = GROUND_TRUTH_FIX.get(inc.classification) == patch.field

        iterations.append(
            IterationRecord(
                index=i + 1,
                incident=inc.display_name(),
                classification=inc.classification,
                action_id=patch.action_id,
                rationale=proposal.rationale,
                source=proposal.source,
                regime=prioritizer.regime,
                result=result,
                accepted=accepted_now,
                ground_truth_correct=gt_correct,
            )
        )
        if accepted_now:
            cfg = candidate
            accepted.append(patch.action_id)
        prioritizer.maybe_fit()

    base_fail, base_tot = _failure_counts(
        baseline or RunConfig(), eval_runs, eval_turns, eval_seed
    )
    fin_fail, fin_tot = _failure_counts(cfg, eval_runs, eval_turns, eval_seed)
    overall = two_proportion_test(base_fail, base_tot, fin_fail, fin_tot, alpha=alpha)

    return LoopResult(
        baseline_config=baseline or RunConfig(),
        final_config=cfg,
        overall=overall,
        iterations=iterations,
        accepted=accepted,
        prioritizer_metrics=prioritizer.evaluate(),
        final_regime=prioritizer.regime,
    )


def _select(
    incidents: list[ScoredIncident],
    proposer: HeuristicProposer | LLMProposer,
    prioritizer: Prioritizer,
    cfg: RunConfig,
    tried: set[tuple],
) -> tuple[ScoredIncident, Proposal] | None:
    """Pick the highest-priority incident with an untried, effective patch."""
    sig = _cfg_sig(cfg)
    for inc in prioritizer.rank(incidents):
        for proposal in proposer.propose(inc):
            patch: Patch = proposal.patch
            if not patch.changed(cfg):
                continue
            if (sig, patch.action_id) in tried:
                continue
            return inc, proposal
    return None
