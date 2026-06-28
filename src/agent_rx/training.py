"""training.py - offline pretraining + evaluation of the prioritizer.

The in-loop prioritizer learns online, but a single run only produces a
handful of labels. To get a model worth deploying - and to *evaluate* it
honestly - we generate many independent remediation episodes across randomized
buggy configurations, label each incident by whether its best available fix
produced a significant improvement, then fit and test the learned model on a
held-out split.

This is the part that answers an interviewer's "where is the ML, and how do
you know it works": a real train/test split with AUC and Brier, learned vs the
heuristic prior, on self-generated labels.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from agent_rx.environment import simulate
from agent_rx.prioritizer import (
    Prioritizer,
    _LogReg,
    auc_score,
    brier_score,
    extract_features,
    heuristic_score_from_features,
)
from agent_rx.proposer import HeuristicProposer
from agent_rx.schema import RunConfig
from agent_rx.severity import score_events
from agent_rx.stats import two_proportion_test


def random_buggy_config(rng: random.Random) -> RunConfig:
    """Sample a plausibly-broken starting config to remediate."""
    return RunConfig(
        planner_consistency=round(rng.uniform(0.45, 0.85), 3),
        navigator_retries=rng.choice([0, 0, 0, 1]),
        timeout_ms=rng.choice([600, 800, 1000, 1200]),
        friction_rate=round(rng.uniform(0.1, 0.3), 3),
    )


def _failures(cfg: RunConfig, runs: int, turns: int, seed: int) -> tuple[int, int]:
    events = simulate(cfg, num_runs=runs, num_turns=turns, seed=seed)
    fail = sum(1 for e in events if (not e.action_succeeded) or e.failure_classification is not None)
    return fail, len(events)


@dataclass
class Dataset:
    X: list[list[float]]
    y: list[int]


def generate_dataset(
    n_episodes: int = 60,
    *,
    seed: int = 7,
    eval_runs: int = 24,
    eval_turns: int = 20,
) -> Dataset:
    """One labelled example per (episode, incident): did its best fix help?

    Each incident is evaluated independently against its own config baseline,
    so examples are i.i.d.-ish rather than entangled by sequential patching.
    """
    rng = random.Random(seed)
    proposer = HeuristicProposer()
    X: list[list[float]] = []
    y: list[int] = []

    for ep in range(n_episodes):
        cfg = random_buggy_config(rng)
        eval_seed = 5000 + ep * 7
        events = simulate(cfg, num_runs=12, num_turns=20, seed=100 + ep)
        for inc in score_events(events):
            proposals = [p for p in proposer.propose(inc) if p.patch.changed(cfg)]
            if not proposals:
                # No lever moves this incident (e.g. environment_constraint):
                # a real negative the model should learn to deprioritize.
                X.append(extract_features(inc))
                y.append(0)
                continue
            patch = proposals[0].patch
            c_fail, c_tot = _failures(cfg, eval_runs, eval_turns, eval_seed)
            t_fail, t_tot = _failures(patch.apply(cfg), eval_runs, eval_turns, eval_seed)
            res = two_proportion_test(c_fail, c_tot, t_fail, t_tot)
            X.append(extract_features(inc))
            y.append(1 if res.significant else 0)

    return Dataset(X=X, y=y)


@dataclass
class TrainReport:
    n_train: int
    n_test: int
    positive_rate: float
    auc_learned: float
    auc_heuristic: float
    brier_learned: float
    brier_baserate: float

    def summary(self) -> str:
        return (
            f"train={self.n_train} test={self.n_test} pos_rate={self.positive_rate:.2f} | "
            f"AUC learned={self.auc_learned:.3f} vs heuristic={self.auc_heuristic:.3f} | "
            f"Brier learned={self.brier_learned:.3f} vs base-rate={self.brier_baserate:.3f}"
        )


def train_and_evaluate(
    dataset: Dataset | None = None,
    *,
    test_frac: float = 0.3,
    seed: int = 0,
) -> tuple[Prioritizer, TrainReport]:
    """Fit the learned prioritizer and report held-out metrics vs the prior."""
    ds = dataset or generate_dataset()
    idx = list(range(len(ds.y)))
    random.Random(seed).shuffle(idx)
    cut = int(len(idx) * (1 - test_frac))
    train_idx, test_idx = idx[:cut], idx[cut:]

    Xtr = [ds.X[i] for i in train_idx]
    ytr = [ds.y[i] for i in train_idx]
    Xte = [ds.X[i] for i in test_idx]
    yte = [ds.y[i] for i in test_idx]

    model = _LogReg()
    model.fit(Xtr, ytr)
    probs = model.predict_proba(Xte)
    heur = [heuristic_score_from_features(x) for x in Xte]
    base_rate = sum(ytr) / len(ytr) if ytr else 0.0

    report = TrainReport(
        n_train=len(ytr),
        n_test=len(yte),
        positive_rate=sum(ds.y) / len(ds.y) if ds.y else 0.0,
        auc_learned=auc_score(yte, probs),
        auc_heuristic=auc_score(yte, heur),
        brier_learned=brier_score(yte, probs),
        brier_baserate=brier_score(yte, [base_rate] * len(yte)) if yte else float("nan"),
    )

    # Return a prioritizer primed with the full dataset and fitted model.
    primed = Prioritizer(min_train=1)
    primed._X = list(ds.X)
    primed._y = list(ds.y)
    primed.maybe_fit()
    return primed, report
