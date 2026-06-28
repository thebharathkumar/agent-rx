"""prioritizer.py - learn which incidents are worth fixing.

This is the ML core. agent-triage ranks incidents by a hand-tuned severity
formula. That answers "what is worst" but not "what is worth my next fix
attempt" - a high-severity incident that no available lever can move is not a
good use of the loop's budget.

The prioritizer learns P(an attempted fix yields a statistically significant
improvement) from the loop's own outcome log. The labels are self-generated:
every iteration the loop tries a fix and records whether it worked, so the
system manufactures its own training set as it runs (closed-loop / online).

Model: L2-regularized logistic regression implemented from scratch in numpy
(gradient descent on standardized features). No scikit-learn - the point is to
show the modeling, calibration, and evaluation explicitly. Until enough labels
exist, a transparent heuristic prior is used and the loop reports which regime
it is in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_rx.severity import CLASSIFICATION_WEIGHTS, ScoredIncident

# Classifications used for one-hot features, in a fixed order.
_CLASS_ORDER = (
    "coordination_failure",
    "agent_error",
    "information_lag",
    "environment_constraint",
)

FEATURE_NAMES = (
    "frequency_score",
    "severity_score",
    "recovery_rate",
    "final_score",
    "confidence",
    "run_coverage",
    "recovery_latency",
    *(f"is_{c}" for c in _CLASS_ORDER),
)

# Minimum labelled examples before the learned model overrides the heuristic.
MIN_TRAIN = 12


def extract_features(inc: ScoredIncident) -> list[float]:
    """Map a scored incident to the fixed-length feature vector."""
    coverage = inc.runs_seen_in / inc.runs_total if inc.runs_total else 0.0
    latency = inc.median_recovery_latency if inc.median_recovery_latency is not None else -1.0
    feats = [
        inc.frequency_score,
        inc.severity_score,
        inc.recovery_rate,
        inc.final_score,
        inc.confidence,
        coverage,
        latency,
    ]
    feats.extend(1.0 if inc.classification == c else 0.0 for c in _CLASS_ORDER)
    return feats


def heuristic_score(inc: ScoredIncident) -> float:
    """Cold-start prior: severe, unrecovered, fixable-looking incidents first.

    Normalized to [0, 1]. Environment constraints are damped because they
    self-resolve and there is no lever that helps.
    """
    base = CLASSIFICATION_WEIGHTS.get(inc.classification, 0.3)
    unrecovered = 1.0 - inc.recovery_rate
    severity_term = (inc.final_score / 16.0)  # 16 ~ max plausible final_score
    score = 0.5 * severity_term + 0.3 * unrecovered + 0.2 * base
    return max(0.0, min(1.0, score))


@dataclass
class _LogReg:
    """Minimal L2 logistic regression with feature standardization."""

    l2: float = 1.0
    lr: float = 0.2
    epochs: int = 600
    weights: Any = None
    bias: float = 0.0
    mean: Any = None
    std: Any = None

    def fit(self, X: list[list[float]], y: list[int]) -> None:
        import numpy as np

        Xa = np.asarray(X, dtype=float)
        ya = np.asarray(y, dtype=float)
        self.mean = Xa.mean(axis=0)
        self.std = Xa.std(axis=0)
        self.std[self.std == 0] = 1.0
        Xs = (Xa - self.mean) / self.std

        n, d = Xs.shape
        w = np.zeros(d)
        b = 0.0
        for _ in range(self.epochs):
            z = Xs @ w + b
            p = 1.0 / (1.0 + np.exp(-z))
            err = p - ya
            grad_w = (Xs.T @ err) / n + self.l2 * w / n
            grad_b = err.mean()
            w -= self.lr * grad_w
            b -= self.lr * grad_b
        self.weights = w
        self.bias = b

    def predict_proba(self, X: list[list[float]]) -> list[float]:
        import numpy as np

        if self.weights is None:
            raise RuntimeError("model not fitted")
        Xa = np.asarray(X, dtype=float)
        Xs = (Xa - self.mean) / self.std
        z = Xs @ self.weights + self.bias
        return (1.0 / (1.0 + np.exp(-z))).tolist()


def auc_score(y_true: list[int], y_score: list[float]) -> float:
    """Rank-based ROC AUC (Mann-Whitney U). 0.5 == random."""
    pos = [s for s, y in zip(y_score, y_true) if y == 1]
    neg = [s for s, y in zip(y_score, y_true) if y == 0]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def brier_score(y_true: list[int], y_prob: list[float]) -> float:
    return sum((p - y) ** 2 for p, y in zip(y_prob, y_true)) / len(y_true)


@dataclass
class Prioritizer:
    """Ranks incidents by learned (or heuristic) expected fix value."""

    min_train: int = MIN_TRAIN
    _X: list[list[float]] = field(default_factory=list)
    _y: list[int] = field(default_factory=list)
    _model: _LogReg | None = None

    @property
    def n_examples(self) -> int:
        return len(self._y)

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    @property
    def regime(self) -> str:
        return "learned" if self.is_trained else "heuristic"

    def add_outcome(self, inc: ScoredIncident, fix_succeeded: bool) -> None:
        """Record a (features -> did the fix help) example from the loop."""
        self._X.append(extract_features(inc))
        self._y.append(1 if fix_succeeded else 0)

    def maybe_fit(self) -> bool:
        """Retrain once there are enough examples spanning both classes."""
        if len(self._y) < self.min_train:
            return False
        if len(set(self._y)) < 2:  # need positives and negatives
            return False
        model = _LogReg()
        model.fit(self._X, self._y)
        self._model = model
        return True

    def score(self, inc: ScoredIncident) -> float:
        if self._model is None:
            return heuristic_score(inc)
        return self._model.predict_proba([extract_features(inc)])[0]

    def rank(self, incidents: list[ScoredIncident]) -> list[ScoredIncident]:
        return sorted(incidents, key=self.score, reverse=True)

    def evaluate(self) -> dict[str, float]:
        """Offline metrics on the accumulated log: learned vs heuristic.

        Reported so the loop can show that learning beats the prior rather
        than asserting it. AUC compares ranking quality; Brier compares
        calibrated probability quality.
        """
        out: dict[str, float] = {"n": float(len(self._y))}
        if len(set(self._y)) < 2:
            return out
        heur = [heuristic_score_from_features(x) for x in self._X]
        out["auc_heuristic"] = auc_score(self._y, heur)
        if self._model is not None:
            probs = self._model.predict_proba(self._X)
            out["auc_learned"] = auc_score(self._y, probs)
            out["brier_learned"] = brier_score(self._y, probs)
        return out


def heuristic_score_from_features(x: list[float]) -> float:
    """Heuristic score reconstructed from a feature vector (for offline eval).

    Mirrors ``heuristic_score`` using the same feature columns so the AUC
    comparison is apples-to-apples.
    """
    # indices follow FEATURE_NAMES
    recovery_rate = x[2]
    final_score = x[3]
    # class one-hots start after the 7 numeric features
    class_flags = x[7:7 + len(_CLASS_ORDER)]
    base = 0.3
    for flag, cls in zip(class_flags, _CLASS_ORDER):
        if flag >= 0.5:
            base = CLASSIFICATION_WEIGHTS.get(cls, 0.3)
            break
    severity_term = final_score / 16.0
    unrecovered = 1.0 - recovery_rate
    return max(0.0, min(1.0, 0.5 * severity_term + 0.3 * unrecovered + 0.2 * base))
