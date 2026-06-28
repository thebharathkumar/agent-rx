"""stats.py - is the before/after difference real, or just noise?

The loop accepts a fix only if the treatment run beats the control run by a
margin unlikely to come from randomness. We use a two-proportion z-test on
the action-level failure rate (failures / total actions). No scipy: the
normal CDF comes from math.erfc, which is plenty for a go/no-go gate.

Small samples are flagged ``tentative`` so a thin-evidence win is not treated
as a verified one. This mirrors agent-triage's handling of low-occurrence
deltas.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Below this count of failures on either side, call the result tentative.
TENTATIVE_MIN_FAILURES = 5


@dataclass
class SignificanceResult:
    control_rate: float
    treatment_rate: float
    abs_delta: float          # treatment - control (negative == improvement)
    rel_delta: float          # fraction reduction vs control
    z: float
    p_value: float
    significant: bool         # improved AND p < alpha
    tentative: bool

    def summary(self) -> str:
        direction = "improved" if self.abs_delta < 0 else "regressed"
        verdict = "significant" if self.significant else "not significant"
        tag = " (tentative)" if self.tentative else ""
        return (
            f"{self.control_rate:.1%} -> {self.treatment_rate:.1%} "
            f"({self.rel_delta:+.0%}, {direction}, p={self.p_value:.3f}, {verdict}){tag}"
        )


def _normal_sf(z: float) -> float:
    """Upper-tail survival function of the standard normal."""
    return 0.5 * math.erfc(z / math.sqrt(2))


def two_proportion_test(
    control_failures: int,
    control_total: int,
    treatment_failures: int,
    treatment_total: int,
    alpha: float = 0.05,
) -> SignificanceResult:
    """Pooled two-proportion z-test, two-sided p-value."""
    p_c = control_failures / control_total if control_total else 0.0
    p_t = treatment_failures / treatment_total if treatment_total else 0.0

    pooled = (
        (control_failures + treatment_failures) / (control_total + treatment_total)
        if (control_total + treatment_total)
        else 0.0
    )
    se = math.sqrt(
        pooled * (1 - pooled) * (1 / control_total + 1 / treatment_total)
    ) if control_total and treatment_total and 0 < pooled < 1 else 0.0

    z = (p_t - p_c) / se if se > 0 else 0.0
    p_value = 2 * _normal_sf(abs(z))

    abs_delta = p_t - p_c
    rel_delta = (abs_delta / p_c) if p_c > 0 else 0.0
    improved = abs_delta < 0
    tentative = min(control_failures, treatment_failures) < TENTATIVE_MIN_FAILURES

    return SignificanceResult(
        control_rate=p_c,
        treatment_rate=p_t,
        abs_delta=abs_delta,
        rel_delta=rel_delta,
        z=z,
        p_value=p_value,
        significant=improved and p_value < alpha,
        tentative=tentative,
    )
