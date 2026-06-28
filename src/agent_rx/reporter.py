"""reporter.py - render a remediation report from a LoopResult."""

from __future__ import annotations

from agent_rx.analyze import AnalysisReport
from agent_rx.loop import LoopResult


def render_report(result: LoopResult) -> str:
    o = result.overall
    lines: list[str] = []
    lines.append("# agent-rx remediation report\n")

    verdict = "**significant**" if o.significant else "not significant"
    lines.append(
        f"**Outcome:** failure rate {o.control_rate:.1%} -> {o.treatment_rate:.1%} "
        f"({o.rel_delta:+.0%}), p={o.p_value:.4f} ({verdict}).\n"
    )
    lines.append(
        f"**Fixes accepted:** {len(result.accepted)} "
        f"({', '.join(result.accepted) if result.accepted else 'none'}).  "
        f"**Targeting precision:** {result.precision:.0%} of accepted fixes hit the "
        f"correct ground-truth lever.\n"
    )

    lines.append("## Config diff (baseline -> remediated)\n")
    lines.append("| knob | baseline | remediated |")
    lines.append("|------|----------|------------|")
    base = result.baseline_config.to_dict()
    fin = result.final_config.to_dict()
    for k in base:
        marker = "" if base[k] == fin[k] else "  ←changed"
        lines.append(f"| `{k}` | {base[k]} | {fin[k]}{marker} |")
    lines.append("")

    lines.append("## Iterations\n")
    lines.append("| # | incident | action | regime | result | accepted | right lever |")
    lines.append("|---|----------|--------|--------|--------|----------|-------------|")
    for it in result.iterations:
        acc = "yes" if it.accepted else "no"
        gt = "yes" if it.ground_truth_correct else "no"
        lines.append(
            f"| {it.index} | {it.incident} | `{it.action_id}` | {it.regime} | "
            f"{it.result.summary()} | {acc} | {gt} |"
        )
    lines.append("")

    m = result.prioritizer_metrics
    lines.append("## Prioritizer\n")
    lines.append(f"Final regime: **{result.final_regime}**, trained on {int(m.get('n', 0))} "
                 "self-generated outcome labels.\n")
    if "auc_learned" in m:
        lines.append(
            f"Offline ranking quality (AUC): learned **{m['auc_learned']:.2f}** vs "
            f"heuristic prior {m.get('auc_heuristic', float('nan')):.2f}"
            + (f", Brier {m['brier_learned']:.3f}." if "brier_learned" in m else ".")
        )
        lines.append("")
    elif "auc_heuristic" in m:
        lines.append(
            f"Heuristic-only AUC on the log: {m['auc_heuristic']:.2f} "
            "(not enough labels yet to train the learned model).\n"
        )

    return "\n".join(lines)


def render_plan(report: AnalysisReport) -> str:
    """Render a prioritized remediation plan from a real-trace analysis."""
    lines: list[str] = []
    lines.append("# agent-rx remediation plan\n")
    lines.append(
        f"Analyzed **{report.n_events}** events, found **{report.n_incidents}** "
        f"incident patterns at a **{report.failure_rate:.1%}** failure rate. "
        f"Ranked by the **{report.regime}** prioritizer (expected fix value).\n"
    )
    lines.append(
        "> Read-only mode: these are recommended fixes ranked by priority. "
        "Verification (the A/B significance gate) requires re-running the system "
        "under each patch, which static trace files can't provide.\n"
    )

    lines.append("| # | priority | incident | recovery | confidence | recommended fix |")
    lines.append("|---|----------|----------|----------|------------|-----------------|")
    for rec in report.recommendations:
        inc = rec.incident
        fix = rec.proposal.patch.action_id if rec.proposal else "no action (self-resolving)"
        lines.append(
            f"| {rec.rank} | {rec.priority:.2f} | {inc.display_name()} | "
            f"{inc.recovery_rate:.0%} | {inc.confidence_label} | `{fix}` |"
        )
    lines.append("")

    for rec in report.recommendations:
        if rec.proposal is None:
            continue
        lines.append(f"**#{rec.rank} {rec.incident.display_name()}** — {rec.proposal.rationale}")
        lines.append("")

    return "\n".join(lines)
