"""proposer.py - diagnose an incident and propose a concrete fix.

Two interchangeable proposers implement the same ``propose`` contract:

  HeuristicProposer  deterministic, offline, no API key. Maps a diagnosed
                     classification to its candidate patches. Used for the
                     reproducible demo and CI.

  LLMProposer        asks Claude to read the incident context and pick an
                     action from the constrained schema, returning structured
                     JSON. Used for the live mode. Falls back cleanly if the
                     anthropic package or API key is missing.

Both return a ranked list of Proposals; the loop applies the top one whose
patch actually changes the config.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from agent_rx.actions import Patch, action_schema, candidate_patches_for
from agent_rx.severity import ScoredIncident


@dataclass
class Proposal:
    patch: Patch
    rationale: str
    source: str


class HeuristicProposer:
    source = "heuristic"

    def propose(self, inc: ScoredIncident) -> list[Proposal]:
        proposals = []
        for patch in candidate_patches_for(inc.classification):
            rationale = (
                f"{inc.classification} on [{inc.agent_id}] {inc.tool_name} with "
                f"{inc.recovery_rate:.0%} recovery over {inc.frequency} events; "
                f"{patch.description}"
            )
            proposals.append(Proposal(patch=patch, rationale=rationale, source=self.source))
        return proposals


_LLM_SYSTEM = """\
You are a site-reliability engineer for multi-agent systems. Given one failure
incident and a fixed catalog of remediation actions, choose the single action
most likely to reduce the failure, or "none" if no action fits.

Rules:
- Pick exactly one action_id from the catalog, or "none".
- Output strict JSON: {"action_id": "...", "rationale": "<= 2 sentences"}.
- Do not invent action_ids. Do not output anything except the JSON object.
"""


class LLMProposer:
    """Claude-backed proposer. Optional; requires anthropic + ANTHROPIC_API_KEY."""

    source = "llm"

    def __init__(self, model: str = "claude-haiku-4-5-20251001", api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

    def available(self) -> bool:
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False
        return bool(self.api_key)

    def propose(self, inc: ScoredIncident) -> list[Proposal]:
        import anthropic

        client = anthropic.Anthropic(api_key=self.api_key)
        user = (
            f"Incident: {inc.display_name()}\n"
            f"classification: {inc.classification}\n"
            f"frequency: {inc.frequency}, recovery_rate: {inc.recovery_rate:.0%}, "
            f"confidence: {inc.confidence_label}\n"
            f"divergence_fields: {list(inc.divergence_fields) or 'none'}\n\n"
            f"Action catalog:\n{json.dumps(action_schema(), indent=2)}\n\n"
            "Choose one action_id (or 'none')."
        )
        msg = client.messages.create(
            model=self.model,
            max_tokens=200,
            system=[{"type": "text", "text": _LLM_SYSTEM,
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
        )
        text = getattr(msg.content[0], "text", "{}")
        return self._parse(text, inc)

    def _parse(self, text: str, inc: ScoredIncident) -> list[Proposal]:
        from agent_rx.actions import action_by_id

        try:
            data = json.loads(text.strip())
            action_id = data.get("action_id")
            rationale = data.get("rationale", "")
        except (json.JSONDecodeError, AttributeError):
            return []
        if action_id in (None, "none"):
            return []
        patch = action_by_id(str(action_id))
        if patch is None:
            return []
        return [Proposal(patch=patch, rationale=rationale or "(llm)", source=self.source)]


def get_proposer(prefer_llm: bool = False) -> HeuristicProposer | LLMProposer:
    """Return the LLM proposer when usable, else the heuristic one."""
    if prefer_llm:
        llm = LLMProposer()
        if llm.available():
            return llm
    return HeuristicProposer()
