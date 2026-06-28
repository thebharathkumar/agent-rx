"""agent-rx: close the loop on agent-triage.

agent-triage ranks multi-agent failures. agent-rx consumes that ranking,
proposes a fix from a constrained action space, A/B-tests it against the
unpatched system, and accepts it only on a statistically significant
improvement - learning, over time, which incidents are worth fixing.
"""

from agent_rx.loop import LoopResult, run_loop
from agent_rx.schema import RunConfig, TraceEvent

__version__ = "0.1.0"
__all__ = ["run_loop", "LoopResult", "RunConfig", "TraceEvent"]
