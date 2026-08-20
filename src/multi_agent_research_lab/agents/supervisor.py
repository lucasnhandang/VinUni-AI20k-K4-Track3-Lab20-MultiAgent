from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import AgentName
from multi_agent_research_lab.core.state import ResearchState


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and enforces stopping guardrails."""

    name = "supervisor"

    def __init__(self, max_iterations: int | None = None, enable_critic: bool = True) -> None:
        settings = get_settings()
        self.max_iterations = max_iterations or settings.max_iterations
        self.enable_critic = enable_critic

    def decide_route(self, state: ResearchState) -> str:
        """Determine next node: 'researcher', 'analyst', 'writer', 'critic', or 'done'."""
        # 1. Guardrail: Max iterations reached
        if state.iteration >= self.max_iterations:
            return "done"

        # 2. Missing research sources
        if not state.sources:
            return "researcher"

        # 3. Missing structured analysis notes
        if not state.analysis_notes:
            return "analyst"

        # 4. Missing final synthesized answer
        if not state.final_answer:
            return "writer"

        # 5. Critic audit step (if enabled and not yet executed)
        has_critic_result = any(r.agent == AgentName.CRITIC for r in state.agent_results)
        if self.enable_critic and not has_critic_result:
            return "critic"

        # 6. Everything completed
        return "done"

    def run(self, state: ResearchState) -> ResearchState:
        """Evaluate state, determine next route, and record trace."""
        route = self.decide_route(state)
        state.record_route(route)
        state.add_trace_event(
            "supervisor.routed",
            {"route": route, "iteration": state.iteration},
        )
        return state
