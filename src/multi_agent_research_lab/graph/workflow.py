from typing import Any

from langgraph.graph import END, START, StateGraph

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient


class MultiAgentWorkflow:
    """Builds and executes the LangGraph multi-agent research workflow."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        search_client: SearchClient | None = None,
        max_iterations: int | None = None,
        enable_critic: bool = True,
    ) -> None:
        self.llm_client = llm_client or LLMClient()
        self.search_client = search_client or SearchClient()
        self.supervisor = SupervisorAgent(
            max_iterations=max_iterations, enable_critic=enable_critic
        )
        self.researcher = ResearcherAgent(search_client=self.search_client)
        self.analyst = AnalystAgent(llm_client=self.llm_client)
        self.writer = WriterAgent(llm_client=self.llm_client)
        self.critic = CriticAgent(llm_client=self.llm_client)

    def _wrap_agent(self, agent: BaseAgent) -> Any:
        def _node_fn(state: ResearchState) -> dict[str, Any]:
            updated_state = agent.run(state)
            return updated_state.model_dump()

        return _node_fn

    def build(self) -> Any:
        """Create and compile the LangGraph workflow graph."""
        builder = StateGraph(ResearchState)

        # Register nodes
        builder.add_node("supervisor", self._wrap_agent(self.supervisor))
        builder.add_node("researcher", self._wrap_agent(self.researcher))
        builder.add_node("analyst", self._wrap_agent(self.analyst))
        builder.add_node("writer", self._wrap_agent(self.writer))
        builder.add_node("critic", self._wrap_agent(self.critic))

        # Workflow entrypoint
        builder.add_edge(START, "supervisor")

        # Conditional routing from supervisor
        def _route_next(state: ResearchState) -> str:
            history = state.route_history
            if not history:
                return END
            last = history[-1]
            if last in ("researcher", "analyst", "writer", "critic"):
                return last
            return END

        builder.add_conditional_edges(
            "supervisor",
            _route_next,
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                "critic": "critic",
                END: END,
            },
        )

        # Workers return control back to supervisor
        builder.add_edge("researcher", "supervisor")
        builder.add_edge("analyst", "supervisor")
        builder.add_edge("writer", "supervisor")
        builder.add_edge("critic", "supervisor")

        return builder.compile()

    def run(self, state: ResearchState) -> ResearchState:
        """Execute compiled LangGraph workflow and return updated ResearchState."""
        graph = self.build()
        output = graph.invoke(state.model_dump())
        if isinstance(output, ResearchState):
            return output
        return ResearchState.model_validate(output)
