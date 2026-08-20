from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights and critical analysis."""

    name = "analyst"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes` from research sources."""
        if not state.sources and not state.research_notes:
            error_msg = "AnalystAgent: No sources or research notes available for analysis."
            state.errors.append(error_msg)
            return state

        sources_text = state.research_notes or "\n".join(
            f"[{idx}] {doc.title}: {doc.snippet}" for idx, doc in enumerate(state.sources, start=1)
        )

        system_prompt = (
            "You are an expert Research Analyst. Your role is to examine collected evidence, "
            "synthesize key insights, compare differing perspectives, evaluate evidence "
            "credibility, and highlight practical implications. Structure your response clearly "
            "with headings:\n"
            "- Key Findings & Core Concepts\n"
            "- Comparative Analysis & Trade-offs\n"
            "- Evidence Strength & Knowledge Gaps\n"
            "- Key Takeaways"
        )

        user_prompt = (
            f"User Research Query: {state.request.query}\n"
            f"Target Audience: {state.request.audience}\n\n"
            f"Collected Research Evidence:\n{sources_text}\n\n"
            "Please provide a thorough analytical breakdown."
        )

        response = self.llm_client.complete(system_prompt=system_prompt, user_prompt=user_prompt)
        state.analysis_notes = response.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.ANALYST,
                content=response.content,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
        )
        state.add_trace_event(
            "analyst.done",
            {
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
            },
        )
        return state
