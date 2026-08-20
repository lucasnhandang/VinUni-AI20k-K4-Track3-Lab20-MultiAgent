from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes with rigorous citations."""

    name = "writer"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`."""
        sources_list: list[str] = []
        for idx, doc in enumerate(state.sources, start=1):
            url_info = f" ({doc.url})" if doc.url else ""
            sources_list.append(f"[{idx}] {doc.title}{url_info}\n    Excerpt: {doc.snippet}")
        sources_context = (
            "\n".join(sources_list) if sources_list else "No direct sources available."
        )

        analysis_context = (
            state.analysis_notes or state.research_notes or "No prior analysis notes."
        )

        system_prompt = (
            "You are a Principal Technical Writer and Research Synthesis Expert. "
            "Your objective is to produce an authoritative, comprehensive, and well-structured "
            "final research report based on the provided analysis and sources.\n\n"
            "Style & Grounding Rules:\n"
            "1. Tailor the tone, depth, and terminology for the specified audience.\n"
            "2. Ensure strict factual grounding: every factual claim MUST include bracketed "
            "numerical citations (e.g. [1], [2]) corresponding to the provided sources.\n"
            "3. Conclude with a dedicated '### References' section listing all cited sources.\n"
            "4. Organize the report with clear headings, bullet points, and actionable summaries."
        )

        user_prompt = (
            f"Research Question: {state.request.query}\n"
            f"Target Audience: {state.request.audience}\n\n"
            f"Analyst Insights:\n{analysis_context}\n\n"
            f"Available Sources for Citation:\n{sources_context}\n\n"
            "Write the comprehensive final report now."
        )

        response = self.llm_client.complete(system_prompt=system_prompt, user_prompt=user_prompt)
        state.final_answer = response.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=response.content,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
        )
        state.add_trace_event(
            "writer.done",
            {
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
            },
        )
        return state
