from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class CriticAgent(BaseAgent):
    """Fact-checking, citation verification, and quality audit agent."""

    name = "critic"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final answer and append audit findings."""
        if not state.final_answer:
            state.errors.append("CriticAgent: No final_answer to audit.")
            return state

        sources_summary = "\n".join(
            f"[{idx}] {doc.title}: {doc.snippet}" for idx, doc in enumerate(state.sources, start=1)
        )

        system_prompt = (
            "You are a Senior Fact-Checker, Peer Reviewer, and Quality Assurance Critic. "
            "Your role is to evaluate a technical research report for:\n"
            "1. Factual Consistency: Check if statements align with source snippets.\n"
            "2. Citation Discipline: Check if bracketed citations (e.g. [1]) match sources.\n"
            "3. Hallucination Detection: Highlight claims unsupported by evidence.\n"
            "4. Completeness: Confirm that the user query is thoroughly answered.\n\n"
            "Output a structured review with:\n"
            "- Overall Assessment (PASS / MINOR ISSUES / MAJOR ISSUES)\n"
            "- Factual & Citation Integrity Analysis\n"
            "- Strengths & Weaknesses\n"
            "- Recommendations for refinement"
        )

        user_prompt = (
            f"Original Query: {state.request.query}\n\n"
            f"Ground Truth Sources:\n{sources_summary}\n\n"
            f"Draft Final Report Under Review:\n{state.final_answer}\n\n"
            "Perform your critical evaluation."
        )

        response = self.llm_client.complete(system_prompt=system_prompt, user_prompt=user_prompt)

        state.agent_results.append(
            AgentResult(
                agent=AgentName.CRITIC,
                content=response.content,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
        )
        state.add_trace_event(
            "critic.done",
            {
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
            },
        )
        return state
