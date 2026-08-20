from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.search_client import SearchClient


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(self, search_client: SearchClient | None = None) -> None:
        self.search_client = search_client or SearchClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""
        docs = self.search_client.search(state.request.query, max_results=state.request.max_sources)
        state.sources = docs

        notes_lines: list[str] = [
            f"Retrieved {len(docs)} sources for query: '{state.request.query}'",
            "",
        ]
        for idx, doc in enumerate(docs, start=1):
            url_str = f" ({doc.url})" if doc.url else ""
            notes_lines.append(f"[{idx}] {doc.title}{url_str}")
            notes_lines.append(f"    Summary: {doc.snippet.strip()}")

        state.research_notes = "\n".join(notes_lines)
        state.agent_results.append(
            AgentResult(
                agent=AgentName.RESEARCHER,
                content=state.research_notes,
                metadata={"sources_count": len(docs)},
            )
        )
        state.add_trace_event("researcher.done", {"sources_count": len(docs)})
        return state
