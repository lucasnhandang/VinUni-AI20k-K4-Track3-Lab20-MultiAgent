from unittest.mock import MagicMock

from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.services.llm_client import LLMResponse


def test_multi_agent_workflow_end_to_end() -> None:
    mock_search = MagicMock()
    mock_search.search.return_value = [
        SourceDocument(
            title="GraphRAG Overview",
            url="https://example.com/graphrag",
            snippet="GraphRAG combines knowledge graphs with LLM retrieval.",
        ),
        SourceDocument(
            title="Agent Architecture",
            url="https://example.com/agents",
            snippet="Multi-agent collaboration outperforms single-agent on complex tasks.",
        ),
    ]

    mock_llm = MagicMock()
    mock_llm.complete.side_effect = [
        # 1. Analyst call
        LLMResponse(
            content="Key Finding: GraphRAG improves grounded retrieval over baseline RAG.",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.0001,
        ),
        # 2. Writer call
        LLMResponse(
            content=(
                "Research Report:\n"
                "GraphRAG achieves high accuracy [1] via multi-agent design [2].\n"
                "### References\n"
                "[1] GraphRAG Overview\n"
                "[2] Agent Architecture"
            ),
            input_tokens=150,
            output_tokens=80,
            cost_usd=0.0002,
        ),
        # 3. Critic call
        LLMResponse(
            content="Critique: PASS. All 2 sources cited properly.",
            input_tokens=120,
            output_tokens=40,
            cost_usd=0.0001,
        ),
    ]

    workflow = MultiAgentWorkflow(
        llm_client=mock_llm,
        search_client=mock_search,
        max_iterations=6,
        enable_critic=True,
    )

    init_state = ResearchState(
        request=ResearchQuery(query="Explain GraphRAG state of the art", max_sources=2)
    )

    final_state = workflow.run(init_state)

    assert final_state.final_answer is not None
    assert "GraphRAG" in final_state.final_answer
    assert len(final_state.sources) == 2
    assert "researcher" in final_state.route_history
    assert "analyst" in final_state.route_history
    assert "writer" in final_state.route_history
    assert "critic" in final_state.route_history
    assert "done" in final_state.route_history
