from unittest.mock import MagicMock

from multi_agent_research_lab.agents import (
    AnalystAgent,
    CriticAgent,
    ResearcherAgent,
    SupervisorAgent,
    WriterAgent,
)
from multi_agent_research_lab.core.schemas import AgentName, ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMResponse


def test_supervisor_routing_cycle() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    supervisor = SupervisorAgent(max_iterations=6, enable_critic=True)

    # 1. Initially no sources -> routes to researcher
    assert supervisor.decide_route(state) == "researcher"

    # 2. Has sources, no analysis -> routes to analyst
    state.sources = [
        SourceDocument(title="Paper A", url="https://example.com/a", snippet="Doc content A")
    ]
    assert supervisor.decide_route(state) == "analyst"

    # 3. Has analysis, no final answer -> routes to writer
    state.analysis_notes = "Key analysis insights."
    assert supervisor.decide_route(state) == "writer"

    # 4. Has final answer, no critic -> routes to critic
    state.final_answer = "Final report draft."
    assert supervisor.decide_route(state) == "critic"

    # 5. Critic done -> routes to done
    state.agent_results.append(
        MagicMock(agent=AgentName.CRITIC, content="Review pass", metadata={})
    )
    assert supervisor.decide_route(state) == "done"


def test_supervisor_max_iterations_guardrail() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"), iteration=6)
    supervisor = SupervisorAgent(max_iterations=6)
    assert supervisor.decide_route(state) == "done"


def test_researcher_agent() -> None:
    mock_search = MagicMock()
    mock_search.search.return_value = [
        SourceDocument(title="Doc 1", url="https://test.com/1", snippet="Snippet 1")
    ]
    agent = ResearcherAgent(search_client=mock_search)
    state = ResearchState(request=ResearchQuery(query="Test query"))
    updated = agent.run(state)

    assert len(updated.sources) == 1
    assert "Doc 1" in (updated.research_notes or "")
    assert updated.agent_results[-1].agent == AgentName.RESEARCHER


def test_analyst_agent() -> None:
    mock_llm = MagicMock()
    mock_llm.complete.return_value = LLMResponse(
        content="Analytical Summary", input_tokens=50, output_tokens=30, cost_usd=0.0001
    )
    agent = AnalystAgent(llm_client=mock_llm)
    state = ResearchState(
        request=ResearchQuery(query="Test query"),
        sources=[SourceDocument(title="Doc 1", snippet="Snippet 1")],
    )
    updated = agent.run(state)

    assert updated.analysis_notes == "Analytical Summary"
    assert updated.agent_results[-1].agent == AgentName.ANALYST


def test_writer_agent() -> None:
    mock_llm = MagicMock()
    mock_llm.complete.return_value = LLMResponse(
        content="Comprehensive Report [1]\n### References\n[1] Doc 1",
        input_tokens=100,
        output_tokens=80,
        cost_usd=0.0002,
    )
    agent = WriterAgent(llm_client=mock_llm)
    state = ResearchState(
        request=ResearchQuery(query="Test query"),
        sources=[SourceDocument(title="Doc 1", snippet="Snippet 1")],
        analysis_notes="Key insights",
    )
    updated = agent.run(state)

    assert updated.final_answer is not None
    assert "[1]" in updated.final_answer
    assert updated.agent_results[-1].agent == AgentName.WRITER


def test_critic_agent() -> None:
    mock_llm = MagicMock()
    mock_llm.complete.return_value = LLMResponse(
        content="Critique: PASS. Citations valid.",
        input_tokens=80,
        output_tokens=40,
        cost_usd=0.0001,
    )
    agent = CriticAgent(llm_client=mock_llm)
    state = ResearchState(
        request=ResearchQuery(query="Test query"),
        sources=[SourceDocument(title="Doc 1", snippet="Snippet 1")],
        final_answer="Draft report [1]",
    )
    updated = agent.run(state)

    assert any(r.agent == AgentName.CRITIC for r in updated.agent_results)
