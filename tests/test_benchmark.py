from multi_agent_research_lab.core.schemas import (
    AgentName,
    AgentResult,
    ResearchQuery,
    SourceDocument,
)
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import compute_citation_coverage, run_benchmark


def test_compute_citation_coverage() -> None:
    state = ResearchState(
        request=ResearchQuery(query="Test query"),
        sources=[
            SourceDocument(title="Doc Alpha", url="https://example.com/alpha", snippet="..."),
            SourceDocument(title="Doc Beta", url="https://example.com/beta", snippet="..."),
        ],
        final_answer="According to recent studies [1], alpha is verified. Beta is also noted.",
    )
    cov = compute_citation_coverage(state)
    assert cov == 1.0  # Both [1] and Beta are cited


def test_run_benchmark_metrics() -> None:
    def dummy_runner(q: str) -> ResearchState:
        st = ResearchState(
            request=ResearchQuery(query=q),
            sources=[SourceDocument(title="Source 1", snippet="Snippet 1")],
            final_answer=(
                "Comprehensive research finding [1] with detailed sections.\n"
                "### References\n"
                "[1] Source 1"
            ),
        )
        st.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=st.final_answer,
                metadata={"cost_usd": 0.0005},
            )
        )
        return st

    state, metrics = run_benchmark("test_run", "Test query for benchmark", dummy_runner)
    assert metrics.run_name == "test_run"
    assert metrics.latency_seconds >= 0.0
    assert metrics.estimated_cost_usd == 0.0005
    assert metrics.citation_coverage == 1.0
    assert metrics.quality_score is not None and metrics.quality_score >= 8.0
    assert metrics.failure_rate == 0.0
