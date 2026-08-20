import re
from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

Runner = Callable[[str], ResearchState]


def compute_citation_coverage(state: ResearchState) -> float:
    """Calculate the ratio of retrieved sources cited in the final answer."""
    if not state.sources or not state.final_answer:
        return 0.0

    answer = state.final_answer.lower()
    cited_count = 0

    for idx, doc in enumerate(state.sources, start=1):
        idx_pattern = rf"\[{idx}\]"
        has_idx = bool(re.search(idx_pattern, answer))
        has_title = bool(doc.title and doc.title.lower() in answer)

        # Match significant title words
        title_words = (
            [
                w.lower()
                for w in doc.title.split()
                if len(w) > 3 and w.lower() not in ("from", "with", "that", "this", "about")
            ]
            if doc.title
            else []
        )
        has_title_words = any(w in answer for w in title_words) if title_words else False

        has_url = False
        if doc.url:
            url_lower = doc.url.lower()
            url_slug = url_lower.rstrip("/").split("/")[-1]
            has_url = url_lower in answer or (len(url_slug) > 3 and url_slug in answer)

        if has_idx or has_title or has_title_words or has_url:
            cited_count += 1

    return round(cited_count / len(state.sources), 2)


def run_benchmark(
    run_name: str, query: str, runner: Runner
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Execute runner and evaluate comprehensive latency, cost, citation, and quality metrics."""
    started = perf_counter()
    try:
        state = runner(query)
        failure_rate = 1.0 if (state.errors or not state.final_answer) else 0.0
    except Exception as exc:
        latency = perf_counter() - started
        metrics = BenchmarkMetrics(
            run_name=run_name,
            latency_seconds=round(latency, 3),
            failure_rate=1.0,
            notes=f"Failed with exception: {exc}",
        )
        raise exc

    latency = perf_counter() - started

    # Aggregate token costs from agent execution results
    total_cost: float = 0.0
    for res in state.agent_results:
        cost = res.metadata.get("cost_usd")
        if cost and isinstance(cost, (int, float)):
            total_cost += float(cost)

    citation_cov = compute_citation_coverage(state)

    # Calculate heuristic quality score (0 - 10)
    quality = 0.0
    if state.final_answer and not state.errors:
        quality += 4.0  # Base completion
        if len(state.final_answer) > 300:
            quality += 2.0  # Depth
        if citation_cov >= 0.5:
            quality += 2.0  # Citation grounding
        if "references" in state.final_answer.lower() or "### " in state.final_answer:
            quality += 2.0  # Structure & formatting

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=round(latency, 3),
        estimated_cost_usd=round(total_cost, 6) if total_cost > 0 else None,
        quality_score=round(quality, 1),
        citation_coverage=citation_cov,
        failure_rate=failure_rate,
        notes=f"Iterations: {state.iteration}, Sources: {len(state.sources)}",
    )
    return state, metrics
