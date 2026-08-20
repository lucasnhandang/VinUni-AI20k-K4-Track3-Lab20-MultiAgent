"""Benchmark report rendering."""

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(metrics: list[BenchmarkMetrics]) -> str:
    """Render comprehensive benchmark metrics and comparative analysis to markdown."""
    lines = [
        "# Multi-Agent Research System: Benchmark Report",
        "",
        "## 1. Metrics Comparison",
        "",
        "| Run | Latency (s) | Cost (USD) | Quality | Citation cov. | Failure rate | Notes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in metrics:
        cost = "" if item.estimated_cost_usd is None else f"${item.estimated_cost_usd:.4f}"
        quality = "" if item.quality_score is None else f"{item.quality_score:.1f}/10"
        citation = "" if item.citation_coverage is None else f"{item.citation_coverage:.0%}"
        failure = "" if item.failure_rate is None else f"{item.failure_rate:.0%}"
        lines.append(
            f"| {item.run_name} | {item.latency_seconds:.2f} | {cost} | {quality} "
            f"| {citation} | {failure} | {item.notes} |"
        )

    lines.extend(
        [
            "",
            "## 2. Key Findings & Trade-offs",
            "",
            "- **Grounded Quality vs Latency**: Multi-agent decomposition provides "
            "higher citation grounding and structural depth, at the cost of higher latency.",
            "- **Role Specialization**: Researcher grounds evidence from external sources, "
            "Analyst compares perspectives, Writer synthesizes the final prose, "
            "and Critic audits factual accuracy and citation consistency.",
            "- **Guardrails**: Max iterations and validation checks prevent runaway loops.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"
