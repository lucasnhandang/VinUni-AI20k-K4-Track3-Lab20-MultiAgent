"""Command-line entrypoint for the Multi-Agent Research Lab."""

import os
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import AgentName, AgentResult, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.storage import LocalArtifactStore

app = typer.Typer(help="Multi-Agent Research Lab CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)

    if settings.langsmith_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
        if settings.langsmith_endpoint:
            os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith_endpoint
            os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
        elif "apac.smith" in (settings.langsmith_api_key or ""):
            os.environ["LANGCHAIN_ENDPOINT"] = "https://apac.api.smith.langchain.com"
            os.environ["LANGSMITH_ENDPOINT"] = "https://apac.api.smith.langchain.com"


def _parse_query(
    query: str, max_sources: int = 5, audience: str = "technical learners"
) -> ResearchQuery:
    try:
        return ResearchQuery(query=query, max_sources=max_sources, audience=audience)
    except ValidationError as exc:
        console.print(
            Panel.fit(
                f"Invalid query: {exc.errors()[0]['msg']}",
                title="Input Error",
                style="red",
            )
        )
        raise typer.Exit(code=1) from exc


def run_single_agent(query_text: str) -> ResearchState:
    """Baseline: Single LLM call without multi-agent decomposition."""
    request = _parse_query(query_text)
    state = ResearchState(request=request)
    llm = LLMClient()
    system_prompt = (
        "You are a general AI assistant. Write a concise, comprehensive research report "
        "answering the user's query."
    )
    user_prompt = f"Query: {query_text}\nTarget Audience: {state.request.audience}"
    resp = llm.complete(system_prompt, user_prompt)
    state.final_answer = resp.content
    state.agent_results.append(
        AgentResult(
            agent=AgentName.WRITER,
            content=resp.content,
            metadata={
                "input_tokens": resp.input_tokens,
                "output_tokens": resp.output_tokens,
                "cost_usd": resp.cost_usd,
            },
        )
    )
    state.record_route("single_agent")
    return state


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    audience: Annotated[
        str, typer.Option("--audience", "-a", help="Target audience")
    ] = "technical learners",
) -> None:
    """Run a single-agent LLM baseline."""
    _init()
    state = run_single_agent(query)
    console.print(
        Panel.fit(
            state.final_answer or "", title="Single-Agent Baseline Response", border_style="cyan"
        )
    )


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    max_sources: Annotated[
        int, typer.Option("--max-sources", "-s", help="Max sources to retrieve")
    ] = 5,
    audience: Annotated[
        str, typer.Option("--audience", "-a", help="Target audience")
    ] = "technical learners",
) -> None:
    """Run the Multi-Agent Research System (Supervisor + Researcher + Analyst + Writer + Critic)."""
    _init()
    request = _parse_query(query, max_sources=max_sources, audience=audience)
    state = ResearchState(request=request)
    workflow = MultiAgentWorkflow()

    result = workflow.run(state)

    # Route summary
    summary_text = (
        f"Execution Steps: {' -> '.join(result.route_history)}\n"
        f"Total Iterations: {result.iteration}\n"
        f"Sources Retrieved: {len(result.sources)}"
    )
    console.print(
        Panel.fit(
            summary_text,
            title="Workflow Routing Summary",
            border_style="green",
        )
    )

    # Final report
    console.print(
        Panel.fit(
            result.final_answer or "No answer produced.",
            title="Final Research Report (Multi-Agent)",
            border_style="bright_blue",
        )
    )

    # Critic feedback if present
    critic_results = [r for r in result.agent_results if r.agent == AgentName.CRITIC]
    if critic_results:
        console.print(
            Panel.fit(
                critic_results[-1].content,
                title="Critic Review & Fact-Check",
                border_style="magenta",
            )
        )


@app.command()
def benchmark(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    max_sources: Annotated[int, typer.Option("--max-sources", "-s", help="Max sources")] = 5,
    output_file: Annotated[
        str, typer.Option("--output", "-o", help="Report output filename")
    ] = "benchmark_report.md",
) -> None:
    """Run benchmark comparing Single-Agent vs Multi-Agent system."""
    _init()

    def _multi_agent_runner(q: str) -> ResearchState:
        req = _parse_query(q, max_sources=max_sources)
        st = ResearchState(request=req)
        return MultiAgentWorkflow().run(st)

    metrics_list = []

    console.print("[bold yellow]Running Single-Agent Baseline...[/bold yellow]")
    _, m_single = run_benchmark("Single-Agent", query, run_single_agent)
    metrics_list.append(m_single)

    console.print("[bold green]Running Multi-Agent System...[/bold green]")
    _, m_multi = run_benchmark("Multi-Agent (LangGraph)", query, _multi_agent_runner)
    metrics_list.append(m_multi)

    # Render console table
    table = Table(title="Benchmark Comparison")
    table.add_column("Run Name", style="bold")
    table.add_column("Latency (s)", justify="right")
    table.add_column("Cost (USD)", justify="right")
    table.add_column("Quality", justify="right")
    table.add_column("Citation Cov.", justify="right")
    table.add_column("Failure Rate", justify="right")

    for m in metrics_list:
        cost_str = f"${m.estimated_cost_usd:.4f}" if m.estimated_cost_usd is not None else "N/A"
        qual_str = f"{m.quality_score:.1f}/10" if m.quality_score is not None else "N/A"
        cit_str = f"{m.citation_coverage:.0%}" if m.citation_coverage is not None else "N/A"
        fail_str = f"{m.failure_rate:.0%}" if m.failure_rate is not None else "N/A"
        table.add_row(m.run_name, f"{m.latency_seconds:.2f}", cost_str, qual_str, cit_str, fail_str)

    console.print(table)

    # Save to markdown report
    md_content = render_markdown_report(metrics_list)
    store = LocalArtifactStore()
    saved_path = store.write_text(output_file, md_content)
    console.print(f"[bold green]Report written to:[/bold green] {saved_path}")


if __name__ == "__main__":
    app()
