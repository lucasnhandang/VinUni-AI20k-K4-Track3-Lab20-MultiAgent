import json
import logging
from pathlib import Path
from typing import Any

import certifi
import httpx

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import SourceDocument

logger = logging.getLogger(__name__)


class SearchClient:
    """Search client supporting Tavily API with offline corpus fallback."""

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float | None = None,
        corpus_dir: Path | None = None,
    ) -> None:
        settings = get_settings()
        self.api_key = api_key or settings.tavily_api_key
        self.timeout = timeout or float(settings.timeout_seconds)
        self.corpus_dir = corpus_dir or (
            Path.cwd() / "ai_agent_offline_research_corpus_v2" / "topics"
        )

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents via Tavily API or fallback to local offline corpus."""
        if self.api_key:
            try:
                return self._search_tavily(query, max_results)
            except Exception as exc:
                logger.warning("Tavily search failed (%s), falling back to local corpus", exc)

        return self._search_local_corpus(query, max_results)

    def _search_tavily(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
        }
        with httpx.Client(verify=certifi.where(), timeout=self.timeout) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()

        results: list[SourceDocument] = []
        for item in data.get("results", []):
            results.append(
                SourceDocument(
                    title=item.get("title", "Untitled Document"),
                    url=item.get("url"),
                    snippet=item.get("content", ""),
                    metadata={"score": item.get("score")},
                )
            )
        return results[:max_results]

    def _search_local_corpus(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search through offline benchmark corpus files."""
        if not self.corpus_dir.exists():
            # Fallback mock document if corpus directory does not exist
            return [
                SourceDocument(
                    title=f"Overview on {query}",
                    url="https://offline-corpus.local/article-1",
                    snippet=(
                        f"Key concepts, architectural patterns, and trade-offs regarding {query}."
                    ),
                    metadata={"source": "mock_fallback"},
                )
            ]

        query_terms = [t.lower() for t in query.split() if len(t) > 2]
        scored_docs: list[tuple[int, SourceDocument]] = []

        for file_path in self.corpus_dir.glob("*.json"):
            try:
                with file_path.open("r", encoding="utf-8") as f:
                    topic_data: dict[str, Any] = json.load(f)
            except Exception:
                continue

            # Check embedded source documents
            for doc in topic_data.get("source_documents", []):
                title = doc.get("title", "")
                snippet = doc.get("snippet", "") or doc.get("summary", "")
                text_to_match = f"{title} {snippet}".lower()
                score = sum(1 for term in query_terms if term in text_to_match)
                if score > 0 or not scored_docs:
                    scored_docs.append(
                        (
                            score,
                            SourceDocument(
                                title=title or "Corpus Document",
                                url=doc.get("url"),
                                snippet=snippet,
                                metadata={
                                    "source_id": doc.get("source_id"),
                                    "is_synthetic": doc.get("is_synthetic", False),
                                },
                            ),
                        )
                    )

            # Check long-form articles
            for article in topic_data.get("articles", []):
                title = article.get("title", "")
                content = article.get("content", "") or article.get("summary", "")
                text_to_match = f"{title} {content}".lower()
                score = sum(1 for term in query_terms if term in text_to_match)
                if score > 0:
                    scored_docs.append(
                        (
                            score,
                            SourceDocument(
                                title=title,
                                url=f"offline://corpus/{article.get('article_id', 'article')}",
                                snippet=content[:400],
                                metadata={"article_id": article.get("article_id")},
                            ),
                        )
                    )

        # Sort by relevance score descending
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        results = [doc for _, doc in scored_docs]
        return (
            results[:max_results]
            if results
            else [
                SourceDocument(
                    title=f"Reference on {query}",
                    url="https://offline-corpus.local/reference",
                    snippet=(
                        f"Detailed background analysis and key design considerations for {query}."
                    ),
                    metadata={"source": "offline_default"},
                )
            ]
        )
