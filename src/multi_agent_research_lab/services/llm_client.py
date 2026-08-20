import logging
from dataclasses import dataclass
from typing import Any

import httpx
from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError, LabError

logger = logging.getLogger(__name__)

# Approximate pricing per 1M tokens (USD)
_MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    "o1-mini": (3.00, 12.00),
    "o1-preview": (15.00, 60.00),
    "claude-3.5-sonnet": (3.00, 15.00),
    "claude-3-haiku": (0.25, 1.25),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-1.5-flash": (0.075, 0.30),
    "deepseek-chat": (0.14, 0.28),
    "deepseek-r1": (0.55, 2.19),
    "llama-3.3-70b-instruct": (0.40, 0.40),
}


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """Provider-agnostic LLM client using OpenAI SDK compatible with OpenAI and OpenRouter."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        client: Any | None = None,
    ) -> None:
        settings = get_settings()
        self.api_key = api_key or settings.openrouter_api_key or settings.openai_api_key

        if base_url:
            self.base_url: str | None = base_url
        elif settings.openai_base_url:
            self.base_url = settings.openai_base_url
        elif settings.openrouter_api_key or (self.api_key and self.api_key.startswith("sk-or-")):
            self.base_url = "https://openrouter.ai/api/v1"
        else:
            self.base_url = None

        self.model = model or settings.openai_model
        self.timeout = timeout or float(settings.timeout_seconds)

        headers: dict[str, str] = {}
        if self.base_url and "openrouter.ai" in self.base_url:
            headers = {
                "HTTP-Referer": "https://github.com/VinUni-AI/multi-agent-research-lab",
                "X-Title": "Multi-Agent Research Lab",
            }
        self.default_headers = headers if headers else None

        if client is not None:
            self._client = client
        elif self.api_key:
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
                default_headers=self.default_headers,
            )
        else:
            self._client = None

    def _estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        # Normalize model identifier (e.g., openai/gpt-4o-mini -> gpt-4o-mini)
        clean_model = self.model.split("/")[-1].lower()
        input_price_per_m, output_price_per_m = _MODEL_PRICING.get(
            clean_model, _MODEL_PRICING["gpt-4o-mini"]
        )
        cost = (input_tokens / 1_000_000.0) * input_price_per_m + (
            output_tokens / 1_000_000.0
        ) * output_price_per_m
        return round(cost, 6)

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion with retry and token metrics."""
        if not self._client:
            if not self.api_key:
                raise LabError(
                    "API key is not set. Please set OPENAI_API_KEY or OPENROUTER_API_KEY in .env."
                )
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
                default_headers=self.default_headers,
            )

        @retry(
            reraise=True,
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
        )
        def _call_api() -> LLMResponse:
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    timeout=self.timeout,
                )
                content = response.choices[0].message.content or ""
                input_tokens = response.usage.prompt_tokens if response.usage else None
                output_tokens = response.usage.completion_tokens if response.usage else None
                cost_usd = (
                    self._estimate_cost(input_tokens, output_tokens)
                    if (input_tokens is not None and output_tokens is not None)
                    else None
                )
                return LLMResponse(
                    content=content,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost_usd,
                )
            except Exception as exc:
                logger.error("LLM call failed: %s", exc)
                raise AgentExecutionError(f"LLM call failed: {exc}") from exc

        return _call_api()
