"""OpenRouter API client with tenacity retry and structured logging.

Per VALIDATION_REPORT.md M10/M11: all API calls must use exponential backoff
retry (tenacity) and accumulate cost toward a hard-stop limit.
"""

import json
import logging
from pathlib import Path

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

_RETRY_POLICY = dict(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPStatusError)),
    reraise=True,
)


@retry(**_RETRY_POLICY)
async def _post_json(
    client: httpx.AsyncClient,
    url: str,
    payload: dict,
    headers: dict,
) -> dict:
    """Single HTTP POST with retry. Decorated at module level so tenacity can
    instrument it correctly for async functions."""
    response = await client.post(url, json=payload, headers=headers, timeout=60.0)
    response.raise_for_status()
    return response.json()


class OpenRouterClient:
    """Thin async wrapper around the OpenRouter chat completions endpoint.

    Handles:
    - Authorization headers
    - Qwen3 enable_thinking=False parameter (M2)
    - Tenacity retry via _post_json (M10)
    - Raw response logging to logs/raw/ (R5)
    - Approximate cost estimation from usage tokens
    """

    # Approximate token prices (USD per token) for cost estimation.
    # OpenRouter does not always return cost in the response body.
    _PRICE_PER_TOKEN: dict[str, float] = {
        "qwen/qwen3-8b": 1e-7,              # ~$0.10/1M tokens
        "deepseek/deepseek-v4-flash": 5e-7,  # ~$0.50/1M tokens (flash)
        "default": 5e-7,
    }

    def __init__(
        self,
        api_key: str,
        base_url: str,
        log_raw_dir: Path | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.log_raw_dir = log_raw_dir
        if log_raw_dir:
            log_raw_dir.mkdir(parents=True, exist_ok=True)

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/openscriptura/openscriptura",
            "X-Title": "OpenScriptura",
        }

    async def chat(
        self,
        client: httpx.AsyncClient,
        model: str,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int = 42,
        enable_thinking: bool = False,
        log_key: str | None = None,
    ) -> dict:
        """Call chat completions.  Returns the raw OpenRouter response dict."""
        payload: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
        }
        # Qwen3: disable chain-of-thought thinking tokens (M2).
        # Passed directly in the request body per OpenRouter's model-specific
        # parameter convention.
        if not enable_thinking:
            payload["enable_thinking"] = False

        result = await _post_json(
            client=client,
            url=f"{self.base_url}/chat/completions",
            payload=payload,
            headers=self._headers(),
        )

        if self.log_raw_dir and log_key:
            raw_path = self.log_raw_dir / f"{log_key}.json"
            raw_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.debug("Raw response logged to %s", raw_path)

        return result

    @staticmethod
    def extract_text(response: dict) -> str:
        """Extract assistant message content, stripping Qwen3 <think> tags."""
        content: str = response["choices"][0]["message"]["content"]
        if "<think>" in content and "</think>" in content:
            content = content.split("</think>", 1)[-1].strip()
        return content

    def estimate_cost_usd(self, response: dict, model: str) -> float:
        """Approximate cost in USD from usage token counts."""
        usage = response.get("usage", {})
        total_tokens = usage.get("total_tokens", 0)
        price = self._PRICE_PER_TOKEN.get(model, self._PRICE_PER_TOKEN["default"])
        return total_tokens * price
