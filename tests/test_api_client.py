"""Tests for utils.api_client — defensive extraction, single-judge contract, pricing.

Covers the changes from the single-flash-judge work:
  - chat() no longer exposes a cross-model fallback (comparability lock)
  - extract_text() is defensive vs empty choices / null content (-> "" parse error)
  - _price_for() normalizes dated/snapshot model ids to the right tier
"""

import inspect

from utils.api_client import OpenRouterClient


def _client():
    return OpenRouterClient(api_key="k", base_url="https://example.invalid")


# ---------------------------------------------------------------------------
# single-judge contract: no cross-model fallback
# ---------------------------------------------------------------------------

def test_chat_has_no_fallback_param():
    # A cross-model fallback would make the judge a non-deterministic mixture and
    # break HARD RULE #3 — the param must not exist.
    assert "fallback_model" not in inspect.signature(OpenRouterClient.chat).parameters


# ---------------------------------------------------------------------------
# extract_text — defensive
# ---------------------------------------------------------------------------

def test_extract_text_valid():
    resp = {"choices": [{"message": {"content": "Rating: 4"}}]}
    assert OpenRouterClient.extract_text(resp) == "Rating: 4"


def test_extract_text_empty_choices_returns_empty():
    assert OpenRouterClient.extract_text({"choices": []}) == ""
    assert OpenRouterClient.extract_text({}) == ""


def test_extract_text_null_content_returns_empty():
    assert OpenRouterClient.extract_text({"choices": [{"message": {"content": None}}]}) == ""


def test_extract_text_strips_think_block():
    resp = {"choices": [{"message": {"content": "<think>hmm</think>Rating: 2"}}]}
    assert OpenRouterClient.extract_text(resp) == "Rating: 2"


# ---------------------------------------------------------------------------
# actual_model + pricing
# ---------------------------------------------------------------------------

def test_actual_model_reads_response_else_requested():
    assert OpenRouterClient.actual_model({"model": "x-2026"}, "x") == "x-2026"
    assert OpenRouterClient.actual_model({}, "fallback") == "fallback"


def test_price_for_exact_keys():
    c = _client()
    assert c._price_for("qwen/qwen3-8b") == 1e-7
    assert c._price_for("deepseek/deepseek-v4-flash") == 5e-7
    assert c._price_for("deepseek/deepseek-v4-pro") == 7e-7


def test_price_for_dated_snapshot_ids():
    c = _client()
    # OpenRouter returns a dated/provider-suffixed id; it must still resolve.
    assert c._price_for("deepseek/deepseek-v4-flash-20260423") == 5e-7
    assert c._price_for("deepseek/deepseek-v4-pro-20260423") == 7e-7
    assert c._price_for("qwen/qwen3-8b-04-28") == 1e-7


def test_price_for_unknown_falls_to_default():
    assert _client()._price_for("some/unknown-model") == 5e-7


def test_estimate_cost_uses_served_model_price():
    c = _client()
    resp = {"model": "deepseek/deepseek-v4-pro-20260423", "usage": {"total_tokens": 1000}}
    assert abs(c.estimate_cost_usd(resp, "deepseek/deepseek-v4-pro") - 1000 * 7e-7) < 1e-12
