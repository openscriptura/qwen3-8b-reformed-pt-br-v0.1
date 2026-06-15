"""Tests for the hard-stop cost guardrail (VALIDATION_REPORT.md M11)."""

import pytest

from utils.cost_tracker import CostLimitExceeded, CostTracker


def test_accumulates_under_limit():
    tracker = CostTracker(limit_usd=1.00)
    assert tracker.add(0.40) == pytest.approx(0.40)
    assert tracker.add(0.30) == pytest.approx(0.70)
    assert tracker.total == pytest.approx(0.70)


def test_raises_when_limit_crossed():
    tracker = CostTracker(limit_usd=0.50)
    tracker.add(0.49)
    with pytest.raises(CostLimitExceeded):
        tracker.add(0.02)


def test_exactly_at_limit_does_not_raise():
    tracker = CostTracker(limit_usd=0.50)
    # add() raises only when total strictly exceeds the limit.
    assert tracker.add(0.50) == pytest.approx(0.50)
