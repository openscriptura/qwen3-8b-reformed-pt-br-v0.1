import threading


class CostLimitExceeded(Exception):
    """Raised when accumulated API spend exceeds the configured hard limit."""


class CostTracker:
    """Thread-safe accumulated cost tracker with a hard-stop limit.

    Per VALIDATION_REPORT.md M11: every script must enforce a cost guardrail
    that raises an exception rather than silently overspending.

    Usage:
        tracker = CostTracker(limit_usd=2.00)
        tracker.add(0.000182)   # raises CostLimitExceeded if total > limit
        print(tracker.total)    # current accumulated spend
    """

    def __init__(self, limit_usd: float) -> None:
        self.limit_usd = limit_usd
        self._accumulated = 0.0
        self._lock = threading.Lock()

    def add(self, cost_usd: float) -> float:
        with self._lock:
            self._accumulated += cost_usd
            if self._accumulated > self.limit_usd:
                raise CostLimitExceeded(
                    f"Accumulated spend ${self._accumulated:.4f} exceeds "
                    f"limit ${self.limit_usd:.2f}. "
                    "Run with --resume to continue from the last checkpoint."
                )
            return self._accumulated

    @property
    def total(self) -> float:
        with self._lock:
            return self._accumulated
