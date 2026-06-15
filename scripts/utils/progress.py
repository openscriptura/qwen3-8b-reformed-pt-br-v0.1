"""progress.py — TTY-aware progress bar ported from pastor-ai/_progress.py."""

import sys
import time


class ProgressBar:
    """Unicode progress bar for TTY; ASCII fallback for log files.

    TTY:      [████████████░░░░░░░░░░░░░░░░░░] 40%  16/40 prompts processed
    Non-TTY:  [############------------------] 40%  16/40 prompts processed
    """

    def __init__(self, total: int, label: str = "", width: int = 30) -> None:
        self.total = total
        self.label = label
        self.width = width
        self.current = 0
        self._last_render = 0.0

    def update(self, n: int = 1) -> None:
        self.current += n
        now = time.monotonic()
        # Throttle to max 10 redraws/second; always render on completion.
        if now - self._last_render < 0.1 and self.current < self.total:
            return
        self._last_render = now
        self._render()

    def _render(self) -> None:
        pct = self.current / self.total if self.total else 0
        filled = int(pct * self.width)
        empty = self.width - filled

        if sys.stdout.isatty():
            bar = "█" * filled + "░" * empty
            print(
                f"\r[{bar}] {pct*100:5.1f}%  {self.current}/{self.total} {self.label}",
                end="",
                flush=True,
            )
        else:
            bar = "#" * filled + "-" * empty
            print(
                f"[{bar}] {pct*100:5.1f}%  {self.current}/{self.total} {self.label}",
                flush=True,
            )

    def done(self) -> None:
        self.current = self.total
        self._render()
        if sys.stdout.isatty():
            print()  # move to next line after inline bar
