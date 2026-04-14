from __future__ import annotations

import sys
import time


class ProgressBar:
    """Lightweight terminal progress bar (no external dependencies)."""

    def __init__(self, total: int, prefix: str = "", width: int = 28, file=None):
        self.total = max(int(total), 1)
        self.prefix = prefix
        self.width = int(width)
        self.file = file or sys.stderr
        self.current = 0
        self._start = time.perf_counter()
        self._last_print = 0.0
        self._print_interval = 0.15

    def update(self, n: int = 1, suffix: str = ""):
        self.current = min(self.current + int(n), self.total)
        now = time.perf_counter()
        if self.current < self.total and now - self._last_print < self._print_interval:
            return
        self._last_print = now
        self._render(suffix)

    def set(self, value: int, suffix: str = ""):
        self.current = min(int(value), self.total)
        now = time.perf_counter()
        if self.current < self.total and now - self._last_print < self._print_interval:
            return
        self._last_print = now
        self._render(suffix)

    def _render(self, suffix: str = ""):
        frac = self.current / self.total
        filled = int(self.width * frac)
        bar = "█" * filled + "░" * (self.width - filled)
        elapsed = time.perf_counter() - self._start
        pct = frac * 100.0
        parts = [f"\r{self.prefix} |{bar}| {pct:5.1f}% [{self.current}/{self.total}]"]
        if elapsed >= 0.5:
            parts.append(f" {elapsed:.1f}s")
            if self.current > 0 and self.current < self.total:
                eta = elapsed / self.current * (self.total - self.current)
                parts.append(f" ETA {eta:.1f}s")
        if suffix:
            parts.append(f" {suffix}")
        line = "".join(parts)
        self.file.write(line + "  ")
        self.file.flush()

    def close(self, final_suffix: str = "", *, suffix: str = ""):
        self.current = self.total
        self._render(final_suffix or suffix)
        self.file.write("\n")
        self.file.flush()


def stage_print(msg: str, file=None):
    f = file or sys.stderr
    f.write(f"[vcc-iris] {msg}\n")
    f.flush()
