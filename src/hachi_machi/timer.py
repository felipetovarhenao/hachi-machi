import time


class Timer:
    def __init__(self):
        self._start: float | None = None
        self._elapsed: float = 0.0

    def start(self):
        if self._start is None:
            self._start = time.perf_counter()

    def stop(self):
        if self._start is not None:
            self._elapsed += time.perf_counter() - self._start
            self._start = None

    def reset(self):
        self._start = None
        self._elapsed = 0.0

    @property
    def elapsed(self) -> float:
        if self._start is not None:
            return self._elapsed + (time.perf_counter() - self._start)
        return self._elapsed

    @property
    def is_running(self) -> bool:
        return self._start is not None

    def __str__(self) -> str:
        total_seconds = int(self.elapsed)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"[{hours:02d}:{minutes:02d}:{seconds:02d}]"

    def __repr__(self) -> str:
        return f"Timer(elapsed={self.elapsed:.3f}s, running={self.is_running})"
