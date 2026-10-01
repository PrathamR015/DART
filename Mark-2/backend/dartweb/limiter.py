"""A small in-memory sliding-window rate limiter, one window per client key (per server instance)."""
import time
from collections import deque

WINDOW_SECONDS = 60.0
MAX_TRACKED_KEYS = 10_000


class RateLimiter:
    def __init__(self, per_minute, clock=time.monotonic):
        self.per_minute = per_minute
        self.clock = clock
        self.hits = {}

    def allow(self, key):
        """True if this call is within the limit (and records it). A limit of 0 or less disables limiting."""
        if self.per_minute <= 0:
            return True
        now = self.clock()
        window = self.hits.setdefault(key, deque())
        while window and now - window[0] >= WINDOW_SECONDS:
            window.popleft()
        if len(window) >= self.per_minute:
            return False
        window.append(now)
        self._prune(now)
        return True

    def _prune(self, now):
        if len(self.hits) <= MAX_TRACKED_KEYS:
            return
        self.hits = {key: window for key, window in self.hits.items() if window and now - window[-1] < WINDOW_SECONDS}
