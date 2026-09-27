"""Temporarily deprioritize provider/model targets whose quota is exhausted."""

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.json_types import JsonObject, JsonValue


def is_quota_failure(failure: ExecutionFailure) -> bool:
    """Return whether a failure means the target is out of rate limit or credits."""
    return failure.kind is FailureKind.RATE_LIMIT or (
        failure.kind is FailureKind.PERMISSION and failure.status_code == 402
    )


@dataclass(frozen=True, slots=True)
class _Cooldown:
    until: float
    failure_kind: str
    status_code: int


class ModelCooldowns:
    """Track provider/model refs that recently hit a quota limit."""

    def __init__(self, *, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._cooldowns: dict[str, _Cooldown] = {}

    def mark(
        self, provider_model_ref: str, failure: ExecutionFailure, *, seconds: float
    ) -> None:
        """Cool down ``provider_model_ref`` for ``seconds`` after a quota failure."""
        if seconds <= 0:
            return
        self._cooldowns[provider_model_ref] = _Cooldown(
            until=self._clock() + seconds,
            failure_kind=failure.kind.value,
            status_code=failure.status_code,
        )

    def clear(self, provider_model_ref: str) -> None:
        """Forget any cooldown after ``provider_model_ref`` served a request."""
        self._cooldowns.pop(provider_model_ref, None)

    def is_cooling(self, provider_model_ref: str) -> bool:
        cooldown = self._cooldowns.get(provider_model_ref)
        if cooldown is None:
            return False
        if cooldown.until <= self._clock():
            del self._cooldowns[provider_model_ref]
            return False
        return True

    def order[T](self, targets: Sequence[T], ref: Callable[[T], str]) -> list[T]:
        """Move cooling targets after available ones, preserving relative order.

        Cooling targets are never dropped, so a request still runs when every
        configured model is cooling down.
        """
        available: list[T] = []
        cooling: list[T] = []
        for target in targets:
            (cooling if self.is_cooling(ref(target)) else available).append(target)
        return available + cooling

    def snapshot(self) -> list[JsonValue]:
        """Return active cooldowns as JSON-safe rows."""
        now = self._clock()
        rows: list[JsonValue] = []
        for model_ref, cooldown in sorted(self._cooldowns.items()):
            if cooldown.until <= now:
                continue
            row: JsonObject = {
                "model": model_ref,
                "until": cooldown.until,
                "failure_kind": cooldown.failure_kind,
                "status_code": cooldown.status_code,
            }
            rows.append(row)
        return rows
