"""In-process token usage accounting per provider/model target."""

import json
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.core.json_types import JsonObject, JsonValue

_MAX_RECENT_FALLBACKS = 20
# Terminal Responses events carry the final ``response.usage`` object.
_RESPONSES_TERMINAL_EVENTS = frozenset(
    {"response.completed", "response.incomplete", "response.failed"}
)


class StreamUsageObserver:
    """Read final token usage from Anthropic Messages or OpenAI Responses SSE."""

    def __init__(self) -> None:
        self._pending: list[str] = []
        self.input_tokens = 0
        self.output_tokens = 0

    def feed(self, chunk: str) -> None:
        self._pending.append(chunk.replace("\r\n", "\n"))
        text = "".join(self._pending)
        *events, tail = text.split("\n\n")
        self._pending = [tail] if tail else []
        for event in events:
            self._handle_event(event)

    def _handle_event(self, event: str) -> None:
        data_lines = [
            line[5:].lstrip() for line in event.split("\n") if line.startswith("data:")
        ]
        if not data_lines:
            return
        try:
            payload = json.loads("\n".join(data_lines))
        except ValueError:
            return
        if not isinstance(payload, dict):
            return
        payload_type = payload.get("type")
        if payload_type == "message_start":
            message = payload.get("message")
            if isinstance(message, dict):
                self._apply(message.get("usage"))
        elif payload_type == "message_delta":
            self._apply(payload.get("usage"))
        elif payload_type in _RESPONSES_TERMINAL_EVENTS:
            response = payload.get("response")
            if isinstance(response, dict):
                self._apply(response.get("usage"))

    def _apply(self, usage: object) -> None:
        if not isinstance(usage, dict):
            return
        input_tokens = _token_count(usage.get("input_tokens"))
        if input_tokens:
            self.input_tokens = input_tokens
        output_tokens = _token_count(usage.get("output_tokens"))
        if output_tokens is not None and output_tokens >= self.output_tokens:
            self.output_tokens = output_tokens


def _token_count(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


@dataclass(slots=True)
class _ModelUsage:
    requests: int = 0
    failures: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    last_used_at: float | None = None
    last_failure_kind: str | None = None


@dataclass(frozen=True, slots=True)
class _FallbackEvent:
    at: float
    request_id: str
    from_model: str
    to_model: str
    failure_kind: str
    status_code: int


class UsageLedger:
    """Accumulate token consumption, failures, and fallback switches in memory."""

    def __init__(self, *, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._started_at = clock()
        self._models: dict[str, _ModelUsage] = {}
        self._fallbacks: deque[_FallbackEvent] = deque(maxlen=_MAX_RECENT_FALLBACKS)
        self._fallback_count = 0

    def record_usage(
        self, provider_model_ref: str, *, input_tokens: int, output_tokens: int
    ) -> None:
        """Record one request served by ``provider_model_ref``."""
        entry = self._entry(provider_model_ref)
        entry.requests += 1
        entry.input_tokens += input_tokens
        entry.output_tokens += output_tokens
        entry.last_used_at = self._clock()

    def record_failure(
        self, provider_model_ref: str, failure: ExecutionFailure
    ) -> None:
        """Record one attempt that failed before producing output."""
        entry = self._entry(provider_model_ref)
        entry.failures += 1
        entry.last_failure_kind = failure.kind.value

    def record_fallback(
        self,
        *,
        request_id: str,
        from_model: str,
        to_model: str,
        failure: ExecutionFailure,
    ) -> None:
        """Record an automatic switch to the next configured fallback model."""
        self._fallback_count += 1
        self._fallbacks.append(
            _FallbackEvent(
                at=self._clock(),
                request_id=request_id,
                from_model=from_model,
                to_model=to_model,
                failure_kind=failure.kind.value,
                status_code=failure.status_code,
            )
        )

    def reset(self) -> None:
        """Clear all counters and restart the accounting window."""
        self._models.clear()
        self._fallbacks.clear()
        self._fallback_count = 0
        self._started_at = self._clock()

    def snapshot(self) -> JsonObject:
        """Return a JSON-safe view of the current accounting window."""
        models: list[JsonValue] = [
            {
                "model": model_ref,
                "requests": entry.requests,
                "failures": entry.failures,
                "input_tokens": entry.input_tokens,
                "output_tokens": entry.output_tokens,
                "total_tokens": entry.input_tokens + entry.output_tokens,
                "last_used_at": entry.last_used_at,
                "last_failure_kind": entry.last_failure_kind,
            }
            for model_ref, entry in sorted(
                self._models.items(),
                key=lambda item: item[1].input_tokens + item[1].output_tokens,
                reverse=True,
            )
        ]
        input_tokens = sum(entry.input_tokens for entry in self._models.values())
        output_tokens = sum(entry.output_tokens for entry in self._models.values())
        recent: list[JsonValue] = [
            {
                "at": event.at,
                "request_id": event.request_id,
                "from_model": event.from_model,
                "to_model": event.to_model,
                "failure_kind": event.failure_kind,
                "status_code": event.status_code,
            }
            for event in reversed(self._fallbacks)
        ]
        return {
            "since": self._started_at,
            "totals": {
                "requests": sum(entry.requests for entry in self._models.values()),
                "failures": sum(entry.failures for entry in self._models.values()),
                "fallbacks": self._fallback_count,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
            "models": models,
            "recent_fallbacks": recent,
        }

    def _entry(self, provider_model_ref: str) -> _ModelUsage:
        entry = self._models.get(provider_model_ref)
        if entry is None:
            entry = _ModelUsage()
            self._models[provider_model_ref] = entry
        return entry
