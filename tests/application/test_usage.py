"""Token usage observer and ledger contracts."""

import json

from free_claude_code.application.usage import StreamUsageObserver, UsageLedger
from free_claude_code.core.failures import ExecutionFailure, FailureKind


def _event(name: str, payload: dict[str, object]) -> str:
    return f"event: {name}\ndata: {json.dumps(payload)}\n\n"


def _rate_limit() -> ExecutionFailure:
    return ExecutionFailure(
        kind=FailureKind.RATE_LIMIT,
        status_code=429,
        message="quota exhausted",
        retryable=True,
    )


def test_observer_reads_anthropic_start_and_cumulative_delta_usage() -> None:
    observer = StreamUsageObserver()
    observer.feed(
        _event(
            "message_start",
            {"type": "message_start", "message": {"usage": {"input_tokens": 12}}},
        )
    )
    observer.feed(
        _event(
            "message_delta", {"type": "message_delta", "usage": {"output_tokens": 5}}
        )
    )
    observer.feed(
        _event(
            "message_delta",
            {
                "type": "message_delta",
                "usage": {"input_tokens": 15, "output_tokens": 9},
            },
        )
    )

    assert (observer.input_tokens, observer.output_tokens) == (15, 9)


def test_observer_reads_responses_terminal_usage() -> None:
    observer = StreamUsageObserver()
    observer.feed(
        _event(
            "response.created",
            {"type": "response.created", "response": {"usage": None}},
        )
    )
    observer.feed(
        _event(
            "response.completed",
            {
                "type": "response.completed",
                "response": {"usage": {"input_tokens": 40, "output_tokens": 7}},
            },
        )
    )

    assert (observer.input_tokens, observer.output_tokens) == (40, 7)


def test_observer_handles_events_split_across_chunks_and_crlf() -> None:
    raw = _event(
        "message_delta",
        {"type": "message_delta", "usage": {"input_tokens": 3, "output_tokens": 4}},
    ).replace("\n", "\r\n")
    observer = StreamUsageObserver()
    for index in range(0, len(raw), 7):
        observer.feed(raw[index : index + 7])

    assert (observer.input_tokens, observer.output_tokens) == (3, 4)


def test_observer_ignores_malformed_and_unrelated_events() -> None:
    observer = StreamUsageObserver()
    observer.feed("event: ping\ndata: not-json\n\n")
    observer.feed(": comment\n\n")
    observer.feed('data: ["list"]\n\n')
    observer.feed(_event("message_delta", {"type": "message_delta", "usage": "x"}))
    observer.feed(
        _event(
            "message_delta",
            {"type": "message_delta", "usage": {"output_tokens": True}},
        )
    )

    assert (observer.input_tokens, observer.output_tokens) == (0, 0)


def test_ledger_accumulates_per_model_totals_and_fallbacks() -> None:
    now = [100.0]
    ledger = UsageLedger(clock=lambda: now[0])
    ledger.record_failure("nvidia_nim/a", _rate_limit())
    ledger.record_fallback(
        request_id="req_1",
        from_model="nvidia_nim/a",
        to_model="groq/b",
        failure=_rate_limit(),
    )
    now[0] = 101.0
    ledger.record_usage("groq/b", input_tokens=10, output_tokens=5)
    ledger.record_usage("groq/b", input_tokens=2, output_tokens=1)
    ledger.record_usage("nvidia_nim/a", input_tokens=1, output_tokens=1)

    snapshot = ledger.snapshot()

    assert snapshot["since"] == 100.0
    assert snapshot["totals"] == {
        "requests": 3,
        "failures": 1,
        "fallbacks": 1,
        "input_tokens": 13,
        "output_tokens": 7,
        "total_tokens": 20,
    }
    assert snapshot["models"] == [
        {
            "model": "groq/b",
            "requests": 2,
            "failures": 0,
            "input_tokens": 12,
            "output_tokens": 6,
            "total_tokens": 18,
            "last_used_at": 101.0,
            "last_failure_kind": None,
        },
        {
            "model": "nvidia_nim/a",
            "requests": 1,
            "failures": 1,
            "input_tokens": 1,
            "output_tokens": 1,
            "total_tokens": 2,
            "last_used_at": 101.0,
            "last_failure_kind": "rate_limit",
        },
    ]
    assert snapshot["recent_fallbacks"] == [
        {
            "at": 100.0,
            "request_id": "req_1",
            "from_model": "nvidia_nim/a",
            "to_model": "groq/b",
            "failure_kind": "rate_limit",
            "status_code": 429,
        }
    ]


def test_ledger_keeps_bounded_recent_fallbacks_newest_first() -> None:
    ledger = UsageLedger(clock=lambda: 0.0)
    for index in range(25):
        ledger.record_fallback(
            request_id=f"req_{index}",
            from_model="a/x",
            to_model="b/y",
            failure=_rate_limit(),
        )

    snapshot = ledger.snapshot()
    recent = snapshot["recent_fallbacks"]

    assert isinstance(recent, list)
    assert len(recent) == 20
    assert recent[0] == {
        "at": 0.0,
        "request_id": "req_24",
        "from_model": "a/x",
        "to_model": "b/y",
        "failure_kind": "rate_limit",
        "status_code": 429,
    }
    assert snapshot["totals"] == {
        "requests": 0,
        "failures": 0,
        "fallbacks": 25,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }


def test_ledger_reset_clears_counters_and_restarts_window() -> None:
    now = [5.0]
    ledger = UsageLedger(clock=lambda: now[0])
    ledger.record_usage("a/x", input_tokens=1, output_tokens=1)
    ledger.record_fallback(
        request_id="r", from_model="a/x", to_model="b/y", failure=_rate_limit()
    )
    now[0] = 9.0

    ledger.reset()
    snapshot = ledger.snapshot()

    assert snapshot["since"] == 9.0
    assert snapshot["models"] == []
    assert snapshot["recent_fallbacks"] == []
    assert snapshot["totals"] == {
        "requests": 0,
        "failures": 0,
        "fallbacks": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
