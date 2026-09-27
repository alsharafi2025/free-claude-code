"""Quota cooldown contracts."""

import pytest

from free_claude_code.application.cooldown import ModelCooldowns, is_quota_failure
from free_claude_code.core.failures import ExecutionFailure, FailureKind


def _failure(kind: FailureKind, status_code: int) -> ExecutionFailure:
    return ExecutionFailure(
        kind=kind, status_code=status_code, message="x", retryable=False
    )


@pytest.mark.parametrize(
    ("kind", "status_code", "expected"),
    (
        (FailureKind.RATE_LIMIT, 429, True),
        (FailureKind.PERMISSION, 402, True),
        (FailureKind.PERMISSION, 403, False),
        (FailureKind.OVERLOADED, 529, False),
        (FailureKind.AUTHENTICATION, 401, False),
        (FailureKind.UPSTREAM, 500, False),
    ),
)
def test_quota_failures_are_rate_limits_and_billing(
    kind: FailureKind, status_code: int, expected: bool
) -> None:
    assert is_quota_failure(_failure(kind, status_code)) is expected


def test_mark_expires_after_seconds() -> None:
    now = [0.0]
    cooldowns = ModelCooldowns(clock=lambda: now[0])
    cooldowns.mark("a/x", _failure(FailureKind.RATE_LIMIT, 429), seconds=30)

    now[0] = 29.9
    assert cooldowns.is_cooling("a/x")
    now[0] = 30.0
    assert not cooldowns.is_cooling("a/x")
    assert cooldowns.snapshot() == []


def test_zero_seconds_disables_cooldown() -> None:
    cooldowns = ModelCooldowns(clock=lambda: 0.0)
    cooldowns.mark("a/x", _failure(FailureKind.RATE_LIMIT, 429), seconds=0)

    assert not cooldowns.is_cooling("a/x")


def test_clear_removes_cooldown() -> None:
    cooldowns = ModelCooldowns(clock=lambda: 0.0)
    cooldowns.mark("a/x", _failure(FailureKind.RATE_LIMIT, 429), seconds=60)
    cooldowns.clear("a/x")
    cooldowns.clear("missing/model")

    assert not cooldowns.is_cooling("a/x")


def test_order_moves_cooling_targets_last_without_dropping_them() -> None:
    cooldowns = ModelCooldowns(clock=lambda: 0.0)
    cooldowns.mark("a/x", _failure(FailureKind.RATE_LIMIT, 429), seconds=60)
    cooldowns.mark("c/z", _failure(FailureKind.PERMISSION, 402), seconds=60)

    assert cooldowns.order(["a/x", "b/y", "c/z", "d/w"], str) == [
        "b/y",
        "d/w",
        "a/x",
        "c/z",
    ]
    assert cooldowns.order(["a/x", "c/z"], str) == ["a/x", "c/z"]


def test_snapshot_lists_active_cooldowns_sorted() -> None:
    cooldowns = ModelCooldowns(clock=lambda: 10.0)
    cooldowns.mark("z/m", _failure(FailureKind.PERMISSION, 402), seconds=5)
    cooldowns.mark("a/m", _failure(FailureKind.RATE_LIMIT, 429), seconds=60)

    assert cooldowns.snapshot() == [
        {
            "model": "a/m",
            "until": 70.0,
            "failure_kind": "rate_limit",
            "status_code": 429,
        },
        {
            "model": "z/m",
            "until": 15.0,
            "failure_kind": "permission",
            "status_code": 402,
        },
    ]
