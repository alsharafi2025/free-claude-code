"""Token usage accounting through the public API and local Admin API."""

from fastapi.testclient import TestClient

from free_claude_code.core.failures import ExecutionFailure, FailureKind
from tests.api.model_fallback_support import (
    ControlledFallbackProvider,
    fallback_client,
    messages_payload,
    responses_payload,
)
from tests.api.support import create_test_app


def _quota_exhausted() -> ExecutionFailure:
    return ExecutionFailure(
        kind=FailureKind.RATE_LIMIT,
        status_code=429,
        message="quota exhausted",
        retryable=True,
    )


def _admin_client(client: TestClient) -> TestClient:
    return TestClient(client.app, client=("127.0.0.1", 50000))


def test_quota_failure_falls_back_and_admin_reports_usage() -> None:
    primary = ControlledFallbackProvider(failure=_quota_exhausted())
    fallback = ControlledFallbackProvider(text="fallback worked")

    with fallback_client(primary, fallback) as client:
        response = client.post("/v1/messages", json=messages_payload(stream=True))
        usage = _admin_client(client).get("/admin/api/usage")

    assert response.status_code == 200
    assert "fallback worked" in response.text
    assert usage.status_code == 200
    assert usage.headers["cache-control"] == "no-store"
    body = usage.json()
    assert body["totals"] == {
        "requests": 1,
        "failures": 1,
        "fallbacks": 1,
        "input_tokens": 3,
        "output_tokens": 4,
        "total_tokens": 7,
    }
    models = {model["model"]: model for model in body["models"]}
    assert models["groq/fallback-model"]["requests"] == 1
    assert models["groq/fallback-model"]["total_tokens"] == 7
    assert models["nvidia_nim/primary-model"]["failures"] == 1
    assert models["nvidia_nim/primary-model"]["last_failure_kind"] == "rate_limit"
    [event] = body["recent_fallbacks"]
    assert event["from_model"] == "nvidia_nim/primary-model"
    assert event["to_model"] == "groq/fallback-model"
    assert (event["failure_kind"], event["status_code"]) == ("rate_limit", 429)


def test_responses_usage_is_recorded_for_serving_model() -> None:
    primary = ControlledFallbackProvider(text="primary worked")
    fallback = ControlledFallbackProvider(text="unused")

    with fallback_client(primary, fallback) as client:
        response = client.post("/v1/responses", json=responses_payload())
        body = _admin_client(client).get("/admin/api/usage").json()

    assert response.status_code == 200
    assert fallback.stream_models == []
    assert [model["model"] for model in body["models"]] == ["nvidia_nim/primary-model"]
    assert body["models"][0]["requests"] == 1
    assert body["totals"]["fallbacks"] == 0


def test_usage_reset_clears_counters() -> None:
    primary = ControlledFallbackProvider(text="ok")
    fallback = ControlledFallbackProvider(text="unused")

    with fallback_client(primary, fallback) as client:
        client.post("/v1/messages", json=messages_payload(stream=True))
        admin = _admin_client(client)
        reset = admin.post("/admin/api/usage/reset")
        after = admin.get("/admin/api/usage").json()

    assert reset.status_code == 200
    assert reset.json()["models"] == []
    assert after["totals"]["requests"] == 0


def test_usage_admin_api_is_local_only() -> None:
    app = create_test_app()
    client = TestClient(app, client=("203.0.113.5", 50000))

    assert client.get("/admin/api/usage").status_code == 403
    assert client.post("/admin/api/usage/reset").status_code == 403
