"""Rendered token usage and fallback history for the local Admin UI."""

from playwright.sync_api import Page, expect

from free_claude_code.application.cooldown import ModelCooldowns
from free_claude_code.application.usage import UsageLedger
from free_claude_code.core.failures import ExecutionFailure, FailureKind


def _open_usage(page: Page, admin_base_url: str) -> None:
    page.emulate_media(reduced_motion="reduce")
    page.goto(f"{admin_base_url}/admin")
    expect(page.locator("#messageArea")).to_have_text("")
    page.get_by_role("button", name="Usage", exact=True).click()
    expect(page.locator("#pageTitle")).to_have_text("Usage")


def test_usage_view_renders_totals_models_and_fallbacks(
    page: Page,
    admin_base_url: str,
    admin_usage_ledger: UsageLedger,
) -> None:
    admin_usage_ledger.record_failure(
        "open_router/vendor/model-a",
        ExecutionFailure(
            kind=FailureKind.RATE_LIMIT,
            status_code=429,
            message="quota exhausted",
            retryable=True,
        ),
    )
    admin_usage_ledger.record_fallback(
        request_id="req_e2e",
        from_model="open_router/vendor/model-a",
        to_model="groq/model-b",
        failure=ExecutionFailure(
            kind=FailureKind.RATE_LIMIT,
            status_code=429,
            message="quota exhausted",
            retryable=True,
        ),
    )
    admin_usage_ledger.record_usage("groq/model-b", input_tokens=1200, output_tokens=34)

    _open_usage(page, admin_base_url)

    totals = page.locator("#usageTotals")
    expect(totals.locator(".usage-stat").first).to_contain_text("1,234")
    expect(totals).to_contain_text("Fallbacks")
    models = page.locator("#usageModels tr")
    expect(models).to_have_count(2)
    expect(models.first).to_contain_text("groq/model-b")
    expect(models.nth(1)).to_contain_text("1 (rate_limit)")
    fallback_row = page.locator("#usageFallbacks tr")
    expect(fallback_row).to_have_count(1)
    expect(fallback_row).to_contain_text("open_router/vendor/model-a")
    expect(fallback_row).to_contain_text("rate_limit (429)")


def test_usage_reset_clears_rendered_counters(
    page: Page,
    admin_base_url: str,
    admin_usage_ledger: UsageLedger,
) -> None:
    admin_usage_ledger.record_usage("groq/model-b", input_tokens=5, output_tokens=5)

    _open_usage(page, admin_base_url)
    expect(page.locator("#usageModels tr")).to_contain_text("groq/model-b")

    page.get_by_role("button", name="Reset", exact=True).click()

    expect(page.locator("#usageModels")).to_have_text("No requests yet")
    expect(page.locator("#usageFallbacks")).to_have_text("No fallbacks yet")
    expect(page.locator("#usageTotals .usage-stat").first).to_contain_text("0")


def test_usage_view_shows_quota_cooldown_status(
    page: Page,
    admin_base_url: str,
    admin_cooldowns: ModelCooldowns,
) -> None:
    admin_cooldowns.mark(
        "open_router/vendor/model-a",
        ExecutionFailure(
            kind=FailureKind.RATE_LIMIT,
            status_code=429,
            message="quota exhausted",
            retryable=True,
        ),
        seconds=600,
    )

    _open_usage(page, admin_base_url)

    row = page.locator("#usageModels tr")
    expect(row).to_have_count(1)
    expect(row).to_contain_text("open_router/vendor/model-a")
    expect(row).to_contain_text("Cooling down until")
    expect(row).to_contain_text("(429)")
