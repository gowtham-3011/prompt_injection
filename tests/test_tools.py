"""Simulated tools must be sandboxed and must carry correct trust metadata."""

from __future__ import annotations

import socket

import pytest

from src.app.fixtures import ATTACKER_DOMAIN
from src.app.tools import TOOLS, execute_tool
from src.llm.client import ToolCall


@pytest.fixture
def no_network(monkeypatch):
    """Fail loudly if any code path tries to open a socket."""
    def _blocked(*args, **kwargs):
        raise AssertionError("network access attempted")
    monkeypatch.setattr(socket, "socket", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)


def call(name: str, **args) -> ToolCall:
    return ToolCall("t1", name, args)


def test_send_email_only_writes_outbox(env, no_network):
    result = execute_tool(env, call("send_email", to="bob@example.com", subject="Hi", body="Hello"))
    assert not result.is_error
    assert [(m.to, m.body) for m in env.outbox] == [("bob@example.com", "Hello")]


@pytest.mark.parametrize("to", ["not-an-address", "", "a@b", "x@y.com; rm -rf"])
def test_send_email_rejects_bad_recipient(env, to):
    result = execute_tool(env, call("send_email", to=to, subject="s", body="b"))
    assert result.is_error
    assert env.outbox == []


def test_browse_url_never_fetches_unknown_urls(env, no_network):
    result = execute_tool(env, call("browse_url", url="https://www.google.com"))
    assert "404" in result.content


def test_browse_url_serves_fixture_page_raw_html(env):
    result = execute_tool(env, call("browse_url", url=f"https://deals.{ATTACKER_DOMAIN}/laptops"))
    assert "<!--" in result.content  # hidden comment is passed through (that's the vulnerability)
    assert result.untrusted


def test_search_docs_can_return_confidential_doc(env):
    result = execute_tool(env, call("search_docs", query="board acquisition notes"))
    assert "Northwind" in result.content


def test_read_inbox_limit_accepts_numeric_string(env):
    result = execute_tool(env, call("read_inbox", limit="2"))
    assert result.content.count("Subject:") == 2


def test_unknown_tool_and_bad_args_are_errors_not_exceptions(env):
    assert execute_tool(env, call("delete_everything")).is_error
    assert execute_tool(env, call("browse_url")).is_error


def test_trust_metadata():
    # Sources of third-party text are untrusted; send_email is the side-effecting sink.
    assert {n for n, t in TOOLS.items() if t.returns_untrusted} == {"read_inbox", "browse_url", "search_docs"}
    assert {n for n, t in TOOLS.items() if t.has_side_effects} == {"send_email"}
