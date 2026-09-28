from __future__ import annotations

import pytest

from src.utils import deep_merge, load_config, parse_overrides, stable_hash


def test_stable_hash_is_deterministic_and_sensitive():
    assert stable_hash("a", 1) == stable_hash("a", 1) != stable_hash("a", 2)


def test_parse_overrides_types_values():
    assert parse_overrides(["llm.backend=mock", "seed=7", "x.y=null"]) == {
        "llm": {"backend": "mock"}, "seed": 7, "x": {"y": None}}
    with pytest.raises(ValueError):
        parse_overrides(["no_equals_sign"])


def test_deep_merge_does_not_mutate_base():
    base = {"a": {"b": 1, "c": 2}}
    assert deep_merge(base, {"a": {"b": 9}}) == {"a": {"b": 9, "c": 2}}
    assert base == {"a": {"b": 1, "c": 2}}


def test_load_config_cli_override_wins(monkeypatch):
    monkeypatch.setenv("PIF_LLM_BACKEND", "ollama")
    cfg = load_config(overrides=["llm.backend=mock"])
    assert cfg["llm"]["backend"] == "mock"
