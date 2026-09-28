"""Shared fixtures. All tests use MockLLM or no LLM: no network, no API keys."""

from __future__ import annotations

import pytest

from src.app.fixtures import Environment, build_environment

AGENT_CFG = {"user_name": "Alex Chen", "user_email": "alex.chen@acme-corp.example"}


@pytest.fixture
def env() -> Environment:
    return build_environment(AGENT_CFG, seed=42)
