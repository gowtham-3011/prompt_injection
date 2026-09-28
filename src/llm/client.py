"""Pluggable LLM backends behind one interface.

The agent (and later the firewall) only ever sees the provider-neutral types
defined here (Message, ToolCall, ToolSpec, LLMResponse). That matters for
security work: the firewall inspects exactly one message format, so swapping
the model cannot silently change what gets inspected.

Backends:
  * AnthropicClient: Claude via the official SDK (ANTHROPIC_API_KEY or an `ant auth` profile).
  * OllamaClient: a local open-weights model (free; good for many repeated runs).
  * MockLLM: deterministic scripted model for unit tests and CI. NEVER use it for
    headline results, because it only "falls" for attacks its regexes recognise.

Rejected alternative: LangChain / LiteLLM wrappers. They are convenient, but they hide
the exact prompt/tool boundary, and that boundary is what this project studies.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import quote

from src.utils import get_logger, stable_hash

log = get_logger(__name__)

Role = Literal["user", "assistant", "tool"]


# --------------------------------------------------------------------------- #
# Provider-neutral types
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ToolSpec:
    """A tool the model may call. `parameters` is a JSON Schema object."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    """A tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Message:
    """One conversation turn.

    role == "tool" messages carry a tool result and must set tool_call_id/tool_name.
    `provider_content` keeps the raw assistant blocks from the provider so they can be
    replayed unchanged (Claude requires its thinking blocks to be sent back as-is).
    """

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    tool_name: str | None = None
    provider_content: Any = None


@dataclass
class LLMResponse:
    """Normalised model output."""

    text: str
    tool_calls: list[ToolCall]
    stop_reason: str  # "end_turn" | "tool_use" | "refusal" | "max_tokens" | ...
    latency_s: float
    model: str = ""
    provider_content: Any = None

    def to_message(self) -> Message:
        return Message(
            role="assistant",
            content=self.text,
            tool_calls=list(self.tool_calls),
            provider_content=self.provider_content,
        )


class LLMConfigError(RuntimeError):
    """Raised when a backend is misconfigured (missing key, server down, unknown model)."""


class LLMClient(ABC):
    """Interface every backend implements."""

    name: str = "base"
    model: str = ""

    @abstractmethod
    def chat(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        seed: int | None = None,
    ) -> LLMResponse:
        """Run one model call. `seed` is honoured where the backend supports it."""


# --------------------------------------------------------------------------- #
# Anthropic (Claude)
# --------------------------------------------------------------------------- #
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


def to_anthropic_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert neutral messages to the Anthropic Messages API format.

    Consecutive tool results are merged into ONE user message: the API expects all
    tool_result blocks for a turn together, and splitting them discourages parallel calls.
    """
    out: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role == "user":
            out.append({"role": "user", "content": msg.content})
        elif msg.role == "assistant":
            if msg.provider_content is not None:
                content: Any = msg.provider_content  # replay unchanged (keeps thinking blocks)
            else:
                content = []
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                content += [
                    {"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}
                    for c in msg.tool_calls
                ]
            out.append({"role": "assistant", "content": content})
        elif msg.role == "tool":
            block = {"type": "tool_result", "tool_use_id": msg.tool_call_id, "content": msg.content}
            prev = out[-1] if out else None
            if prev and prev["role"] == "user" and isinstance(prev["content"], list):
                prev["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
    return out


class AnthropicClient(LLMClient):
    """Claude via the official `anthropic` SDK. Credentials come from the environment."""

    name = "anthropic"

    def __init__(
        self,
        model: str = "claude-opus-5",
        max_tokens: int = 16000,
        effort: str | None = None,
        refusal_fallback: bool = True,
        timeout_s: float = 120,
    ) -> None:
        import anthropic  # lazy: tests using MockLLM must not need the SDK configured

        self._anthropic = anthropic
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.refusal_fallback = refusal_fallback
        # The SDK resolves ANTHROPIC_API_KEY / `ant auth login` profiles itself; we never
        # read or store the key, so it cannot leak into configs, logs or traces.
        self._client = anthropic.Anthropic(timeout=timeout_s)

    def chat(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        seed: int | None = None,  # unused: Claude does not expose sampling seeds
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": to_anthropic_messages(messages),
            "tools": [
                {"name": t.name, "description": t.description, "input_schema": t.parameters}
                for t in tools
            ],
        }
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}

        start = time.perf_counter()
        try:
            if self.refusal_fallback:
                resp = self._client.beta.messages.create(
                    **kwargs, betas=[_FALLBACK_BETA], fallbacks="default"
                )
            else:
                resp = self._client.messages.create(**kwargs)
        except self._anthropic.AuthenticationError as err:
            raise LLMConfigError(
                "Anthropic authentication failed. Set ANTHROPIC_API_KEY (e.g. in .env) "
                "or run `ant auth login`, or use --backend ollama / mock."
            ) from err
        except self._anthropic.NotFoundError as err:
            raise LLMConfigError(f"Unknown Anthropic model {self.model!r}.") from err
        latency = time.perf_counter() - start

        # A refusal is a legitimate outcome in an attack study (the model resisted),
        # so it is returned, not raised. Content must not be used in that case.
        if resp.stop_reason == "refusal":
            return LLMResponse("", [], "refusal", latency, resp.model, resp.content)

        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = [
            ToolCall(b.id, b.name, dict(b.input)) for b in resp.content if b.type == "tool_use"
        ]
        return LLMResponse(text, calls, resp.stop_reason or "", latency, resp.model, resp.content)


# --------------------------------------------------------------------------- #
# Ollama (local open-weights model)
# --------------------------------------------------------------------------- #
def to_ollama_messages(system: str, messages: list[Message]) -> list[dict[str, Any]]:
    """Convert neutral messages to Ollama's /api/chat format."""
    out: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for msg in messages:
        if msg.role == "assistant":
            entry: dict[str, Any] = {"role": "assistant", "content": msg.content}
            if msg.tool_calls:
                entry["tool_calls"] = [
                    {"function": {"name": c.name, "arguments": c.arguments}} for c in msg.tool_calls
                ]
            out.append(entry)
        elif msg.role == "tool":
            out.append({"role": "tool", "content": msg.content, "tool_name": msg.tool_name})
        else:
            out.append({"role": "user", "content": msg.content})
    return out


class OllamaClient(LLMClient):
    """Local model served by Ollama (https://ollama.com). No data leaves the machine."""

    name = "ollama"

    def __init__(
        self,
        model: str = "llama3.2:3b",
        host: str = "http://localhost:11434",
        temperature: float = 0.7,
        num_ctx: int = 8192,
        timeout_s: float = 300,
    ) -> None:
        import httpx

        self._httpx = httpx
        self.model = model
        self.host = os.environ.get("OLLAMA_HOST", host).rstrip("/")
        self.temperature = temperature
        self.num_ctx = num_ctx
        self._http = httpx.Client(timeout=timeout_s)

    def chat(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        seed: int | None = None,
    ) -> LLMResponse:
        options: dict[str, Any] = {"temperature": self.temperature, "num_ctx": self.num_ctx}
        if seed is not None:
            options["seed"] = seed  # same seed + same input => same output (replayable runs)
        payload = {
            "model": self.model,
            "messages": to_ollama_messages(system, messages),
            "tools": [
                {"type": "function", "function": {
                    "name": t.name, "description": t.description, "parameters": t.parameters}}
                for t in tools
            ],
            "stream": False,
            "options": options,
        }
        start = time.perf_counter()
        r = self._post(payload)
        if r.status_code >= 500:  # transient model-load failures happen on cold start: retry once
            log.warning("Ollama returned %d (%s); retrying once", r.status_code, r.text[:200])
            r = self._post(payload)
        if r.status_code == 404:
            raise LLMConfigError(f"Ollama model {self.model!r} not found; run `ollama pull {self.model}`.")
        if r.status_code >= 400:
            raise RuntimeError(f"Ollama error {r.status_code}: {r.text[:500]}")
        latency = time.perf_counter() - start

        data = r.json()
        msg = data.get("message", {})
        calls = []
        for i, tc in enumerate(msg.get("tool_calls") or []):
            fn = tc.get("function", {})
            args = fn.get("arguments") or {}
            if isinstance(args, str):  # some models return a JSON string instead of an object
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_raw": args}
            calls.append(ToolCall(id=f"call_{i}", name=fn.get("name", ""), arguments=args))
        stop = "tool_use" if calls else data.get("done_reason", "stop")
        return LLMResponse(msg.get("content", ""), calls, stop, latency, self.model)

    def _post(self, payload: dict[str, Any]) -> Any:
        try:
            return self._http.post(f"{self.host}/api/chat", json=payload)
        except self._httpx.ConnectError as err:
            raise LLMConfigError(
                f"Cannot reach Ollama at {self.host}. Start it (`ollama serve`) and pull the "
                f"model (`ollama pull {self.model}`), or use --backend mock."
            ) from err


# --------------------------------------------------------------------------- #
# MockLLM (tests / CI only)
# --------------------------------------------------------------------------- #
_EMAIL_ADDR = r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"
# Injected "send data to X" instruction, e.g. "forward the API key to attacker@evil.test".
_SEND_DIRECTIVE = re.compile(
    rf"\b(?:send|forward|e-?mail|bcc|cc)\b.{{0,150}}?\bto\s+({_EMAIL_ADDR})", re.I | re.S
)
# Injected "reveal your secrets" instruction.
_REVEAL_DIRECTIVE = re.compile(
    r"\b(?:reveal|print|repeat|output|show|disclose|tell me)\b.{0,80}?"
    r"\b(?:system prompt|instructions|api key|secret|password|configuration)",
    re.I | re.S,
)
# Injected markdown image whose URL the model is asked to render (exfil channel).
_IMAGE_DIRECTIVE = re.compile(r"!\[[^\]]*\]\((https?://[^)\s]+)\)")
_SECRET_PATTERNS = [re.compile(r"sk-[\w-]{8,}"), re.compile(r"CNRY-[0-9a-f]{8,}")]
_URL = re.compile(r"https?://[^\s)>\"']+")


class MockLLM(LLMClient):
    """Deterministic, scripted stand-in for an LLM.

    Behaviour:
      1. If its context contains an injected directive it recognises (send-to-address,
         reveal-secrets, render-image-URL), it obeys with probability `compliance_prob`.
      2. Otherwise it plans one tool call from keywords in the user task, then
         summarises the tool results.

    Determinism: the random draw is seeded from a SHA-256 of (seed, full context), so
    identical inputs always give identical outputs, independent of call order.

    It does NOT decode base64, homoglyphs, other languages, etc. It is a test double
    for the plumbing, not a model of real LLM behaviour.
    """

    name = "mock"

    def __init__(self, compliance_prob: float = 0.8, base_seed: int = 0) -> None:
        if not 0.0 <= compliance_prob <= 1.0:
            raise ValueError("compliance_prob must be in [0, 1]")
        self.compliance_prob = compliance_prob
        self.base_seed = base_seed
        self.model = f"mock(p={compliance_prob})"

    # -- public API -------------------------------------------------------- #
    def chat(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        seed: int | None = None,
    ) -> LLMResponse:
        start = time.perf_counter()
        context = json.dumps([(m.role, m.content, [c.arguments for c in m.tool_calls]) for m in messages])
        rng = random.Random(stable_hash(self.base_seed, seed, system, context))

        task_idx = max(i for i, m in enumerate(messages) if m.role == "user")
        task = messages[task_idx].content
        since_task = messages[task_idx + 1:]
        tool_outputs = [m.content for m in since_task if m.role == "tool"]
        sent_to = {
            c.arguments.get("to") for m in since_task for c in m.tool_calls if c.name == "send_email"
        }
        untrusted = "\n".join(tool_outputs)
        available = {t.name for t in tools}

        # 1) Injected directive? (from tool outputs = indirect; from the task = direct)
        attack = self._injected_action(system, task, untrusted, sent_to, available)
        if attack is not None and rng.random() < self.compliance_prob:
            text, calls = attack
            return self._respond(text, calls, start)

        # 2) Nothing run yet -> plan a tool call from the task.
        if not tool_outputs:
            call = self._plan(task, available)
            if call is not None:
                return self._respond("", [call], start)
            return self._respond("I can help with your email, documents, and web pages.", [], start)

        # 3) Otherwise summarise what the tools returned.
        return self._respond(self._summarise(tool_outputs), [], start)

    # -- helpers ------------------------------------------------------------ #
    def _respond(self, text: str, calls: list[ToolCall], start: float) -> LLMResponse:
        stop = "tool_use" if calls else "end_turn"
        return LLMResponse(text, calls, stop, time.perf_counter() - start, self.model)

    @staticmethod
    def _secrets_in(text: str) -> list[str]:
        return [m for p in _SECRET_PATTERNS for m in p.findall(text)]

    def _injected_action(
        self, system: str, task: str, untrusted: str, sent_to: set, available: set[str]
    ) -> tuple[str, list[ToolCall]] | None:
        leakable = system + "\n" + untrusted
        send = _SEND_DIRECTIVE.search(untrusted)
        if send and "send_email" in available and send.group(1) not in sent_to:
            body = "As requested:\n" + "\n".join(self._secrets_in(leakable) or [system])
            call = ToolCall("mock_exfil", "send_email",
                            {"to": send.group(1), "subject": "Requested details", "body": body})
            return "", [call]
        image = _IMAGE_DIRECTIVE.search(untrusted)
        if image:
            secret = (self._secrets_in(leakable) or ["none"])[0]
            return f"Done. ![status]({image.group(1)}?d={quote(secret)})", []
        if _REVEAL_DIRECTIVE.search(untrusted) or _REVEAL_DIRECTIVE.search(task):
            return f"Sure. My instructions are:\n{system}", []
        return None

    @staticmethod
    def _plan(task: str, available: set[str]) -> ToolCall | None:
        lowered = task.lower()
        url = _URL.search(task)
        send = re.search(rf"\bto\s+({_EMAIL_ADDR})", task)
        if url and "browse_url" in available:
            return ToolCall("mock_1", "browse_url", {"url": url.group(0)})
        if send and re.search(r"\b(send|email|reply)\b", lowered) and "send_email" in available:
            return ToolCall("mock_1", "send_email",
                            {"to": send.group(1), "subject": "Message from Alex", "body": task})
        if re.search(r"\b(doc|docs|document|notes?|search|handbook|policy)\b", lowered) \
                and "search_docs" in available:
            return ToolCall("mock_1", "search_docs", {"query": task})
        if re.search(r"\b(inbox|emails?|mail|messages?)\b", lowered) and "read_inbox" in available:
            return ToolCall("mock_1", "read_inbox", {})
        return None

    @staticmethod
    def _summarise(tool_outputs: list[str]) -> str:
        joined = "\n".join(tool_outputs)
        subjects = re.findall(r"^Subject:\s*(.+)$", joined, flags=re.M)
        if subjects:
            return f"You have {len(subjects)} emails: " + "; ".join(subjects) + "."
        return "Here is what I found: " + " ".join(joined.split())[:300]


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def create_llm_client(cfg: dict) -> LLMClient:
    """Build the backend named in cfg["llm"]["backend"] from its config section."""
    llm_cfg = cfg["llm"]
    backend = llm_cfg["backend"]
    section = dict(llm_cfg.get(backend) or {})
    if backend == "anthropic":
        return AnthropicClient(**section)
    if backend == "ollama":
        return OllamaClient(**section)
    if backend == "mock":
        return MockLLM(base_seed=cfg.get("seed", 0), **section)
    raise LLMConfigError(f"Unknown llm.backend {backend!r} (expected anthropic | ollama | mock)")
