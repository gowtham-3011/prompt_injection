"""The deliberately vulnerable email assistant: a plain tool-calling agent loop.

    user task -> [model -> tool calls -> tool results -> model -> ...] -> final answer

Why it is vulnerable: tool results (emails, web pages, documents) are appended to the
same context as the user's instructions, so the model cannot reliably tell data apart
from commands. That is the root cause of indirect prompt injection.

The firewall (Phases 3-6) will hook into three points of this loop:
  (1) before untrusted content enters the context  -> detection / sanitisation
  (2) before a tool call executes                  -> taint-aware gating
  (3) before the final answer reaches the user     -> output filtering
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from src.app.fixtures import Environment, SentEmail, build_system_prompt
from src.app.tools import execute_tool, tool_specs
from src.llm.client import LLMClient, Message, ToolSpec
from src.utils import get_logger

log = get_logger(__name__)

EventKind = Literal["user", "assistant", "tool_call", "tool_result"]


@dataclass
class TraceEvent:
    """One observable step. Goal checkers (Phase 2) judge attacks from these events."""

    step: int
    kind: EventKind
    content: str = ""
    tool_name: str | None = None
    arguments: dict[str, Any] | None = None
    untrusted: bool = False
    is_error: bool = False


@dataclass
class AgentTrace:
    task: str
    backend: str
    model: str
    seed: int | None
    events: list[TraceEvent] = field(default_factory=list)
    final_text: str = ""
    outbox: list[SentEmail] = field(default_factory=list)
    stop_reason: str = ""  # final_answer | max_steps | refusal
    steps: int = 0
    wall_time_s: float = 0.0
    llm_time_s: float = 0.0

    @property
    def tool_calls(self) -> list[TraceEvent]:
        return [e for e in self.events if e.kind == "tool_call"]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EmailAssistant:
    """Runs one task against one Environment with one LLM backend."""

    def __init__(
        self,
        llm: LLMClient,
        env: Environment,
        max_steps: int = 6,
        assistant_name: str = "MailPilot",
        tools: list[ToolSpec] | None = None,
    ) -> None:
        self.llm = llm
        self.env = env
        self.max_steps = max_steps
        self.system_prompt = build_system_prompt(env, assistant_name)
        self.tools = tools if tools is not None else tool_specs()

    def run(self, task: str, seed: int | None = None) -> AgentTrace:
        """Run the agent loop until a final answer, a refusal, or max_steps model calls."""
        trace = AgentTrace(task=task, backend=self.llm.name, model=self.llm.model, seed=seed)
        trace.events.append(TraceEvent(0, "user", task))
        messages: list[Message] = [Message("user", task)]
        start = time.perf_counter()

        for step in range(1, self.max_steps + 1):
            response = self.llm.chat(self.system_prompt, messages, self.tools, seed=seed)
            trace.steps, trace.llm_time_s = step, trace.llm_time_s + response.latency_s
            messages.append(response.to_message())
            trace.events.append(TraceEvent(step, "assistant", response.text))

            if response.stop_reason == "refusal":
                trace.stop_reason = "refusal"
                break
            if not response.tool_calls:
                trace.final_text, trace.stop_reason = response.text, "final_answer"
                break

            for call in response.tool_calls:
                log.debug("step %d tool call %s %s", step, call.name, call.arguments)
                trace.events.append(TraceEvent(step, "tool_call", tool_name=call.name,
                                               arguments=dict(call.arguments)))
                result = execute_tool(self.env, call)
                trace.events.append(TraceEvent(step, "tool_result", result.content, call.name,
                                               untrusted=result.untrusted, is_error=result.is_error))
                messages.append(Message("tool", result.content, tool_call_id=call.id, tool_name=call.name))
        else:
            trace.stop_reason = "max_steps"

        trace.outbox = list(self.env.outbox)
        trace.wall_time_s = time.perf_counter() - start
        return trace
