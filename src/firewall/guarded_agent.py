"""The email assistant wrapped by the firewall.

Same agent loop as src/app/agent.py, with the firewall inserted at its three enforcement
points. Keeping this as a separate class (rather than options on EmailAssistant) means the
undefended baseline stays untouched, so "firewall off" really is the original app and not
a code path that happens to skip some checks.

A blocked tool call returns an error string to the model instead of executing. That is
deliberate: the model learns the action failed and can tell the user, which is far better
UX than silently dropping the call, and it does not leak why the policy refused.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from src.app.agent import AgentTrace, TraceEvent
from src.app.fixtures import Environment, build_system_prompt
from src.app.tools import execute_tool, tool_specs
from src.firewall.gating import Decision
from src.firewall.pipeline import Firewall
from src.llm.client import LLMClient, Message, ToolSpec
from src.utils import get_logger

log = get_logger(__name__)


@dataclass
class GuardedTrace(AgentTrace):
    """AgentTrace plus the firewall's audit record."""

    firewall_events: list[dict] = field(default_factory=list)
    blocked_tool_calls: int = 0
    tainted: bool = False
    canary_leaked: bool = False
    firewall_time_s: float = 0.0


class GuardedEmailAssistant:
    """EmailAssistant + Firewall."""

    def __init__(
        self,
        llm: LLMClient,
        env: Environment,
        firewall: Firewall,
        max_steps: int = 6,
        assistant_name: str = "MailPilot",
        tools: list[ToolSpec] | None = None,
        auto_confirm: bool = False,
    ) -> None:
        self.llm = llm
        self.env = env
        self.firewall = firewall
        self.max_steps = max_steps
        self.system_prompt = build_system_prompt(env, assistant_name)
        self.tools = tools if tools is not None else tool_specs()
        # auto_confirm simulates a user who approves every confirmation prompt. Used to
        # measure the worst case: a habituated user who clicks "yes" without reading.
        self.auto_confirm = auto_confirm

    def run(self, task: str, seed: int | None = None) -> GuardedTrace:
        trace = GuardedTrace(task=task, backend=self.llm.name, model=self.llm.model, seed=seed)
        trace.events.append(TraceEvent(0, "user", task))
        start = time.perf_counter()

        # The user's own message is an input too: this is where DIRECT injection is caught.
        fw_start = time.perf_counter()
        verdict = self.firewall.inspect_input(task, source="user_message")
        trace.firewall_time_s += time.perf_counter() - fw_start
        if verdict.blocked:
            trace.final_text = verdict.text
            trace.stop_reason = "blocked_by_firewall"
            trace.firewall_events = self.firewall.audit_log()
            trace.wall_time_s = time.perf_counter() - start
            return trace

        messages: list[Message] = [Message("user", verdict.text)]

        for step in range(1, self.max_steps + 1):
            response = self.llm.chat(self.system_prompt, messages, self.tools, seed=seed)
            trace.steps, trace.llm_time_s = step, trace.llm_time_s + response.latency_s
            messages.append(response.to_message())
            trace.events.append(TraceEvent(step, "assistant", response.text))

            if response.stop_reason == "refusal":
                trace.stop_reason = "refusal"
                break

            if not response.tool_calls:
                fw_start = time.perf_counter()
                out = self.firewall.filter_final_output(response.text)
                trace.firewall_time_s += time.perf_counter() - fw_start
                trace.final_text = out.text
                trace.stop_reason = "blocked_by_firewall" if out.blocked else "final_answer"
                break

            for call in response.tool_calls:
                trace.events.append(TraceEvent(step, "tool_call", tool_name=call.name,
                                               arguments=dict(call.arguments)))
                fw_start = time.perf_counter()
                gate = self.firewall.check_tool_call(call.name, call.arguments,
                                                     user_confirmed=self.auto_confirm)
                trace.firewall_time_s += time.perf_counter() - fw_start

                if gate.decision is not Decision.ALLOW:
                    trace.blocked_tool_calls += 1
                    reason = "; ".join(gate.reasons[:2])
                    content = (f"Error: this action was not permitted by the security policy "
                               f"({gate.decision.value}). Reason: {reason}. "
                               f"Tell the user what you were unable to do.")
                    trace.events.append(TraceEvent(step, "tool_result", content, call.name,
                                                   is_error=True))
                    messages.append(Message("tool", content, tool_call_id=call.id,
                                            tool_name=call.name))
                    continue

                result = execute_tool(self.env, call)
                fw_start = time.perf_counter()
                self.firewall.note_tool_result(call.name, result.untrusted)
                content = result.content
                if result.untrusted:
                    inspected = self.firewall.inspect_input(content, source=call.name)
                    content = inspected.text
                trace.firewall_time_s += time.perf_counter() - fw_start

                trace.events.append(TraceEvent(step, "tool_result", result.content, call.name,
                                               untrusted=result.untrusted,
                                               is_error=result.is_error))
                messages.append(Message("tool", content, tool_call_id=call.id,
                                        tool_name=call.name))
        else:
            trace.stop_reason = "max_steps"

        # Any answer produced without passing through the output filter (e.g. max_steps)
        # still gets filtered, so no path bypasses the last enforcement point.
        if trace.stop_reason in ("max_steps", "refusal") and trace.final_text:
            out = self.firewall.filter_final_output(trace.final_text)
            trace.final_text = out.text

        trace.outbox = list(self.env.outbox)
        trace.firewall_events = self.firewall.audit_log()
        trace.tainted = self.firewall.tainted
        trace.canary_leaked = self.firewall.canary_leaked
        trace.wall_time_s = time.perf_counter() - start
        return trace
