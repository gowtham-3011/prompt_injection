"""Layer 3: taint tracking + tool-call gating. The most important layer.

WHY THIS MATTERS MORE THAN DETECTION
------------------------------------
Detection (heuristics, classifiers) tries to answer "is this text malicious?", which is
an open-ended language question. An attacker gets unlimited attempts to rephrase, encode
or translate until the detector misses - and they only need to succeed once. Any detector
with a usable false-positive rate will therefore be bypassed eventually.

Gating asks a different, much easier question: "should THIS tool call be allowed right
now, given where the data came from?" That is a structural property of the session, not a
judgement about language, so it cannot be argued away by clever phrasing. Even when the
model is completely fooled, least privilege caps the damage: mail cannot reach an
attacker-controlled address, and secrets cannot be placed in outgoing arguments.

This is the same principle as SQL injection defense: input filtering helps, but what
actually fixed SQL injection was prepared statements - a structural rule that holds no
matter what the input says.

Taint model:
  * Reading untrusted content (email, web page, document) marks the session TAINTED.
  * Taint is sticky: once attacker text is in the context, it stays in the context.
  * While tainted, side-effecting tools are blocked or require explicit user confirmation,
    and arguments are checked against an allowlist plus a secret scan.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.app.tools import TOOLS


class Decision(str, Enum):
    ALLOW = "allow"
    CONFIRM = "confirm"   # needs explicit human approval before running
    BLOCK = "block"


@dataclass
class TaintState:
    """Provenance tracking for one agent session."""

    tainted: bool = False
    sources: list[str] = field(default_factory=list)  # which tools introduced taint
    sticky: bool = True

    def mark(self, source: str) -> None:
        self.tainted = True
        if source not in self.sources:
            self.sources.append(source)

    def clear(self) -> None:
        """Only meaningful when sticky is False (used for ablation experiments)."""
        if not self.sticky:
            self.tainted = False
            self.sources.clear()


@dataclass(frozen=True)
class GateResult:
    decision: Decision
    reasons: list[str]
    rule: str = ""
    sanitized_arguments: dict[str, Any] | None = None

    @property
    def allowed(self) -> bool:
        return self.decision is Decision.ALLOW


_SECRET_PATTERNS = [
    (re.compile(r"sk-[\w-]{8,}"), "api key"),
    (re.compile(r"CNRY-[0-9a-f]{8,}", re.I), "session canary"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "aws access key id"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "private key"),
]


def find_secrets(text: str, extra: dict[str, str] | None = None) -> list[str]:
    """Return names of secret-looking values found in `text`.

    `extra` maps a literal secret value to its label, so environment-specific secrets
    (this session's canary and API key) are caught even if they do not match a pattern.
    """
    found = [label for pattern, label in _SECRET_PATTERNS if pattern.search(text or "")]
    for value, label in (extra or {}).items():
        if value and value in (text or "") and label not in found:
            found.append(label)
    return found


def _domain_of(address_or_url: str) -> str:
    if "@" in address_or_url:
        return address_or_url.rsplit("@", 1)[-1].lower()
    return re.sub(r"^https?://([^/]+).*$", r"\1", address_or_url).lower()


def _on_allowlist(value: str, allowlist: list[str]) -> bool:
    domain = _domain_of(value)
    return any(domain == entry.lower() or domain.endswith("." + entry.lower())
               for entry in allowlist)


class ToolGate:
    """Applies the YAML policy to one tool call, given the session's taint state."""

    def __init__(self, policy: dict, secrets: dict[str, str] | None = None) -> None:
        self.policy = policy
        self.secrets = secrets or {}
        self.taint = TaintState(sticky=policy.get("taint", {}).get("sticky", True))
        self._sources = set(policy.get("taint", {}).get("sources", []))

    # -- taint ---------------------------------------------------------------- #
    def note_tool_result(self, tool_name: str, untrusted: bool) -> bool:
        """Call after a tool runs. Marks taint if the tool returned untrusted content."""
        if untrusted or tool_name in self._sources:
            self.taint.mark(tool_name)
            return True
        return False

    # -- gating --------------------------------------------------------------- #
    def check(self, tool_name: str, arguments: dict[str, Any],
              user_confirmed: bool = False) -> GateResult:
        """Decide whether a tool call may run."""
        rules = (self.policy.get("tools") or {}).get(tool_name, {})
        definition = TOOLS.get(tool_name)
        reasons: list[str] = []

        if definition is None:
            return GateResult(Decision.BLOCK, [f"unknown tool {tool_name!r}"], "unknown_tool")

        base = rules.get("when_tainted" if self.taint.tainted else "when_clean", "allow")
        decision = Decision(base)
        if self.taint.tainted and definition.has_side_effects:
            reasons.append(
                f"session is tainted by untrusted content from {', '.join(self.taint.sources)}; "
                f"'{tool_name}' has side effects")

        # Argument-level checks (these can only tighten the decision).
        if tool_name == "send_email":
            decision, extra = self._check_send_email(rules, arguments, decision)
            reasons += extra
        elif tool_name == "browse_url":
            decision, extra = self._check_browse_url(rules, arguments, decision)
            reasons += extra

        # An explicit human confirmation satisfies a CONFIRM decision, but never a BLOCK:
        # the user cannot meaningfully consent to sending a secret they cannot see.
        if decision is Decision.CONFIRM and user_confirmed:
            reasons.append("user explicitly confirmed this action")
            decision = Decision.ALLOW

        if decision is Decision.ALLOW and not reasons:
            reasons.append("no policy rule matched; session clean" if not self.taint.tainted
                           else "tool has no side effects")
        return GateResult(decision, reasons, rule=f"policy:{tool_name}")

    def _check_send_email(self, rules: dict, args: dict, decision: Decision
                          ) -> tuple[Decision, list[str]]:
        reasons: list[str] = []
        recipient = str(args.get("to", ""))
        allowlist = rules.get("recipient_allowlist") or []

        if allowlist and recipient and not _on_allowlist(recipient, allowlist):
            reasons.append(f"recipient {recipient!r} is not on the allowlist {allowlist}")
            # An off-allowlist recipient while tainted is the classic exfil path: block it
            # outright rather than asking a user who has no way to judge.
            decision = Decision.BLOCK if self.taint.tainted else Decision.CONFIRM

        if rules.get("forbid_secrets_in_args", True):
            blob = " ".join(str(v) for v in args.values())
            secrets = find_secrets(blob, self.secrets)
            if secrets:
                reasons.append(f"outgoing arguments contain {', '.join(secrets)}")
                decision = Decision.BLOCK

        max_body = rules.get("max_body_chars")
        if max_body and len(str(args.get("body", ""))) > max_body:
            reasons.append(f"body exceeds {max_body} characters")
            decision = Decision.BLOCK if decision is Decision.ALLOW else decision
        return decision, reasons

    def _check_browse_url(self, rules: dict, args: dict, decision: Decision
                          ) -> tuple[Decision, list[str]]:
        reasons: list[str] = []
        url = str(args.get("url", ""))
        allowlist = rules.get("url_allowlist_when_tainted") or []
        if self.taint.tainted and allowlist and url and not _on_allowlist(url, allowlist):
            # Fetching an arbitrary URL after reading untrusted content can smuggle data
            # out in the path or query string.
            reasons.append(f"tainted session may not fetch {_domain_of(url)!r} "
                           f"(not on {allowlist})")
            decision = Decision.BLOCK
        return decision, reasons


def redact_secrets(text: str, secrets: dict[str, str]) -> str:
    """Replace known secret values and secret-shaped patterns with placeholders."""
    out = text or ""
    for value, label in (secrets or {}).items():
        if value:
            out = out.replace(value, f"[REDACTED-{label.upper().replace(' ', '-')}]")
    for pattern, label in _SECRET_PATTERNS:
        out = pattern.sub(f"[REDACTED-{label.upper().replace(' ', '-')}]", out)
    return out
