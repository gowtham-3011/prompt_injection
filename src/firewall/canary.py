"""Canary tokens: detect that the system prompt leaked.

A canary is a unique random string placed in the system prompt with no other purpose.
It cannot be guessed, so if it appears in a model output, a tool argument, or an outgoing
email, the system prompt has leaked. That is a high-confidence signal: unlike heuristics,
it has essentially no false positives.

What it does NOT do: prevent the leak. By the time the canary is observed, the prompt has
already been exposed to the model's output channel. So canary detection is used to (a)
block that specific output from reaching the user/attacker, and (b) mark the session as
compromised so later actions are restricted. Prevention is the job of Phase 5 (gating).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class CanaryState:
    """Per-session canary tracking."""

    canary: str
    leaked: bool = False
    leak_locations: list[str] = field(default_factory=list)

    def note_leak(self, location: str) -> None:
        self.leaked = True
        if location not in self.leak_locations:
            self.leak_locations.append(location)


def _variants(canary: str) -> list[re.Pattern]:
    """Patterns that catch light mangling of the token.

    Models reformat text: they insert spaces, break lines, or lowercase. An attacker can
    also ask for the token 'with dashes removed'. We match the raw token plus a form that
    tolerates whitespace/punctuation between characters, so trivial reformatting does not
    defeat detection.
    """
    escaped = re.escape(canary)
    loose = r"[\s\-_.]*".join(re.escape(ch) for ch in canary if ch.strip())
    return [re.compile(escaped, re.I), re.compile(loose, re.I)]


def detect(text: str, canary: str) -> bool:
    """True if the canary (or a lightly-mangled form) appears in `text`."""
    if not text or not canary:
        return False
    return any(p.search(text) for p in _variants(canary))


def scan(state: CanaryState, text: str, location: str) -> bool:
    """Check `text` and record a leak at `location`. Returns True if leaked."""
    if detect(text, state.canary):
        state.note_leak(location)
        return True
    return False


def redact(text: str, canary: str, placeholder: str = "[REDACTED-CANARY]") -> str:
    """Remove the canary from text that is about to be shown or sent."""
    if not text or not canary:
        return text
    out = text
    for pattern in _variants(canary):
        out = pattern.sub(placeholder, out)
    return out
