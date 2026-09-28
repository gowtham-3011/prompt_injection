"""Layer 1: fast pattern rules over untrusted text.

Design goals:
  * Cheap (sub-millisecond) so it can run on every chunk of untrusted content.
  * EXPLAINABLE: each rule returns a score in [0,1] and a human-readable reason, so a
    security analyst can audit why something was flagged. A single opaque score would be
    impossible to tune or defend in a review.
  * Conservative: heuristics are the first filter, not the last. They are expected to
    miss paraphrases; that is exactly what Phases 4 and 5 exist for. We would rather
    miss an attack here than block a legitimate user (see hard negatives in data/benign.py).

Rejected alternative: one big regex with a single threshold. It performs similarly but
gives no per-rule attribution, and every tuning change becomes guesswork.
"""

from __future__ import annotations

import base64
import binascii
import re
import unicodedata
from dataclasses import dataclass, field

# Characters that are invisible or direction-changing: used to hide text from humans
# while keeping it visible to the model.
INVISIBLE_CHARS = (
    "​‌‍⁠﻿"  # zero-width space/non-joiner/joiner/word-joiner/BOM
    "‪‫‬‭‮"  # bidi overrides
    "­"                           # soft hyphen
)

_INSTRUCTION_OVERRIDE = re.compile(
    r"\b(?:ignore|disregard|forget|override|bypass)\b[^.\n]{0,40}?"
    r"\b(?:previous|prior|above|earlier|all)\b[^.\n]{0,20}?"
    r"\b(?:instruction|instructions|prompt|prompts|rule|rules|direction|directions)\b",
    re.I,
)
_SYSTEM_PROMPT_TARGET = re.compile(
    r"\b(?:reveal|print|repeat|output|show|disclose|echo|dump|summarize)\b[^.\n]{0,50}?"
    r"\b(?:system prompt|your instructions|configuration|config|api[_ ]?key|session[_ ]?id|"
    r"secret|credentials?|password)\b",
    re.I,
)
_ADDRESSES_THE_MODEL = re.compile(
    r"\b(?:ai assistant|assistant|ai agent|language model|llm|chatbot|gpt|claude)\b\s*[:,]|"
    r"\b(?:message|note|instruction)s?\s+(?:for|to)\s+the\s+(?:ai|assistant|model|agent)\b|"
    r"\b(?:system|developer)\s+(?:message|override|mode)\b",
    re.I,
)
_ROLEPLAY = re.compile(
    r"\b(?:you are now|act as|pretend to be|from now on you|as (?:dan|debugbot|devmode)|"
    r"developer mode|jailbreak|no restrictions|unrestricted mode)\b",
    re.I,
)
_TOOL_ABUSE = re.compile(
    r"\b(?:send|forward|email|transmit|upload|post|exfiltrate)\b[^.\n]{0,80}?"
    r"(?:\bto\b\s*)?[\w.+-]+@[\w-]+\.[\w.-]+",
    re.I,
)
_URGENCY_SECRECY = re.compile(
    r"\b(?:do not (?:mention|tell|inform|reveal this)|without (?:telling|informing|notifying)|"
    r"silently|secretly|before (?:replying|responding|summarizing)|do this first)\b",
    re.I,
)
# "Repeat everything above / verbatim" style context extraction, which does not name the
# system prompt explicitly and so slips past _SYSTEM_PROMPT_TARGET.
_CONTEXT_EXTRACTION = re.compile(
    r"\b(?:repeat|echo|print|output|show|reproduce)\b[^.\n]{0,40}?"
    r"(?:\b(?:everything|all|text|content|message)\b[^.\n]{0,30}?"
    r"\b(?:above|before|preceding|earlier|prior)\b|\bverbatim\b|\bword for word\b)",
    re.I,
)
# A URL whose query string carries secret-looking data: the classic output-channel exfil.
_DATA_BEARING_URL = re.compile(
    r"https?://[^\s)\"'<>]*\?[^\s)\"'<>]*"
    r"(?:api[_-]?key|session[_-]?id|secret|token|password|<[^>]+>|\bDATA\b)",
    re.I,
)
_HTML_COMMENT = re.compile(r"<!--(.*?)-->", re.S)
_HIDDEN_STYLE = re.compile(
    r"style\s*=\s*[\"'][^\"']*(?:display\s*:\s*none|visibility\s*:\s*hidden|"
    r"font-size\s*:\s*0|opacity\s*:\s*0)", re.I)
_MARKDOWN_IMAGE = re.compile(r"!\[[^\]]*\]\((https?://[^)\s]+)\)")
# Trailing '=' padding is not a word character, so a \b anchor at the end would force the
# match to drop the padding and break the length%4 check. Use a negative lookahead instead.
_B64_BLOB = re.compile(r"\b[A-Za-z0-9+/]{24,}={0,2}(?![A-Za-z0-9+/=])")
_HEX_BLOB = re.compile(r"\b(?:[0-9a-fA-F]{2}[\s:]?){16,}\b")


@dataclass(frozen=True)
class RuleHit:
    rule: str
    score: float
    reason: str
    excerpt: str = ""


@dataclass
class HeuristicResult:
    """Aggregate verdict for one piece of text."""

    score: float                       # max of rule scores, in [0,1]
    hits: list[RuleHit] = field(default_factory=list)

    @property
    def reasons(self) -> list[str]:
        return [f"{h.rule}: {h.reason}" for h in self.hits]

    def flagged(self, threshold: float) -> bool:
        return self.score >= threshold


def _excerpt(text: str, match: re.Match, width: int = 60) -> str:
    start = max(0, match.start() - 10)
    return text[start:match.end() + width].replace("\n", " ")[:120]


def _decoded_base64_candidates(text: str) -> list[tuple[str, str]]:
    """Return (blob, decoded) for base64 blobs that decode to readable ASCII text.

    Why decode: attackers encode instructions so keyword rules miss them. We only treat a
    blob as suspicious if it decodes to mostly-printable text (random IDs and hashes do not).
    """
    out = []
    for match in _B64_BLOB.finditer(text):
        blob = match.group(0)
        if len(blob) % 4:
            continue
        try:
            decoded = base64.b64decode(blob, validate=True).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError, ValueError):
            continue
        printable = sum(ch.isprintable() or ch.isspace() for ch in decoded)
        if decoded.strip() and printable / len(decoded) > 0.9 and re.search(r"[a-zA-Z]{3}", decoded):
            out.append((blob, decoded))
    return out


def _normalize_confusables(text: str) -> str:
    """Fold look-alike characters to ASCII so homoglyph attacks hit the same rules.

    NFKD normalization handles many decorated forms; we additionally map a small set of
    Cyrillic/Greek look-alikes that NFKD leaves alone (they are legitimately distinct
    characters, so this is a security-specific transform, not general text processing).
    """
    table = {
        "а": "a", "е": "e", "о": "o", "р": "p", "с": "c",
        "у": "y", "і": "i", "ѕ": "s", "ο": "o", "у": "y",
        "һ": "h", "х": "x", "ԁ": "d",
    }
    folded = "".join(table.get(ch, ch) for ch in text)
    return unicodedata.normalize("NFKD", folded)


def _strip_invisible(text: str) -> str:
    return "".join(ch for ch in text if ch not in INVISIBLE_CHARS)


def analyze(text: str, allowed_domains: tuple[str, ...] = ()) -> HeuristicResult:
    """Score untrusted text. Returns the max rule score plus every rule hit."""
    if not text:
        return HeuristicResult(0.0, [])

    hits: list[RuleHit] = []

    # Analyze three views of the text so obfuscation cannot hide the payload from the
    # keyword rules: raw, invisible-chars removed, and confusables folded to ASCII.
    stripped = _strip_invisible(text)
    folded = _normalize_confusables(stripped)
    views = {"raw": text, "deobfuscated": folded}

    keyword_rules = [
        ("instruction_override", _INSTRUCTION_OVERRIDE, 0.9,
         "phrase attempts to override previous instructions"),
        ("system_prompt_target", _SYSTEM_PROMPT_TARGET, 0.9,
         "asks for the system prompt, configuration or credentials"),
        ("addresses_the_model", _ADDRESSES_THE_MODEL, 0.6,
         "content speaks to the AI assistant rather than the human reader"),
        ("roleplay_jailbreak", _ROLEPLAY, 0.7, "role-play / restriction-removal framing"),
        ("tool_abuse_request", _TOOL_ABUSE, 0.7, "asks to send data to an email address"),
        ("urgency_secrecy", _URGENCY_SECRECY, 0.6, "asks to act silently or hide the action"),
        ("context_extraction", _CONTEXT_EXTRACTION, 0.8,
         "asks the model to repeat its preceding context verbatim"),
    ]
    for view_name, view in views.items():
        for rule, pattern, score, reason in keyword_rules:
            match = pattern.search(view)
            if match and not any(h.rule == rule for h in hits):
                suffix = "" if view_name == "raw" else " (after de-obfuscation)"
                hits.append(RuleHit(rule, score, reason + suffix, _excerpt(view, match)))

    # Structural rules (HTML / markdown / encoding), evaluated on the raw text.
    comment = _HTML_COMMENT.search(text)
    if comment and comment.group(1).strip():
        inner = comment.group(1)
        # A comment is only suspicious if it reads like an instruction.
        score = 0.85 if (_INSTRUCTION_OVERRIDE.search(inner) or _SYSTEM_PROMPT_TARGET.search(inner)
                         or _ADDRESSES_THE_MODEL.search(inner)) else 0.35
        hits.append(RuleHit("html_comment_payload", score,
                            "instruction-like text hidden in an HTML comment",
                            _excerpt(text, comment)))

    hidden = _HIDDEN_STYLE.search(text)
    if hidden:
        hits.append(RuleHit("hidden_html_text", 0.8,
                            "text hidden from human viewers via CSS", _excerpt(text, hidden)))

    invisible_count = sum(ch in INVISIBLE_CHARS for ch in text)
    if invisible_count >= 3:
        hits.append(RuleHit("invisible_unicode", 0.7,
                            f"{invisible_count} invisible/bidi characters embedded in text"))

    if folded != stripped:
        hits.append(RuleHit("homoglyph_text", 0.6,
                            "look-alike (non-ASCII) characters used in Latin-looking words"))

    for blob, decoded in _decoded_base64_candidates(text):
        suspicious = bool(_INSTRUCTION_OVERRIDE.search(decoded) or _SYSTEM_PROMPT_TARGET.search(decoded)
                          or _ADDRESSES_THE_MODEL.search(decoded))
        hits.append(RuleHit("base64_payload", 0.85 if suspicious else 0.4,
                            ("base64 blob decodes to instruction-like text" if suspicious
                             else "base64 blob decodes to readable text"),
                            decoded[:100]))
        break

    hex_blob = _HEX_BLOB.search(text)
    if hex_blob:
        hits.append(RuleHit("hex_blob", 0.3, "long hex blob may hide an encoded payload",
                            _excerpt(text, hex_blob)))

    data_url = _DATA_BEARING_URL.search(text)
    if data_url:
        hits.append(RuleHit("data_bearing_url", 0.8,
                            "URL query string carries secret-looking data (exfil channel)",
                            _excerpt(text, data_url)))

    for match in _MARKDOWN_IMAGE.finditer(text):
        url = match.group(1)
        domain = re.sub(r"^https?://([^/]+).*$", r"\1", url)
        if allowed_domains and any(domain.endswith(d) for d in allowed_domains):
            continue
        hits.append(RuleHit("markdown_image_url", 0.6,
                            f"markdown image points at an external URL ({domain})", url[:120]))
        break

    score = max((h.score for h in hits), default=0.0)
    return HeuristicResult(score, hits)
