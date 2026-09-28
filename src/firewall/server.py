"""FastAPI service exposing the firewall, plus middleware for an existing app.

Two ways to use it:

  1. As a SERVICE - POST text or a tool call and get a decision back. Useful when the
     LLM app is in another language or another process.
         uvicorn src.firewall.server:app --reload

  2. As MIDDLEWARE - mount FirewallMiddleware in front of an existing FastAPI app to
     screen request bodies before they reach the handler.

Design note: the service is stateless per request, but taint is per SESSION, so callers
pass a session_id and the server keeps that session's state. A firewall with no session
memory cannot do taint tracking at all - it would have to decide each tool call as if
nothing had been read before, which is exactly the blind spot attackers use.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from src.firewall.classifier import TfidfClassifier, UntrainedScorer
from src.firewall.pipeline import Firewall, LayerSwitches, Mode, load_policy
from src.utils import PROJECT_ROOT, get_logger

log = get_logger(__name__)

app = FastAPI(title="Prompt Injection Firewall", version="0.1.0")

# Per-session firewalls. In production this belongs in a shared store with a TTL;
# a dict is fine for a single-process demo.
_SESSIONS: dict[str, Firewall] = {}


def _load_classifier():
    path = PROJECT_ROOT / "checkpoints" / "classifier_tfidf.joblib"
    if path.exists():
        return TfidfClassifier.load(path)
    log.warning("no classifier checkpoint; classifier layer disabled")
    return UntrainedScorer()


_CLASSIFIER = None


def get_firewall(session_id: str, canary: str = "", secrets: dict[str, str] | None = None,
                 mode: str = "sanitize", switches: dict[str, bool] | None = None) -> Firewall:
    global _CLASSIFIER
    if _CLASSIFIER is None:
        _CLASSIFIER = _load_classifier()
    if session_id not in _SESSIONS:
        _SESSIONS[session_id] = Firewall(
            policy=load_policy(),
            switches=LayerSwitches(**switches) if switches else LayerSwitches(),
            mode=Mode(mode), classifier=_CLASSIFIER, canary=canary, secrets=secrets or {},
        )
    return _SESSIONS[session_id]


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class InspectRequest(BaseModel):
    session_id: str = Field(..., description="Groups requests so taint can persist")
    text: str
    source: str = "tool"
    untrusted: bool = True
    canary: str = ""
    mode: str = "sanitize"


class InspectResponse(BaseModel):
    text: str
    blocked: bool
    score: float
    reasons: list[str]
    tainted: bool


class ToolCallRequest(BaseModel):
    session_id: str
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    user_confirmed: bool = False


class ToolCallResponse(BaseModel):
    decision: str
    allowed: bool
    reasons: list[str]
    tainted: bool


class OutputRequest(BaseModel):
    session_id: str
    text: str


class OutputResponse(BaseModel):
    text: str
    blocked: bool
    reasons: list[str]
    canary_leaked: bool


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/health")
def health() -> dict:
    return {"status": "ok", "sessions": len(_SESSIONS)}


@app.post("/v1/inspect_input", response_model=InspectResponse)
def inspect_input(req: InspectRequest) -> InspectResponse:
    """Screen untrusted content before it enters an LLM context."""
    fw = get_firewall(req.session_id, canary=req.canary, mode=req.mode)
    verdict = fw.inspect_input(req.text, source=req.source, untrusted=req.untrusted)
    if req.untrusted:
        fw.note_tool_result(req.source, True)
    reasons = [r for e in verdict.events for r in e.reasons]
    return InspectResponse(text=verdict.text, blocked=verdict.blocked, score=verdict.score,
                           reasons=reasons, tainted=fw.tainted)


@app.post("/v1/check_tool_call", response_model=ToolCallResponse)
def check_tool_call(req: ToolCallRequest) -> ToolCallResponse:
    """Gate a tool call against the policy and the session's taint state."""
    fw = get_firewall(req.session_id)
    verdict = fw.check_tool_call(req.tool, req.arguments, user_confirmed=req.user_confirmed)
    return ToolCallResponse(decision=verdict.decision.value, allowed=verdict.allowed,
                            reasons=verdict.reasons, tainted=fw.tainted)


@app.post("/v1/filter_output", response_model=OutputResponse)
def filter_output_route(req: OutputRequest) -> OutputResponse:
    """Filter a model answer before it reaches the user."""
    fw = get_firewall(req.session_id)
    verdict = fw.filter_final_output(req.text)
    reasons = [r for e in verdict.events for r in e.reasons]
    return OutputResponse(text=verdict.text, blocked=verdict.blocked, reasons=reasons,
                          canary_leaked=fw.canary_leaked)


@app.get("/v1/audit/{session_id}")
def audit(session_id: str) -> dict:
    """Full decision log for a session, for a security analyst to review."""
    fw = _SESSIONS.get(session_id)
    if fw is None:
        raise HTTPException(status_code=404, detail="unknown session")
    return {"session_id": session_id, "tainted": fw.tainted,
            "canary_leaked": fw.canary_leaked, "events": fw.audit_log()}


@app.delete("/v1/session/{session_id}")
def end_session(session_id: str) -> dict:
    """Drop a session's state (taint does not survive, by design)."""
    existed = _SESSIONS.pop(session_id, None) is not None
    return {"deleted": existed}


# --------------------------------------------------------------------------- #
# Middleware
# --------------------------------------------------------------------------- #
class FirewallMiddleware(BaseHTTPMiddleware):
    """Screens JSON request bodies on selected paths before they reach the app.

    Deliberately limited: middleware sees the request, not the tool calls the model will
    later make, so it can only do the INPUT layer. Real protection needs the gating layer
    inside the agent loop. This exists so an existing service can get input screening
    with one line, not as a substitute for the full pipeline.
    """

    def __init__(self, app, field: str = "message", paths: tuple[str, ...] = ("/chat",),
                 mode: str = "sanitize") -> None:
        super().__init__(app)
        self.field = field
        self.paths = paths
        self.mode = mode

    async def dispatch(self, request: Request, call_next):
        if request.method != "POST" or not any(request.url.path.startswith(p) for p in self.paths):
            return await call_next(request)

        body = await request.body()
        try:
            payload = json.loads(body or b"{}")
        except json.JSONDecodeError:
            return await call_next(request)

        text = payload.get(self.field)
        if isinstance(text, str) and text:
            session_id = payload.get("session_id", request.client.host if request.client else "anon")
            fw = get_firewall(session_id, mode=self.mode)
            verdict = fw.inspect_input(text, source="http_request", untrusted=True)
            if verdict.blocked:
                return JSONResponse(status_code=403, content={
                    "error": "request blocked by prompt injection firewall",
                    "reasons": [r for e in verdict.events for r in e.reasons],
                })
            payload[self.field] = verdict.text
            new_body = json.dumps(payload).encode()

            async def receive() -> dict:
                return {"type": "http.request", "body": new_body, "more_body": False}

            request._receive = receive  # replace the stream with the sanitized body
        return await call_next(request)
