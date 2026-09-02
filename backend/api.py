"""
backend/api.py
FastAPI backend for AC-RAG React UI.

Exposes:
  GET  /health   → liveness probe
  GET  /status   → readiness (is a document loaded?)
  POST /upload   → ingest a document, build vector store, load pipeline
  POST /ask      → SSE streaming response for a query

SSE event types (in order):
  agent_start  — a pipeline agent has begun executing
  start        — route metadata (fired after entry_router completes)
  agent_done   — a pipeline agent has finished (with structured output)
  token        — one word of the answer (word-by-word simulation)
  scores       — critic self-reflection scores
  sources      — attribution entries
  trace        — pipeline stage log
  done         — terminal event
"""

import asyncio
import json
import logging
import sys
import tempfile
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ── Add project root to sys.path ───────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import SUPPORTED_EXTENSIONS, VECTORSTORE_DIR

logger = logging.getLogger("ac_rag.api")

# Upload guardrails. This backend is single-tenant (one shared in-memory vector
# store for all clients — no per-user isolation, see _state below), so these
# limits exist to stop a single bad upload from exhausting memory/disk, not to
# provide multi-tenant security. Document that limitation if this is ever
# exposed beyond a local/trusted demo.
MAX_UPLOAD_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB — generous for a text/PDF document

# ── Pipeline node → UI agent-id mapping ───────────────────────────────────────
# retrieval_planner + retriever are both part of the "retrieval_planning" agent.
# retrieval_planner starts the agent (active), retriever completes it (done).
NODE_TO_AGENT = {
    "query_analyzer":    "query_understanding",
    "retriever":         "retrieval_planning",   # completes the retrieval_planning agent
    "validator":         "evidence_validation",
    "context_refiner":   "context_refinement",
    "generator":         "answer_generation",
    "critic":            "self_reflection",
}

# Ordered sequence used to determine "next agent" after each step (6 agents)
RAG_SEQUENCE = [
    "query_understanding",
    "retrieval_planning",
    "evidence_validation",
    "context_refinement",
    "answer_generation",
    "self_reflection",
]

# ── Global state ───────────────────────────────────────────────────────────────
_state: dict = {
    "vsm":      None,
    "pipeline": None,
    "doc_name": None,
    "doc_list": [],
    "ready":    False,
}

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(title="AC-RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ──────────────────────────────────────────────────
class ConversationTurn(BaseModel):
    role: str      # "user" | "assistant"
    content: str


class AskRequest(BaseModel):
    query: str
    # Prior turns for follow-up/pronoun resolution (see GAPS_AND_IMPROVEMENTS.md C.15).
    # Optional — omit for a fresh conversation. Only the last few turns are used
    # (query_analyzer.py caps it), so the client can send the full local history
    # without worrying about prompt bloat.
    history: list[ConversationTurn] = []


# ── Helper: extract structured output from each node's state delta ─────────────

def _extract_output(node_name: str, state: dict) -> dict:
    """Return a display-friendly output dict for the given node."""
    if node_name == "query_analyzer":
        return {
            "rewritten_query":  state.get("rewritten_query"),
            "intent":           state.get("intent"),
            "complexity_score": state.get("complexity_score"),
        }
    if node_name == "retriever":
        # retriever completes the retrieval_planning agent; show plan + retrieved docs
        plan = state.get("retrieval_plan") or {}
        docs = state.get("retrieved_docs") or []
        return {
            "k":              plan.get("k"),
            "depth":          plan.get("retrieval_depth"),
            "modality":       plan.get("modality_filter"),
            "multi_query":    plan.get("use_multi_query"),
            "retrieved_count": len(docs),
        }
    if node_name == "validator":
        docs   = state.get("scored_docs") or []
        passed = state.get("validation_passed")
        scores = [round(d.get("score", 0), 2) for d in docs if isinstance(d, dict)]
        return {"scored_count": len(docs), "validation_passed": passed, "scores": scores}
    if node_name == "context_refiner":
        ctx = state.get("refined_context") or ""
        return {
            "context_length": len(ctx),
            "preview":        (ctx[:120] + "…") if len(ctx) > 120 else ctx,
        }
    if node_name == "generator":
        ans = state.get("answer") or ""
        return {
            "answer_length": len(ans),
            "preview":       (ans[:120] + "…") if len(ans) > 120 else ans,
        }
    if node_name == "critic":
        scores = state.get("critic_scores") or {}
        passed = state.get("critic_passed")
        return {"scores": scores, "passed": passed}
    return {}


def _make_summary(node_name: str, output: dict) -> str:
    """Return a one-line human-readable summary for each node's output."""
    if node_name == "query_analyzer":
        return (
            f"intent: {output.get('intent', '?')} · "
            f"complexity: {output.get('complexity_score', '?')}"
        )
    if node_name == "retriever":
        return (
            f"k={output.get('k', '?')} · modality={output.get('modality', '?')} · "
            f"{output.get('retrieved_count', '?')} passages retrieved"
        )
    if node_name == "validator":
        p = "passed" if output.get("validation_passed") else "failed"
        return f"validation {p} · {output.get('scored_count', '?')} scored"
    if node_name == "context_refiner":
        return f"context: {output.get('context_length', '?')} chars"
    if node_name == "generator":
        return f"answer: {output.get('answer_length', '?')} chars"
    if node_name == "critic":
        scores  = output.get("scores") or {}
        overall = scores.get("overall", "?")
        passed  = "PASS" if output.get("passed") else "FAIL"
        return f"overall: {overall} · {passed}"
    return ""


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"ok": True}


@app.get("/status")
def status():
    return {
        "ready":    _state["ready"],
        "doc_name": _state["doc_name"],
        "doc_list": _state["doc_list"],
    }


@app.get("/docs")
def list_docs():
    return {"doc_list": _state["doc_list"]}


@app.delete("/docs")
def clear_docs():
    _state["vsm"]      = None
    _state["pipeline"] = None
    _state["doc_name"] = None
    _state["doc_list"] = []
    _state["ready"]    = False

    # Also remove the persisted index from disk — previously this only cleared
    # in-memory references, so uploaded document content silently survived on
    # disk after a user asked to "clear all" (see GAPS_AND_IMPROVEMENTS.md C.19).
    import shutil
    index_dir = Path(VECTORSTORE_DIR)
    if index_dir.exists():
        shutil.rmtree(index_dir, ignore_errors=True)
        logger.info("Deleted persisted index at %s", index_dir)

    logger.info("All documents cleared.")
    return {"ok": True}


@app.post("/upload")
async def upload(request: Request, file: UploadFile):
    fname = file.filename or ""
    suffix = Path(fname).suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix or '(none)'}'. "
                   f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}",
        )

    # Reject oversized uploads before reading the body into memory, when the
    # client sends Content-Length (most do). This is a best-effort guard, not
    # a hard limit — a client could omit/lie about Content-Length — so the
    # post-read size check below is the real backstop.
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max {MAX_UPLOAD_SIZE_BYTES // (1024*1024)} MB.",
        )

    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max {MAX_UPLOAD_SIZE_BYTES // (1024*1024)} MB.",
        )

    existing_vsm = _state["vsm"]  # capture before thread

    def _ingest_and_build():
        from ingestion import ingest_documents
        from vectorstore.store import VectorStoreManager
        from pipeline.graph import build_pipeline

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            logger.info("Ingesting document: %s", fname)
            chunks = ingest_documents(tmp_path)

            if not chunks:
                raise RuntimeError("No content extracted from document.")

            logger.info("%d chunks extracted.", len(chunks))

            if existing_vsm is not None:
                # Additive: add to the existing vector store
                logger.info("Adding to existing index (%d new chunks)…", len(chunks))
                existing_vsm.add_documents(chunks)
                existing_vsm.save()
                vsm = existing_vsm
            else:
                # First upload: build from scratch
                logger.info("Building new vector store (%d chunks)…", len(chunks))
                vsm = VectorStoreManager()
                vsm.build(chunks)
                vsm.save()

            logger.info("Compiling pipeline…")
            pipeline = build_pipeline(vsm)
            return vsm, pipeline

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    try:
        vsm, pipeline = await asyncio.to_thread(_ingest_and_build)
    except Exception as exc:
        logger.exception("Upload failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    # Update doc_list (avoid duplicates)
    current_list = _state["doc_list"] or []
    if fname not in current_list:
        current_list = current_list + [fname]

    _state["vsm"]      = vsm
    _state["pipeline"] = pipeline
    _state["doc_name"] = fname
    _state["doc_list"] = current_list
    _state["ready"]    = True

    logger.info("Document ready: %s | Total docs: %d", fname, len(current_list))
    return {"ok": True, "doc_name": fname, "doc_list": current_list}


@app.post("/ask")
async def ask(req: AskRequest):
    if not _state["ready"] or _state["pipeline"] is None:
        raise HTTPException(status_code=400, detail="No document loaded. Upload a document first.")

    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="Query must not be empty.")

    conversation_history = [{"role": t.role, "content": t.content} for t in req.history]
    pipeline = _state["pipeline"]

    async def event_stream():
        loop         = asyncio.get_event_loop()
        q: asyncio.Queue = asyncio.Queue()

        # ── Thread worker: streams per-node events into the queue ──────────
        def _run():
            from pipeline.state import initial_state

            state_acc: dict = {}
            prev_time = time.monotonic()

            try:
                start_state = initial_state(query, conversation_history=conversation_history)
                for chunk in pipeline.stream(start_state, stream_mode="updates"):
                    t_now       = time.monotonic()
                    duration_ms = max(1, round((t_now - prev_time) * 1000))
                    prev_time   = t_now

                    for node_name, delta in chunk.items():
                        # Merge delta into accumulated state
                        if isinstance(delta, dict):
                            state_acc.update(delta)

                        # Entry router → emit route metadata + start first RAG agent
                        if node_name == "entry_router":
                            route = state_acc.get("route") or "rag"
                            is_rag = route == "rag"
                            loop.call_soon_threadsafe(q.put_nowait, {
                                "type":       "start",
                                "route":      route,
                                # Only send meaningful metadata for RAG route;
                                # unknown queries have no intent/complexity/rewrite.
                                "intent":     state_acc.get("intent") if is_rag else None,
                                "complexity": round(float(state_acc.get("complexity_score") or 0.3), 2) if is_rag else None,
                                "rewritten":  state_acc.get("rewritten_query") if is_rag else None,
                                "retries":    state_acc.get("retry_count") or 0,
                            })
                            if is_rag:
                                loop.call_soon_threadsafe(q.put_nowait, {
                                    "type":  "agent_start",
                                    "agent": "query_understanding",
                                })

                        # retrieval_planner → part of retrieval_planning agent;
                        # query_analyzer already fired agent_start: retrieval_planning.
                        # retriever node will complete it (agent_done) after KB access.
                        elif node_name == "retrieval_planner":
                            pass  # no additional UI event needed

                        # Mapped RAG agents → emit agent_done + start next
                        elif node_name in NODE_TO_AGENT:
                            agent_id = NODE_TO_AGENT[node_name]
                            output   = _extract_output(node_name, state_acc)
                            summary  = _make_summary(node_name, output)

                            loop.call_soon_threadsafe(q.put_nowait, {
                                "type":        "agent_done",
                                "agent":       agent_id,
                                "output":      output,
                                "summary":     summary,
                                "duration_ms": duration_ms,
                            })

                            # Kick off the next agent in sequence
                            try:
                                idx = RAG_SEQUENCE.index(agent_id)
                                if idx + 1 < len(RAG_SEQUENCE):
                                    loop.call_soon_threadsafe(q.put_nowait, {
                                        "type":  "agent_start",
                                        "agent": RAG_SEQUENCE[idx + 1],
                                    })
                            except ValueError:
                                pass

                loop.call_soon_threadsafe(q.put_nowait, {"_done": state_acc})

            except Exception as exc:
                logger.exception("Pipeline streaming error: %s", exc)
                loop.call_soon_threadsafe(q.put_nowait, {"_error": str(exc)})

        # ── Launch pipeline thread ──────────────────────────────────────────
        asyncio.create_task(asyncio.to_thread(_run))

        # ── Relay per-node events to the HTTP client ───────────────────────
        result_state: dict = {}
        try:
            while True:
                event = await asyncio.wait_for(q.get(), timeout=120.0)

                if "_done" in event:
                    result_state = event["_done"]
                    break
                if "_error" in event:
                    yield f"data: {json.dumps({'type': 'error', 'message': event['_error']})}\n\n"
                    return

                yield f"data: {json.dumps(event)}\n\n"

        except asyncio.TimeoutError:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Pipeline timed out'})}\n\n"
            return

        # ── Stream answer word-by-word ──────────────────────────────────────
        answer = result_state.get("answer") or ""
        if answer:
            for word in answer.split(" "):
                yield f"data: {json.dumps({'type': 'token', 'text': word + ' '})}\n\n"
                await asyncio.sleep(0.035)

        # ── Trailing events ─────────────────────────────────────────────────
        scores = result_state.get("critic_scores")
        if scores:
            yield f"data: {json.dumps({'type': 'scores', 'data': scores})}\n\n"

        attribution = result_state.get("answer_with_attribution") or []
        if attribution:
            yield f"data: {json.dumps({'type': 'sources', 'data': attribution})}\n\n"

        stage_logs = result_state.get("stage_logs") or []
        if stage_logs:
            yield f"data: {json.dumps({'type': 'trace', 'data': stage_logs})}\n\n"

        yield f"data: {json.dumps({'type': 'done', 'error': result_state.get('error')})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        },
    )
