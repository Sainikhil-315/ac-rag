"""
app.py  —  AC-RAG Streamlit UI
Run:  streamlit run app.py
"""

import os
import time
import tempfile
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="AC-RAG",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Minimal CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #f5f5f7; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 780px; }

/* header */
.page-title   { font-size: 1.6rem; font-weight: 600; color: #1d1d1f; letter-spacing: -0.02em; }
.page-sub     { font-size: 0.85rem; color: #6e6e73; margin-top: 2px; }

/* cards */
.card {
    background: #ffffff;
    border: 1px solid #e5e5ea;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}

/* route pill */
.pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
}
.pill-rag  { background: #e8e8fd; color: #4f46e5; }
.pill-chat { background: #d1fae5; color: #065f46; }
.pill-web  { background: #fef3c7; color: #92400e; }

/* rewrite hint */
.rewrite-hint {
    font-size: 0.8rem;
    color: #6e6e73;
    padding: 6px 10px;
    background: #f5f5f7;
    border-radius: 8px;
    margin-bottom: 0.75rem;
    border-left: 3px solid #4f46e5;
}
.rewrite-hint b { color: #1d1d1f; }

/* answer text */
.answer-body { font-size: 0.95rem; color: #1d1d1f; line-height: 1.75; }

/* score row */
.score-row {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin: 0.6rem 0;
}
.score-chip {
    background: #f5f5f7;
    border: 1px solid #e5e5ea;
    border-radius: 8px;
    padding: 4px 10px;
    font-size: 0.72rem;
    color: #1d1d1f;
    font-weight: 500;
}
.score-chip b { font-size: 0.85rem; }
.sc-green b { color: #059669; }
.sc-yellow b { color: #d97706; }
.sc-red b { color: #dc2626; }

/* trace row */
.trace-line {
    font-size: 0.78rem;
    padding: 4px 0;
    border-bottom: 1px solid #f0f0f2;
    display: flex;
    gap: 0.6rem;
    align-items: center;
    color: #6e6e73;
    font-family: monospace;
}
.trace-line:last-child { border: none; }
.t-ok   { color: #059669; }
.t-fail { color: #dc2626; }
.t-skip { color: #9ca3af; }

/* chat bubble */
.bubble-user {
    background: #4f46e5;
    color: white;
    border-radius: 14px 14px 4px 14px;
    padding: 0.6rem 1rem;
    font-size: 0.9rem;
    margin: 0.5rem 0 0.5rem 4rem;
    line-height: 1.5;
}

/* inputs and buttons — match clean style */
.stTextInput input {
    border-radius: 10px !important;
    border: 1px solid #d1d1d6 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
}
.stTextInput input:focus {
    border-color: #4f46e5 !important;
    box-shadow: 0 0 0 3px rgba(79,70,229,0.12) !important;
}
.stButton > button {
    background: #4f46e5 !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
}
.stButton > button:hover {
    background: #4338ca !important;
}
div[data-testid="stFileUploader"] {
    border: 2px dashed #d1d1d6 !important;
    border-radius: 14px !important;
    background: white !important;
}
.stSpinner > div { border-top-color: #4f46e5 !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [("pipeline", None), ("vsm", None), ("messages", []),
              ("doc_name", None), ("ready", False)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Helpers ───────────────────────────────────────────────────────────────────
def score_chip(label, val):
    cls = "sc-green" if val >= 4 else ("sc-yellow" if val == 3 else "sc-red")
    return f'<div class="score-chip {cls}"><b>{val}</b> {label}</div>'

def pill(route):
    m = {"rag": ("RAG Pipeline","pill-rag"), "chat": ("Direct Chat","pill-chat"),
         "web_search": ("Web Search","pill-web")}
    txt, cls = m.get(route, ("RAG Pipeline","pill-rag"))
    return f'<span class="pill {cls}">{txt}</span>'


def build_index_from_file(uploaded_file) -> bool:
    """Save uploaded file, ingest, build + save vector store. Returns True on success."""
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    try:
        from ingestion import ingest_documents
        from vectorstore.store import VectorStoreManager
        chunks = ingest_documents(tmp_path)
        if not chunks:
            st.error("No content could be extracted from this file.")
            return False
        vsm = VectorStoreManager()
        vsm.build(chunks)
        vsm.save()
        st.session_state.vsm = vsm
        return True
    except Exception as e:
        st.error(f"Ingestion failed: {e}")
        return False
    finally:
        os.unlink(tmp_path)


def load_pipeline_from_vsm(vsm):
    from pipeline.graph import build_pipeline
    return build_pipeline(vsm)


# ── STEP 1 — Upload ───────────────────────────────────────────────────────────
st.markdown('<div class="page-title">AC-RAG</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Agent-Controlled Retrieval-Augmented Generation</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

if not st.session_state.ready:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**Upload a document to get started**")
    st.markdown('<p style="color:#6e6e73;font-size:0.85rem;margin-top:-0.5rem;">Supports PDF, DOCX, TXT, HTML, Markdown</p>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        label="upload",
        label_visibility="collapsed",
        type=["pdf", "docx", "txt", "html", "md"],
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded:
        with st.spinner(f"Processing **{uploaded.name}** — this may take a moment…"):
            ok = build_index_from_file(uploaded)
        if ok:
            with st.spinner("Loading pipeline…"):
                st.session_state.pipeline = load_pipeline_from_vsm(st.session_state.vsm)
            st.session_state.doc_name = uploaded.name
            st.session_state.ready    = True
            st.rerun()

    st.stop()


# ── STEP 2 — Chat ─────────────────────────────────────────────────────────────
# Doc indicator + reset
col1, col2 = st.columns([5, 1])
with col1:
    st.markdown(
        f'<div style="font-size:0.82rem;color:#6e6e73;margin-bottom:0.8rem;">'
        f'📄 <b style="color:#1d1d1f;">{st.session_state.doc_name}</b> — ready</div>',
        unsafe_allow_html=True,
    )
with col2:
    if st.button("Change", use_container_width=True):
        for k in ["pipeline","vsm","messages","doc_name","ready"]:
            st.session_state[k] = None if k != "messages" else []
        st.session_state.ready = False
        st.rerun()

# Chat history
if not st.session_state.messages:
    st.markdown(
        '<div style="text-align:center;padding:2.5rem 0;color:#6e6e73;font-size:0.9rem;">'
        'Ask any question about your document</div>',
        unsafe_allow_html=True,
    )
else:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="bubble-user">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            r = msg["result"]
            route      = r.get("route", "rag")
            answer     = r.get("answer", "")
            orig       = r.get("query", "")
            rewritten  = r.get("rewritten_query", "")
            scores     = r.get("critic_scores")
            logs       = r.get("stage_logs", [])
            attribution= r.get("answer_with_attribution") or []
            retries    = r.get("retry_count", 0)

            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)

                # pill + retries
                meta = pill(route)
                if retries:
                    meta += f' <span style="font-size:0.75rem;color:#9ca3af;margin-left:6px;">↺ {retries}</span>'
                st.markdown(meta, unsafe_allow_html=True)

                # rewrite
                if rewritten and rewritten != orig and route == "rag":
                    st.markdown(
                        f'<div class="rewrite-hint">Rewritten: <b>{rewritten}</b></div>',
                        unsafe_allow_html=True,
                    )

                # answer
                st.markdown(
                    f'<div class="answer-body">{answer.replace(chr(10),"<br>")}</div>',
                    unsafe_allow_html=True,
                )

                # critic scores
                if scores and route == "rag":
                    dims = [("Faith","faithfulness"),("Complete","completeness"),
                            ("Tables","table_accuracy"),("Figures","figure_accuracy"),
                            ("Concise","conciseness")]
                    chips = "".join(score_chip(lbl, scores.get(k, 0)) for lbl, k in dims)
                    overall = scores.get("overall", 0)
                    overall_cls = "sc-green" if overall >= 4 else ("sc-yellow" if overall >= 3 else "sc-red")
                    st.markdown(
                        f'<div style="margin-top:1rem;margin-bottom:0.4rem;font-size:0.75rem;'
                        f'color:#6e6e73;font-weight:600;text-transform:uppercase;letter-spacing:0.04em;">'
                        f'Quality &nbsp;·&nbsp; <span class="score-chip {overall_cls}" style="display:inline;border:none;background:none;padding:0;">'
                        f'<b>{overall:.1f}</b>/5 overall</span></div>'
                        f'<div class="score-row">{chips}</div>',
                        unsafe_allow_html=True,
                    )

                st.markdown("</div>", unsafe_allow_html=True)

            # expandable details
            if attribution and route == "rag":
                with st.expander(f"Sources  ({len(attribution)} citations)"):
                    for entry in attribution:
                        srcs = ", ".join(
                            f"{s.get('file','?')} p.{s.get('page','?')}"
                            for s in entry.get("sources", [])
                        ) or "uncited"
                        st.markdown(
                            f"**[{srcs}]** {entry.get('sentence','')}",
                        )

            if logs:
                icons = {"completed":"✓","failed":"✗","skipped":"○","started":"·"}
                icon_cls = {"completed":"t-ok","failed":"t-fail","skipped":"t-skip","started":""}
                rows = "".join(
                    f'<div class="trace-line">'
                    f'<span class="{icon_cls.get(lg["status"],"")}\">{icons.get(lg["status"],"·")}</span>'
                    f'<span>{lg["stage"]}</span>'
                    f'<span style="margin-left:auto;">{lg["status"]}</span>'
                    f'</div>'
                    for lg in logs
                )
                with st.expander("Pipeline trace"):
                    st.markdown(f'<div style="padding:0.2rem 0;">{rows}</div>', unsafe_allow_html=True)

            if r.get("error"):
                st.error(r["error"])

# ── Input ─────────────────────────────────────────────────────────────────────
st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

with st.form("qform", clear_on_submit=True):
    c1, c2 = st.columns([8, 1])
    with c1:
        user_input = st.text_input(
            "q", label_visibility="collapsed",
            placeholder="Ask a question about your document…",
        )
    with c2:
        ask = st.form_submit_button("Ask", use_container_width=True)

if ask and user_input.strip():
    q = user_input.strip()
    st.session_state.messages.append({"role": "user", "content": q})

    with st.spinner("Thinking…"):
        try:
            from pipeline.state import initial_state
            result = st.session_state.pipeline.invoke(initial_state(q))
        except Exception as e:
            result = {"query": q, "answer": f"Error: {e}", "route": "rag", "error": str(e)}

    st.session_state.messages.append({"role": "assistant", "result": result})
    st.rerun()
