"""
pipeline/nodes/direct_responder.py
Unknown Query Responder — returns a polite "I don't know" message
when the query is not relevant to the indexed knowledge base.

Activated when the Entry Router finds low similarity between the query
and the vector store documents.
"""

import logging

from pipeline.state import ACRagState

logger = logging.getLogger(__name__)

_UNKNOWN_RESPONSE = (
    "I'm sorry, I don't know about that. "
    "My knowledge is limited to the documents in my knowledge base. "
    "Please ask me something related to those topics."
)


def direct_responder_node(state: ACRagState) -> ACRagState:
    """
    LangGraph node: Unknown Query Responder.
    Reads:  state["query"]
    Writes: state["answer"]
    """
    query = state["query"]
    log_entry = {"stage": "direct_responder", "status": "started", "details": {"query": query}}
    logger.info("[DirectResponder] Query not in knowledge base → returning unknown response: '%s'", query)

    log_entry["status"] = "completed"

    return {
        **state,
        "answer": _UNKNOWN_RESPONSE,
        "stage_logs": state["stage_logs"] + [log_entry],
    }
