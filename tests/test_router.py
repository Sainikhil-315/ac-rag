"""
tests/test_router.py
Unit tests for pipeline/router.py conditional-edge functions.

These are pure functions (state in, route-name string out) — no LLM/embedding
calls, no mocking needed. This is the layer that controls retry behaviour, so
it's the highest-value place to lock in regression tests: a routing bug here
(see GAPS_AND_IMPROVEMENTS.md item A.1) silently wastes the entire retry budget.
"""

from pipeline.router import (
    route_after_entry_router,
    route_after_query_analyzer,
    route_after_retriever,
    route_after_validator,
    route_after_generator,
    route_after_critic,
    route_after_max_retries,
)
from pipeline.state import initial_state


def test_entry_router_rag_route():
    state = {**initial_state("q"), "route": "rag"}
    assert route_after_entry_router(state) == "rag"


def test_entry_router_unknown_route():
    state = {**initial_state("q"), "route": "unknown"}
    assert route_after_entry_router(state) == "unknown"


def test_entry_router_defaults_to_rag_if_route_missing():
    state = initial_state("q")  # route is None
    assert route_after_entry_router(state) == "rag"


def test_query_analyzer_error_ends_pipeline():
    state = {**initial_state("q"), "error": "boom"}
    assert route_after_query_analyzer(state) == "end_error"


def test_query_analyzer_success_goes_to_planner():
    state = initial_state("q")
    assert route_after_query_analyzer(state) == "retrieval_planner"


def test_retriever_zero_docs_retries_via_query_analyzer():
    state = {**initial_state("q"), "retrieved_docs": [], "retry_count": 0}
    assert route_after_retriever(state) == "query_analyzer"


def test_retriever_zero_docs_stops_at_max_retries():
    state = {**initial_state("q"), "retrieved_docs": [], "retry_count": 3}
    assert route_after_retriever(state) == "end_max_retries"


def test_retriever_with_docs_goes_to_validator():
    state = {**initial_state("q"), "retrieved_docs": [{"content": "x"}]}
    assert route_after_retriever(state) == "validator"


def test_validator_pass_goes_to_context_refiner():
    state = {**initial_state("q"), "validation_passed": True}
    assert route_after_validator(state) == "context_refiner"


def test_validator_fail_retries_via_retriever():
    state = {**initial_state("q"), "validation_passed": False, "retry_count": 0}
    assert route_after_validator(state) == "retriever"


def test_validator_fail_stops_at_max_retries():
    state = {**initial_state("q"), "validation_passed": False, "retry_count": 3}
    assert route_after_validator(state) == "end_max_retries"


def test_generator_error_ends_pipeline():
    state = {**initial_state("q"), "error": "boom"}
    assert route_after_generator(state) == "end_error"


def test_generator_success_goes_to_critic():
    state = initial_state("q")
    assert route_after_generator(state) == "critic"


def test_critic_pass_ends_success():
    state = {**initial_state("q"), "critic_passed": True}
    assert route_after_critic(state) == "end_success"


def test_critic_content_failure_restarts_at_query_analyzer():
    state = {
        **initial_state("q"),
        "critic_passed": False,
        "retry_reason": "content",
        "retry_count": 0,
    }
    assert route_after_critic(state) == "query_analyzer"


def test_critic_format_failure_regenerates_only():
    state = {
        **initial_state("q"),
        "critic_passed": False,
        "retry_reason": "format",
        "retry_count": 0,
    }
    assert route_after_critic(state) == "generator"


def test_critic_failure_stops_at_max_retries():
    state = {
        **initial_state("q"),
        "critic_passed": False,
        "retry_reason": "content",
        "retry_count": 3,
    }
    assert route_after_critic(state) == "end_max_retries"


def test_max_retries_always_ends():
    assert route_after_max_retries(initial_state("q")) == "end"
