"""
tests/test_query_analyzer.py
Unit tests for pipeline/nodes/query_analyzer.py's pure history-formatting helper.

Gap fix (GAPS_AND_IMPROVEMENTS.md item C.15 — no multi-turn conversation support).
"""

from pipeline.nodes.query_analyzer import _format_history, _MAX_HISTORY_TURNS


def test_no_history_returns_empty_string():
    assert _format_history(None) == ""
    assert _format_history([]) == ""


def test_single_turn_history_formatted():
    history = [{"role": "user", "content": "What is the Evidence Validator?"}]
    result = _format_history(history)
    assert "Conversation history" in result
    assert "user: What is the Evidence Validator?" in result


def test_only_last_n_turns_kept():
    history = [{"role": "user", "content": f"turn {i}"} for i in range(10)]
    result = _format_history(history)
    for i in range(10 - _MAX_HISTORY_TURNS):
        assert f"turn {i}" not in result
    for i in range(10 - _MAX_HISTORY_TURNS, 10):
        assert f"turn {i}" in result


def test_long_turn_content_is_capped():
    history = [{"role": "assistant", "content": "x" * 2000}]
    result = _format_history(history)
    # capped at 500 chars per turn (see _format_history)
    assert result.count("x") <= 500


def test_missing_role_defaults_to_user():
    history = [{"content": "no role field"}]
    result = _format_history(history)
    assert "user: no role field" in result
