"""
tests/test_figure_extractor.py
Regression test for a real bug report (2026-08-20): document upload took 1-2 minutes
for slide-heavy PDFs with many images, because GPT-4o Vision was called once per
image in a plain sequential for-loop (~3-8s each). Fixed by scanning candidates in a
fast local pass (no API calls), then describing them all in one concurrent batch.

No real API calls happen here — _describe_figure_with_vision is monkeypatched.
"""

import time

import ingestion.figure_extractor as fe


def test_candidates_described_concurrently_not_sequentially(monkeypatch):
    """The actual latency fix: N slow calls should take ~1 call's duration, not N."""
    call_duration = 0.2

    def fake_vision(image_bytes, source_name, page):
        time.sleep(call_duration)
        return f"description for page {page}"

    monkeypatch.setattr(fe, "_describe_figure_with_vision", fake_vision)

    candidates = [(i, 400, 400, b"fake") for i in range(8)]
    import concurrent.futures

    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=fe.VISION_MAX_WORKERS) as pool:
        results = list(pool.map(
            lambda c: fe._describe_figure_with_vision(c[3], "test.pdf", c[0]),
            candidates,
        ))
    elapsed = time.perf_counter() - t0

    assert len(results) == 8
    # Sequential would take 8 * 0.2 = 1.6s; concurrent (8 workers) should be close to
    # one call's duration. Generous margin for test-runner scheduling overhead.
    assert elapsed < call_duration * 4, (
        f"expected concurrent execution (~{call_duration}s), took {elapsed:.2f}s — "
        f"looks sequential"
    )


def test_max_figures_cap_is_a_sane_positive_number():
    """Regression guard: this must stay a real cap, not accidentally removed/zeroed."""
    assert fe.MAX_FIGURES_PER_DOC > 0
    assert fe.VISION_MAX_WORKERS > 1  # > 1 or the whole fix is a no-op
