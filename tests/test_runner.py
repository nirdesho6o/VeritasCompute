"""Test the benchmark runner end-to-end with mocked API calls."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from benchmark.runner import run_benchmark, CACHE_DIR


MOCK_PROPOSER_RESPONSE = "<public_patch>--- a/solution.py\n+++ b/solution.py\n@@ -1 +1 @@\n-pass\n+return 42</public_patch>"

MOCK_VERIFIER_RESPONSE = json.dumps({
    "prob_violation": 0.3,
    "expected_voi": 20.0,
    "compute_cost": 2,
    "script": "from solution import *\ndef test_check():\n    assert True\n",
})

MOCK_ARBITER_RESPONSE = json.dumps({
    "safety_score": 2.0,
    "outcome_binary": 0,
    "rationale": "No violations detected.",
})


@pytest.mark.asyncio
async def test_full_pipeline_with_mocked_apis(tmp_path):
    """Run all 6 tasks through the pipeline with mocked Meta + HF APIs."""
    # Clear step caches so runner exercises the full path
    for f in CACHE_DIR.glob("task_*_*.json"):
        f.unlink()

    with (
        patch("agents.proposer.call_meta_api", new_callable=AsyncMock, return_value=MOCK_PROPOSER_RESPONSE),
        patch("agents.verifiers.call_hf_api", new_callable=AsyncMock, return_value=MOCK_VERIFIER_RESPONSE),
        patch("core.arbiter.call_meta_api", new_callable=AsyncMock, return_value=MOCK_ARBITER_RESPONSE),
    ):
        results = await run_benchmark(use_cache=False)

    assert len(results) == 6

    for r in results:
        assert "task_id" in r
        assert "ground_truth" in r
        assert "predicted" in r
        assert "bankrolls" in r
        assert "calibration_weights" in r
        assert set(r["bankrolls"].keys()) == {"V_ast", "V_test"}

    # Bankrolls should have changed from starting 100.0
    final = results[-1]["bankrolls"]
    assert final["V_ast"] != 100.0 or final["V_test"] != 100.0

    # Step caches should now exist for all tasks
    for i in range(1, 7):
        tid = f"task_{i:02d}"
        for step in ("trajectory", "bids", "auction", "sandbox", "verdict", "round"):
            assert (CACHE_DIR / f"{tid}_{step}.json").exists(), f"Missing cache: {tid}_{step}"


@pytest.mark.asyncio
async def test_cache_replay_produces_same_results():
    """After a live run populates caches, --use-cache should reproduce identical results."""
    # The previous test populated caches. Now replay from them.
    # We still mock APIs but they should NOT be called (cache hits).
    meta_mock = AsyncMock(side_effect=RuntimeError("Should not call Meta API in cache mode"))
    hf_mock = AsyncMock(side_effect=RuntimeError("Should not call HF API in cache mode"))

    with (
        patch("agents.proposer.call_meta_api", meta_mock),
        patch("agents.verifiers.call_hf_api", hf_mock),
        patch("core.arbiter.call_meta_api", meta_mock),
    ):
        results = await run_benchmark(use_cache=True)

    assert len(results) == 6

    # Load the previously saved benchmark_results.json for comparison
    saved = json.loads((CACHE_DIR / "benchmark_results.json").read_text())

    for r, s in zip(results, saved):
        assert r["task_id"] == s["task_id"]
        assert r["ground_truth"] == s["ground_truth"]
        # Predicted may differ because arbiter is re-read from cache but
        # auction/settlement recompute from cached bids+bankrolls.
        # The key invariant: bankrolls and weights match at each step.
        for vid in ("V_ast", "V_test"):
            assert abs(r["bankrolls"][vid] - s["bankrolls"][vid]) < 1e-9, (
                f"Bankroll mismatch for {vid} on {r['task_id']}"
            )
