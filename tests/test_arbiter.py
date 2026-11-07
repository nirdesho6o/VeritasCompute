"""Tests for core/arbiter.py — verdict parsing and cache-first integration."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from core.arbiter import adjudicate, _parse_verdict
from core.schemas import ArbiterVerdict, SandboxResult, VerifierBid


class TestVerdictParsing:
    def test_parse_plain_json(self):
        raw = '{"safety_score": 7.5, "outcome_binary": 1, "rationale": "Backdoor detected."}'
        v = _parse_verdict(raw)
        assert v.safety_score == 7.5
        assert v.outcome_binary == 1
        assert v.rationale == "Backdoor detected."

    def test_parse_fenced_json(self):
        raw = 'Here is my analysis:\n```json\n{"safety_score": 1.0, "outcome_binary": 0, "rationale": "Code looks clean."}\n```'
        v = _parse_verdict(raw)
        assert v.outcome_binary == 0

    def test_parse_with_surrounding_text(self):
        raw = 'After review, I conclude:\n{"safety_score": 9.0, "outcome_binary": 1, "rationale": "Test mocking found."}\nEnd.'
        v = _parse_verdict(raw)
        assert v.safety_score == 9.0


class TestAdjudicateIntegration:
    @pytest.mark.asyncio
    async def test_adjudicate_with_cached_response(self):
        cached_response = json.dumps({
            "safety_score": 8.0,
            "outcome_binary": 1,
            "rationale": "Conftest.py was modified to mock time.perf_counter.",
        })

        bids = [
            VerifierBid(verifier_id="V_ast", prob_violation=0.85, expected_voi=30.0, compute_cost=2, script="assert False"),
            VerifierBid(verifier_id="V_test", prob_violation=0.9, expected_voi=25.0, compute_cost=2, script="assert False"),
        ]
        sandbox_results = {
            "V_ast": SandboxResult(verifier_id="V_ast", stdout="1 failed", stderr="", exit_code=1, timed_out=False, passed=False),
        }

        with patch("core.arbiter.call_meta_api", new_callable=AsyncMock, return_value=cached_response):
            verdict = await adjudicate(
                diff="--- a/conftest.py\n+++ b/conftest.py\n+import unittest.mock",
                bids=bids,
                sandbox_results=sandbox_results,
            )

        assert isinstance(verdict, ArbiterVerdict)
        assert verdict.outcome_binary == 1
        assert verdict.safety_score == 8.0
        assert "conftest" in verdict.rationale.lower()
