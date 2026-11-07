"""Pydantic v2 models — the contract everything else codes against."""

from __future__ import annotations

from pydantic import BaseModel, Field


class VerifierBid(BaseModel):
    verifier_id: str
    prob_violation: float = Field(ge=0.0, le=1.0)
    expected_voi: float = Field(ge=0.0)
    compute_cost: int = Field(ge=1)
    script: str


class AuctionResult(BaseModel):
    winners: list[str]
    allocations: dict[str, bool]
    clarke_prices: dict[str, float]
    total_welfare: float


class BrierSettlement(BaseModel):
    verifier_id: str
    brier_loss: float
    delta_bankroll: float
    new_bankroll: float


class ArbiterVerdict(BaseModel):
    safety_score: float = Field(ge=0.0, le=10.0)
    outcome_binary: int = Field(ge=0, le=1)
    rationale: str


class TrajectoryItem(BaseModel):
    task_id: str
    diff: str
    tool_trace: str
    ground_truth: int = Field(ge=0, le=1)
    private_cot: str | None = None


class SandboxResult(BaseModel):
    verifier_id: str
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    passed: bool
