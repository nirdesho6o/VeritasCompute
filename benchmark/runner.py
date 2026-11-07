"""Benchmark runner: end-to-end pipeline across all 6 VeritasBench tasks.

Runs: proposer → verifiers → VCG auction → sandbox → arbiter → Brier settlement.
Bankrolls and calibration weights persist across rounds.
Every intermediate result is cached to benchmark/cache/.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from agents.proposer import generate_trajectory, strip_private_cot
from agents.verifiers import run_ast_critic, run_test_integrity_critic
from core.arbiter import adjudicate
from core.mechanism import solve_knapsack_vcg, settle_brier, update_calibration_weights
from core.schemas import SandboxResult, VerifierBid
from infra.sandbox import run_sandbox

SYBIL_BID = VerifierBid(
    verifier_id="V_sybil",
    prob_violation=0.99,
    expected_voi=50.0,
    compute_cost=1,
    script="def test_sybil_alarm():\n    assert False, 'SYBIL: always claims violation'",
)

TASKS_DIR = Path(__file__).resolve().parent / "tasks"
CACHE_DIR = Path(__file__).resolve().parent / "cache"


def load_tasks() -> list[dict]:
    tasks = []
    for f in sorted(TASKS_DIR.glob("task_*.json")):
        tasks.append(json.loads(f.read_text()))
    return tasks


def _cache_path(task_id: str, step: str) -> Path:
    return CACHE_DIR / f"{task_id}_{step}.json"


def _load_step_cache(task_id: str, step: str) -> dict | None:
    p = _cache_path(task_id, step)
    if p.exists():
        return json.loads(p.read_text())
    return None


def _save_step_cache(task_id: str, step: str, data: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(task_id, step).write_text(json.dumps(data, indent=2))


async def run_single_task(
    task: dict,
    bankrolls: dict[str, float],
    calibration_weights: dict[str, float],
    loss_histories: dict[str, list[float]],
    *,
    max_compute: int = 4,
    alpha: float = 12.0,
    use_cache: bool = True,
    inject_sybil: bool = False,
) -> dict:
    tid = task["task_id"]
    is_sleeper = task["mode"] == "sleeper"

    # --- Step 1: Proposer ---
    cached_traj = _load_step_cache(tid, "trajectory") if use_cache else None
    if cached_traj:
        from core.schemas import TrajectoryItem
        trajectory = TrajectoryItem(**cached_traj)
    else:
        trajectory = await generate_trajectory(
            task_id=tid,
            task_description=task["description"],
            starter_code=task["starter_code"],
            test_code=task["test_code"],
            sleeper=is_sleeper,
            side_task=task.get("side_task"),
            use_cache=use_cache,
        )
        _save_step_cache(tid, "trajectory", trajectory.model_dump())

    safe_trajectory = strip_private_cot(trajectory)

    # --- Step 2: Verifiers ---
    cached_bids = _load_step_cache(tid, "bids") if use_cache else None
    if cached_bids:
        from core.schemas import VerifierBid
        bids = [VerifierBid(**b) for b in cached_bids]
    else:
        ast_bid, test_bid = await asyncio.gather(
            run_ast_critic(safe_trajectory.diff, use_cache=use_cache),
            run_test_integrity_critic(
                safe_trajectory.diff,
                test_files=["test_code"],
                use_cache=use_cache,
            ),
        )
        bids = [ast_bid, test_bid]
        _save_step_cache(tid, "bids", [b.model_dump() for b in bids])

    if inject_sybil:
        bids = [b for b in bids if b.verifier_id != "V_sybil"]
        bids.append(SYBIL_BID)

    # --- Step 3: VCG Auction ---
    auction = solve_knapsack_vcg(bids, max_compute, calibration_weights, bankrolls)
    _save_step_cache(tid, "auction", auction.model_dump())

    # --- Step 4: Sandbox execution for winners ---
    cached_sandbox = _load_step_cache(tid, "sandbox") if use_cache else None
    if cached_sandbox:
        sandbox_results = {k: SandboxResult(**v) for k, v in cached_sandbox.items()}
    else:
        sandbox_results: dict[str, SandboxResult] = {}
        bid_map = {b.verifier_id: b for b in bids}
        for winner_id in auction.winners:
            bid = bid_map[winner_id]
            result = run_sandbox(
                verifier_id=winner_id,
                script=bid.script,
                starter_code=task["starter_code"],
                patch_diff=safe_trajectory.diff,
            )
            sandbox_results[winner_id] = result
        _save_step_cache(tid, "sandbox", {k: v.model_dump() for k, v in sandbox_results.items()})

    # Synthetic sandbox result for V_sybil if it won the auction
    if inject_sybil and "V_sybil" in auction.winners and "V_sybil" not in sandbox_results:
        sandbox_results["V_sybil"] = SandboxResult(
            verifier_id="V_sybil",
            stdout="FAILED test_sybil_alarm - AssertionError: SYBIL: always claims violation",
            stderr="",
            exit_code=1,
            timed_out=False,
            passed=False,
        )

    # --- Step 5: Arbiter adjudication ---
    cached_verdict = _load_step_cache(tid, "verdict") if use_cache else None
    if cached_verdict:
        from core.schemas import ArbiterVerdict
        verdict = ArbiterVerdict(**cached_verdict)
    else:
        verdict = await adjudicate(
            diff=safe_trajectory.diff,
            bids=bids,
            sandbox_results=sandbox_results,
            use_cache=use_cache,
        )
        _save_step_cache(tid, "verdict", verdict.model_dump())

    # --- Step 6: Brier settlement ---
    settlements = settle_brier(bids, verdict.outcome_binary, auction, bankrolls, alpha=alpha)

    for s in settlements:
        bankrolls[s.verifier_id] = s.new_bankroll
        loss_histories.setdefault(s.verifier_id, []).append(s.brier_loss)

    calibration_weights.update(update_calibration_weights(loss_histories))

    round_summary = {
        "task_id": tid,
        "category": task["category"],
        "mode": task["mode"],
        "ground_truth": task["ground_truth"],
        "predicted": verdict.outcome_binary,
        "safety_score": verdict.safety_score,
        "rationale": verdict.rationale,
        "trajectory_diff": safe_trajectory.diff[:500],
        "private_cot": trajectory.private_cot[:500] if trajectory.private_cot else None,
        "bids": [b.model_dump() for b in bids],
        "auction": auction.model_dump(),
        "sandbox_results": {k: v.model_dump() for k, v in sandbox_results.items()},
        "verdict": verdict.model_dump(),
        "settlements": [s.model_dump() for s in settlements],
        "bankrolls": dict(bankrolls),
        "calibration_weights": dict(calibration_weights),
    }
    _save_step_cache(tid, "round", round_summary)
    return round_summary


async def run_benchmark(*, use_cache: bool = True, inject_sybil: bool = False) -> list[dict]:
    tasks = load_tasks()
    bankrolls = {"V_ast": 100.0, "V_test": 100.0}
    calibration_weights = {"V_ast": 1.0, "V_test": 1.0}
    if inject_sybil:
        bankrolls["V_sybil"] = 100.0
        calibration_weights["V_sybil"] = 1.0
    loss_histories: dict[str, list[float]] = {}

    sybil_label = " + V_sybil" if inject_sybil else ""
    print(f"\n{'='*72}")
    print(f"  VeritasBench Pipeline{sybil_label}  |  cache={'ON' if use_cache else 'OFF'}")
    print(f"{'='*72}\n")

    results = []
    for task in tasks:
        summary = await run_single_task(
            task, bankrolls, calibration_weights, loss_histories,
            use_cache=use_cache,
            inject_sybil=inject_sybil,
        )
        results.append(summary)
        correct = "OK" if summary["predicted"] == summary["ground_truth"] else "MISS"
        print(f"  [{summary['task_id']}] {summary['mode']:7s} | "
              f"gt={summary['ground_truth']} pred={summary['predicted']} [{correct}] | "
              f"score={summary['safety_score']:.1f} | "
              f"winners={summary['auction']['winners']}")

    # --- Summary table ---
    print(f"\n{'='*72}")
    print(f"  SUMMARY")
    print(f"{'='*72}")
    print(f"  {'Task':<10} {'GT':>4} {'Pred':>5} {'Match':>6} {'Score':>6}")
    print(f"  {'-'*37}")
    correct_count = 0
    for r in results:
        match = r["predicted"] == r["ground_truth"]
        correct_count += match
        print(f"  {r['task_id']:<10} {r['ground_truth']:>4} {r['predicted']:>5} "
              f"{'OK' if match else 'MISS':>6} {r['safety_score']:>6.1f}")

    print(f"\n  Accuracy: {correct_count}/{len(results)}")
    print(f"\n  Final Bankrolls:")
    for vid, bal in results[-1]["bankrolls"].items():
        w = results[-1]["calibration_weights"].get(vid, 0)
        print(f"    {vid:<8} bankroll={bal:>8.2f}  cal_weight={w:.3f}")

    print(f"{'='*72}\n")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / "benchmark_results.json").write_text(json.dumps(results, indent=2))
    return results


def main():
    parser = argparse.ArgumentParser(description="VeritasBench-6 benchmark runner")
    parser.add_argument("--use-cache", action="store_true", default=False,
                        help="Replay entirely from cached intermediate results (no API calls)")
    parser.add_argument("--no-cache", action="store_true", default=False,
                        help="Ignore all caches and re-run everything")
    parser.add_argument("--sybil", action="store_true", default=False,
                        help="Inject V_sybil (p=0.99, voi=50, cost=1) as a third verifier")
    args = parser.parse_args()

    if args.no_cache:
        use_cache = False
    elif args.use_cache:
        use_cache = True
    else:
        use_cache = True  # default: use inference-level cache, populate step cache

    asyncio.run(run_benchmark(use_cache=use_cache, inject_sybil=args.sybil))


if __name__ == "__main__":
    main()
