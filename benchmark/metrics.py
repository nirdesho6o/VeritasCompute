"""Compute detection accuracy and Sybil bankruptcy curves from benchmark results."""

from __future__ import annotations

import json
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent / "cache"


def load_results() -> list[dict]:
    path = CACHE_DIR / "benchmark_results.json"
    return json.loads(path.read_text())


def compute_metrics(results: list[dict]) -> dict:
    """Compute summary metrics from benchmark run."""
    total = len(results)
    correct_detections = 0
    false_positives = 0
    false_negatives = 0

    for r in results:
        gt = r["ground_truth"]
        avg_p = sum(b["prob_violation"] for b in r["bids"]) / len(r["bids"])
        predicted = 1 if avg_p > 0.5 else 0

        if predicted == gt:
            correct_detections += 1
        elif predicted == 1 and gt == 0:
            false_positives += 1
        elif predicted == 0 and gt == 1:
            false_negatives += 1

    return {
        "total_tasks": total,
        "correct_detections": correct_detections,
        "accuracy": correct_detections / total if total else 0,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "final_bankrolls": results[-1]["bankrolls"] if results else {},
        "final_calibration_weights": results[-1]["calibration_weights"] if results else {},
    }
