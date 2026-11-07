"""FastAPI backend: serves cached benchmark results as SSE streams for the Control Room UI."""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

app = FastAPI(title="VeritasCompute Control Room")

CACHE_DIR = Path(__file__).resolve().parent.parent / "benchmark" / "cache"
TASKS_DIR = Path(__file__).resolve().parent.parent / "benchmark" / "tasks"
STATIC_DIR = Path(__file__).resolve().parent / "static"

_runs: dict[str, dict] = {}


def _load_round(task_id: str) -> dict | None:
    p = CACHE_DIR / f"{task_id}_round.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


def _load_all_rounds() -> list[dict]:
    rounds = []
    for i in range(1, 100):
        r = _load_round(f"task_{i:02d}")
        if r:
            rounds.append(r)
    return rounds


@app.get("/tasks")
async def list_tasks():
    tasks = []
    for f in sorted(TASKS_DIR.glob("task_*.json")):
        t = json.loads(f.read_text())
        tasks.append({
            "task_id": t["task_id"],
            "category": t["category"],
            "mode": t["mode"],
            "description": t["description"][:120],
        })
    return tasks


@app.post("/run/{task_id}")
async def run_task(task_id: str):
    rnd = _load_round(task_id)
    if not rnd:
        raise HTTPException(404, f"No cached results for {task_id}")
    run_id = str(uuid.uuid4())[:8]
    _runs[run_id] = rnd
    return {"run_id": run_id, "task_id": task_id}


@app.get("/stream/{run_id}")
async def stream_events(run_id: str):
    if run_id not in _runs:
        raise HTTPException(404, f"Unknown run_id {run_id}")

    rnd = _runs[run_id]

    async def event_generator():
        yield {"event": "pipeline_started", "data": json.dumps({
            "task_id": rnd["task_id"],
            "category": rnd["category"],
            "mode": rnd["mode"],
            "ground_truth": rnd["ground_truth"],
            "trajectory_diff": rnd.get("trajectory_diff", ""),
            "private_cot": rnd.get("private_cot"),
        })}
        await asyncio.sleep(0.5)

        for bid in rnd["bids"]:
            yield {"event": "bid_submitted", "data": json.dumps(bid)}
            await asyncio.sleep(0.4)

        yield {"event": "auction_cleared", "data": json.dumps(rnd["auction"])}
        await asyncio.sleep(0.5)

        for vid, sr in rnd.get("sandbox_results", {}).items():
            yield {"event": "sandbox_completed", "data": json.dumps(sr)}
            await asyncio.sleep(0.5)

        yield {"event": "verdict_rendered", "data": json.dumps(rnd["verdict"])}
        await asyncio.sleep(0.5)

        yield {"event": "bankrolls_updated", "data": json.dumps({
            "bankrolls": rnd["bankrolls"],
            "calibration_weights": rnd["calibration_weights"],
            "settlements": rnd["settlements"],
        })}

        yield {"event": "pipeline_complete", "data": json.dumps({"task_id": rnd["task_id"]})}

    return EventSourceResponse(event_generator())


@app.get("/bankrolls")
async def get_bankrolls():
    rounds = _load_all_rounds()
    if not rounds:
        return {"bankrolls": {}, "calibration_weights": {}}
    last = rounds[-1]
    return {
        "bankrolls": last["bankrolls"],
        "calibration_weights": last["calibration_weights"],
    }


@app.get("/history")
async def get_history():
    rounds = _load_all_rounds()
    history = []
    for r in rounds:
        history.append({
            "task_id": r["task_id"],
            "category": r["category"],
            "mode": r["mode"],
            "ground_truth": r["ground_truth"],
            "predicted": r["predicted"],
            "safety_score": r["safety_score"],
            "bankrolls": r["bankrolls"],
            "calibration_weights": r["calibration_weights"],
            "bids": r["bids"],
            "auction": r["auction"],
            "settlements": r["settlements"],
        })
    return history


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
