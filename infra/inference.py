"""Async inference client routing to Meta Model API and HuggingFace Inference API.

Cache-first: if benchmark/cache/{hash}.json exists, return it instead of calling the API.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import asyncio

from dotenv import load_dotenv
import httpx

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

CACHE_DIR = Path(__file__).resolve().parent.parent / "benchmark" / "cache"

_token_usage: dict[str, int] = {"meta": 0, "hf": 0}


def _cache_key(model: str, system_prompt: str, user_prompt: str) -> str:
    blob = json.dumps({"model": model, "system": system_prompt, "user": user_prompt}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def _read_cache(key: str) -> dict[str, Any] | None:
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def _write_cache(key: str, data: dict[str, Any]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / f"{key}.json").write_text(json.dumps(data, indent=2))


def get_token_usage() -> dict[str, int]:
    return dict(_token_usage)


async def call_meta_api(
    model: str,
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    use_cache: bool = True,
) -> str:
    """Call Meta Model API (OpenAI-compatible). Models: muse-spark-1.3, muse-wonder, muse-code."""
    key = _cache_key(model, system_prompt, user_prompt)
    if use_cache:
        cached = _read_cache(key)
        if cached is not None:
            return cached["content"]

    api_key = os.environ.get("META_API_KEY", "")
    base_url = os.environ.get("META_API_BASE_URL", "https://api.meta.ai/v1")

    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }
    if max_tokens:
        payload["max_tokens"] = max_tokens

    content = await _post_with_retry(
        url=f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        payload=payload,
        provider="meta",
    )

    _write_cache(key, {"model": model, "content": content})
    return content


async def call_hf_api(
    model: str,
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    use_cache: bool = True,
) -> str:
    """Call HuggingFace Inference API (Pro). Used for verifier swarm."""
    key = _cache_key(model, system_prompt, user_prompt)
    if use_cache:
        cached = _read_cache(key)
        if cached is not None:
            return cached["content"]

    hf_token = os.environ.get("HF_TOKEN", "")
    base_url = f"https://api-inference.huggingface.co/models/{model}/v1"

    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    content = await _post_with_retry(
        url=f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {hf_token}", "Content-Type": "application/json"},
        payload=payload,
        provider="hf",
    )

    _write_cache(key, {"model": model, "content": content})
    return content


async def _post_with_retry(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    provider: str,
    max_retries: int = 3,
) -> str:
    """POST with exponential backoff on 429/5xx."""
    resp = None
    async with httpx.AsyncClient(timeout=300.0) as client:
        for attempt in range(max_retries):
            print(f"  [{provider}] attempt {attempt+1}/{max_retries} -> {url} (model={payload.get('model','')})", flush=True)
            try:
                resp = await client.post(url, headers=headers, json=payload)
            except Exception as e:
                print(f"  [{provider}] connection error: {e}", flush=True)
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise
            print(f"  [{provider}] status={resp.status_code}", flush=True)
            if resp.status_code == 200:
                data = resp.json()
                print(f"  [{provider}] choices[0]: {json.dumps(data.get('choices',[{}])[0])[:500]}", flush=True)
                try:
                    content = data["choices"][0]["message"]["content"]
                    if content is None:
                        fr = data["choices"][0].get("finish_reason", "unknown")
                        print(f"  [{provider}] content=null, finish_reason={fr}", flush=True)
                        if attempt < max_retries - 1:
                            await asyncio.sleep(1)
                            continue
                        raise RuntimeError(f"API returned null content (finish_reason={fr})")
                except (KeyError, IndexError, TypeError):
                    print(f"  [{provider}] unexpected response: {json.dumps(data)[:800]}", flush=True)
                    raise RuntimeError(f"Unexpected API response: {json.dumps(data)[:500]}")
                usage = data.get("usage", {})
                _token_usage[provider] += usage.get("total_tokens", 0)
                return content
            if resp.status_code in (429, 500, 502, 503, 504):
                wait = 2 ** attempt
                print(f"  [{provider}] {resp.status_code}, retrying in {wait}s...", flush=True)
                await asyncio.sleep(wait)
                continue
            print(f"  [{provider}] ERROR {resp.status_code}: {resp.text[:500]}", flush=True)
            resp.raise_for_status()
    raise RuntimeError(f"{provider} API failed after {max_retries} retries: {resp.status_code if resp else 'no response'}")
