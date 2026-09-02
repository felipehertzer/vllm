# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark a warm OpenAI or Stanza transcription endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import statistics
import time
from pathlib import Path
from typing import Any

import requests
import soundfile as sf


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "median": statistics.median(values),
        "p95": _percentile(values, 0.95),
        "mean": statistics.mean(values),
        "max": max(values),
    }


def _request(
    session: requests.Session,
    *,
    url: str,
    api: str,
    model: str | None,
    audio_file: Path,
    timeout: float,
    token: str | None,
) -> tuple[float, dict[str, Any]]:
    data: dict[str, str] = {}
    if api == "openai":
        if not model:
            raise ValueError("--model is required for the OpenAI endpoint")
        data = {
            "model": model,
            "response_format": "json",
            "temperature": "0.0",
        }
    headers = {"X-AUTH-TOKEN": token} if token else {}

    started = time.perf_counter()
    with audio_file.open("rb") as audio_stream:
        response = session.post(
            url,
            headers=headers,
            data=data,
            files={
                "file": (
                    audio_file.name,
                    audio_stream,
                    mimetypes.guess_type(audio_file.name)[0] or "audio/wav",
                )
            },
            timeout=timeout,
            allow_redirects=False,
        )
        response.raise_for_status()
        payload = response.json()
    elapsed = time.perf_counter() - started

    if not isinstance(payload, dict):
        raise ValueError("transcription endpoint returned non-object JSON")
    text = payload.get("text") if api == "openai" else payload.get("transcription")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("transcription endpoint returned no usable text")
    return elapsed, payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--api", choices=("openai", "stanza"), required=True)
    parser.add_argument("--model")
    parser.add_argument("--audio-file", type=Path, required=True)
    parser.add_argument("--token-env")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    if args.warmup < 0 or args.runs < 1:
        parser.error("--warmup must be non-negative and --runs must be positive")
    if not args.audio_file.is_file():
        parser.error(f"audio file does not exist: {args.audio_file}")

    token = os.environ.get(args.token_env) if args.token_env else None
    if args.token_env and not token:
        parser.error(f"environment variable is missing or empty: {args.token_env}")

    audio_info = sf.info(args.audio_file)
    audio_seconds = float(audio_info.duration)
    session = requests.Session()
    session.trust_env = False
    try:
        for _ in range(args.warmup):
            _request(
                session,
                url=args.url,
                api=args.api,
                model=args.model,
                audio_file=args.audio_file,
                timeout=args.timeout,
                token=token,
            )

        latencies: list[float] = []
        payloads: list[dict[str, Any]] = []
        for _ in range(args.runs):
            latency, payload = _request(
                session,
                url=args.url,
                api=args.api,
                model=args.model,
                audio_file=args.audio_file,
                timeout=args.timeout,
                token=token,
            )
            latencies.append(latency)
            payloads.append(payload)
    finally:
        session.close()

    payload_hashes = [
        hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        for payload in payloads
    ]
    latency_stats = _stats(latencies)
    last_payload = payloads[-1]
    text = (
        last_payload["text"] if args.api == "openai" else last_payload["transcription"]
    )

    print(
        json.dumps(
            {
                "api": args.api,
                "url": args.url,
                "model": args.model,
                "audio_file": str(args.audio_file),
                "audio_seconds": audio_seconds,
                "audio_bytes": args.audio_file.stat().st_size,
                "warmup": args.warmup,
                "runs": args.runs,
                "latency_seconds": latency_stats,
                "realtime_factor_x": {
                    key: audio_seconds / value for key, value in latency_stats.items()
                },
                "responses_identical": len(set(payload_hashes)) == 1,
                "response_sha256": payload_hashes[-1],
                "transcription": text,
                "captions_chars": len(str(last_payload.get("captions") or "")),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
