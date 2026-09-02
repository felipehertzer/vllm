# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Low-latency OpenAI-compatible server for Nemotron transducer ASR."""

from __future__ import annotations

import argparse
import asyncio
import io
import os
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from safetensors.torch import safe_open
from transformers import PreTrainedTokenizerFast

from vllm.config.compilation import CompilationConfig, CompilationMode
from vllm.model_executor.models.nemotron_asr import NemotronASRForRNNT
from vllm.model_executor.models.parakeet import ParakeetExtractor
from vllm.multimodal.media.audio import load_audio
from vllm.transformers_utils.configs.nemotron_asr import NemotronASRConfig


class _MMConfig:
    def get_limit_per_prompt(self, modality: str) -> int:
        del modality
        return 1


@dataclass(frozen=True, slots=True)
class Transcription:
    text: str
    duration_s: float
    decode_ms: float
    inference_ms: float


class NemotronDirectRuntime:
    """Resident Nemotron runtime without generic token-generation replay."""

    def __init__(
        self,
        model_dir: Path,
        *,
        device: Literal["cpu", "mps"] = "mps",
        warmup_seconds: float = 30.0,
        max_audio_seconds: float = 1_800.0,
        max_audio_bytes: int = 100 * 1024 * 1024,
    ) -> None:
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS is not available")
        if warmup_seconds < 0:
            raise ValueError("warmup_seconds must not be negative")
        if max_audio_seconds <= 0 or max_audio_bytes <= 0:
            raise ValueError("audio limits must be positive")

        self.model_dir = model_dir
        self.device = torch.device(device)
        self.max_audio_seconds = max_audio_seconds
        self.max_audio_bytes = max_audio_bytes
        self.config = NemotronASRConfig.from_pretrained(model_dir)
        self.extractor = ParakeetExtractor(self.config.encoder_config)
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir)
        self.model = self._load_model()
        self._lock = asyncio.Lock()

        if warmup_seconds > 0:
            sample_count = int(round(warmup_seconds * self.config.sample_rate))
            self._transcribe_waveform(np.zeros(sample_count, dtype=np.float32))

    def _load_model(self) -> NemotronASRForRNNT:
        os.environ["NEMOTRON_ASR_MPS_ENCODER"] = (
            "auto" if self.device.type == "mps" else "0"
        )
        multimodal_config = _MMConfig()
        model_config = SimpleNamespace(
            hf_config=self.config,
            dtype=torch.float32,
            multimodal_config=multimodal_config,
        )
        model_config.get_multimodal_config = lambda: multimodal_config
        vllm_config = SimpleNamespace(
            model_config=model_config,
            compilation_config=CompilationConfig(mode=CompilationMode.NONE),
        )
        model = NemotronASRForRNNT(vllm_config=vllm_config)
        with safe_open(
            self.model_dir / "model.safetensors",
            framework="pt",
            device="cpu",
        ) as checkpoint:
            tensor_names = checkpoint.keys()
            model.load_weights(
                (name, checkpoint.get_tensor(name)) for name in tensor_names
            )
        if self.device.type == "cpu":
            model.to(self.device)
        return model.eval()

    def _features(self, waveform: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        lengths = torch.tensor([len(waveform)], dtype=torch.long)
        raw = [torch.from_numpy(waveform)]
        features = self.extractor._pad_raw_speech(raw, len(waveform), "cpu")
        features = self.extractor._apply_preemphasis(features, lengths)
        features = self.extractor._torch_extract_fbank_features(features, "cpu")
        return self.extractor._normalize_mel_features(features, lengths)

    def _transcribe_waveform(self, waveform: np.ndarray) -> Transcription:
        started_at = time.perf_counter()
        features, attention_mask = self._features(waveform)
        if self.device.type == "cpu":
            features = features.to(self.device)
            attention_mask = attention_mask.to(self.device)

        with torch.inference_mode():
            encoder_outputs = self.model.model.get_encoder_outputs(
                features,
                attention_mask,
            )
            decode_started_at = time.perf_counter()
            token_ids = self.model.model.greedy_decode_batch(encoder_outputs)[0]
            decode_ms = (time.perf_counter() - decode_started_at) * 1000

        text = self.model.post_process_output(
            self.tokenizer.decode(token_ids, skip_special_tokens=True)
        )
        return Transcription(
            text=text,
            duration_s=len(waveform) / float(self.config.sample_rate),
            decode_ms=decode_ms,
            inference_ms=(time.perf_counter() - started_at) * 1000,
        )

    def transcribe_bytes(
        self,
        audio_data: bytes,
        *,
        filename: str | None,
        content_type: str | None,
    ) -> Transcription:
        if len(audio_data) > self.max_audio_bytes:
            raise ValueError("Audio file exceeds maximum allowed size")
        with io.BytesIO(audio_data) as stream:
            waveform, sample_rate = load_audio(
                stream,
                sr=int(self.config.sample_rate),
                filename=filename,
                content_type=content_type,
                mono=True,
                max_duration_s=self.max_audio_seconds,
                max_decode_bytes=self.max_audio_bytes,
            )
        waveform = np.asarray(waveform, dtype=np.float32)
        if sample_rate != self.config.sample_rate:
            raise ValueError(f"Unexpected sample rate: {sample_rate}")
        return self._transcribe_waveform(waveform)

    async def transcribe(
        self,
        audio_data: bytes,
        *,
        filename: str | None,
        content_type: str | None,
    ) -> Transcription:
        async with self._lock:
            return await asyncio.to_thread(
                self.transcribe_bytes,
                audio_data,
                filename=filename,
                content_type=content_type,
            )


def _error(message: str, status_code: int) -> JSONResponse:
    error_type = "invalid_request_error" if status_code < 500 else "server_error"
    return JSONResponse(
        {"error": {"message": message, "type": error_type}},
        status_code=status_code,
    )


def _authorized(request: Request, api_key: str | None) -> bool:
    authorization = request.headers.get("authorization")
    return api_key is None or authorization == f"Bearer {api_key}"


def create_app(
    runtime: NemotronDirectRuntime,
    *,
    served_model_name: str,
    api_key: str | None = None,
    max_upload_bytes: int = 100 * 1024 * 1024,
) -> FastAPI:
    if max_upload_bytes <= 0:
        raise ValueError("max_upload_bytes must be positive")
    app = FastAPI()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "model": served_model_name}

    @app.get("/v1/models")
    async def models(request: Request) -> JSONResponse:
        if not _authorized(request, api_key):
            return _error("Unauthorized", 401)
        return JSONResponse(
            {
                "object": "list",
                "data": [
                    {
                        "id": served_model_name,
                        "object": "model",
                        "owned_by": "local",
                    }
                ],
            }
        )

    @app.post("/v1/audio/transcriptions")
    async def transcriptions(request: Request) -> JSONResponse:
        if not _authorized(request, api_key):
            return _error("Unauthorized", 401)

        content_length = request.headers.get("content-length")
        if content_length is not None and int(content_length) > max_upload_bytes:
            return _error("Audio file exceeds maximum allowed size", 413)

        form = await request.form()
        upload = form.get("file")
        if upload is None or not hasattr(upload, "read"):
            return _error("Request is missing the 'file' field", 400)
        response_format = str(form.get("response_format") or "json")
        if response_format != "json":
            return _error("Only response_format=json is supported", 400)
        temperature = str(form.get("temperature") or "0.0")
        if temperature not in {"0", "0.0"}:
            return _error("Only temperature=0 is supported", 400)

        try:
            audio_data = await upload.read()
            if len(audio_data) > max_upload_bytes:
                return _error("Audio file exceeds maximum allowed size", 413)
            result = await runtime.transcribe(
                audio_data,
                filename=getattr(upload, "filename", None),
                content_type=getattr(upload, "content_type", None),
            )
        except ValueError as exc:
            return _error(str(exc), 400)
        except Exception:
            return _error("Transcription failed", 500)

        return JSONResponse(
            {
                "text": result.text,
                "usage": {
                    "type": "duration",
                    "seconds": int(np.ceil(result.duration_s)),
                },
            },
            headers={
                "X-Nemotron-Inference-Ms": f"{result.inference_ms:.2f}",
                "X-Nemotron-Decode-Ms": f"{result.decode_ms:.2f}",
            },
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--served-model-name", default="nemotron-asr")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--warmup-seconds", type=float, default=30.0)
    parser.add_argument("--max-audio-seconds", type=float, default=1_800.0)
    parser.add_argument("--max-upload-mb", type=int, default=100)
    parser.add_argument("--api-key", default=os.getenv("VLLM_API_KEY"))
    args = parser.parse_args()

    runtime = NemotronDirectRuntime(
        args.model,
        device=args.device,
        warmup_seconds=args.warmup_seconds,
        max_audio_seconds=args.max_audio_seconds,
        max_audio_bytes=args.max_upload_mb * 1024 * 1024,
    )
    app = create_app(
        runtime,
        served_model_name=args.served_model_name,
        api_key=args.api_key,
        max_upload_bytes=args.max_upload_mb * 1024 * 1024,
    )
    uvicorn.run(app, host=args.host, port=args.port, workers=1, access_log=False)


if __name__ == "__main__":
    main()
