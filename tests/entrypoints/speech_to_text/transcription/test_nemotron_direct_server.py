# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from fastapi.testclient import TestClient

from vllm.entrypoints.speech_to_text.nemotron_direct_server import (
    Transcription,
    WordTimestamp,
    create_app,
    words_from_token_timesteps,
)


class _FakeRuntime:
    def __init__(self) -> None:
        self.calls: list[tuple[bytes, str | None, str | None]] = []

    async def transcribe(
        self,
        audio_data: bytes,
        *,
        filename: str | None,
        content_type: str | None,
    ) -> Transcription:
        self.calls.append((audio_data, filename, content_type))
        return Transcription(
            text="hello world",
            duration_s=1.2,
            decode_ms=3.25,
            inference_ms=9.5,
            words=(
                WordTimestamp(word="hello", start=0.08, end=0.56),
                WordTimestamp(word="world", start=0.56, end=1.2),
            ),
        )


def test_direct_server_returns_openai_transcription_contract() -> None:
    runtime = _FakeRuntime()
    client = TestClient(create_app(runtime, served_model_name="nemotron"))  # type: ignore[arg-type]

    response = client.post(
        "/v1/audio/transcriptions",
        data={"model": "nemotron", "response_format": "json", "temperature": "0"},
        files={"file": ("sample.wav", b"wave", "audio/wav")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "text": "hello world",
        "words": [
            {"word": "hello", "start": 0.08, "end": 0.56},
            {"word": "world", "start": 0.56, "end": 1.2},
        ],
        "usage": {"type": "duration", "seconds": 2},
    }
    assert response.headers["x-nemotron-inference-ms"] == "9.50"
    assert response.headers["x-nemotron-decode-ms"] == "3.25"
    assert runtime.calls == [(b"wave", "sample.wav", "audio/wav")]


def test_direct_server_enforces_optional_bearer_authentication() -> None:
    runtime = _FakeRuntime()
    client = TestClient(
        create_app(runtime, served_model_name="nemotron", api_key="secret")  # type: ignore[arg-type]
    )

    unauthorized = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("sample.wav", b"wave", "audio/wav")},
    )
    unauthorized_models = client.get("/v1/models")
    authorized = client.post(
        "/v1/audio/transcriptions",
        headers={"Authorization": "Bearer secret"},
        files={"file": ("sample.wav", b"wave", "audio/wav")},
    )
    authorized_models = client.get(
        "/v1/models",
        headers={"Authorization": "Bearer secret"},
    )

    assert unauthorized.status_code == 401
    assert unauthorized_models.status_code == 401
    assert authorized.status_code == 200
    assert authorized_models.status_code == 200


def test_direct_server_rejects_unsupported_generation_options() -> None:
    runtime = _FakeRuntime()
    client = TestClient(create_app(runtime, served_model_name="nemotron"))  # type: ignore[arg-type]

    missing_file = client.post("/v1/audio/transcriptions")
    verbose = client.post(
        "/v1/audio/transcriptions",
        data={"response_format": "verbose_json"},
        files={"file": ("sample.wav", b"wave", "audio/wav")},
    )
    sampled = client.post(
        "/v1/audio/transcriptions",
        data={"temperature": "0.2"},
        files={"file": ("sample.wav", b"wave", "audio/wav")},
    )

    assert missing_file.status_code == 400
    assert verbose.status_code == 400
    assert sampled.status_code == 400
    assert runtime.calls == []


class _FakeTokenizer:
    all_special_ids = [9]

    def convert_ids_to_tokens(self, token_ids):
        vocabulary = {
            1: "▁Good",
            2: "▁morn",
            3: "ing",
            4: ",",
            5: "▁Australia",
            9: "<eos>",
        }
        return [vocabulary[token_id] for token_id in token_ids]


def test_word_timestamps_follow_metaspace_boundaries_and_audio_duration() -> None:
    words = words_from_token_timesteps(
        _FakeTokenizer(),  # type: ignore[arg-type]
        token_ids=[1, 2, 3, 4, 5, 9],
        timesteps=[0, 2, 3, 3, 8],
        frame_shift_seconds=0.08,
        duration_seconds=1.0,
    )

    assert words == (
        WordTimestamp(word="Good", start=0.0, end=0.16),
        WordTimestamp(word="morning,", start=0.16, end=0.64),
        WordTimestamp(word="Australia", start=0.64, end=1.0),
    )


def test_direct_server_rejects_oversized_upload_before_inference() -> None:
    runtime = _FakeRuntime()
    client = TestClient(
        create_app(  # type: ignore[arg-type]
            runtime,
            served_model_name="nemotron",
            max_upload_bytes=3,
        )
    )

    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("sample.wav", b"wave", "audio/wav")},
    )

    assert response.status_code == 413
    assert runtime.calls == []
