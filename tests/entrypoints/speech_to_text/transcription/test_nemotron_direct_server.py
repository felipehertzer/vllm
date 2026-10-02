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
            6: "▁.",
            7: "'s",
            8: "▁",
            9: "<eos>",
            10: "very",
            11: "$",
            12: "78",
            13: "▁than",
        }
        return [vocabulary[token_id] for token_id in token_ids]


def test_each_word_spans_from_the_previous_word_end_to_its_last_spoken_frame() -> None:
    words = words_from_token_timesteps(
        _FakeTokenizer(),  # type: ignore[arg-type]
        token_ids=[1, 2, 3, 4, 5, 7, 8, 10, 9],
        timesteps=[0, 2, 3, 3, 8, 9, 10, 10],
        frame_shift_seconds=0.08,
        duration_seconds=1.0,
    )

    assert words == (
        WordTimestamp(word="Good", start=0.0, end=0.08),
        WordTimestamp(word="morning,", start=0.08, end=0.32),
        WordTimestamp(word="Australia's", start=0.32, end=0.8),
        WordTimestamp(word="very", start=0.8, end=0.88),
    )


def test_late_punctuation_does_not_lengthen_the_word_it_joins() -> None:
    # "Good" is heard by frame 0; its full stop is emitted after a pause, with
    # the next word, which may have started anywhere after "Good" ended.
    words = words_from_token_timesteps(
        _FakeTokenizer(),  # type: ignore[arg-type]
        token_ids=[1, 6, 5],
        timesteps=[0, 20, 20],
        frame_shift_seconds=0.08,
        duration_seconds=2.0,
    )

    assert words == (
        WordTimestamp(word="Good.", start=0.0, end=0.08),
        WordTimestamp(word="Australia", start=0.08, end=1.68),
    )


def test_words_ending_on_the_same_frame_share_their_interval() -> None:
    words = words_from_token_timesteps(
        _FakeTokenizer(),  # type: ignore[arg-type]
        token_ids=[1, 5, 13],
        timesteps=[2, 5, 5],
        frame_shift_seconds=0.08,
        duration_seconds=1.0,
    )

    assert words == (
        WordTimestamp(word="Good", start=0.0, end=0.24),
        WordTimestamp(word="Australia", start=0.24, end=0.48),
        WordTimestamp(word="than", start=0.24, end=0.48),
    )


def test_word_timestamps_keep_signs_with_the_word_they_start() -> None:
    # "than $78": the bare boundary and "$" begin a word; only sentence
    # punctuation attaches to the word before it.
    words = words_from_token_timesteps(
        _FakeTokenizer(),  # type: ignore[arg-type]
        token_ids=[13, 8, 11, 12],
        timesteps=[0, 2, 2, 3],
        frame_shift_seconds=0.08,
        duration_seconds=1.0,
    )

    assert words == (
        WordTimestamp(word="than", start=0.0, end=0.08),
        WordTimestamp(word="$78", start=0.08, end=0.32),
    )


def test_word_timestamps_keep_a_word_emitted_on_the_closing_frame() -> None:
    # 29.994 s of audio has encoder frames up to 375, which starts at 30.0 s.
    words = words_from_token_timesteps(
        _FakeTokenizer(),  # type: ignore[arg-type]
        token_ids=[1, 5],
        timesteps=[370, 375],
        frame_shift_seconds=0.08,
        duration_seconds=29.994,
    )

    assert words == (
        WordTimestamp(word="Good", start=0.0, end=29.68),
        WordTimestamp(word="Australia", start=29.68, end=29.994),
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
