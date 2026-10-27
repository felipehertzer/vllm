# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.model_states.nemotron_asr import NemotronASRModelState

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


class _EncoderRunner:
    def __init__(self) -> None:
        self.prepared: list[str] = []

    def prepare_mm_inputs(self, scheduled_encoder_inputs):
        self.prepared = list(scheduled_encoder_inputs)
        return self.prepared, [("audio", req_id) for req_id in self.prepared]

    def execute_mm_encoder(self, mm_kwargs):
        return [torch.full((2, 3), float(len(req_id))) for _, req_id in mm_kwargs]

    def timed_encoder_operation(self, request_ids):
        return nullcontext()


class _Model:
    def __init__(self) -> None:
        self.prompt_ids: list[int] = []

    def greedy_decode_batch(self, encoder_outputs, prompt_ids):
        self.prompt_ids = prompt_ids.tolist()
        return [[5, 6], [9]][: len(encoder_outputs)]


def _state(runner: _EncoderRunner | None = None) -> NemotronASRModelState:
    state = object.__new__(NemotronASRModelState)
    state.device = torch.device("cpu")
    state.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            eos_token_id=0, prompt_id_for_language=lambda language: 7
        )
    )
    state.model = SimpleNamespace(model=_Model())
    state.encoder_runner = runner or _EncoderRunner()  # type: ignore[assignment]
    state.encoder_cache = SimpleNamespace(
        mm_features={
            "a": [SimpleNamespace(data=object())],
            "b": [SimpleNamespace(data=object())],
        }
    )
    state.encoder_outputs = []
    state.encoder_output_req_ids = []
    state.forced_decoder_sequences = {}
    state.forced_decoder_ids = torch.empty(8, dtype=torch.long)
    state.prompt_ids_by_req_id = {"b": 3}
    state._last_encoder_ms = 0.0
    return state


def test_runner_calls_the_transducer_encoder_hooks():
    # The model runner calls these ModelState hooks; a renamed override is
    # never run and leaves every request decoding to EOS.
    for name in ("prepare_inputs_embeds", "dummy_inputs_embeds"):
        assert hasattr(ModelState, name)
        assert name in NemotronASRModelState.__mro__[1].__dict__


def test_encoder_outputs_force_the_decoded_tokens_in_batch_order():
    runner = _EncoderRunner()
    state = _state(runner)
    # "b" is at its first decode step after a one-token decoder prompt; "a" is
    # prefilling its one-token prompt.
    input_batch = SimpleNamespace(
        req_ids=["b", "a"],
        num_computed_tokens_np=np.array([1, 0]),
        num_scheduled_tokens=np.array([1, 1]),
        prefill_len_np=np.array([1, 1]),
        num_tokens_after_padding=4,
    )
    req_states = SimpleNamespace()

    assert (
        state.prepare_inputs_embeds({"a": [0], "b": [0]}, input_batch, req_states)
        is None
    )
    assert runner.prepared == ["b", "a"]

    model_inputs = state.prepare_inputs(input_batch, req_states)

    assert state.model.model.prompt_ids == [3, 7]
    # "b" decoded to [5, 6] and has one output, so it is forced to 6; "a"
    # decoded to [9] and starts at 9; padding repeats the last id.
    assert model_inputs["forced_decoder_ids"].tolist() == [6, 9, 9, 9]


def test_dummy_runs_keep_input_ids():
    # inputs_embeds of None keeps input_ids, which the transducer forward needs.
    assert _state().dummy_inputs_embeds(4) is None
