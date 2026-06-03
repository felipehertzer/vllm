# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch
import torch.nn as nn

from vllm.model_executor.models.config import (
    MODELS_CONFIG_MAP,
    ParakeetForTDTConfig,
)
from vllm.model_executor.models.parakeet_tdt import (
    ParakeetForTDT,
    ParakeetTDTForcedDecoderState,
    ParakeetTDTModel,
)
from vllm.transformers_utils.configs.parakeet_tdt import ParakeetTDTConfig


def test_parakeet_tdt_forced_tokens_follow_positions():
    state = ParakeetTDTForcedDecoderState(eos_token_id=99)
    state.set_sequence([11, 12, 13])

    forced = state.get_forced_token_ids(
        positions=torch.tensor([2, 0], dtype=torch.long),
        device=torch.device("cpu"),
    )

    assert forced.tolist() == [13, 11]


def test_parakeet_tdt_forced_tokens_fall_back_to_eos_after_sequence_end():
    state = ParakeetTDTForcedDecoderState(eos_token_id=99)
    state.set_sequence([11])

    forced = state.get_forced_token_ids(
        positions=torch.tensor([0, 2], dtype=torch.long),
        device=torch.device("cpu"),
    )

    assert forced.tolist() == [11, 99]


def test_parakeet_tdt_forced_tokens_follow_request_indices():
    state = ParakeetTDTForcedDecoderState(eos_token_id=99)
    state.set_sequences([[11, 12], [21, 22]])

    forced = state.get_forced_token_ids(
        positions=torch.tensor([1, 0], dtype=torch.long),
        request_indices=torch.tensor([0, 1], dtype=torch.long),
        device=torch.device("cpu"),
    )

    assert forced.tolist() == [12, 21]


def test_parakeet_tdt_forward_uses_explicit_forced_decoder_ids():
    model = ParakeetForTDT.__new__(ParakeetForTDT)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(vocab_size=100, eos_token_id=99)

    logits = ParakeetForTDT.forward(
        model,
        input_ids=torch.zeros(2, dtype=torch.long),
        positions=torch.tensor([0, 0], dtype=torch.long),
        forced_decoder_ids=torch.tensor([12, 22], dtype=torch.long),
    )

    assert logits.argmax(dim=-1).tolist() == [12, 22]


def test_parakeet_tdt_forward_uses_per_request_forced_sequences():
    model = ParakeetForTDT.__new__(ParakeetForTDT)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(vocab_size=100, eos_token_id=99)

    logits = ParakeetForTDT.forward(
        model,
        input_ids=torch.zeros(2, dtype=torch.long),
        positions=torch.tensor([0, 1], dtype=torch.long),
        forced_decoder_sequences=[[11, 12], [21, 22]],
        forced_decoder_request_indices=torch.tensor([0, 1], dtype=torch.long),
    )

    assert logits.argmax(dim=-1).tolist() == [11, 22]


def test_parakeet_tdt_forward_without_decoder_state_uses_eos():
    model = ParakeetForTDT.__new__(ParakeetForTDT)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(vocab_size=100, eos_token_id=99)

    logits = ParakeetForTDT.forward(
        model,
        input_ids=torch.zeros(2, dtype=torch.long),
        positions=torch.tensor([0, 0], dtype=torch.long),
    )

    assert logits.argmax(dim=-1).tolist() == [99, 99]


def test_parakeet_tdt_forward_decodes_multiple_encoder_outputs():
    model = ParakeetForTDT.__new__(ParakeetForTDT)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(vocab_size=100, eos_token_id=99)
    model.model = SimpleNamespace(
        greedy_decode_batch=lambda encoder_outputs: [
            [int(encoder_output.item()), 99] for encoder_output in encoder_outputs
        ]
    )

    logits = ParakeetForTDT.forward(
        model,
        input_ids=torch.zeros(2, dtype=torch.long),
        positions=torch.tensor([0, 0], dtype=torch.long),
        encoder_outputs=[torch.tensor(11), torch.tensor(21)],
        forced_decoder_request_indices=torch.tensor([0, 1], dtype=torch.long),
    )

    assert logits.argmax(dim=-1).tolist() == [11, 21]


def test_parakeet_tdt_config_updates_runtime_metadata():
    model_config = SimpleNamespace(
        enforce_eager=False,
        hf_config=SimpleNamespace(eos_token_id=3),
        hf_text_config=SimpleNamespace(eos_token_id=3),
        override_generation_config={},
    )
    scheduler_config = SimpleNamespace(max_num_seqs=8)
    vllm_config = SimpleNamespace(
        model_config=model_config,
        scheduler_config=scheduler_config,
    )

    assert MODELS_CONFIG_MAP["ParakeetForTDT"] is ParakeetForTDTConfig
    ParakeetForTDTConfig.verify_and_update_config(vllm_config)

    assert model_config.enforce_eager is False
    assert scheduler_config.max_num_seqs == 8
    assert model_config.override_generation_config == {"eos_token_id": 3}


def test_parakeet_tdt_config_defines_eos_token_id():
    assert ParakeetTDTConfig().eos_token_id == 3


def test_parakeet_tdt_stt_config_uses_audio_metadata_only():
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(sample_rate=16000, eos_token_id=3)
    )

    stt_config = ParakeetForTDT.get_speech_to_text_config(
        model_config, task_type="transcribe"
    )

    assert stt_config.sample_rate == 16000
    assert stt_config.max_audio_clip_s == 30


def test_parakeet_tdt_pads_variable_length_audio_features():
    model = ParakeetForTDT.__new__(ParakeetForTDT)
    nn.Module.__init__(model)

    input_features, attention_mask = model._parse_and_validate_audio_input(
        input_features=[torch.ones(2, 3), torch.full((4, 3), 2.0)],
        attention_mask=[
            torch.ones(2, dtype=torch.bool),
            torch.ones(4, dtype=torch.bool),
        ],
    )

    assert input_features.shape == (2, 4, 3)
    assert attention_mask.shape == (2, 4)
    assert input_features[0, 2:].tolist() == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    assert attention_mask[0].tolist() == [True, True, False, False]


def test_parakeet_tdt_greedy_decode_advances_blank_duration_zero():
    class FakeDecoder:
        def __init__(self) -> None:
            self.calls = 0

        def predict(self, token_id, state, device):
            del token_id, state, device
            self.calls += 1
            return torch.zeros(1, 4), None

    decoder = FakeDecoder()
    fake_model = SimpleNamespace(
        config=SimpleNamespace(
            vocab_size=3,
            blank_token_id=2,
            eos_token_id=1,
            durations=[0, 1],
            max_symbols_per_step=4,
        ),
        decoder=decoder,
        encoder_projector=lambda encoder_output: encoder_output,
        _joint_logits=lambda encoder_frame, pred_state: torch.tensor(
            [[0.0, 0.0, 1.0, 1.0, 0.0]], device=encoder_frame.device
        ),
    )

    token_ids = ParakeetTDTModel.greedy_decode(fake_model, torch.zeros(2, 4))

    assert token_ids == [1]
    assert decoder.calls == 2


def test_parakeet_tdt_greedy_decode_projects_encoder_once():
    class FakeDecoder:
        def predict(self, token_id, state, device):
            del token_id, state, device
            return torch.zeros(1, 4), None

    class FakeProjector:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, encoder_output):
            self.calls += 1
            return encoder_output

    class FakeModel:
        def __init__(self) -> None:
            self.config = SimpleNamespace(
                vocab_size=3,
                blank_token_id=2,
                eos_token_id=1,
                durations=[0, 1],
                max_symbols_per_step=4,
            )
            self.decoder = FakeDecoder()
            self.encoder_projector = FakeProjector()

        def joint(self, encoder_state, pred_state):
            del pred_state
            return torch.tensor(
                [[0.0, 0.0, 1.0, 0.0, 1.0]],
                device=encoder_state.device,
            )

        def _joint_logits(self, encoder_state, pred_state):
            return ParakeetTDTModel._joint_logits(self, encoder_state, pred_state)

    fake_model = FakeModel()

    ParakeetTDTModel.greedy_decode(fake_model, torch.zeros(2, 4))

    assert fake_model.encoder_projector.calls == 1


def test_parakeet_tdt_greedy_decode_does_not_skip_after_positive_duration():
    class FakeDecoder:
        def predict(self, token_id, state, device):
            del token_id, state, device
            return torch.zeros(1, 4), None

    def joint_logits(encoder_frame, pred_state):
        del pred_state
        token_id = int(encoder_frame[0, 0].item())
        logits = torch.full((1, 11), -1.0, device=encoder_frame.device)
        logits[0, token_id] = 1.0
        logits[0, 10] = 1.0
        return logits

    fake_model = SimpleNamespace(
        config=SimpleNamespace(
            vocab_size=10,
            blank_token_id=8,
            eos_token_id=9,
            durations=[1],
            max_symbols_per_step=1,
        ),
        decoder=FakeDecoder(),
        encoder_projector=lambda encoder_output: encoder_output,
        _joint_logits=joint_logits,
    )

    token_ids = ParakeetTDTModel.greedy_decode(
        fake_model,
        torch.tensor([[0.0], [1.0], [2.0]]),
    )

    assert token_ids == [0, 1, 2, 9]


def test_parakeet_tdt_greedy_decode_batch_matches_single_decode():
    class FakeDecoder:
        def predict(self, token_id, state, device):
            del token_id, state
            next_state = (
                torch.zeros(1, 1, 1, device=device),
                torch.zeros(1, 1, 1, device=device),
            )
            return torch.zeros(1, 1, device=device), next_state

        def predict_batch(self, token_ids, state, rows):
            del state
            next_state = (
                torch.zeros(1, rows.numel(), 1, device=token_ids.device),
                torch.zeros(1, rows.numel(), 1, device=token_ids.device),
            )
            return torch.zeros(rows.numel(), 1, device=token_ids.device), next_state

    def joint_logits(encoder_frame, pred_state):
        del pred_state
        token_ids = encoder_frame[:, 0].to(dtype=torch.long)
        logits = torch.full(
            (encoder_frame.shape[0], 11), -1.0, device=encoder_frame.device
        )
        logits[torch.arange(encoder_frame.shape[0]), token_ids] = 1.0
        logits[:, 10] = 1.0
        return logits

    fake_model = SimpleNamespace(
        config=SimpleNamespace(
            vocab_size=10,
            blank_token_id=8,
            eos_token_id=9,
            durations=[1],
            max_symbols_per_step=1,
        ),
        decoder=FakeDecoder(),
        encoder_projector=lambda encoder_output: encoder_output,
        _joint_logits=joint_logits,
    )
    encoder_outputs = [
        torch.tensor([[0.0], [1.0], [2.0]]),
        torch.tensor([[3.0], [4.0]]),
    ]

    batch_token_ids = ParakeetTDTModel.greedy_decode_batch(fake_model, encoder_outputs)
    single_token_ids = [
        ParakeetTDTModel.greedy_decode(fake_model, encoder_output)
        for encoder_output in encoder_outputs
    ]

    assert batch_token_ids == single_token_ids
