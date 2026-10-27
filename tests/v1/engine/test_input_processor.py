# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest

from vllm.inputs import tokens_input
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.input_processor import InputProcessor

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def test_input_processor_applies_override_generation_config_eos():
    model_config = Mock(
        max_model_len=16,
        max_logprobs=20,
        logits_processors=None,
        is_diffusion=False,
        return_sampling_mask=False,
        supports_multimodal_inputs=False,
        override_generation_config={"eos_token_id": 123},
    )
    model_config.get_vocab_size.return_value = 256
    model_config.try_get_generation_config.return_value = {}
    vllm_config = Mock(
        model_config=model_config,
        speculative_config=None,
        structured_outputs_config=None,
    )
    renderer = Mock(tokenizer=None)
    renderer.get_eos_token_id.return_value = None
    input_processor = InputProcessor(vllm_config, renderer, mm_registry=Mock())
    input_processor._validate_params = lambda params, supported_tasks: False
    input_processor._validate_model_inputs = lambda encoder_inputs, decoder_inputs: None

    request = input_processor.process_inputs(
        "request-id",
        tokens_input([1]),
        SamplingParams(max_tokens=4),
        supported_tasks=("generate",),
    )

    assert request.sampling_params is not None
    assert request.sampling_params.stop_token_ids == [123]
    assert request.sampling_params.all_stop_token_ids == {123}
