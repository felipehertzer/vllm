# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.v1.worker.gpu.model_states.transducer_asr import TransducerASRModelState


class NemotronASRModelState(TransducerASRModelState):
    profile_env_names = ("NEMOTRON_ASR_PROFILE", "TRANSDUCER_ASR_PROFILE")
    profile_label = "Nemotron ASR"
    decode_metric_name = "rnnt_decode_ms"
    uses_prompt_ids = True
