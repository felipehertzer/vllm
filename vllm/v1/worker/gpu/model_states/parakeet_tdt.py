# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.v1.worker.gpu.model_states.transducer_asr import TransducerASRModelState


class ParakeetTDTModelState(TransducerASRModelState):
    profile_env_names = ("PARAKEET_PROFILE", "TRANSDUCER_ASR_PROFILE")
    profile_label = "Parakeet TDT"
    decode_metric_name = "tdt_decode_ms"
