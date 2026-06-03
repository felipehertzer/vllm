# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.states import RequestState


class ParakeetTDTModelState(DefaultModelState):
    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        self.encoder_outputs: list[torch.Tensor] = []
        self.encoder_output_req_ids: list[str] = []
        self.forced_decoder_sequences: dict[str, list[int]] = {}
        self.forced_decoder_ids = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            dtype=torch.long,
            device=device,
        )

    def get_supported_generation_tasks(self):
        return ("transcription",)

    def remove_request(self, req_id: str) -> None:
        self.forced_decoder_sequences.pop(req_id, None)

    def _ordered_encoder_inputs(
        self,
        scheduled_encoder_inputs: dict[str, list[int]],
        input_batch: InputBatch,
    ) -> tuple[dict[str, list[int]], list[str]]:
        encoder_inputs: dict[str, list[int]] = {}
        encoder_req_ids: list[str] = []

        for req_id in input_batch.req_ids:
            req_encoder_inputs = scheduled_encoder_inputs.get(req_id, [])
            if not req_encoder_inputs:
                continue

            encoder_inputs[req_id] = req_encoder_inputs
            mm_features = self.encoder_cache.mm_features[req_id]
            for mm_input_id in req_encoder_inputs:
                mm_feature = mm_features[mm_input_id]
                if mm_feature.data is None:
                    continue
                encoder_req_ids.append(req_id)

        return encoder_inputs, encoder_req_ids

    def get_mm_embeddings(
        self,
        scheduled_encoder_inputs: dict[str, list[int]],
        input_batch: InputBatch,
    ) -> None:
        encoder_inputs, encoder_req_ids = self._ordered_encoder_inputs(
            scheduled_encoder_inputs, input_batch
        )
        _, mm_kwargs = self.encoder_runner.prepare_mm_inputs(encoder_inputs)
        if mm_kwargs:
            self.encoder_outputs = self.encoder_runner.execute_mm_encoder(mm_kwargs)
            self.encoder_output_req_ids = encoder_req_ids
        else:
            self.encoder_outputs = []
            self.encoder_output_req_ids = []
        return None

    def _decode_encoder_outputs(self) -> None:
        if not self.encoder_outputs:
            return
        if len(self.encoder_outputs) != len(self.encoder_output_req_ids):
            raise ValueError(
                "Parakeet TDT encoder output count does not match scheduled "
                "request count: "
                f"{len(self.encoder_outputs)} != {len(self.encoder_output_req_ids)}."
            )

        sequences = self.model.model.greedy_decode_batch(self.encoder_outputs)
        for req_id, sequence in zip(
            self.encoder_output_req_ids,
            sequences,
            strict=True,
        ):
            self.forced_decoder_sequences[req_id] = sequence

        self.encoder_outputs = []
        self.encoder_output_req_ids = []

    def _build_forced_decoder_ids(
        self,
        req_ids: list[str],
        num_scheduled_tokens: torch.Tensor | Any,
        output_token_counts: torch.Tensor | Any,
        num_tokens: int,
    ) -> torch.Tensor:
        eos_token_id = int(self.model_config.hf_config.eos_token_id)
        forced_decoder_ids: list[int] = []

        for req_index, req_id in enumerate(req_ids):
            num_scheduled = int(num_scheduled_tokens[req_index])
            start_pos = int(output_token_counts[req_index])
            sequence = self.forced_decoder_sequences.get(req_id, ())
            for position in range(start_pos, start_pos + num_scheduled):
                if 0 <= position < len(sequence):
                    forced_decoder_ids.append(sequence[position])
                else:
                    forced_decoder_ids.append(eos_token_id)

        if len(forced_decoder_ids) < num_tokens:
            pad_token_id = (
                forced_decoder_ids[-1] if forced_decoder_ids else eos_token_id
            )
            forced_decoder_ids.extend(
                [pad_token_id] * (num_tokens - len(forced_decoder_ids))
            )

        self.forced_decoder_ids[:num_tokens].copy_(
            torch.tensor(
                forced_decoder_ids[:num_tokens],
                dtype=torch.long,
                device=self.device,
            )
        )
        return self.forced_decoder_ids[:num_tokens]

    def prepare_inputs(
        self, input_batch: InputBatch, req_states: RequestState
    ) -> dict[str, torch.Tensor]:
        self._decode_encoder_outputs()
        output_token_counts = (
            req_states.total_len.np[input_batch.idx_mapping_np[: input_batch.num_reqs]]
            - input_batch.prefill_len_np
        )
        return {
            "forced_decoder_ids": self._build_forced_decoder_ids(
                input_batch.req_ids,
                input_batch.num_scheduled_tokens,
                output_token_counts,
                input_batch.num_tokens_after_padding,
            )
        }

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        del num_reqs
        eos_token_id = int(self.model_config.hf_config.eos_token_id)
        self.forced_decoder_ids[:num_tokens].fill_(eos_token_id)
        return {"forced_decoder_ids": self.forced_decoder_ids[:num_tokens]}
