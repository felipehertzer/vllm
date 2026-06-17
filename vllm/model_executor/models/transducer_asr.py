# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared helpers for transducer-style ASR models.

This module contains the pieces that are not specific to a concrete acoustic
encoder: forced-token playback for vLLM generation and greedy batched decode for
RNN-T/TDT heads.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn


class TransducerASRForcedDecoderState:
    """Token sequences produced by a transducer decoder for a batch."""

    def __init__(self, eos_token_id: int) -> None:
        self.eos_token_id = eos_token_id
        self._sequences: list[list[int]] = []

    def set_sequence(self, sequence: Sequence[int]) -> None:
        self.set_sequences([sequence])

    def set_sequences(self, sequences: Sequence[Sequence[int]]) -> None:
        self._sequences = [list(sequence) for sequence in sequences]

    def get_forced_token_ids(
        self,
        *,
        positions: torch.Tensor,
        device: torch.device,
        request_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if positions.ndim == 0:
            positions = positions.reshape(1)
        if positions.ndim > 1:
            positions = positions[0]
        if request_indices is None:
            request_indices = torch.zeros_like(positions)
        elif request_indices.ndim == 0:
            request_indices = request_indices.reshape(1)
        if request_indices.ndim > 1:
            request_indices = request_indices[0]

        forced_token_ids: list[int] = []
        for request_index, position in zip(request_indices, positions, strict=True):
            req_idx = int(request_index.item())
            seq_idx = int(position.item())
            sequence = (
                self._sequences[req_idx] if 0 <= req_idx < len(self._sequences) else []
            )
            if 0 <= seq_idx < len(sequence):
                forced_token_ids.append(sequence[seq_idx])
            else:
                forced_token_ids.append(self.eos_token_id)

        return torch.tensor(forced_token_ids, dtype=torch.long, device=device)


@dataclass(frozen=True)
class TransducerDecodeConfig:
    vocab_size: int
    blank_token_id: int
    eos_token_id: int
    max_symbols_per_step: int
    durations: Sequence[int] = ()


class TransducerPredictionDecoder(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.decoder_projector = nn.Linear(hidden_size, hidden_size)

    def predict(
        self,
        token_id: int,
        state: tuple[torch.Tensor, torch.Tensor] | None,
        device: torch.device,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        label = torch.tensor([[token_id]], dtype=torch.long, device=device)
        hidden_states = self.embedding(label)
        hidden_states, state = self.lstm(hidden_states, state)
        hidden_states = self.decoder_projector(hidden_states)
        return hidden_states[:, 0, :], state

    def predict_batch(
        self,
        token_ids: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None,
        rows: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hidden_states = self.embedding(token_ids.unsqueeze(1))
        row_state = None
        if state is not None:
            row_state = (
                state[0][:, rows, :].contiguous(),
                state[1][:, rows, :].contiguous(),
            )
        hidden_states, next_state = self.lstm(hidden_states, row_state)
        hidden_states = self.decoder_projector(hidden_states)
        return hidden_states[:, 0, :], next_state


def greedy_decode_transducer_batch(
    *,
    encoder_projected: torch.Tensor,
    lengths: torch.Tensor,
    decoder: TransducerPredictionDecoder,
    joint_logits,
    config: TransducerDecodeConfig,
) -> list[list[int]]:
    """Greedy batched RNN-T/TDT decode.

    ``encoder_projected`` is ``[batch, time, hidden]``. If ``config.durations``
    is empty, decode uses RNN-T semantics. Otherwise it uses TDT semantics with
    the provided duration values.
    """

    if encoder_projected.shape[0] == 0:
        return []

    device = encoder_projected.device
    batch_size = int(encoder_projected.shape[0])
    max_encoder_frames = int(encoder_projected.shape[1])
    durations = tuple(int(duration) for duration in config.durations)
    is_tdt = bool(durations)
    duration_values = (
        torch.tensor(durations, dtype=torch.long, device=device) if is_tdt else None
    )

    time_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
    last_tokens = torch.full(
        (batch_size,),
        config.blank_token_id,
        dtype=torch.long,
        device=device,
    )
    has_last_token = torch.zeros(batch_size, dtype=torch.bool, device=device)
    state: tuple[torch.Tensor, torch.Tensor] | None = None
    max_output_tokens = max_encoder_frames * config.max_symbols_per_step + 1
    output_ids = torch.full(
        (batch_size, max_output_tokens),
        config.eos_token_id,
        dtype=torch.long,
        device=device,
    )
    output_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)

    for _ in range(max_encoder_frames):
        active = torch.nonzero(time_idx < lengths, as_tuple=False).flatten()
        if active.numel() == 0:
            break
        symbols_added = torch.zeros(active.shape[0], dtype=torch.long, device=device)
        needs_loop = torch.ones(active.shape[0], dtype=torch.bool, device=device)
        still_looping_after_limit = torch.zeros(
            active.shape[0], dtype=torch.bool, device=device
        )

        for _symbol_step in range(config.max_symbols_per_step):
            loop_positions = torch.nonzero(needs_loop, as_tuple=False).flatten()
            if loop_positions.numel() == 0:
                break
            loop_rows = active[loop_positions]
            labels = torch.where(
                has_last_token[loop_rows],
                last_tokens[loop_rows],
                torch.full_like(loop_rows, config.blank_token_id),
            )

            pred_state, next_state = decoder.predict_batch(labels, state, loop_rows)
            encoder_state = encoder_projected[loop_rows, time_idx[loop_rows]]
            logits = joint_logits(encoder_state, pred_state)

            token_logits = logits[:, : config.vocab_size].float()
            tokens = token_logits.argmax(dim=1)
            blank_tokens = tokens == config.blank_token_id

            if is_tdt:
                assert duration_values is not None
                duration_logits = logits[:, config.vocab_size :].float()
                duration_indices = duration_logits.argmax(dim=1)
                skips = duration_values[duration_indices]
                skips = torch.where(
                    blank_tokens & (skips == 0),
                    torch.ones_like(skips),
                    skips,
                )
                needs_next_loop = skips == 0
            else:
                skips = torch.where(
                    blank_tokens,
                    torch.ones_like(tokens, dtype=torch.long),
                    torch.zeros_like(tokens, dtype=torch.long),
                )
                needs_next_loop = ~blank_tokens

            nonblank_tokens = ~blank_tokens
            nonblank_rows = loop_rows[nonblank_tokens]
            emit_positions = output_lengths[nonblank_rows]
            output_ids[nonblank_rows, emit_positions] = tokens[nonblank_tokens]
            output_lengths[nonblank_rows] += 1

            last_tokens[nonblank_rows] = tokens[nonblank_tokens]
            has_last_token[nonblank_rows] = True
            if state is None:
                num_layers = next_state[0].shape[0]
                hidden_size = next_state[0].shape[-1]
                state = (
                    torch.zeros(
                        num_layers,
                        batch_size,
                        hidden_size,
                        dtype=next_state[0].dtype,
                        device=device,
                    ),
                    torch.zeros(
                        num_layers,
                        batch_size,
                        hidden_size,
                        dtype=next_state[1].dtype,
                        device=device,
                    ),
                )
            state[0][:, nonblank_rows, :] = next_state[0][:, nonblank_tokens, :]
            state[1][:, nonblank_rows, :] = next_state[1][:, nonblank_tokens, :]

            time_idx[loop_rows] += skips
            symbols_added[loop_positions] += 1
            still_looping_after_limit[loop_positions] = needs_next_loop
            needs_loop[loop_positions] = needs_next_loop & (
                symbols_added[loop_positions] < config.max_symbols_per_step
            )

        advance_after_limit = active[
            still_looping_after_limit & (symbols_added >= config.max_symbols_per_step)
        ]
        time_idx[advance_after_limit] += 1

    eos_positions = output_lengths.clamp(max=max_output_tokens - 1)
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device)
    output_ids[batch_indices, eos_positions] = config.eos_token_id
    output_lengths = eos_positions + 1

    output_ids_cpu = output_ids.cpu()
    output_lengths_cpu = output_lengths.cpu().tolist()
    return [
        output_ids_cpu[row, :output_length].tolist()
        for row, output_length in enumerate(output_lengths_cpu)
    ]


def strip_asr_special_tokens(text: str) -> str:
    """Remove ASR control markers while preserving normal text spacing."""

    pieces: list[str] = []
    i = 0
    while i < len(text):
        char = text[i]
        if char in "<[":
            close = ">" if char == "<" else "]"
            end = text.find(close, i + 1)
            if end != -1:
                i = end + 1
                continue
        pieces.append(char)
        i += 1
    return " ".join("".join(pieces).split())
