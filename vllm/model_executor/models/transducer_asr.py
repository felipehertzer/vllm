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
import torch.nn.functional as F


@torch.compile(fullgraph=True)
def _lstm_step_from_input_gates(
    input_gates: torch.Tensor,
    hidden: torch.Tensor,
    cell: torch.Tensor,
    recurrent_weight: torch.Tensor,
    recurrent_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    gates = input_gates + F.linear(hidden, recurrent_weight, recurrent_bias)
    input_gate, forget_gate, candidate, output_gate = gates.chunk(4, dim=-1)
    next_cell = torch.sigmoid(forget_gate) * cell + torch.sigmoid(
        input_gate
    ) * torch.tanh(candidate)
    return torch.sigmoid(output_gate) * torch.tanh(next_cell), next_cell


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


@dataclass(frozen=True)
class TransducerHypothesis:
    token_ids: list[int]
    timesteps: list[int]


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
        self._cpu_input_gate_cache: torch.Tensor | None = None
        self._cpu_input_gate_key: tuple[tuple[int, int], ...] | None = None

    def _input_gate_table(self) -> torch.Tensor | None:
        parameters = (
            self.embedding.weight,
            self.lstm.weight_ih_l0,
            self.lstm.bias_ih_l0,
        )
        # This path is for resident, frozen CPU decoders with small vocabularies.
        # Bound the table rather than scaling its memory with arbitrary models.
        if (
            self.training
            or torch.is_grad_enabled()
            or self.embedding.weight.device.type != "cpu"
            or self.embedding.weight.dtype != torch.float32
            or self.embedding.num_embeddings * 4 * self.lstm.hidden_size * 4
            > 64 * 1024 * 1024
            or any(parameter.is_inference() for parameter in parameters)
        ):
            return None
        key = tuple(
            (parameter.data_ptr(), parameter._version) for parameter in parameters
        )
        if key != self._cpu_input_gate_key:
            self._cpu_input_gate_cache = F.linear(*parameters)
            self._cpu_input_gate_key = key
        return self._cpu_input_gate_cache

    def _predict_from_input_gates(
        self,
        input_gates: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None,
        rows: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if state is None:
            hidden = input_gates.new_zeros(
                self.lstm.num_layers, input_gates.shape[0], self.lstm.hidden_size
            )
            cell = torch.zeros_like(hidden)
        else:
            hidden, cell = state[0][:, rows, :], state[1][:, rows, :]
        next_hidden, next_cell = [], []
        output = input_gates
        for layer in range(self.lstm.num_layers):
            if layer:
                output = F.linear(
                    output,
                    getattr(self.lstm, f"weight_ih_l{layer}"),
                    getattr(self.lstm, f"bias_ih_l{layer}"),
                )
            output, cell_output = _lstm_step_from_input_gates(
                output,
                hidden[layer],
                cell[layer],
                getattr(self.lstm, f"weight_hh_l{layer}"),
                getattr(self.lstm, f"bias_hh_l{layer}"),
            )
            next_hidden.append(output)
            next_cell.append(cell_output)
        return self.decoder_projector(output), (
            torch.stack(next_hidden),
            torch.stack(next_cell),
        )

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
        if token_ids.device.type == "cpu" and token_ids.numel() == 1:
            input_gates = self._input_gate_table()
            if input_gates is not None:
                return self._predict_from_input_gates(
                    F.embedding(token_ids, input_gates), state, rows
                )
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
    return [
        hypothesis.token_ids
        for hypothesis in greedy_decode_transducer_batch_with_timestamps(
            encoder_projected=encoder_projected,
            lengths=lengths,
            decoder=decoder,
            joint_logits=joint_logits,
            config=config,
        )
    ]


def greedy_decode_transducer_batch_with_timestamps(
    *,
    encoder_projected: torch.Tensor,
    lengths: torch.Tensor,
    decoder: TransducerPredictionDecoder,
    joint_logits,
    config: TransducerDecodeConfig,
) -> list[TransducerHypothesis]:
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
    if not is_tdt:
        return _greedy_decode_rnnt_batch_with_timestamps(
            encoder_projected=encoder_projected,
            lengths=lengths,
            decoder=decoder,
            joint_logits=joint_logits,
            config=config,
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
    output_timesteps = torch.zeros(
        (batch_size, max_output_tokens),
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
            output_timesteps[nonblank_rows, emit_positions] = time_idx[nonblank_rows]
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
    output_timesteps_cpu = output_timesteps.cpu()
    output_lengths_cpu = output_lengths.cpu().tolist()
    return [
        TransducerHypothesis(
            token_ids=output_ids_cpu[row, :output_length].tolist(),
            timesteps=output_timesteps_cpu[row, : max(0, output_length - 1)].tolist(),
        )
        for row, output_length in enumerate(output_lengths_cpu)
    ]


def _greedy_decode_rnnt_batch_with_timestamps(
    *,
    encoder_projected: torch.Tensor,
    lengths: torch.Tensor,
    decoder: TransducerPredictionDecoder,
    joint_logits,
    config: TransducerDecodeConfig,
) -> list[TransducerHypothesis]:
    device = encoder_projected.device
    batch_size = int(encoder_projected.shape[0])
    max_encoder_frames = int(encoder_projected.shape[1])
    if device.type == "cpu" and batch_size == 1:
        return [
            _greedy_decode_rnnt_single_cpu_with_timestamps(
                encoder_projected=encoder_projected[0],
                length=int(lengths[0].item()),
                decoder=decoder,
                joint_logits=joint_logits,
                config=config,
            )
        ]

    time_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
    last_tokens = torch.full(
        (batch_size,),
        config.blank_token_id,
        dtype=torch.long,
        device=device,
    )
    has_last_token = torch.zeros(batch_size, dtype=torch.bool, device=device)
    state: tuple[torch.Tensor, torch.Tensor] | None = None
    cached_pred_state: torch.Tensor | None = None
    cached_next_state: tuple[torch.Tensor, torch.Tensor] | None = None
    cached_prediction_valid = torch.zeros(batch_size, dtype=torch.bool, device=device)

    max_output_tokens = max_encoder_frames * config.max_symbols_per_step + 1
    output_ids = torch.full(
        (batch_size, max_output_tokens),
        config.eos_token_id,
        dtype=torch.long,
        device=device,
    )
    output_timesteps = torch.zeros(
        (batch_size, max_output_tokens),
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
            uncached_positions = torch.nonzero(
                ~cached_prediction_valid[loop_rows],
                as_tuple=False,
            ).flatten()
            if uncached_positions.numel() > 0:
                uncached_rows = loop_rows[uncached_positions]
                labels = torch.where(
                    has_last_token[uncached_rows],
                    last_tokens[uncached_rows],
                    torch.full_like(uncached_rows, config.blank_token_id),
                )
                pred_state, next_state = decoder.predict_batch(
                    labels,
                    state,
                    uncached_rows,
                )
                if cached_pred_state is None:
                    num_layers = next_state[0].shape[0]
                    hidden_size = next_state[0].shape[-1]
                    cached_pred_state = torch.zeros(
                        batch_size,
                        hidden_size,
                        dtype=pred_state.dtype,
                        device=device,
                    )
                    cached_next_state = (
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
                    if state is None:
                        state = (
                            torch.zeros_like(cached_next_state[0]),
                            torch.zeros_like(cached_next_state[1]),
                        )

                assert cached_pred_state is not None
                assert cached_next_state is not None
                cached_pred_state[uncached_rows] = pred_state
                cached_next_state[0][:, uncached_rows, :] = next_state[0]
                cached_next_state[1][:, uncached_rows, :] = next_state[1]
                cached_prediction_valid[uncached_rows] = True

            assert cached_pred_state is not None
            pred_state = cached_pred_state[loop_rows]
            encoder_state = encoder_projected[loop_rows, time_idx[loop_rows]]
            logits = joint_logits(encoder_state, pred_state)

            token_logits = logits[:, : config.vocab_size].float()
            tokens = token_logits.argmax(dim=1)
            blank_tokens = tokens == config.blank_token_id
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
            output_timesteps[nonblank_rows, emit_positions] = time_idx[nonblank_rows]
            output_lengths[nonblank_rows] += 1

            last_tokens[nonblank_rows] = tokens[nonblank_tokens]
            has_last_token[nonblank_rows] = True
            if nonblank_rows.numel() > 0:
                assert state is not None
                assert cached_next_state is not None
                state[0][:, nonblank_rows, :] = cached_next_state[0][
                    :, nonblank_rows, :
                ]
                state[1][:, nonblank_rows, :] = cached_next_state[1][
                    :, nonblank_rows, :
                ]
                cached_prediction_valid[nonblank_rows] = False

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
    output_timesteps_cpu = output_timesteps.cpu()
    output_lengths_cpu = output_lengths.cpu().tolist()
    return [
        TransducerHypothesis(
            token_ids=output_ids_cpu[row, :output_length].tolist(),
            timesteps=output_timesteps_cpu[row, : max(0, output_length - 1)].tolist(),
        )
        for row, output_length in enumerate(output_lengths_cpu)
    ]


def _greedy_decode_rnnt_single_cpu_with_timestamps(
    *,
    encoder_projected: torch.Tensor,
    length: int,
    decoder: TransducerPredictionDecoder,
    joint_logits,
    config: TransducerDecodeConfig,
) -> TransducerHypothesis:
    assert encoder_projected.device.type == "cpu"

    output_ids: list[int] = []
    timesteps: list[int] = []
    state: tuple[torch.Tensor, torch.Tensor] | None = None
    cached_pred_state: torch.Tensor | None = None
    cached_next_state: tuple[torch.Tensor, torch.Tensor] | None = None
    cached_prediction_valid = False
    label = torch.tensor([config.blank_token_id], dtype=torch.long)
    rows = torch.tensor([0], dtype=torch.long)

    time_idx = 0
    length = min(length, int(encoder_projected.shape[0]))
    while time_idx < length:
        symbols_added = 0
        while symbols_added < config.max_symbols_per_step:
            if not cached_prediction_valid:
                pred_state, next_state = decoder.predict_batch(label, state, rows)
                cached_pred_state = pred_state
                cached_next_state = next_state
                cached_prediction_valid = True

            assert cached_pred_state is not None
            logits = joint_logits(
                encoder_projected[time_idx : time_idx + 1],
                cached_pred_state,
            )
            token = int(logits[0, : config.vocab_size].float().argmax().item())

            if token == config.blank_token_id:
                time_idx += 1
                break

            output_ids.append(token)
            timesteps.append(time_idx)
            label[0] = token
            state = cached_next_state
            cached_prediction_valid = False
            symbols_added += 1
        else:
            time_idx += 1

    output_ids.append(config.eos_token_id)
    return TransducerHypothesis(token_ids=output_ids, timesteps=timesteps)


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
