# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark Parakeet TDT greedy batch decode before/after GPU-first changes.

This is a focused microbenchmark for the Python decode loop. It uses a small
synthetic Parakeet-like decoder/joint network so the benchmark can run without
model weights while still exercising the same control flow as production.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from types import SimpleNamespace

import torch
import torch.nn as nn


class BenchmarkDecoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int) -> None:
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


class BenchmarkParakeet(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            vocab_size=vocab_size,
            blank_token_id=vocab_size - 1,
            eos_token_id=vocab_size - 2,
            durations=[0, 1, 2, 4],
            max_symbols_per_step=4,
        )
        self.encoder_projector = nn.Linear(hidden_size, hidden_size)
        self.decoder = BenchmarkDecoder(vocab_size, hidden_size, num_layers)
        self.joint = nn.Linear(hidden_size, vocab_size + len(self.config.durations))
        self.to(device=device)

    def _joint_logits(
        self,
        encoder_state: torch.Tensor,
        decoder_state: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.joint(torch.relu(encoder_state + decoder_state))
        logits[:, self.config.vocab_size :] = -1.0e9
        logits[:, self.config.vocab_size + 1] = 1.0e9
        return logits


def single_greedy_decode(
    model: BenchmarkParakeet,
    encoder_output: torch.Tensor,
) -> list[int]:
    cfg = model.config
    device = encoder_output.device
    token_ids: list[int] = []
    state: tuple[torch.Tensor, torch.Tensor] | None = None
    last_token: int | None = None
    encoder_projected = model.encoder_projector(encoder_output)

    time_idx = 0
    out_len = int(encoder_output.shape[0])
    while time_idx < out_len:
        encoder_state = encoder_projected[time_idx : time_idx + 1]
        symbols_added = 0
        need_loop = True
        skip = 1

        while need_loop and symbols_added < cfg.max_symbols_per_step:
            label = cfg.blank_token_id if last_token is None else last_token
            pred_state, next_state = model.decoder.predict(label, state, device)
            logits = model._joint_logits(encoder_state, pred_state)[0]

            token_logits = logits[: cfg.vocab_size].float()
            duration_logits = logits[cfg.vocab_size :].float()
            token = token_logits.argmax()

            duration_idx = int(duration_logits.argmax().item())
            skip = cfg.durations[duration_idx]
            token_id = int(token.item())
            if token_id == cfg.blank_token_id and skip == 0:
                skip = 1

            if token_id != cfg.blank_token_id:
                token_ids.append(token_id)
                state = next_state
                last_token = token_id

            symbols_added += 1
            time_idx += skip
            need_loop = skip == 0

        if need_loop:
            time_idx += 1

    token_ids.append(cfg.eos_token_id)
    return token_ids


def old_greedy_decode_batch(
    model: BenchmarkParakeet,
    encoder_outputs: Sequence[torch.Tensor],
) -> list[list[int]]:
    if not encoder_outputs:
        return []
    if len(encoder_outputs) == 1:
        return [single_greedy_decode(model, encoder_outputs[0])]

    cfg = model.config
    device = encoder_outputs[0].device
    lengths = torch.tensor(
        [int(encoder_output.shape[0]) for encoder_output in encoder_outputs],
        dtype=torch.long,
        device=device,
    )
    encoder_projected = model.encoder_projector(
        nn.utils.rnn.pad_sequence(list(encoder_outputs), batch_first=True)
    )

    batch_size = len(encoder_outputs)
    token_ids: list[list[int]] = [[] for _ in range(batch_size)]
    time_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
    last_tokens = torch.full(
        (batch_size,),
        cfg.blank_token_id,
        dtype=torch.long,
        device=device,
    )
    has_last_token = torch.zeros(batch_size, dtype=torch.bool, device=device)
    state: tuple[torch.Tensor, torch.Tensor] | None = None

    while bool(torch.any(time_idx < lengths).item()):
        active = torch.nonzero(time_idx < lengths, as_tuple=False).flatten()
        symbols_added = torch.zeros(active.shape[0], dtype=torch.long, device=device)
        needs_loop = torch.ones(active.shape[0], dtype=torch.bool, device=device)

        while active.numel() and bool(torch.any(needs_loop).item()):
            loop_positions = torch.nonzero(needs_loop, as_tuple=False).flatten()
            loop_rows = active[loop_positions]
            labels = torch.where(
                has_last_token[loop_rows],
                last_tokens[loop_rows],
                torch.full_like(loop_rows, cfg.blank_token_id),
            )

            pred_state, next_state = model.decoder.predict_batch(
                labels, state, loop_rows
            )
            encoder_state = encoder_projected[loop_rows, time_idx[loop_rows]]
            logits = model._joint_logits(encoder_state, pred_state)

            token_logits = logits[:, : cfg.vocab_size].float()
            duration_logits = logits[:, cfg.vocab_size :].float()
            tokens = token_logits.argmax(dim=1)
            duration_indices = duration_logits.argmax(dim=1).tolist()
            skips = torch.tensor(
                [cfg.durations[index] for index in duration_indices],
                dtype=torch.long,
                device=device,
            )
            blank_tokens = tokens == cfg.blank_token_id
            skips = torch.where(
                blank_tokens & (skips == 0),
                torch.ones_like(skips),
                skips,
            )
            nonblank_tokens = ~blank_tokens

            if bool(torch.any(nonblank_tokens).item()):
                nonblank_rows = loop_rows[nonblank_tokens]
                nonblank_tokens_cpu = tokens[nonblank_tokens].tolist()
                for row, token_id in zip(
                    nonblank_rows.tolist(), nonblank_tokens_cpu, strict=True
                ):
                    token_ids[row].append(int(token_id))

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
            needs_loop[loop_positions] = (skips == 0) & (
                symbols_added[loop_positions] < cfg.max_symbols_per_step
            )

        if active.numel():
            still_looping = active[needs_loop]
            if still_looping.numel():
                time_idx[still_looping] += 1

    for sequence in token_ids:
        sequence.append(cfg.eos_token_id)

    return token_ids


def new_greedy_decode_batch(
    model: BenchmarkParakeet,
    encoder_outputs: Sequence[torch.Tensor],
) -> list[list[int]]:
    if not encoder_outputs:
        return []

    cfg = model.config
    device = encoder_outputs[0].device
    batch_size = len(encoder_outputs)
    max_encoder_frames = max(
        int(encoder_output.shape[0]) for encoder_output in encoder_outputs
    )
    lengths = torch.tensor(
        [int(encoder_output.shape[0]) for encoder_output in encoder_outputs],
        dtype=torch.long,
        device=device,
    )
    encoder_projected = model.encoder_projector(
        nn.utils.rnn.pad_sequence(list(encoder_outputs), batch_first=True)
    )

    time_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
    last_tokens = torch.full(
        (batch_size,),
        cfg.blank_token_id,
        dtype=torch.long,
        device=device,
    )
    has_last_token = torch.zeros(batch_size, dtype=torch.bool, device=device)
    state: tuple[torch.Tensor, torch.Tensor] | None = None
    duration_values = torch.tensor(cfg.durations, dtype=torch.long, device=device)
    max_output_tokens = max_encoder_frames * cfg.max_symbols_per_step + 1
    output_ids = torch.full(
        (batch_size, max_output_tokens),
        cfg.eos_token_id,
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

        for _symbol_step in range(cfg.max_symbols_per_step):
            loop_positions = torch.nonzero(needs_loop, as_tuple=False).flatten()
            if loop_positions.numel() == 0:
                break
            loop_rows = active[loop_positions]
            labels = torch.where(
                has_last_token[loop_rows],
                last_tokens[loop_rows],
                torch.full_like(loop_rows, cfg.blank_token_id),
            )

            pred_state, next_state = model.decoder.predict_batch(
                labels, state, loop_rows
            )
            encoder_state = encoder_projected[loop_rows, time_idx[loop_rows]]
            logits = model._joint_logits(encoder_state, pred_state)

            token_logits = logits[:, : cfg.vocab_size].float()
            duration_logits = logits[:, cfg.vocab_size :].float()
            tokens = token_logits.argmax(dim=1)
            duration_indices = duration_logits.argmax(dim=1)
            skips = duration_values[duration_indices]
            blank_tokens = tokens == cfg.blank_token_id
            skips = torch.where(
                blank_tokens & (skips == 0),
                torch.ones_like(skips),
                skips,
            )
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
            still_looping_after_limit[loop_positions] = skips == 0
            needs_loop[loop_positions] = (skips == 0) & (
                symbols_added[loop_positions] < cfg.max_symbols_per_step
            )

        advance_after_limit = active[
            still_looping_after_limit & (symbols_added >= cfg.max_symbols_per_step)
        ]
        time_idx[advance_after_limit] += 1

    eos_positions = output_lengths.clamp(max=max_output_tokens - 1)
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device)
    output_ids[batch_indices, eos_positions] = cfg.eos_token_id
    output_lengths = eos_positions + 1

    output_ids_cpu = output_ids.cpu()
    output_lengths_cpu = output_lengths.cpu().tolist()
    return [
        output_ids_cpu[row, :output_length].tolist()
        for row, output_length in enumerate(output_lengths_cpu)
    ]


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_decode(
    fn,
    model: BenchmarkParakeet,
    encoder_outputs: Sequence[torch.Tensor],
    *,
    device: torch.device,
    iterations: int,
    warmup: int,
) -> tuple[float, int]:
    last_output: list[list[int]] = []
    with torch.inference_mode():
        for _ in range(warmup):
            last_output = fn(model, encoder_outputs)
        sync(device)
        started_at = time.perf_counter()
        for _ in range(iterations):
            last_output = fn(model, encoder_outputs)
        sync(device)
    elapsed_ms = (time.perf_counter() - started_at) * 1000 / iterations
    total_tokens = sum(len(sequence) for sequence in last_output)
    return elapsed_ms, total_tokens


def parse_csv_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", default="2,4,8,12")
    parser.add_argument("--encoder-frames", type=int, default=750)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--devices", default="auto", help="'auto' or csv: cpu,cuda")
    args = parser.parse_args()

    torch.manual_seed(1234)
    devices = ["cpu"]
    if args.devices == "auto":
        if torch.cuda.is_available():
            devices.append("cuda")
    else:
        devices = [device.strip() for device in args.devices.split(",") if device]

    print("device,batch_size,encoder_frames,impl,decode_ms,tokens_per_sec,total_tokens")
    for device_name in devices:
        device = torch.device(device_name)
        model = BenchmarkParakeet(
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            device=device,
        )
        model.eval()
        for batch_size in parse_csv_ints(args.batches):
            encoder_outputs = [
                torch.randn(args.encoder_frames, args.hidden_size, device=device)
                for _ in range(batch_size)
            ]
            old_ms, old_tokens = time_decode(
                old_greedy_decode_batch,
                model,
                encoder_outputs,
                device=device,
                iterations=args.iterations,
                warmup=args.warmup,
            )
            new_ms, new_tokens = time_decode(
                new_greedy_decode_batch,
                model,
                encoder_outputs,
                device=device,
                iterations=args.iterations,
                warmup=args.warmup,
            )
            if old_greedy_decode_batch(
                model, encoder_outputs
            ) != new_greedy_decode_batch(model, encoder_outputs):
                raise AssertionError("old and new decode outputs diverged")
            for impl, decode_ms, total_tokens in (
                ("old", old_ms, old_tokens),
                ("new", new_ms, new_tokens),
            ):
                tokens_per_sec = total_tokens / (decode_ms / 1000)
                print(
                    f"{device_name},{batch_size},{args.encoder_frames},{impl},"
                    f"{decode_ms:.3f},{tokens_per_sec:.2f},{total_tokens}"
                )


if __name__ == "__main__":
    main()
