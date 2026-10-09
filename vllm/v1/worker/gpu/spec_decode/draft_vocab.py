# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Restricted draft vocabularies for drafters that share the target's LM head.

A drafter such as DFlash computes its draft logits with the target's full LM
head, which for a 248k-token vocabulary is the largest GEMM it runs. Drafting
only from a frequency-ranked subset of the vocabulary (FR-Spec, VocabTrim)
replaces it with a much smaller GEMM. The target still verifies every draft
against its full vocabulary, so generated tokens cannot change: a target token
outside the subset is simply never drafted, which costs acceptance only.
"""

import json
from pathlib import Path

import torch
import torch.nn.functional as F


def load_draft_vocab(path: str, vocab_size: int) -> torch.Tensor:
    """The sorted, de-duplicated token ids listed in a JSON file.

    The file holds a JSON array of target-vocabulary token ids.
    """
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, list) or not raw:
        raise ValueError(
            f"draft_vocab_path {path!r} must contain a non-empty JSON list"
        )
    if not all(
        isinstance(token_id, int) and not isinstance(token_id, bool) for token_id in raw
    ):
        raise ValueError(f"draft_vocab_path {path!r} must list integer token ids")
    token_ids = sorted(set(raw))
    if token_ids[0] < 0 or token_ids[-1] >= vocab_size:
        raise ValueError(
            f"draft_vocab_path {path!r} lists token ids outside the target "
            f"vocabulary [0, {vocab_size})"
        )
    return torch.tensor(token_ids, dtype=torch.long)


def draft_head_weight(
    lm_head_weight: torch.Tensor, token_ids: torch.Tensor
) -> torch.Tensor:
    """The rows of the LM head weight for the drafted token ids, as a copy."""
    return lm_head_weight.index_select(
        0, token_ids.to(lm_head_weight.device)
    ).contiguous()


def greedy_draft_tokens(
    hidden_states: torch.Tensor,
    head_weight: torch.Tensor,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    """Greedy draft tokens, as target-vocabulary ids, chosen within the subset.

    Positive logit scaling cannot change an argmax, so the head's logit scale
    is not applied.
    """
    logits = F.linear(hidden_states, head_weight)
    return token_ids[logits.argmax(dim=-1)]
