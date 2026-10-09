# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import dataclasses
import json
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from vllm.config.speculative import SpeculativeConfig
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3ForCausalLM
from vllm.v1.worker.gpu.spec_decode.draft_vocab import (
    draft_head_weight,
    greedy_draft_tokens,
    load_draft_vocab,
)
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator


def _write(tmp_path, value) -> str:
    path = tmp_path / "draft_vocab.json"
    path.write_text(json.dumps(value))
    return str(path)


def test_load_draft_vocab_sorts_and_deduplicates(tmp_path):
    token_ids = load_draft_vocab(_write(tmp_path, [7, 2, 7, 0]), vocab_size=8)

    assert token_ids.tolist() == [0, 2, 7]
    assert token_ids.dtype == torch.long


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ([], "non-empty JSON list"),
        ({"ids": [1]}, "non-empty JSON list"),
        ([1, "2"], "integer token ids"),
        ([True, 2], "integer token ids"),
        ([1, 8], "outside the target vocabulary"),
        ([-1, 2], "outside the target vocabulary"),
    ],
)
def test_load_draft_vocab_rejects_invalid_lists(tmp_path, value, message):
    with pytest.raises(ValueError, match=message):
        load_draft_vocab(_write(tmp_path, value), vocab_size=8)


def test_greedy_draft_tokens_choose_within_the_subset_and_map_to_target_ids():
    weight = torch.eye(6, 4)[[0, 1, 2, 3, 0, 1]] * torch.arange(1, 7).unsqueeze(1)
    token_ids = torch.tensor([1, 3, 4])
    hidden = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    )

    drafts = greedy_draft_tokens(
        hidden, draft_head_weight(weight, token_ids), token_ids
    )

    full = (hidden @ weight.T).argmax(dim=-1)
    # Row 0's full-vocabulary best (3) is in the subset, so the drafts agree.
    assert full[0].item() == 3 and drafts[0].item() == 3
    # Row 1's best (4) is in the subset too. Row 2 prefers 5, outside it, so the
    # drafter falls back to the best token it may propose.
    assert full[1:].tolist() == [4, 5]
    assert drafts[1:].tolist() == [4, 1]


def _dflash_with_head(weight: torch.Tensor) -> DFlashQwen3ForCausalLM:
    model = object.__new__(DFlashQwen3ForCausalLM)  # type: ignore[type-abstract]
    nn.Module.__init__(model)
    model.lm_head = SimpleNamespace(weight=weight)
    model.draft_id_to_target_id = None
    return model


def test_dflash_set_draft_vocab_drafts_from_a_copy_of_the_shared_head():
    weight = torch.randn(10, 4)
    model = _dflash_with_head(weight)
    token_ids = torch.tensor([2, 5, 9])

    model.set_draft_vocab(token_ids)
    hidden = torch.randn(3, 4)

    torch.testing.assert_close(model.draft_head_weight, weight[token_ids])
    expected = token_ids[(hidden @ weight[token_ids].T).argmax(dim=-1)]
    assert model.greedy_draft_tokens(hidden).tolist() == expected.tolist()
    assert "draft_vocab_ids" not in model.state_dict()


def test_dflash_set_draft_vocab_refuses_a_drafter_with_its_own_vocabulary():
    model = _dflash_with_head(torch.randn(10, 4))
    model.draft_id_to_target_id = torch.zeros(4, dtype=torch.long)

    with pytest.raises(ValueError, match="its own reduced vocabulary"):
        model.set_draft_vocab(torch.tensor([1, 2]))


def test_dflash_set_draft_vocab_needs_an_unsplit_head():
    model = _dflash_with_head(torch.randn(10, 4))
    model.lm_head.tp_size = 2

    with pytest.raises(ValueError, match="tensor-parallel"):
        model.set_draft_vocab(torch.tensor([1, 2]))


def test_dflash_set_draft_vocab_needs_an_unquantized_head():
    model = _dflash_with_head(torch.zeros(10, 4, dtype=torch.int8))

    with pytest.raises(ValueError, match="unquantized shared LM head"):
        model.set_draft_vocab(torch.tensor([1, 2]))


def _sample_greedy_draft(model: Any, hidden: torch.Tensor) -> torch.Tensor:
    speculator = SimpleNamespace(
        use_local_argmax_reduction=False,
        model=model,
        acceptance_estimator=None,
    )
    speculator.compute_draft_logits = lambda h, step: model.compute_logits(h)
    speculator._maybe_predict_acceptance = lambda *args: None
    no_tensor = torch.empty(0)
    return DraftModelSpeculator.sample_draft(
        speculator,  # type: ignore[arg-type]
        hidden,
        sample_src_positions=no_tensor,
        idx_mapping=no_tensor,
        temperature=no_tensor,
        seeds=no_tensor,
        draft_step=no_tensor,
        draft_logits=None,
    )


def test_greedy_sample_draft_uses_the_restricted_vocabulary():
    model = SimpleNamespace(
        draft_vocab_ids=torch.tensor([1, 2]),
        greedy_draft_tokens=lambda h: torch.tensor([7, 8]),
        compute_logits=lambda h: pytest.fail("full-vocabulary logits computed"),
    )

    assert _sample_greedy_draft(model, torch.randn(2, 4)).tolist() == [7, 8]


def test_greedy_sample_draft_without_a_restricted_vocabulary_uses_full_logits():
    logits = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    model = SimpleNamespace(compute_logits=lambda h: logits)

    assert _sample_greedy_draft(model, torch.randn(2, 4)).tolist() == [1, 0]


def _verify_args(**overrides):
    # SpeculativeConfig's defaults, without resolving a draft model.
    values = {
        field.name: field.default
        for field in dataclasses.fields(SpeculativeConfig)
        if field.default is not dataclasses.MISSING
    }
    values.update(
        method="dflash",
        num_speculative_tokens=4,
        draft_model_config=None,
        draft_vocab_path="/tmp/draft_vocab.json",
    )
    values.update(overrides)
    config = object.__new__(SpeculativeConfig)
    config.__dict__.update(values)
    return config._verify_args()  # type: ignore[operator]


def test_draft_vocab_path_accepts_greedy_dflash():
    _verify_args()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"method": "eagle3"}, "only works with method='dflash'"),
        (
            {"draft_sample_method": "probabilistic"},
            "only supports greedy draft sampling",
        ),
        ({"use_local_argmax_reduction": True}, "use_local_argmax_reduction"),
        ({"enable_adaptive_verification": True}, "Adaptive verification"),
    ],
)
def test_draft_vocab_path_is_validated(overrides, message):
    with pytest.raises(ValueError, match=message):
        _verify_args(**overrides)
