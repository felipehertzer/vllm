import torch

from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import (
    PLACEHOLDER_TOKEN_ID,
    _expand_batch_to_tokens_torch,
    _rejection_sample_torch,
)


def _sampling_metadata() -> SamplingMetadata:
    return SamplingMetadata(
        temperature=None,
        all_greedy=True,
        all_random=False,
        top_p=None,
        top_k=None,
        generators={},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.empty(0),
        presence_penalties=torch.empty(0),
        repetition_penalties=torch.empty(0),
        output_token_ids=[],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )


def test_expand_batch_to_tokens_torch_repeats_and_replaces() -> None:
    expanded = _expand_batch_to_tokens_torch(
        torch.tensor([0.0, 0.5, 1.0]),
        torch.tensor([2, 5, 6], dtype=torch.int32),
        num_tokens=6,
        replace_from=0,
        replace_to=1,
    )

    assert expanded.tolist() == [1.0, 1.0, 0.5, 0.5, 0.5, 1.0]


def test_rejection_sample_torch_keeps_accepted_and_replaces_rejected() -> None:
    output = torch.full((1, 4), PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
    draft_token_ids = torch.tensor([1, 2, 3], dtype=torch.int32)
    target_logits = torch.full((3, 8), -10.0)
    target_logits[0, 1] = 10.0
    target_logits[1, 2] = 10.0
    target_logits[2, 5] = 10.0

    result = _rejection_sample_torch(
        output_token_ids=output,
        draft_token_ids=draft_token_ids,
        num_draft_tokens=[3],
        max_spec_len=3,
        cu_num_draft_tokens=torch.tensor([3], dtype=torch.int32),
        draft_probs=None,
        target_logits=target_logits,
        bonus_token_ids=torch.tensor([7], dtype=torch.int32),
        is_greedy=None,
        sampling_metadata=_sampling_metadata(),
    )

    assert result.tolist() == [[1, 2, 5, PLACEHOLDER_TOKEN_ID]]


def test_rejection_sample_torch_adds_bonus_when_all_drafts_match() -> None:
    output = torch.full((1, 4), PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
    draft_token_ids = torch.tensor([1, 2, 3], dtype=torch.int32)
    target_logits = torch.full((3, 8), -10.0)
    target_logits[0, 1] = 10.0
    target_logits[1, 2] = 10.0
    target_logits[2, 3] = 10.0

    result = _rejection_sample_torch(
        output_token_ids=output,
        draft_token_ids=draft_token_ids,
        num_draft_tokens=[3],
        max_spec_len=3,
        cu_num_draft_tokens=torch.tensor([3], dtype=torch.int32),
        draft_probs=None,
        target_logits=target_logits,
        bonus_token_ids=torch.tensor([7], dtype=torch.int32),
        is_greedy=None,
        sampling_metadata=_sampling_metadata(),
    )

    assert result.tolist() == [[1, 2, 3, 7]]
