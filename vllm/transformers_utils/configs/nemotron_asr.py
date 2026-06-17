# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for NVIDIA Nemotron 3.5 ASR streaming checkpoints."""

from __future__ import annotations

from typing import Any

from transformers import ParakeetEncoderConfig, PretrainedConfig

NEMOTRON_ASR_DEFAULT_LANGUAGE = "en-US"

NEMOTRON_ASR_LANGUAGE_ALIASES = {
    "en": "en-US",
    "pt": "pt-BR",
    "es": "es-US",
    "fr": "fr-FR",
    "it": "it-IT",
    "nl": "nl-NL",
    "de": "de-DE",
    "tr": "tr-TR",
    "ru": "ru-RU",
    "ar": "ar-AR",
    "hi": "hi-IN",
    "ja": "ja-JP",
    "ko": "ko-KR",
    "vi": "vi-VN",
    "uk": "uk-UA",
    "pl": "pl-PL",
    "sv": "sv-SE",
    "cs": "cs-CZ",
    "nb": "nb-NO",
    "da": "da-DK",
    "bg": "bg-BG",
    "fi": "fi-FI",
    "hr": "hr-HR",
    "sk": "sk-SK",
    "zh": "zh-CN",
    "hu": "hu-HU",
    "ro": "ro-RO",
    "et": "et-EE",
    "el": "el-GR",
    "lt": "lt-LT",
    "lv": "lv-LV",
    "mt": "mt-MT",
    "sl": "sl-SI",
    "he": "he-IL",
    "th": "th-TH",
    "nn": "nn-NO",
}


class NemotronASRConfig(PretrainedConfig):
    """vLLM/HF layout for Nemotron 3.5 ASR.

    The public checkpoint is a NeMo ``.nemo`` archive. The companion converter
    writes this config plus safetensors so the runtime can stay normal vLLM
    PyTorch instead of importing NeMo for inference.
    """

    model_type = "nemotron_asr"
    sub_configs = {"encoder_config": ParakeetEncoderConfig}

    def __init__(
        self,
        *,
        encoder_config: dict[str, Any] | ParakeetEncoderConfig | None = None,
        vocab_size: int = 8193,
        blank_token_id: int = 8192,
        decoder_hidden_size: int = 640,
        num_decoder_layers: int = 2,
        max_symbols_per_step: int = 10,
        sample_rate: int = 16000,
        hidden_act: str = "relu",
        prompt_dim: int = 128,
        prompt_default_language: str = NEMOTRON_ASR_DEFAULT_LANGUAGE,
        prompt_language_to_id: dict[str, int] | None = None,
        chunk_ms: int = 1120,
        realtime_chunk_ms: int = 320,
        frame_shift_seconds: float = 0.08,
        pad_token_id: int = 0,
        eos_token_id: int | None = None,
        **kwargs,
    ) -> None:
        eos_token_id = blank_token_id if eos_token_id is None else eos_token_id
        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            bos_token_id=blank_token_id,
            decoder_start_token_id=blank_token_id,
            **kwargs,
        )

        if isinstance(encoder_config, dict):
            encoder_config = ParakeetEncoderConfig(**encoder_config)
        elif encoder_config is None:
            encoder_config = ParakeetEncoderConfig()

        encoder_config.sampling_rate = sample_rate

        self.encoder_config = encoder_config
        self.vocab_size = vocab_size
        self.blank_token_id = blank_token_id
        self.decoder_hidden_size = decoder_hidden_size
        self.hidden_size = decoder_hidden_size
        self.num_decoder_layers = num_decoder_layers
        self.num_hidden_layers = 0
        self.num_attention_heads = 0
        self.max_symbols_per_step = max_symbols_per_step
        self.sample_rate = sample_rate
        self.hidden_act = hidden_act
        self.prompt_dim = prompt_dim
        self.prompt_default_language = prompt_default_language
        self.prompt_language_to_id = prompt_language_to_id or {
            "auto": 0,
            NEMOTRON_ASR_DEFAULT_LANGUAGE: 1,
        }
        self.chunk_ms = chunk_ms
        self.realtime_chunk_ms = realtime_chunk_ms
        self.frame_shift_seconds = frame_shift_seconds
        self.is_encoder_decoder = True
        self.is_cache_aware_streaming = True

    def resolve_prompt_language(self, language: str | None) -> str:
        if language in (None, ""):
            language = self.prompt_default_language
        elif language != "auto":
            language = NEMOTRON_ASR_LANGUAGE_ALIASES.get(language, language)

        if language not in self.prompt_language_to_id:
            raise ValueError(
                f"Unsupported Nemotron ASR language {language!r}; expected one of "
                f"{sorted(self.prompt_language_to_id)}."
            )
        return language

    def prompt_id_for_language(self, language: str | None) -> int:
        return int(self.prompt_language_to_id[self.resolve_prompt_language(language)])


__all__ = [
    "NEMOTRON_ASR_DEFAULT_LANGUAGE",
    "NEMOTRON_ASR_LANGUAGE_ALIASES",
    "NemotronASRConfig",
]
