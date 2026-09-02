# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from copy import deepcopy
from functools import cache
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import _TYPES as _SAFETENSORS_TO_TORCH_DTYPE
from transformers import PretrainedConfig

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe  # noqa
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear import (
    MPLinearLayerConfig,
    choose_mp_linear_kernel,
)
from vllm.model_executor.layers.fused_moe import (
    FusedMoEConfig,
    FusedMoEExpertsModular,
    FusedMoEMethodBase,
    FusedMoEQuantConfig,
    FusedMoeWeightScaleSupported,
    RoutedExperts,
    SharedExperts,
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
    WNA16MoEBackend,
    convert_to_wna16_moe_kernel_format,
    make_wna16_moe_kernel,
    select_wna16_moe_backend,
)
from vllm.model_executor.layers.linear import LinearMethodBase, set_weight_attrs
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.layers.quantization.utils.gptq_utils import (
    get_dynamic_override,
    get_linear_quant_method,
    override_config,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    check_moe_marlin_supports_layer,
    get_marlin_input_dtype,
    marlin_make_workspace_new,
    marlin_repeat_scales_on_all_ranks,
    verify_marlin_supported,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kInt4StaticGroupScale,
    kInt8StaticGroupScale,
)
from vllm.model_executor.parameter import (
    ChannelQuantScaleParameter,
    GroupQuantScaleParameter,
    PackedColumnParameter,
    PackedvLLMParameter,
    RowvLLMParameter,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.transformers_utils.config import get_safetensors_params_metadata
from vllm.utils.collection_utils import is_list_of

logger = init_logger(__name__)


_MPS_GPTQ_BACKEND_NAME = "MPSDequantLinearKernel"
_MPS_GPTQ_DIRECT_MIN_OUTPUT_SIZE = 8192
_MPS_GPTQ_DIRECT_ZERO_WORD = 0x77777777
_MPS_GPTQ_PROFILE_TOTALS: dict[str, float] = {}
_MPS_GPTQ_PROFILE_COUNTS: dict[str, int] = {}


_MPS_GPTQ_DIRECT_GEMV_SRC = r"""
#include <metal_stdlib>
using namespace metal;

#define SIMD_THREADS 32
#define VEC8 8

kernel void gptq_gemv_gidx_vec8_simd32(device const half* x,
                                       device const int* qweight,
                                       device const half* scales,
                                       device const int* g_idx,
                                       device half* out,
                                       constant uint& N,
                                       constant uint& K,
                                       uint tid [[thread_index_in_threadgroup]],
                                       uint lane [[thread_index_in_simdgroup]],
                                       uint2 tg [[threadgroup_position_in_grid]]) {
  uint n0 = tg.y * VEC8;
  float acc[VEC8];

  for (uint j = 0; j < VEC8; ++j) {
    acc[j] = 0.0f;
  }

  for (uint k = tid; k < K; k += SIMD_THREADS) {
    uint k_pack = k >> 3;
    uint k_shift = (k & 7u) << 2;
    uint group = uint(g_idx[k]);
    float xv = float(x[k]);
    for (uint j = 0; j < VEC8; ++j) {
      uint n = n0 + j;
      if (n < N) {
        int packed = qweight[k_pack * N + n];
        int w = (packed >> k_shift) & 0xF;
        float wv = (float(w) - 8.0f) * float(scales[group * N + n]);
        acc[j] = fma(xv, wv, acc[j]);
      }
    }
  }

  for (uint j = 0; j < VEC8; ++j) {
    uint n = n0 + j;
    if (n < N) {
      float total = simd_sum(acc[j]);
      if (lane == 0) {
        out[n] = half(total);
      }
    }
  }
}
"""


@cache
def _get_mps_gptq_direct_gemv_lib():
    return torch.mps.compile_shader(_MPS_GPTQ_DIRECT_GEMV_SRC)


def _mps_gptq_profile_active(x: torch.Tensor) -> bool:
    return x.device.type == "mps" and envs.VLLM_MPS_PROFILE


def _mps_gptq_profile_start(x: torch.Tensor) -> float:
    torch.mps.synchronize()
    return time.perf_counter()


def _mps_gptq_profile_log(name: str, started: float, x: torch.Tensor) -> None:
    torch.mps.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000
    _MPS_GPTQ_PROFILE_TOTALS[name] = (
        _MPS_GPTQ_PROFILE_TOTALS.get(name, 0.0) + elapsed_ms
    )
    _MPS_GPTQ_PROFILE_COUNTS[name] = _MPS_GPTQ_PROFILE_COUNTS.get(name, 0) + 1
    count = _MPS_GPTQ_PROFILE_COUNTS[name]
    if count % 200 == 0:
        logger.info(
            "MPS profile auto_gptq %s: total=%.2fms count=%d avg=%.3fms",
            name,
            _MPS_GPTQ_PROFILE_TOTALS[name],
            count,
            _MPS_GPTQ_PROFILE_TOTALS[name] / count,
        )


def _unpack_gptq_rows(packed: torch.Tensor, bits: int) -> torch.Tensor:
    pack_factor = 32 // bits
    shifts = torch.arange(pack_factor, dtype=torch.int32) * bits
    unpacked = (packed.to(torch.int32).unsqueeze(-1) >> shifts) & ((1 << bits) - 1)
    return unpacked.permute(0, 2, 1).reshape(packed.shape[0] * pack_factor, -1)


def _unpack_gptq_cols(packed: torch.Tensor, bits: int) -> torch.Tensor:
    pack_factor = 32 // bits
    shifts = torch.arange(pack_factor, dtype=torch.int32) * bits
    unpacked = (packed.to(torch.int32).unsqueeze(-1) >> shifts) & ((1 << bits) - 1)
    return unpacked.reshape(packed.shape[0], packed.shape[1] * pack_factor)


def _dequantize_gptq_weight(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    g_idx: torch.Tensor,
    bits: int,
    group_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    qweight_cpu = qweight.detach().cpu()
    scales_cpu = scales.detach().cpu().to(torch.float32)
    qzeros_cpu = qzeros.detach().cpu()
    g_idx_cpu = g_idx.detach().cpu().to(torch.long)

    weight = _unpack_gptq_rows(qweight_cpu, bits).to(torch.float32)
    zeros = _unpack_gptq_cols(qzeros_cpu, bits).to(torch.float32) + 1
    input_size = weight.shape[0]

    if g_idx_cpu.numel() == input_size:
        group_ids = g_idx_cpu.clamp_(0, scales_cpu.shape[0] - 1)
    else:
        normalized_group_size = input_size if group_size == -1 else group_size
        group_ids = torch.arange(input_size, dtype=torch.long) // normalized_group_size
        group_ids.clamp_(0, scales_cpu.shape[0] - 1)

    weight = (weight - zeros[group_ids]) * scales_cpu[group_ids]
    # vLLM linear weights are [out_features, in_features].
    return weight.t().contiguous().to(device=device, dtype=dtype)


def get_moe_quant_method(
    config: "AutoGPTQConfig",
    layer: RoutedExperts,
    prefix: str,
    moe_method_cls: type,
):
    cloned_config = deepcopy(config)

    assert isinstance(layer, RoutedExperts)
    # False = skip module, None = no override, else = Positive match
    if (
        get_dynamic_override(  # noqa: E712
            cloned_config,  # noqa: E712
            layer_name=prefix,
        )
        == False
    ):  # noqa: E712
        return UnquantizedFusedMoEMethod(layer.moe_config)

    if prefix:
        # Dynamic per module/layer rules may override base config
        override_config(cloned_config, prefix=prefix)

    return moe_method_cls(cloned_config, layer.moe_config)


class AutoGPTQConfig(QuantizationConfig):
    """Config class for AutoGPTQ quantization using Marlin kernels."""

    # (num_bits, is_sym) -> quant_type
    TYPE_MAP = {
        (4, True): scalar_types.uint4b8,
        (8, True): scalar_types.uint8b128,
    }

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        desc_act: bool,
        is_sym: bool,
        lm_head_quantized: bool,
        dynamic: dict[str, dict[str, int | bool]],
        full_config: dict[str, Any],
        modules_in_block_to_quantize: list[str] | None = None,
    ) -> None:
        super().__init__()
        if desc_act and group_size == -1:
            # In this case, act_order == True is the same as act_order == False
            # (since we have only one group per output channel)
            desc_act = False

        # GPTQModel use `dynamic` config property to allow per module
        # quantization config so each module can be individually optimized.
        # Format is dict[str, dict] where key is a regex string that can
        # perform both positive ("+:" prefixed) or negative ("-:" prefixed)
        # matching of a module.
        # Default to positive match, override base quant config mode, if no
        # prefix is used. Value is in dict format of field key and override
        # value.
        # Negative matching will skip quantization init for this module
        # entirely:
        # non-quantized inference. More details and quantization examples can be
        # found at: https://github.com/ModelCloud/GPTQModel
        # Example:
        #  # last 1/2 of the layers 10-21 has 8bit vs 4bit for 0-9
        #  # last 1/4 of the layers 16-21 has 8bit and group_size 64
        # dynamic = {
        #  #`.*\.` matches the layers_node prefix
        #  # positive match layer 10-15
        #  r"+:.*\.(?:1[0-5])\..*": {"bits": 8,},
        #  # positive match layer 16-21
        #  r"+:.*\.(?:1[6-9]|20|21)\..*": {"bits": 8, "group_size": 64,},
        #  r"-:.*\.moe\..*": {}, # negative match (skip) all `moe` layers
        # }
        self.dynamic = dynamic

        self.weight_bits = weight_bits
        self.is_sym = is_sym

        self.pack_factor = 32 // weight_bits  # packed into int32
        self.group_size = group_size
        self.desc_act = desc_act
        self.lm_head_quantized = lm_head_quantized
        self.full_config = full_config

        if (weight_bits, is_sym) not in self.TYPE_MAP:
            raise ValueError(
                f"Unsupported quantization config: bits={weight_bits}, sym={is_sym}"
            )

        self.quant_type = self.TYPE_MAP[(weight_bits, is_sym)]

        self.modules_in_block_to_quantize = modules_in_block_to_quantize or []
        # used to identify GPTQ model quantized by autoround
        self.autoround_version = full_config.get("autoround_version", "")

    def __repr__(self) -> str:
        return (
            f"AutoGPTQConfig(quant_type={self.quant_type}, "
            f"group_size={self.group_size}, "
            f"desc_act={self.desc_act}, "
            f"lm_head_quantized={self.lm_head_quantized}, "
            f"dynamic={self.dynamic}, "
            f"modules_in_block_to_quantize={self.modules_in_block_to_quantize})"
        )

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "auto_gptq"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 60

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "AutoGPTQConfig":
        dynamic = cls.get_from_keys_or(config, ["dynamic"], default={})
        dynamic = {} if dynamic is None else dynamic

        weight_bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        desc_act = cls.get_from_keys(config, ["desc_act"])
        is_sym = cls.get_from_keys(config, ["sym"])
        lm_head_quantized = cls.get_from_keys_or(config, ["lm_head"], default=False)
        modules_in_block_to_quantize = cls.get_from_keys_or(
            config, ["modules_in_block_to_quantize"], default=None
        )
        return cls(
            weight_bits,
            group_size,
            desc_act,
            is_sym,
            lm_head_quantized,
            dynamic,
            config,
            modules_in_block_to_quantize,
        )

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg, user_quant, hf_config=None
    ) -> QuantizationMethods | None:
        """Override to use AutoGPTQ for compatible GPTQ models."""
        quant_method = hf_quant_cfg.get("quant_method", "").lower()

        if quant_method != "gptq":
            return None

        is_valid_user_quant = user_quant is None or user_quant in (
            "gptq",
            "gptq_marlin",
            "auto_gptq",
            "marlin",
        )

        if is_valid_user_quant:
            return cls.get_name()

        return None

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase | None":
        if isinstance(layer, RoutedExperts):
            from vllm.model_executor.layers.quantization.moe_wna16 import MoeWNA16Config

            if not check_moe_marlin_supports_layer(
                layer, self.group_size, allow_tile_padding=not self.desc_act
            ):
                logger.warning_once(
                    f"Layer '{prefix}' is not supported by GPTQMoeMarlin. "
                    "Falling back to Moe WNA16 kernels."
                )
                return MoeWNA16Config.from_config(self.full_config).get_quant_method(
                    layer, prefix
                )
            moe_quant_method = get_moe_quant_method(
                self, layer, prefix, AutoGPTQMoEMethod
            )
            if moe_quant_method is None:
                return None
            moe_quant_method.input_dtype = get_marlin_input_dtype(prefix)
            return moe_quant_method

        quant_method = get_linear_quant_method(
            self, layer, prefix, AutoGPTQLinearMethod
        )
        if quant_method is None:
            return None
        quant_method.input_dtype = get_marlin_input_dtype(prefix)
        return quant_method

    def apply_vllm_mapper(self, hf_to_vllm_mapper):
        if self.modules_in_block_to_quantize is not None:
            self.modules_in_block_to_quantize = hf_to_vllm_mapper.apply_list(
                self.modules_in_block_to_quantize
            )

    def maybe_update_config(
        self,
        model_name: str,
        hf_config: PretrainedConfig | None = None,
        revision: str | None = None,
    ):
        if self.modules_in_block_to_quantize:
            if is_list_of(self.modules_in_block_to_quantize, list):
                # original modules_in_block_to_quantize: list[list[str]]
                # flatten original modules_in_block_to_quantize
                self.modules_in_block_to_quantize = [
                    item
                    for sublist in self.modules_in_block_to_quantize
                    for item in sublist
                ]
            return

        unquant_dtypes = [torch.float16, torch.bfloat16, torch.float32]
        metadata = get_safetensors_params_metadata(model_name, revision=revision)
        quant_layers: set[str] = {
            param_name.rsplit(".", 1)[0]
            for param_name, info in metadata.items()
            if (dtype := info.get("dtype", None))
            and _SAFETENSORS_TO_TORCH_DTYPE[dtype] not in unquant_dtypes
        }
        self.modules_in_block_to_quantize = list(quant_layers)


class AutoGPTQLinearMethod(LinearMethodBase):
    """Linear method for AutoGPTQ using Marlin kernels.

    Args:
        quant_config: The AutoGPTQ quantization config.
    """

    _kernel_backends_being_used: set[str] = set()

    def __init__(self, quant_config: AutoGPTQConfig) -> None:
        self.quant_config = quant_config
        self.input_dtype = None
        self.quant_type = self.quant_config.quant_type
        self._use_mps_dequant = current_platform.is_mps()
        self._use_mps_direct_gemv = False

        # Verify supported on platform.
        if not self._use_mps_dequant:
            verify_marlin_supported(
                quant_type=self.quant_config.quant_type,
                group_size=self.quant_config.group_size,
            )

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)
        is_row_parallel = input_size != input_size_per_partition
        weight_loader = extra_weight_attrs.get("weight_loader")
        input_dtype = self.input_dtype

        if self._use_mps_dequant:
            kernel_type = None
            if _MPS_GPTQ_BACKEND_NAME not in self._kernel_backends_being_used:
                logger.info("Using %s for AutoGPTQLinearMethod", _MPS_GPTQ_BACKEND_NAME)
                self._kernel_backends_being_used.add(_MPS_GPTQ_BACKEND_NAME)
        else:
            mp_linear_kernel_config = MPLinearLayerConfig(
                full_weight_shape=(input_size, output_size),
                partition_weight_shape=(
                    input_size_per_partition,
                    output_size_per_partition,
                ),
                weight_type=self.quant_config.quant_type,
                act_type=params_dtype if input_dtype is None else input_dtype,
                group_size=self.quant_config.group_size,
                zero_points=False,
                has_g_idx=self.quant_config.desc_act,
            )

            kernel_type = choose_mp_linear_kernel(mp_linear_kernel_config)

            if kernel_type.__name__ not in self._kernel_backends_being_used:
                logger.info("Using %s for AutoGPTQLinearMethod", kernel_type.__name__)
                self._kernel_backends_being_used.add(kernel_type.__name__)

        # Normalize group_size
        if self.quant_config.group_size != -1:
            group_size = self.quant_config.group_size
        else:
            group_size = input_size

        # Determine sharding
        if marlin_repeat_scales_on_all_ranks(
            self.quant_config.desc_act, self.quant_config.group_size, is_row_parallel
        ):
            # By setting scale_dim == None, weight_loader will
            # repeat the scales on each GPU in TP>1 case.
            scales_and_zp_input_dim = None
            scales_and_zp_size = input_size // group_size
        else:
            # By setting scale_dim == 0, weight_loader will
            # shard the scales in TP>1 case.
            scales_and_zp_input_dim = 0
            scales_and_zp_size = input_size_per_partition // group_size

        # Quantized weights
        qweight = PackedvLLMParameter(
            data=torch.empty(
                input_size_per_partition // self.quant_config.pack_factor,
                output_size_per_partition,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=0,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=weight_loader,
        )

        # Activation order
        g_idx = RowvLLMParameter(
            data=torch.empty(
                input_size_per_partition,
                dtype=torch.int32,
            ),
            input_dim=0,
            weight_loader=weight_loader,
        )

        qzeros_args = {
            "data": torch.empty(
                scales_and_zp_size,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            "weight_loader": weight_loader,
        }
        weight_scale_args = {
            "data": torch.empty(
                scales_and_zp_size,
                output_size_per_partition,
                dtype=params_dtype,
            ),
            "weight_loader": weight_loader,
        }

        if scales_and_zp_input_dim is None:
            scales = ChannelQuantScaleParameter(output_dim=1, **weight_scale_args)
            qzeros = PackedColumnParameter(
                output_dim=1,
                packed_dim=1,
                packed_factor=self.quant_config.pack_factor,
                **qzeros_args,
            )

        else:
            scales = GroupQuantScaleParameter(
                output_dim=1, input_dim=0, **weight_scale_args
            )
            qzeros = PackedvLLMParameter(
                input_dim=0,
                output_dim=1,
                packed_dim=1,
                packed_factor=self.quant_config.pack_factor,
                **qzeros_args,
            )

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("g_idx", g_idx)
        layer.register_parameter("scales", scales)
        layer.register_parameter("qzeros", qzeros)

        if kernel_type is not None:
            self.kernel = kernel_type(
                mp_linear_kernel_config,
                w_q_param_name="qweight",
                w_s_param_name="scales",
                w_zp_param_name="qzeros",
                w_gidx_param_name="g_idx",
            )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self._use_mps_dequant:
            self._use_mps_direct_gemv = self._can_use_mps_direct_gemv(layer)
            if self._use_mps_direct_gemv:
                layer.qweight.data = layer.qweight.data.contiguous()
                layer.scales.data = layer.scales.data.contiguous()
                layer.g_idx.data = layer.g_idx.data.contiguous()

            weight = _dequantize_gptq_weight(
                layer.qweight.data,
                layer.scales.data,
                layer.qzeros.data,
                layer.g_idx.data,
                self.quant_config.quant_type.size_bits,
                self.quant_config.group_size,
                layer.params_dtype,
                layer.qweight.device,
            )
            names_to_remove = (
                ("qzeros",)
                if self._use_mps_direct_gemv
                else ("qweight", "scales", "qzeros", "g_idx")
            )
            for name in names_to_remove:
                if hasattr(layer, name):
                    delattr(layer, name)
            layer.register_parameter(
                "weight", torch.nn.Parameter(weight, requires_grad=False)
            )
            return

        self.kernel.process_weights_after_loading(layer)

    def _can_use_mps_direct_gemv(self, layer: torch.nn.Module) -> bool:
        if not envs.VLLM_MPS_GPTQ_DIRECT_GEMV:
            return False
        if self.quant_config.quant_type.size_bits != 4:
            return False
        if self.quant_config.group_size != 128 or not self.quant_config.is_sym:
            return False
        if layer.params_dtype != torch.float16:
            return False

        input_size = layer.qweight.shape[0] * self.quant_config.pack_factor
        output_size = layer.qweight.shape[1]
        if input_size % self.quant_config.pack_factor != 0:
            return False
        if output_size < _MPS_GPTQ_DIRECT_MIN_OUTPUT_SIZE:
            return False

        qzeros = layer.qzeros.detach().cpu()
        if qzeros.numel() == 0:
            return False
        first_zero_word = int(qzeros.flatten()[0])
        if first_zero_word != _MPS_GPTQ_DIRECT_ZERO_WORD:
            return False
        return bool(torch.all(qzeros == first_zero_word).item())

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._use_mps_dequant:
            profile = _mps_gptq_profile_active(x)
            if self._use_mps_direct_gemv:
                started = _mps_gptq_profile_start(x) if profile else 0.0
                direct_output = self._apply_mps_direct_gemv(layer, x, bias)
                if direct_output is not None:
                    if profile:
                        rows = x.reshape(-1, x.shape[-1]).shape[0]
                        _mps_gptq_profile_log(
                            f"direct rows={rows} in={x.shape[-1]} "
                            f"out={layer.qweight.shape[1]}",
                            started,
                            x,
                        )
                    return direct_output
            started = _mps_gptq_profile_start(x) if profile else 0.0
            output = F.linear(x, layer.weight, bias)
            if profile:
                rows = x.reshape(-1, x.shape[-1]).shape[0]
                _mps_gptq_profile_log(
                    f"dense rows={rows} in={x.shape[-1]} out={layer.weight.shape[0]}",
                    started,
                    x,
                )
            return output
        return self.kernel.apply_weights(layer, x, bias)

    def _apply_mps_direct_gemv(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if x.device.type != "mps" or x.dtype != torch.float16:
            return None
        input_size = layer.qweight.shape[0] * self.quant_config.pack_factor
        if x.shape[-1] != input_size:
            return None
        if not x.is_contiguous():
            return None
        x_2d = x.reshape(-1, input_size)
        rows = x_2d.shape[0]
        if rows != 1:
            return None

        output_size = layer.qweight.shape[1]
        out = torch.empty((rows, output_size), device=x.device, dtype=x.dtype)
        lib = _get_mps_gptq_direct_gemv_lib()
        lib.gptq_gemv_gidx_vec8_simd32(
            x_2d,
            layer.qweight,
            layer.scales,
            layer.g_idx,
            out,
            output_size,
            input_size,
            threads=(32, (output_size + 7) // 8, 1),
            group_size=(32, 1, 1),
        )
        out = out.reshape(*x.shape[:-1], output_size)
        return out if bias is None else out + bias


class AutoGPTQMoEMethod(FusedMoEMethodBase):
    """MoE Marlin method with quantization."""

    def __init__(
        self,
        quant_config: AutoGPTQConfig,
        moe: FusedMoEConfig,
    ) -> None:
        super().__init__(moe)
        self.quant_config = quant_config
        if self.quant_config.quant_type.size_bits == 4:
            quant_type = scalar_types.uint4b8
            scale = kInt4StaticGroupScale
        elif self.quant_config.quant_type.size_bits == 8:
            quant_type = scalar_types.uint8b128
            scale = kInt8StaticGroupScale
        else:
            raise ValueError("AutoGPTQMoEMethod only supports int4 and int8 now.")
        self.input_dtype = None
        self.use_marlin = True
        weight_key = QuantKey(quant_type, scale)

        self.wna16_moe_backend, self.experts_cls = select_wna16_moe_backend(
            moe,
            weight_key,
            quant_config=self.quant_config,
            may_have_zp=not self.quant_config.is_sym,
            may_have_bias=True,
            allow_tile_padding=not self.quant_config.desc_act,
        )

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        layer.input_dtype = self.input_dtype
        is_a_8bit = self.input_dtype is not None and self.input_dtype.itemsize == 1

        if is_a_8bit:
            assert self.quant_config.quant_type.size_bits == 8, (
                "W8A8-INT8 is not supported by marlin kernel."
            )

        intermediate_size_full = extra_weight_attrs.pop("intermediate_size_full")

        self.is_k_full = (not self.quant_config.desc_act) or (
            intermediate_size_per_partition == intermediate_size_full
        )

        if self.quant_config.group_size != -1:
            scales_size13 = hidden_size // self.quant_config.group_size
            w2_scales_size = (
                intermediate_size_full
                if self.quant_config.desc_act
                else intermediate_size_per_partition
            )
            scales_size2 = w2_scales_size // self.quant_config.group_size
            strategy = FusedMoeWeightScaleSupported.GROUP.value
        else:
            scales_size13 = 1
            scales_size2 = 1
            strategy = FusedMoeWeightScaleSupported.CHANNEL.value

        layer.num_groups_w13 = scales_size13
        layer.num_groups_w2 = scales_size2

        extra_weight_attrs.update({"quant_method": strategy, "is_transposed": True})
        # Fused gate_up_proj (column parallel)
        w13_qweight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size // self.quant_config.pack_factor,
                self.moe.w13_num_shards * intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_qweight", w13_qweight)
        set_weight_attrs(w13_qweight, extra_weight_attrs)
        # down_proj (row parallel)
        w2_qweight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition // self.quant_config.pack_factor,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_qweight", w2_qweight)
        set_weight_attrs(w2_qweight, extra_weight_attrs)
        # up_proj scales
        w13_scales = torch.nn.Parameter(
            torch.empty(
                num_experts,
                scales_size13,
                self.moe.w13_num_shards * intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_scales", w13_scales)
        set_weight_attrs(w13_scales, extra_weight_attrs)
        # down_proj scales
        w2_scales = torch.nn.Parameter(
            torch.empty(num_experts, scales_size2, hidden_size, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w2_scales", w2_scales)
        set_weight_attrs(w2_scales, extra_weight_attrs)
        # don't shard the w2 scales when running act order
        set_weight_attrs(w2_scales, {"load_full_w2": self.quant_config.desc_act})
        # up_proj zero points
        w13_qzeros = torch.nn.Parameter(
            torch.empty(
                num_experts,
                scales_size13,
                self.moe.w13_num_shards
                * intermediate_size_per_partition
                // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_qzeros", w13_qzeros)
        set_weight_attrs(w13_qzeros, extra_weight_attrs)
        # down_proj zero points
        w2_qzeros = torch.nn.Parameter(
            torch.empty(
                num_experts,
                scales_size2,
                hidden_size // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_qzeros", w2_qzeros)
        set_weight_attrs(w2_qzeros, extra_weight_attrs)
        # don't shard the w2 scales when running act order
        set_weight_attrs(w2_qzeros, {"load_full_w2": self.quant_config.desc_act})
        w13_g_idx = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_g_idx", w13_g_idx)
        set_weight_attrs(w13_g_idx, extra_weight_attrs)
        w2_g_idx = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_g_idx", w2_g_idx)
        set_weight_attrs(w2_g_idx, extra_weight_attrs)
        w13_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_g_idx_sort_indices", w13_g_idx_sort_indices)
        set_weight_attrs(w13_g_idx_sort_indices, extra_weight_attrs)
        w2_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_g_idx_sort_indices", w2_g_idx_sort_indices)
        set_weight_attrs(w2_g_idx_sort_indices, extra_weight_attrs)

        # Some GPTQ checkpoints contain expert biases even when the model
        # architecture does not declare them. Zero initialization keeps
        # checkpoints without biases equivalent to the bias-free path.
        w13_bias = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                self.moe.w13_num_shards * intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_bias", w13_bias)
        set_weight_attrs(w13_bias, extra_weight_attrs)
        w2_bias = torch.nn.Parameter(
            torch.zeros(num_experts, hidden_size, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w2_bias", w2_bias)
        set_weight_attrs(w2_bias, extra_weight_attrs)

        if self.experts_cls is not None and issubclass(
            self.experts_cls, FusedMoEExpertsModular
        ):
            device = layer.w13_qweight.device
            layer.workspace = marlin_make_workspace_new(device, 4)

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        def replace_or_register(name: str, val: torch.Tensor | None):
            if val is None:
                return

            if hasattr(layer, name):
                replace_parameter(layer, name, val)
            else:
                layer.register_parameter(
                    name, torch.nn.Parameter(val, requires_grad=False)
                )

        is_a_8bit = self.input_dtype is not None and self.input_dtype.itemsize == 1

        assert not is_a_8bit or self.quant_config.quant_type.size_bits == 8, (
            "W8A8-INT8 is not supported by marlin kernel."
        )

        w13_bias = getattr(layer, "w13_bias", None)
        if "w13_bias" not in layer._loaded_expert_biases:
            layer.register_parameter("w13_bias", None)
            w13_bias = None
        w2_bias = getattr(layer, "w2_bias", None)
        if "w2_bias" not in layer._loaded_expert_biases:
            layer.register_parameter("w2_bias", None)
            w2_bias = None

        converted = convert_to_wna16_moe_kernel_format(
            backend=self.wna16_moe_backend,
            layer=layer,
            quant_config=self.quant_config,
            input_dtype=self.input_dtype,
            w13=layer.w13_qweight,
            w2=layer.w2_qweight,
            w13_scale=layer.w13_scales,
            w2_scale=layer.w2_scales,
            w13_g_idx=layer.w13_g_idx,
            w2_g_idx=layer.w2_g_idx,
            w13_bias=w13_bias,
            w2_bias=w2_bias,
            w13_qzeros=getattr(layer, "w13_qzeros", None),
            w2_qzeros=getattr(layer, "w2_qzeros", None),
        )

        if converted is None:
            # Backend rewrote the layer's params in place (e.g. Humming).
            self._setup_kernel(layer)
            return

        (
            w13,
            w2,
            w13_scale,
            w2_scale,
            w13_g_idx,
            w2_g_idx,
            w13_g_idx_sort_indices,
            w2_g_idx_sort_indices,
            w13_qzeros,
            w2_qzeros,
            w13_input_global_scale,
            w2_input_global_scale,
            w13_bias,
            w2_bias,
        ) = converted

        replace_parameter(layer, "w13_qweight", w13)
        replace_parameter(layer, "w2_qweight", w2)
        replace_parameter(layer, "w13_scales", w13_scale)
        replace_parameter(layer, "w2_scales", w2_scale)
        replace_parameter(layer, "w13_g_idx", w13_g_idx)
        replace_parameter(layer, "w2_g_idx", w2_g_idx)
        replace_parameter(layer, "w13_g_idx_sort_indices", w13_g_idx_sort_indices)
        replace_parameter(layer, "w2_g_idx_sort_indices", w2_g_idx_sort_indices)
        replace_or_register("w13_input_global_scale", w13_input_global_scale)
        replace_or_register("w2_input_global_scale", w2_input_global_scale)
        replace_or_register("w13_bias", w13_bias)
        replace_or_register("w2_bias", w2_bias)
        replace_or_register("w13_qzeros", w13_qzeros)
        replace_or_register("w2_qzeros", w2_qzeros)

        # The modular kernel reads w13_weight/w2_weight; marlin keeps *_qweight.
        layer.w13_weight = layer.w13_qweight
        layer.w2_weight = layer.w2_qweight

        self._setup_kernel(layer)

    def _setup_kernel(self, layer: RoutedExperts) -> None:
        """Build the FusedMoEKernel for this layer."""

        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        self.moe_kernel = make_wna16_moe_kernel(
            moe_quant_config=self.moe_quant_config,
            moe_config=self.moe,
            experts_cls=self.experts_cls,
            backend=self.wna16_moe_backend,
            is_k_full=self.is_k_full,
            w13_g_idx=getattr(layer, "w13_g_idx", None),
            w2_g_idx=getattr(layer, "w2_g_idx", None),
            w13_g_idx_sort_indices=getattr(layer, "w13_g_idx_sort_indices", None),
            w2_g_idx_sort_indices=getattr(layer, "w2_g_idx_sort_indices", None),
            routing_tables=layer._expert_routing_tables(),
        )

    def get_fused_moe_quant_config(self, layer: RoutedExperts) -> FusedMoEQuantConfig:
        if self.wna16_moe_backend == WNA16MoEBackend.HUMMING:
            from vllm.model_executor.layers.quantization.utils.humming_utils import (
                get_humming_moe_quant_config,
            )

            return get_humming_moe_quant_config(
                layer,
                gemm1_alpha=getattr(layer, "swiglu_alpha", None),
                gemm1_beta=getattr(layer, "swiglu_beta", None),
                gemm1_clamp_limit=getattr(layer, "swiglu_limit", None),
            )

        from vllm.model_executor.layers.fused_moe.config import (
            gptq_marlin_moe_quant_config,
        )

        # CPU fused_experts_cpu requires zero points even for symmetric quant
        use_zp = (
            not self.quant_config.is_sym
            or self.wna16_moe_backend == WNA16MoEBackend.CPU
        )
        return gptq_marlin_moe_quant_config(
            w1_scale=layer.w13_scales,
            w2_scale=layer.w2_scales,
            weight_bits=self.quant_config.weight_bits,
            group_size=self.quant_config.group_size,
            w1_zp=getattr(layer, "w13_qzeros", None) if use_zp else None,
            w2_zp=getattr(layer, "w2_qzeros", None) if use_zp else None,
            w1_bias=getattr(layer, "w13_bias", None),
            w2_bias=getattr(layer, "w2_bias", None),
        )

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        assert not self.is_monolithic
        assert self.moe_kernel is not None
        return self.moe_kernel.apply(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            expert_map=layer.expert_map,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )

    def apply_monolithic(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.is_monolithic
        assert self.moe_kernel is not None
        return self.moe_kernel.apply_monolithic(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            router_logits=router_logits,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            num_expert_group=layer.num_expert_group,
            topk_group=layer.topk_group,
            e_score_correction_bias=layer.e_score_correction_bias,
            routed_scaling_factor=layer.routed_scaling_factor,
        )
