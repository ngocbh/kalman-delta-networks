# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Copyright Lightning AI. Licensed under the Apache License 2.0,
# see LICENSE file at https://github.com/Lightning-AI/litgpt/blob/main/LICENSE

from dataclasses import dataclass
from typing import Any, Literal, Optional, Type

import torch
from typing_extensions import Self

import lit_gpt.model
from lit_gpt.utils import find_multiple


@dataclass
class Config:
    org: str = "Lightning-AI"
    name: str = "lit-GPT"
    block_size: int = 4096
    vocab_size: int = 50254
    padding_multiple: int = 64
    padded_vocab_size: Optional[int] = None
    n_layer: int = 16
    n_head: int = 32
    n_embd: int = 4096
    rotary_percentage: float = 0.25
    parallel_residual: bool = True
    bias: bool = True
    local_window: int = -1
    mlp: bool = True
    gdn2_per_layer: int = -1
    mixer_type: str = "gdn2"
    kdn_omega_min: float = 0.0
    kdn_r_min: float = 0.01
    kdn_info_scale: Optional[float] = None
    nope: bool = False
    mamba_init: bool = False
    # to use multi-head attention (MHA), set this to `n_head` (default)
    # to use multi-query attention (MQA), set this to 1
    # to use grouped-query attention (GQA), set this to a value in between
    # Example with `n_head=4`
    # ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
    # │ v ││ v ││ v ││ v │     │ v │    │ v │             │ v │
    # └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
    #   │    │    │    │         │        │                 │
    # ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
    # │ k ││ k ││ k ││ k │     │ k │    │ k │             │ k │
    # └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
    #   │    │    │    │      ┌──┴──┐  ┌──┴──┐      ┌────┬──┴─┬────┐
    # ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐
    # │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │
    # └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘
    # ◀──────────────────▶  ◀──────────────────▶  ◀──────────────────▶
    #         MHA                    GQA                   MQA
    #   n_query_groups=4       n_query_groups=2      n_query_groups=1
    #
    # credit https://arxiv.org/pdf/2305.13245.pdf
    n_query_groups: Optional[int] = None
    shared_attention_norm: bool = False
    _norm_class: Literal["LayerNorm", "RMSNorm", "FusedRMSNorm"] = "LayerNorm"
    norm_eps: float = 1e-5
    _mlp_class: Literal["LLaMAMLP"] = "LLaMAMLP"
    intermediate_size: Optional[int] = None
    condense_ratio: int = 1

    def __post_init__(self):
        # error checking
        assert self.n_embd % self.n_head == 0
        # vocab size should be a power of 2 to be optimal on hardware. compute the closest value
        if self.padded_vocab_size is None:
            self.padded_vocab_size = find_multiple(self.vocab_size, self.padding_multiple)
        # compute the number of query groups
        if self.n_query_groups is not None:
            assert self.n_head % self.n_query_groups == 0
        else:
            self.n_query_groups = self.n_head
        # compute the intermediate size for MLP if not set
        if self.intermediate_size is None:
            if self._mlp_class == "LLaMAMLP":
                raise ValueError("The config needs to set the `intermediate_size`")
            self.intermediate_size = 4 * self.n_embd

    @property
    def head_size(self) -> int:
        return self.n_embd // self.n_head

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        conf_dict = name_to_config[name].copy()
        conf_dict.update(kwargs)
        return cls(**conf_dict)

    @property
    def mlp_class(self) -> Type:
        # `self._mlp_class` cannot be the type to keep the config json serializable
        return getattr(lit_gpt.model, self._mlp_class)

    @property
    def norm_class(self) -> Type:
        # `self._norm_class` cannot be the type to keep the config json serializable
        if self._norm_class == "RMSNorm":
            from lit_gpt.rmsnorm import RMSNorm

            return RMSNorm
        elif self._norm_class == "FusedRMSNorm":
            from lit_gpt.rmsnorm import FusedRMSNorm

            return FusedRMSNorm
        return getattr(torch.nn, self._norm_class)


configs = [
    # ---------------- GatedDeltaNet2 (gdn2) ----------------
    dict(
        org="NVIDIA",
        name="gdn2_1.3B", # Total parameters 1,302,638,112
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        gdn2_per_layer=1,
        n_layer=18,
        n_head=18,
        n_embd=2304,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=6208,
        local_window=2048,
        mamba_init=True,
    ),
    # ---------------- Isotropic KDN -----------------------------------------
    dict(
        org="yale",
        name="iso_kdn_1.3B",  # Total 1,449,848,592; transformer.h 1,302,390,288
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        gdn2_per_layer=1,
        mixer_type="iso_kdn",
        kdn_omega_min=0.0,
        kdn_r_min=0.01,
        kdn_info_scale=128.0,
        n_layer=18,
        n_head=18,
        n_embd=2304,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=7208,
        local_window=2048,
        mamba_init=True,
    ),
    # ---------------- Diagonal KDN ------------------------------------------
    dict(
        org="yale",
        name="diag_kdn_1.3B",  # Total 1,449,806,796; transformer.h 1,302,348,492
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        gdn2_per_layer=1,
        mixer_type="diag_kdn",
        kdn_omega_min=0.0,
        kdn_r_min=0.01,
        kdn_info_scale=128.0,
        n_layer=18,
        n_head=18,
        n_embd=2304,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=7128,
        local_window=2048,
        mamba_init=True,
    ),
    dict(
        org="NVIDIA",
        name="swa_gdn2_1.3B", # Total parameters 1,300,314,384
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        gdn2_per_layer=2,
        n_layer=18,
        n_head=18,
        n_embd=2304,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=6784,
        local_window=2048,
        mamba_init=True,
    )
]

name_to_config = {config["name"]: config for config in configs}
