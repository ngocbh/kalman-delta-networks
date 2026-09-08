# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Copyright Lightning AI. Licensed under the Apache License 2.0,
# see LICENSE file at https://github.com/Lightning-AI/litgpt/blob/main/LICENSE

import math
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import Self

from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input

from lit_gpt.config import Config
from .fused_rotary_embedding import apply_rotary_emb_func

from .gdn2 import GatedDeltaNet2

RoPECache = Tuple[torch.Tensor, torch.Tensor]
KVCache = Tuple[torch.Tensor, torch.Tensor]


class GPT(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        assert config.padded_vocab_size is not None
        self.config = config

        self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=False)
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
                h=nn.ModuleList(Block(config, i) for i in range(config.n_layer)),
                ln_f=config.norm_class(config.n_embd, eps=config.norm_eps),
            )
        )
        self.rope_cache: Optional[RoPECache] = None
        self.mask_cache: Optional[torch.Tensor] = None
        self.kv_caches: List[Optional[KVCache]] = []
        self.max_len = self.config.block_size
        self.mamba_init = config.mamba_init

    def _init_weights(self, module: nn.Module, n_layer) -> None:
        """Meant to be used with `gpt.apply(gpt._init_weights)`."""
        # GPT-NeoX  https://arxiv.org/pdf/2204.06745.pdf
        if isinstance(module, nn.Embedding):
            if self.mamba_init:
                torch.nn.init.normal_(module.weight, std=0.02)
            else:
                torch.nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / self.config.n_embd))
        elif isinstance(module, nn.Linear):
            if self.mamba_init:
                if module.bias is not None:
                    if not getattr(module.bias, "_no_reinit", False):
                        nn.init.zeros_(module.bias)
            else:
                torch.nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / self.config.n_embd))
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
        # GPT-NeoX
        for name, p in module.named_parameters():
            if (
                name in ["out_proj.weight", "fc2.weight"]
                or (name == "proj.weight" and isinstance(module, LLaMAMLP))
                or (name == "w3.weight" and isinstance(module, SwiGLU))
                or (name == "proj.weight" and isinstance(module, CausalSelfAttention))
            ):
                if self.mamba_init:
                    n_residuals_per_layer = 1 if not self.config.mlp else 2
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        p /= math.sqrt(n_residuals_per_layer * n_layer)
                else:
                    nn.init.normal_(p, mean=0.0, std=1 / math.sqrt(self.config.n_embd) / n_layer)
        # DiagKDN starts from isotropic process noise. Apply this after the
        # ordinary initializer so non-Mamba configurations preserve its RNG
        # consumption and the frozen checkpoint initialization order.
        if getattr(module, "_kdn_omega_zero", False):
            nn.init.zeros_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def reset_cache(self) -> None:
        self.max_len = self.config.block_size
        self.kv_caches.clear()
        if self.mask_cache is not None and self.mask_cache.device.type == "xla":
            # https://github.com/Lightning-AI/lit-gpt/pull/83#issuecomment-1558150179
            self.rope_cache = None
            self.mask_cache = None

    def forward(
        self,
        idx: torch.Tensor,
        max_seq_length: Optional[int] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T = idx.size()
        use_kv_cache = input_pos is not None
        if use_kv_cache and self.config.mixer_type in {"iso_kdn", "diag_kdn"}:
            raise ValueError("KDN mixers do not support KV caching")

        block_size = self.config.block_size
        if max_seq_length is None:
            max_seq_length = block_size
        if use_kv_cache:  # not relevant otherwise
            assert (
                max_seq_length >= T
            ), f"Cannot forward sequence of length {T}, max seq length is only {max_seq_length}"

        if not self.config.nope:
            if self.rope_cache is None:
                self.rope_cache = self.build_rope_cache(idx, self.max_len)
            elif T > self.max_len:
                self.max_len = T
                self.rope_cache = self.build_rope_cache(idx, self.max_len)
            cos, sin = self.rope_cache

        if use_kv_cache and self.mask_cache is None:
            self.mask_cache = self.build_mask_cache(idx)

        if use_kv_cache:
            if not self.config.nope:
                cos = cos.index_select(0, input_pos)
                sin = sin.index_select(0, input_pos)
            mask = self.mask_cache.index_select(2, input_pos)
            mask = mask[:, :, :, :max_seq_length]
        else:
            if not self.config.nope:
                cos = cos[:T]
                sin = sin[:T]
            mask = None

        rope = None if self.config.nope else (cos, sin)

        # forward the model itself
        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)

        if not use_kv_cache:
            for block in self.transformer.h:
                x, *_ = block(x, rope, max_seq_length)
        else:
            start_pos = int(input_pos[0].item())
            if start_pos == 0:
                self.kv_caches = []

            if self.config.nope:
                self.kv_caches = self.kv_caches or self.build_kv_caches(x, max_seq_length, None)
            else:
                self.kv_caches = self.kv_caches or self.build_kv_caches(x, max_seq_length, cos.size(-1) * 2)

            for i, block in enumerate(self.transformer.h):
                x, self.kv_caches[i] = block(
                    x, rope, max_seq_length, mask, input_pos, self.kv_caches[i],
                )

        x = self.transformer.ln_f(x)
        return self.lm_head(x)  # (b, t, vocab_size)

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(Config.from_name(name, **kwargs))

    def build_rope_cache(self, idx: torch.Tensor, seq_len: int) -> RoPECache:
        return build_rope_cache(
            seq_len=seq_len,
            n_elem=int(self.config.rotary_percentage * self.config.head_size),
            dtype=torch.float32,
            device=idx.device,
            condense_ratio=self.config.condense_ratio,
        )

    def build_mask_cache(self, idx: torch.Tensor) -> torch.Tensor:
        ones = torch.ones((self.config.block_size, self.config.block_size), device=idx.device, dtype=torch.bool)
        return torch.tril(ones).unsqueeze(0).unsqueeze(0)

    def build_kv_caches(
        self, idx: torch.Tensor, max_seq_length: int, rope_cache_length: Optional[int]
    ) -> List[Optional[KVCache]]:
        if self.config.mixer_type in {"iso_kdn", "diag_kdn"}:
            raise ValueError("KDN mixers do not support KV caching")
        B = idx.size(0)
        heads = 1 if self.config.n_query_groups == 1 else self.config.n_query_groups
        if rope_cache_length is not None:
            k_cache_shape = (
                B,
                max_seq_length,
                heads,
                rope_cache_length + self.config.head_size - int(self.config.rotary_percentage * self.config.head_size),
            )
        else:
            k_cache_shape = (
                B,
                max_seq_length,
                heads,
                self.config.head_size,
            )
        v_cache_shape = (B, max_seq_length, heads, self.config.head_size)
        device = idx.device

        caches: List[Optional[KVCache]] = []
        for i in range(self.config.n_layer):
            block = self.transformer.h[i]
            if block.use_gdn2:
                caches.append(None)
            else:
                caches.append((
                    torch.zeros(k_cache_shape, device=device),
                    torch.zeros(v_cache_shape, device=device),
                ))
        return caches


class Block(nn.Module):
    def __init__(self, config: Config, layer_idx: int) -> None:
        super().__init__()
        self.norm_1 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.use_gdn2 = layer_idx % config.gdn2_per_layer == 0 if config.gdn2_per_layer > 0 else False
        self.is_iso_kdn = self.use_gdn2 and config.mixer_type == "iso_kdn"
        self.is_diag_kdn = self.use_gdn2 and config.mixer_type == "diag_kdn"
        if self.use_gdn2:
            if config.mixer_type == "iso_kdn":
                from lit_gpt.iso_kdn import IsotropicKalmanDeltaNetwork

                self.attn = IsotropicKalmanDeltaNetwork(
                    hidden_size=config.n_embd,
                    head_dim=config.head_size,
                    num_heads=config.n_head,
                    expand_v=1.0,
                    omega_min=config.kdn_omega_min,
                    r_min=config.kdn_r_min,
                    info_scale=config.kdn_info_scale,
                    norm_eps=config.norm_eps,
                    layer_idx=layer_idx,
                )
            elif config.mixer_type == "diag_kdn":
                from lit_gpt.diag_kdn import DiagonalKalmanDeltaNetwork

                self.attn = DiagonalKalmanDeltaNetwork(
                    hidden_size=config.n_embd,
                    head_dim=config.head_size,
                    num_heads=config.n_head,
                    expand_v=1.0,
                    omega_min=config.kdn_omega_min,
                    r_min=config.kdn_r_min,
                    info_scale=config.kdn_info_scale,
                    norm_eps=config.norm_eps,
                    layer_idx=layer_idx,
                )
            elif config.mixer_type == "gdn2":
                self.attn = GatedDeltaNet2(hidden_size=config.n_embd)
            else:
                raise ValueError(f"unsupported recurrent mixer_type: {config.mixer_type!r}")
        else:
            self.attn = CausalSelfAttention(config, n_embd=config.n_embd, layer_idx=layer_idx)
        if not config.shared_attention_norm and config.mlp and not config.parallel_residual:
            self.norm_2 = config.norm_class(config.n_embd, eps=config.norm_eps)
        if config.mlp:
            self.mlp = config.mlp_class(config)
        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPECache,
        max_seq_length: int,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        n_1 = self.norm_1(x)
        if self.is_iso_kdn or self.is_diag_kdn:
            h, _, new_kv_cache = self.attn(
                n_1,
                attention_mask=mask,
                past_key_values=kv_cache,
                use_cache=input_pos is not None,
            )
        elif self.use_gdn2:
            h, _, new_kv_cache = self.attn(n_1, attention_mask=None)
        else:
            h, new_kv_cache = self.attn(n_1, rope, max_seq_length, mask, input_pos, kv_cache)

        if self.config.parallel_residual:
            assert self.config.shared_attention_norm
            if self.config.mlp:
                h = h + self.mlp(n_1)
            x = x + h
        else:
            x = x + h
            if self.config.mlp:
                n_2 = self.norm_2(x)
                h = self.mlp(n_2)
                x = x + h
        return x, new_kv_cache


class CausalSelfAttention(nn.Module):
    def __init__(self, config: Config, layer_idx: int, n_embd: int, head_size=None) -> None:
        super().__init__()
        if head_size is not None:
            self.head_size = head_size
            self.n_head = n_embd // head_size
            self.n_query_groups = self.n_head
        else:
            self.head_size = config.head_size
            self.n_head = config.n_head
            self.n_query_groups = config.n_query_groups
        shape = (self.n_head + 2 * self.n_query_groups) * self.head_size
        # key, query, value projections for all heads, but in a batch
        self.attn = nn.Linear(n_embd, shape, bias=config.bias)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd, bias=config.bias)
        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPECache,
        max_seq_length: int,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        qkv = self.attn(x)
        # assemble into a number of query groups to support MHA, MQA and GQA together (see `config.n_query_groups`)
        q_per_kv = self.n_head // self.n_query_groups
        total_qkv = q_per_kv + 2  # each group has 1+ queries, 1 key, and 1 value
        qkv = qkv.view(B, T, self.n_query_groups, total_qkv, self.head_size)
        # split batched computation into three
        q, k, v = qkv.split((q_per_kv, 1, 1), dim=-2)
        q = q.reshape(B, T, -1)
        k = k.reshape(B, T, -1)
        v = v.reshape(B, T, -1)
        q = q.reshape(B, T, -1, self.head_size)
        k = k.reshape(B, T, -1, self.head_size)
        v = v.reshape(B, T, -1, self.head_size)
        if not self.config.nope:
            cos, sin = rope
            # apply rope in fp32 significantly stabilizes training
            # fused rope expects (batch_size, seqlen, nheads, headdim)
            q = apply_rotary_emb_func(q, cos, sin, False, True)
            k = apply_rotary_emb_func(k, cos, sin, False, True)

        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            cache_k, cache_v = cache_k.to(dtype=k.dtype), cache_v.to(dtype=v.dtype)
            # check if reached token limit
            if input_pos[-1] >= max_seq_length:
                input_pos = torch.tensor(max_seq_length - 1, device=input_pos.device)
                # shift 1 position to the left
                cache_k = torch.roll(cache_k, -1, dims=1)
                cache_v = torch.roll(cache_v, -1, dims=1)

            k = cache_k.index_copy_(1, input_pos, k)
            v = cache_v.index_copy_(1, input_pos, v)
            kv_cache = k, v

        if mask is not None:
            q, k, v, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(q, k, v, mask, T)
            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_q, max_seqlen_k = max_seq_lens
            o = flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                causal=True,
                window_size=(-1, -1) if self.config.local_window is None else (self.config.local_window - 1, 0),
            )
            o = pad_input(o, indices_q, B, T)
        else:
            o = flash_attn_func(
                q, k, v,
                causal=True,
                window_size=(-1, -1) if self.config.local_window is None else (self.config.local_window - 1, 0),
            )
        o = o.reshape(B, T, -1)  # re-assemble all head outputs side by side
        o = self.proj(o)
        return o, kv_cache

    def _upad_input(self, q, k, v, attention_mask, q_len):
        seqlens = attention_mask.sum(-1, dtype=torch.int32)
        indices_k = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
        max_seqlen_k = seqlens.max().item()
        cu_seqlens_k = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
        batch_size, seq_len, num_key_value_heads, head_dim = k.shape

        k = index_first_axis(k.reshape(batch_size * seq_len, num_key_value_heads, head_dim), indices_k)
        v = index_first_axis(v.reshape(batch_size * seq_len, num_key_value_heads, head_dim), indices_k)
        if q_len == seq_len:
            q = index_first_axis(q.reshape(batch_size * seq_len, self.n_head, head_dim), indices_k)
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_q = max_seqlen_k
            indices_q = indices_k
        elif q_len == 1:
            max_seqlen_q = 1
            # There is a memcpy here, that is very bad.
            cu_seqlens_q = torch.arange(batch_size + 1, dtype=torch.int32, device=q.device)
            indices_q = cu_seqlens_q[:-1]
            q = q.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -q_len:]
            q, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(q, attention_mask)

        return q, k, v, indices_q, (cu_seqlens_q, cu_seqlens_k), (max_seqlen_q, max_seqlen_k)


class LLaMAMLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.swiglu = SwiGLU(config.n_embd, config.intermediate_size, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.swiglu(x)
        return x


def build_rope_cache(
    seq_len: int,
    n_elem: int,
    dtype: torch.dtype,
    device: torch.device,
    base: int = 10000,
    condense_ratio: int = 1,
) -> RoPECache:
    """Enhanced Transformer with Rotary Position Embedding.

    Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
    transformers/rope/__init__.py. MIT License:
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
    """
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device) / n_elem))

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, device=device) / condense_ratio

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta)

    cos, sin = torch.cos(idx_theta), torch.sin(idx_theta)
    return cos, sin


class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features, bias=False):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w2 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, in_features, bias=bias)

    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        x = F.silu(x1) * x2
        x = self.w3(x)
        return x
