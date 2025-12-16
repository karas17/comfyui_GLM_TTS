# Copyright (c) 2025 Zhipu AI Inc (authors: CogAudio Group Members)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core modules for the DiT (Diffusion Transformer) model.
Includes position embeddings, normalization, attention mechanisms, and transformer blocks.
"""

from __future__ import annotations
from typing import Optional
import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from x_transformers.x_transformers import apply_rotary_pos_emb

# -----------------------------------------------------------------------------
# Position Embeddings
# -----------------------------------------------------------------------------

class SinusPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ConvPositionEmbedding(nn.Module):
    def __init__(self, dim, kernel_size=31, groups=16):
        super().__init__()
        assert kernel_size % 2 != 0
        self.conv1d = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            mask: Boolean mask of shape (batch, seq_len)
        """
        if mask is not None:
            mask = mask[..., None]
            x = x.masked_fill(~mask, 0.)

        x = rearrange(x, 'b n d -> b d n')
        x = self.conv1d(x)
        out = rearrange(x, 'b d n -> b n d')

        if mask is not None:
            out = out.masked_fill(~mask, 0.)

        return out


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, theta_rescale_factor=1.):
    """
    Precompute Rotary Positional Embeddings (RoPE).
    References:
    https://github.com/lucidrains/rotary-embedding-torch
    """
    theta *= theta_rescale_factor ** (dim / (dim - 2))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device) 
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return torch.cat([freqs_cos, freqs_sin], dim=-1)


def get_pos_embed_indices(start, length, max_pos, scale=1.):
    """
    Calculates indices for positional embeddings, handling potential scalar scaling.
    """
    scale = scale * torch.ones_like(start, dtype=torch.float32)
    pos = start.unsqueeze(1) + (
            torch.arange(length, device=start.device, dtype=torch.float32).unsqueeze(0) *
            scale.unsqueeze(1)).long()
    # Clamp to avoid index out of bounds
    pos = torch.where(pos < max_pos, pos, max_pos - 1)
    return pos


class TimestepEmbedding(nn.Module):
    def __init__(self, dim, freq_embed_dim=256):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(freq_embed_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, timestep: torch.Tensor):
        """
        Args:
            timestep: Tensor of shape (batch,)
        Returns:
            Time embedding of shape (batch, dim)
        """
        time_hidden = self.time_embed(timestep)
        time = self.time_mlp(time_hidden) 
        return time


# -----------------------------------------------------------------------------
# Normalization & Layers
# -----------------------------------------------------------------------------

class GRN(nn.Module):
    """ Global Response Normalization (GRN) layer """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class ConvNeXtV2Block(nn.Module):
    """ 
    ConvNeXt-V2 Block.
    Reference: https://github.com/facebookresearch/ConvNeXt-V2
    """
    def __init__(self, dim: int, intermediate_dim: int, dilation: int = 1):
        super().__init__()
        padding = (dilation * (7 - 1)) // 2
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=padding, groups=dim, dilation=dilation) 
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = x.transpose(1, 2)  # (b, n, d) -> (b, d, n)
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (b, d, n) -> (b, n, d)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        return residual + x


class AdaLayerNormZero(nn.Module):
    """
    Adaptive LayerNorm with modulation (scale/shift) from an embedding (e.g., time/condition).
    Outputs modulation parameters for MSA (Multi-Head Self Attention) and MLP (FeedForward).
    """
    def __init__(self, dim, additional_dim=0):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim + additional_dim, dim * 6)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb=None):
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(emb, 6, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp

class AdaLayerNormZero_Final(nn.Module):
    """
    Adaptive LayerNorm for the final layer.
    Only outputs modulation for the current normalization, no downstream parameters.
    """
    def __init__(self, dim, additional_dim=0):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim + additional_dim, dim * 2)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb):
        emb = self.linear(self.silu(emb))
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x

class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, dropout=0., approximate: str = 'none'):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        activation = nn.GELU(approximate=approximate)

        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            activation
        )

        self.ff = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.ff(x)


# -----------------------------------------------------------------------------
# Attention Mechanisms
# -----------------------------------------------------------------------------

class AttnProcessorCausalV2:
    def __init__(self):
        pass

    def __call__(
            self,
            attn: AttentionV2,
            x: torch.Tensor,
            padding_mask: Optional[torch.Tensor] = None,
            rope=None,
            attn_mask=None,
    ) -> torch.Tensor:
        
        batch_size = x.shape[0]

        # Projections
        query = attn.to_q(x)
        key = attn.to_k(x)
        value = attn.to_v(x)

        # Apply Rotary Position Embedding (RoPE)
        if rope is not None:
            freqs, xpos_scale = rope
            q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale ** -1.) if xpos_scale is not None else (1., 1.)
            query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
            key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)

        # Reshape for multi-head attention
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        
        # (batch, seq, heads, dim) -> (batch, heads, seq, dim)
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # Scaled Dot Product Attention
        # is_causal=False because masking is handled explicitly via attn_mask
        x = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        
        # Reshape back: (batch, heads, seq, dim) -> (batch, seq, inner_dim)
        x = x.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        x = x.to(query.dtype)

        # Output projection and dropout
        x = attn.to_out[0](x)
        x = attn.to_out[1](x)

        # Apply padding mask if provided
        if padding_mask is not None:
            padding_mask = rearrange(padding_mask, 'b 1 n -> b n 1').bool()
            x = x.masked_fill(~padding_mask, 0.)

        return x


class AttentionV2(nn.Module):
    def __init__(
            self,
            processor: AttnProcessorCausalV2,
            dim: int,
            heads: int = 8,
            dim_head: int = 64,
            dropout: float = 0.0,
            context_dim: Optional[int] = None,
            context_pre_only=None,
    ):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("Attention requires PyTorch 2.0+.")

        self.processor = processor
        self.dim = dim
        self.heads = heads
        self.inner_dim = dim_head * heads
        self.dropout = dropout
        self.context_dim = context_dim
        self.context_pre_only = context_pre_only

        self.to_q = nn.Linear(dim, self.inner_dim)
        self.to_k = nn.Linear(dim, self.inner_dim)
        self.to_v = nn.Linear(dim, self.inner_dim)

        # Context Attention (if applicable)
        if self.context_dim is not None:
            self.to_k_c = nn.Linear(context_dim, self.inner_dim)
            self.to_v_c = nn.Linear(context_dim, self.inner_dim)
            if self.context_pre_only is not None:
                self.to_q_c = nn.Linear(context_dim, self.inner_dim)

        self.to_out = nn.ModuleList([
            nn.Linear(self.inner_dim, dim),
            nn.Dropout(dropout)
        ])

        if self.context_pre_only is not None and not self.context_pre_only:
            self.to_out_c = nn.Linear(self.inner_dim, dim)

    def forward(
            self,
            x: torch.Tensor,
            c: torch.Tensor = None,
            padding_mask: Optional[torch.Tensor] = None,
            rope=None,
            c_rope=None,
            attn_mask=None,
    ) -> torch.Tensor:
        if c is not None:
            # Note: The current flow logic mainly uses self-attention without 'c' here,
            # but the interface supports it.
            return self.processor(self, x, c=c, padding_mask=padding_mask, rope=rope, c_rope=c_rope,
                                  attn_mask=attn_mask)
        else:
            return self.processor(self, x, padding_mask=padding_mask, rope=rope, attn_mask=attn_mask)


# -----------------------------------------------------------------------------
# Transformer Blocks
# -----------------------------------------------------------------------------

class DiTBlockCausalV2(nn.Module):
    def __init__(self, dim, heads, dim_head, ff_mult=4, dropout=0.1, additional_condition_dim=0):
        super().__init__()

        self.attn_norm = AdaLayerNormZero(dim, additional_condition_dim)
        self.attn = AttentionV2(
            processor=AttnProcessorCausalV2(),
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
        )

        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh")

    def forward(self, x, t, padding_mask=None, rope=None, attn_mask=None):
        """
        Args:
            x: Noised input tensor
            t: Time embedding (or concatenated time+condition embedding)
            padding_mask: Mask for padding
            rope: Rotary positional embeddings
            attn_mask: Attention mask (e.g., for causality or blocking)
        """
        # 1. Pre-norm & Modulation for Attention
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)

        # 2. Attention
        attn_output = self.attn(x=norm, padding_mask=padding_mask, rope=rope, attn_mask=attn_mask)

        # 3. Residual Connection (modulated)
        x = x + gate_msa.unsqueeze(1) * attn_output

        # 4. Pre-norm & Modulation for FeedForward
        norm = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm)

        # 5. Residual Connection (modulated)
        x = x + gate_mlp.unsqueeze(1) * ff_output

        return x