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
DiT (Diffusion Transformer) model definition.
Combines embedding layers, transformer blocks, and output projections.
"""

from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from utils import block_mask_util
from x_transformers.x_transformers import RotaryEmbedding

from flow.modules import (
    TimestepEmbedding,
    ConvNeXtV2Block,
    ConvPositionEmbedding,
    AdaLayerNormZero_Final,
    precompute_freqs_cis,
    get_pos_embed_indices,
    DiTBlockCausalV2
)


class TextEmbedding(nn.Module):
    """
    Embeds text/speech tokens. Supports optional ConvNeXt modeling.
    """
    def __init__(self, text_num_embeds, output_dim, conv_layers=0, conv_mult=2, length_align="fill"):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, output_dim)  # use 0 as filler token
        self.length_align = length_align

        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = 4096  # ~44s of 24khz audio
            self.register_buffer("freqs_cis", precompute_freqs_cis(output_dim, self.precompute_max_pos),
                                 persistent=False)
            self.text_blocks = nn.Sequential(
                *[ConvNeXtV2Block(output_dim, output_dim * conv_mult) for _ in range(conv_layers)]
            )
        else:
            self.extra_modeling = False

    def forward(self, text_bt, aim_seq_len):
        # Debug check for vocab size
        if text_bt.max() > self.text_embed.num_embeddings:
             raise ValueError(f"Token ID {text_bt.max()} exceeds vocabulary size {self.text_embed.num_embeddings}")

        if self.length_align == "fill":
            text_bt = text_bt[:, :aim_seq_len]
            _, text_len = text_bt.shape
            # Pad with 0 if shorter than target
            text_bt = F.pad(text_bt, (0, aim_seq_len - text_len), value=0)
        elif self.length_align == "interpolate_token":
            # Interpolate directly on tokens (nearest neighbor)
            text_bt = F.interpolate(text_bt.unsqueeze(1).float(), size=aim_seq_len, mode='nearest').squeeze(1).long()

        hidden_btd = self.text_embed(text_bt)

        if self.length_align == "interpolate_feature":
             # Interpolate on the feature dimension
            hidden_btd = F.interpolate(hidden_btd.permute(0, 2, 1), size=aim_seq_len, mode='nearest').permute(0, 2, 1)

        # Extra modeling (ConvNeXt + RoPE)
        if self.extra_modeling:
            batch = text_bt.shape[0]
            batch_start = torch.zeros((batch,), dtype=torch.long, device=text_bt.device)
            pos_idx = get_pos_embed_indices(batch_start, aim_seq_len, max_pos=self.precompute_max_pos)
            
            text_pos_embed = self.freqs_cis[pos_idx]
            hidden_btd = hidden_btd + text_pos_embed
            hidden_btd = self.text_blocks(hidden_btd)

        return hidden_btd


class EmbeddingConcater(nn.Module):
    """
    Concatenates input noisy audio, condition audio, and text embeddings, 
    then projects them to the transformer dimension.
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, text_embed: torch.Tensor,
                drop_audio_cond=False):
        
        if drop_audio_cond:
            cond = torch.zeros_like(cond)

        # Concatenate along the feature dimension
        x = self.proj(torch.cat((x, cond, text_embed), dim=-1))
        # Add convolutional position embedding
        x = self.conv_pos_embed(x) + x
        return x


class DiT(nn.Module):
    def __init__(
            self,
            *,
            trans_dim,
            depth=8,
            heads=8,
            dim_head=64,
            dropout=0.1,
            ff_mult=4,
            mel_dim=80,
            text_vocab_size=100000,
            text_emb_dim=None,
            conv_layers=0,
            long_skip_connection=False,
            condition_dim=0,
            attention_mask_type=None,
            spkr_emb_adaLN=False,
            wav_lm_emb=False,
    ):
        super().__init__()

        self.time_embed = TimestepEmbedding(trans_dim)
        self.text_emb_layer = TextEmbedding(text_vocab_size,
                                            text_emb_dim,
                                            conv_layers=conv_layers, 
                                            length_align="interpolate_token")
        
        in_dim = mel_dim + text_emb_dim + condition_dim
        self.emb_concator = EmbeddingConcater(in_dim, trans_dim)

        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = trans_dim
        self.depth = depth
        self.num_heads = heads

        self.spkr_emb_adaLN = spkr_emb_adaLN
        self.attention_mask_type = attention_mask_type

        # Determine speaker embedding dimension
        if spkr_emb_adaLN:
            # 192 (std spk emb) + 256 (optional wavlm)
            spkr_dim = 192 + 256 if wav_lm_emb else 192
        else:
            spkr_dim = 0

        self.transformer_blocks = nn.ModuleList(
            [DiTBlockCausalV2(
                dim=trans_dim, 
                heads=heads, 
                dim_head=dim_head, 
                ff_mult=ff_mult, 
                dropout=dropout,
                additional_condition_dim=spkr_dim
             ) for _ in range(depth)]
        )

        self.long_skip_connection = nn.Linear(trans_dim * 2, trans_dim, bias=False) if long_skip_connection else None

        self.norm_out = AdaLayerNormZero_Final(trans_dim, additional_dim=spkr_dim)
        self.proj_out = nn.Linear(trans_dim, mel_dim)

    def forward(
            self,
            middle_point_btd,
            condition_btd,  # masked audio condition
            text,
            time_step_1d,
            padding_mask_bt,
            is_causal=False,
            spkr_emb_bd=None,
            block_pattern=None,
    ):
        """
        Forward pass of the DiT model.
        """
        bs, seq_len = middle_point_btd.shape[0], middle_point_btd.shape[1]
        
        # 1. Time Embedding
        time_emb_bd = self.time_embed(time_step_1d)
        
        # 2. Speaker Embedding (if Adaptive LayerNorm is used)
        if self.spkr_emb_adaLN:
            assert spkr_emb_bd is not None
            time_emb_bd = torch.cat([time_emb_bd, spkr_emb_bd], dim=-1)

        # 3. Text/Speech Token Embedding
        text_embed = self.text_emb_layer(text, seq_len)

        # 4. Input Concatenation & Projection
        middle_point_btd = self.emb_concator(middle_point_btd, condition_btd, text_embed, drop_audio_cond=False)

        # 5. Rotary Embeddings
        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        # 6. Long Skip Connection (Save residual)
        if self.long_skip_connection is not None:
            residual = middle_point_btd

        # 7. Create Attention Mask
        if is_causal:
            # Default block pattern if None
            if block_pattern is None:
                block_pattern = [25, 50, 200]
                
            attn_mask = self.create_attn_mask(bs, seq_len,
                                              padding_mask_bt.unsqueeze(1),
                                              padding_mask_bt.device,
                                              n_heads=self.num_heads, 
                                              block_pattern=block_pattern)
        else:
            attn_mask = None

        # 8. Transformer Blocks
        for block in self.transformer_blocks:
            middle_point_btd = block(middle_point_btd, time_emb_bd,
                                     padding_mask=padding_mask_bt.unsqueeze(1),
                                     rope=rope,
                                     attn_mask=attn_mask)

        # 9. Long Skip Connection (Apply)
        if self.long_skip_connection is not None:
            middle_point_btd = self.long_skip_connection(torch.cat((middle_point_btd, residual), dim=-1))

        # 10. Final Norm & Output Projection
        middle_point_btd = self.norm_out(middle_point_btd, time_emb_bd)
        output = self.proj_out(middle_point_btd)

        return output

    def create_attn_mask(self, bs, seq_len, padding_mask_b1t, device, n_heads, block_pattern):
        """
        Creates a custom attention mask, likely for block-causal attention.
        """
        block_list = [self.token_size_to_mel_size(i) for i in block_pattern]
        mask_tt = block_mask_util.create_with_cache(block_list, seq_len).bool().to(device)
        block_mask_btt = mask_tt[None].repeat(bs, 1, 1)
        
        # Combine with padding mask
        block_mask_btt = block_mask_btt * padding_mask_b1t.bool()
        
        # Expand for heads
        attn_mask_bhtt = block_mask_btt.unsqueeze(1).repeat(1, n_heads, 1, 1)
        return attn_mask_bhtt

    def token_size_to_mel_size(self, i):
        # Convert token steps to mel-spectrogram frames.
        # Assumptions: 12.5 Hz token rate, 22050 Hz sample rate, 256 hop length.
        # Calculation: i / token_rate * (sample_rate / hop_length)
        return int(i / 12.5 * 22050 / 256)