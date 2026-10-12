# Copyright (c) 2025 ASLP-lab
# MIT license.
# Source: https://github.com/ASLP-lab/SenSE/blob/main/src/sense/model/backbones/dit.py
# License included under licenses/LICENSE_sense.


"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from x_transformers.x_transformers import RotaryEmbedding

from .modules import (
    AdaLayerNorm_Final,
    ConvNeXtV2Block,
    ConvPositionEmbedding,
    DiTBlock,
    TimestepEmbedding,
    get_pos_embed_indices,
    precompute_freqs_cis,
)


# Text embedding: projecting phonetic embedding to phonetic condition
class TextEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim, mask_padding=True, conv_layers=0, conv_mult=2):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

        self.mask_padding = mask_padding  # mask filler and batch padding tokens or not
        
        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = 4096
            self.register_buffer("freqs_cis", precompute_freqs_cis(out_dim, self.precompute_max_pos), persistent=False)
            self.text_blocks = nn.Sequential(
                *[ConvNeXtV2Block(out_dim, out_dim * conv_mult) for _ in range(conv_layers)]
            )
        else:
            self.extra_modeling = False

    def forward(self, text: float["b nt d"], seq_len, drop_text=False):  # noqa: F722
        """text: phonetic representations, (B, Nt, D)"""
        text = text[:, :seq_len]  # curtail if character tokens are more than the mel spec tokens
        batch, text_len = text.shape[0], text.shape[1]
        text = F.pad(text, (0,0, 0, seq_len - text_len), value=0.)
        if self.mask_padding:
            text_mask = torch.zeros((batch, seq_len), dtype=torch.bool).to(text.device)
            text_mask[:, text_len:] = True  # padding positions

        if drop_text:  # cfg for text
            text = torch.zeros_like(text)

        text = self.proj(text)  # b n d -> b n d

        # possible extra modeling
        if self.extra_modeling:
            # sinus pos emb
            batch_start = torch.zeros((batch,), dtype=torch.long)
            pos_idx = get_pos_embed_indices(batch_start, seq_len, max_pos=self.precompute_max_pos)
            text_pos_embed = self.freqs_cis[pos_idx]
            text = text + text_pos_embed

            # convnextv2 blocks
            if self.mask_padding:
                text = text.masked_fill(text_mask.unsqueeze(-1).expand(-1, -1, text.size(-1)), 0.0)
                for block in self.text_blocks:
                    text = block(text)
                    text = text.masked_fill(text_mask.unsqueeze(-1).expand(-1, -1, text.size(-1)), 0.0)
            else:
                text = self.text_blocks(text)

        return text


# noised input audio and context mixing embedding
class InputEmbedding(nn.Module):
    def __init__(self, mel_dim, text_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(mel_dim * 3 + text_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(self, x: float["b n d"], cond: float["b n d"], cond_noisy: float["b n d"], text_embed: float["b n d"], drop_audio_cond=False, drop_noisy_audio_cond=False):  # noqa: F722
        if drop_audio_cond:  # cfg for cond audio
            cond = torch.zeros_like(cond)
        if drop_noisy_audio_cond:  # cfg for cond noisy audio
            cond_noisy = torch.zeros_like(cond_noisy)

        x = self.proj(torch.cat((x, cond, cond_noisy, text_embed), dim=-1))
        x = self.conv_pos_embed(x) + x
        return x


# Transformer backbone using DiT blocks
class DiT(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        mel_dim=100,
        text_in_dim=256,
        text_dim=None,
        text_mask_padding=True,
        qk_norm=None,
        conv_layers=0,
        pe_attn_head=None,
        long_skip_connection=False,
        checkpoint_activations=False,
    ):
        super().__init__()

        self.time_embed = TimestepEmbedding(dim)
        if text_dim is None:
            text_dim = mel_dim
        self.text_embed = TextEmbedding(
            text_in_dim, text_dim, mask_padding=text_mask_padding, conv_layers=conv_layers
        )
        self.text_cond, self.text_uncond = None, None  # text cache
        self.input_embed = InputEmbedding(mel_dim, text_dim, dim)

        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth

        self.transformer_blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    ff_mult=ff_mult,
                    dropout=dropout,
                    qk_norm=qk_norm,
                    pe_attn_head=pe_attn_head,
                )
                for _ in range(depth)
            ]
        )
        self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None

        self.norm_out = AdaLayerNorm_Final(dim)  # final modulation
        self.proj_out = nn.Linear(dim, mel_dim)

        self.checkpoint_activations = checkpoint_activations

        self.initialize_weights()

    def initialize_weights(self):
        # Zero-out AdaLN layers in DiT blocks:
        for block in self.transformer_blocks:
            nn.init.constant_(block.attn_norm.linear.weight, 0)
            nn.init.constant_(block.attn_norm.linear.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.norm_out.linear.bias, 0)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    def ckpt_wrapper(self, module):
        # https://github.com/chuanyangjin/fast-DiT/blob/main/models.py
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs

        return ckpt_forward

    def clear_cache(self):
        self.text_cond, self.text_uncond = None, None

    def forward(
        self,
        x: float["b n d"],  # nosied input audio  # noqa: F722
        cond: float["b n d"],  # masked cond audio  # noqa: F722
        cond_noisy: float["b n d"],  # masked cond noisy audio  # noqa: F722
        text: int["b nt"],  # text  # noqa: F722
        time: float["b"] | float[""],  # time step  # noqa: F821 F722
        drop_audio_cond,  # cfg for cond audio
        drop_noisy_audio_cond,  # cfg for cond noisy audio
        drop_text,  # cfg for text
        mask: bool["b n"] | None = None,  # noqa: F722
        cache=False,
    ):
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)

        # t: conditioning time, text: text, x: noised audio + cond audio + text
        t = self.time_embed(time)
        if cache:
            if drop_text:
                if self.text_uncond is None:
                    self.text_uncond = self.text_embed(text, seq_len, drop_text=True)
                text_embed = self.text_uncond
            else:
                if self.text_cond is None:
                    self.text_cond = self.text_embed(text, seq_len, drop_text=False)
                text_embed = self.text_cond
        else:
            text_embed = self.text_embed(text, seq_len, drop_text=drop_text)
        x = self.input_embed(x, cond, cond_noisy, text_embed, drop_audio_cond=drop_audio_cond, drop_noisy_audio_cond=drop_noisy_audio_cond)

        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        if self.long_skip_connection is not None:
            residual = x

        for block in self.transformer_blocks:
            if self.checkpoint_activations:
                # https://pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.checkpoint
                x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, t, mask, rope, use_reentrant=False)
            else:
                x = block(x, t, mask=mask, rope=rope)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, t)
        output = self.proj_out(x)

        return output


if __name__ == "__main__":
    from omegaconf import OmegaConf
    
    config = OmegaConf.load('configs/cfg_train_flow.yaml')
    dit_kwargs = config.cfm_config.dit_kwargs
    model = DiT(**dit_kwargs).cuda()
    
    x = torch.randn(2, 50, 100).cuda()  # gausian noised input
    cond = torch.randn(2, 50, 100).cuda()  # masked clean cond audio
    cond_noisy = torch.randn(2, 50, 100).cuda()  # masked noisy cond audio
    text = torch.randn(2, 40, 1024).cuda()  # phonetic cond representations
    time = torch.rand((2,)).cuda()
    
    output = model(x, cond, cond_noisy, text, time, drop_audio_cond=False, drop_noisy_audio_cond=False, drop_text=False)
    
    print(output.shape)