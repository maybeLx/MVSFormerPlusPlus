# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging

import math
import pdb

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
logger = logging.getLogger("dinov2")

try:
    from xformers.ops import memory_efficient_attention, unbind, fmha

    XFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("xFormers not available")
    XFORMERS_AVAILABLE = False

try:
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_func

    FLASH_AVAILABLE = True
except ImportError:
    logger.warning("FLASH ATTENTION2 not available")
    FLASH_AVAILABLE = False


def get_attention_type(attention_type):
    if attention_type == 'Linear':
        attention_class = CrossLinearAttention
    elif attention_type == 'FLASH2':
        attention_class = CrossFlashAttention2
    elif attention_type == "XFormers":
        attention_class = CrossXFormersAttention
    else:
        raise NotImplementedError('Unkown attention type', attention_type)
    return attention_class


class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
            **kwargs
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.attn_drop_rate = attn_drop

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        # additional settings
        self.softmax_scale = kwargs.get('softmax_scale', None)
        self.train_avg_length = kwargs.get('train_avg_length', None)


    def forward(self, x: Tensor, return_attn=False) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]
        softmax_scale = self.scale
        if self.softmax_scale == "entropy_invariance":
            softmax_scale *= math.log(N, self.train_avg_length)

        # q *= softmax_scale
        #
        # attn = q @ k.transpose(-2, -1)
        #
        # attn = attn.softmax(dim=-1)
        # # attn = self.attn_drop(attn)
        #
        # x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # use F.scaled_dot_product_attention for torch>=2.1 with custom scale
        x = F.scaled_dot_product_attention(q, k, v, scale=softmax_scale)
        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x




class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None, return_attn=False, positions=None):
        if not XFORMERS_AVAILABLE:
            assert attn_bias is None, "xFormers is required for nested tensors usage"
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        if self.softmax_scale is None:
            softmax_scale = None
        else:
            softmax_scale = self.scale * math.log(N, self.train_avg_length)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias, scale=softmax_scale)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attn:
            # q:[B,L,nhead,C]
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            # [B,nhead,1,C]x[B,nhead,C,L-1]=[B,nhead,1,L-1]
            a = q[:, :, 0:1] @ k[:, :, 1:].transpose(-2, -1) * self.scale if softmax_scale is None else softmax_scale
            a = a.squeeze(2)  # [B,nhead,L-1]
            a = torch.softmax(a, dim=2)
            return x, a
        else:
            return x


class FlashAttention2(Attention):
    def forward(self, x: Tensor, attn_bias=None, positions=None, return_attn=False):

        if not FLASH_AVAILABLE:
            assert attn_bias is None, "FLASH-ATTENTION2 is required for nested tensors usage"
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        # qkv: (batch_size, seqlen, 3, nheads, headdim)
        # dropout_p: float. Dropout probability. use 0 for inference
        if self.training:
            dropout_p = self.attn_drop_rate
        else:
            dropout_p = 0.0

        if self.softmax_scale is None:
            softmax_scale = None
        else:
            softmax_scale = self.scale * math.log(N, self.train_avg_length)

        # out: (batch_size, seqlen, nheads, headdim)
        x = flash_attn_qkvpacked_func(qkv, dropout_p=dropout_p, softmax_scale=softmax_scale, causal=False)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class CrossAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
            **kwargs
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.attn_drop_rate = attn_drop

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        # additional settings
        self.softmax_scale = kwargs.get('softmax_scale', None)
        self.train_avg_length = kwargs.get('train_avg_length', None)

        # if kwargs.get("attention_type") == 'TransNormer':
        #     self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x: Tensor, key=None, value=None, **kwargs) -> Tensor:
        B, N, C = x.shape
        key = x if key is None else key
        value = x if value is None else value
        q = self.q_proj(x).reshape(B, N, self.num_heads, C // self.num_heads)
        k = self.k_proj(key).reshape(B, N, self.num_heads, C // self.num_heads)
        v = self.v_proj(value).reshape(B, N, self.num_heads, C // self.num_heads)

        softmax_scale = self.scale
        if self.softmax_scale == "entropy_invariance":
            softmax_scale *= math.log(N, self.train_avg_length)
        q *= softmax_scale

        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossFlashAttention2(CrossAttention):
    def forward(self, x: Tensor, key=None, value=None, **kwargs):

        if not FLASH_AVAILABLE:
            raise NotImplementedError("FLASH-ATTENTION2 is not working!")

        B, N, C = x.shape
        key = x if key is None else key
        value = x if value is None else value
        q = self.q_proj(x).reshape(B, N, self.num_heads, C // self.num_heads)
        k = self.k_proj(key).reshape(B, N, self.num_heads, C // self.num_heads)
        v = self.v_proj(value).reshape(B, N, self.num_heads, C // self.num_heads)

        # dropout_p: float. Dropout probability. use 0 for inference
        if self.training:
            dropout_p = self.attn_drop_rate
        else:
            dropout_p = 0.0

        if self.softmax_scale is None:
            softmax_scale = None
        else:
            softmax_scale = self.scale * math.log(N, self.train_avg_length)

        # out: (batch_size, seqlen, nheads, headdim)
        x = flash_attn_func(q, k, v, dropout_p=dropout_p, softmax_scale=softmax_scale, causal=False)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class CrossLinearAttention(CrossAttention):
    def forward(self, x: Tensor, key=None, value=None, **kwargs):
        eps = 1e-6

        B, N, C = x.shape
        key = x if key is None else key
        value = x if value is None else value
        q = self.q_proj(x).reshape(B, N, self.num_heads, C // self.num_heads).to(dtype=torch.float32)
        k = self.k_proj(key).reshape(B, N, self.num_heads, C // self.num_heads).to(dtype=torch.float32)
        v = self.v_proj(value).reshape(B, N, self.num_heads, C // self.num_heads).to(dtype=torch.float32)
        q = torch.nn.functional.elu(q) + 1
        k = torch.nn.functional.elu(k) + 1

        # Compute the KV matrix, namely the dot product of keys and values so
        # that we never explicitly compute the attention matrix and thus
        # decrease the complexity [B,L,nh,C]x[B,L,nh,C]->[B,nh,C,C]
        with torch.cuda.amp.autocast(enabled=False):
            KV = torch.einsum("nshd,nshm->nhmd", k, v)

            # Compute the normalizer [B,L,nh,C]x[B,nh,C]->[B,L,nh]
            Z = 1 / (torch.einsum("nlhd,nhd->nlh", q, k.sum(dim=1)) + eps)

            # Finally compute and return the new values
            V = torch.einsum("nlhd,nhmd,nlh->nlhm", q, KV, Z)  # [B,L,nh,C]

        V = V.reshape(B, N, C).contiguous()

        V = self.proj(V)
        V = self.proj_drop(V)

        return V


class CrossXFormersAttention(Attention):
    def forward(self, x: Tensor, key=None, value=None, attn_bias=None, **kwargs):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        if self.softmax_scale is None:
            softmax_scale = None
        else:
            softmax_scale = self.scale * math.log(N, self.train_avg_length)

        if attn_bias is not None:
            attn_bias = attn_bias.unsqueeze(1).repeat(1, self.num_heads, 1, 1).to(q.dtype)  # [B,HW,HW]->[B,nh,HW,HW]
        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias, scale=softmax_scale)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)

        return x



if __name__ == '__main__':
    N= 128
    x = torch.rand(1,N,64)
    Atten = Attention(dim=64)
    train_avg_length = 64
    scale = Atten.scale * math.log(N, train_avg_length)
    pdb.set_trace()
