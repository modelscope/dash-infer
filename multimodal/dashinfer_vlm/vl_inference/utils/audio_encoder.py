'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    audio_encoder.py
'''
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers.models.qwen2_audio.modeling_qwen2_audio import Qwen2AudioEncoder


@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int
    att_type: str = "default"


class LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


class Conv1d(nn.Conv1d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
        **kwargs,
    ):
        is_pad_mask = kwargs.get("is_pad_mask", False)

        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk = self.qkv_attention(q, k, v, mask, is_pad_mask=is_pad_mask)
        return self.out(wv), qk

    def qkv_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Optional[Tensor] = None,
        **kwargs,
    ):
        kwargs.get("is_pad_mask", False)
        n_batch, n_ctx, n_state = q.shape
        head_dim = n_state // self.n_head
        scale = (n_state // self.n_head) ** -0.25

        # q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        # k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        # v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        # qk = q @ k
        # if mask is not None:
        #     if not is_pad_mask:
        #         qk = qk + mask[:n_ctx, :n_ctx]
        #     else:
        #         mask = mask.unsqueeze(1).eq(0)  # (batch, 1, t, 1)
        #         min_value = -float(
        #             "inf"
        #         )  # min_value = float(np.finfo(torch.tensor(0, dtype=qk.dtype).numpy().dtype).min)
        #         qk = qk.masked_fill(mask, min_value)

        # qk = qk.float()

        # w = F.softmax(qk, dim=-1).to(q.dtype)
        # if mask is not None and is_pad_mask:
        #     w = w.masked_fill(mask, 0.0)
        # return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()

        n = self.n_head
        d = head_dim
        # attention_21 start
        b = v.size(0)
        q = q.view(b, -1, n, d)
        k = k.view(b, -1, n, d)
        v = v.view(b, -1, n, d)

        q = q.permute(0, 2, 1, 3) * scale
        k = k.permute(0, 2, 3, 1) * scale
        v = v.permute(0, 2, 1, 3)

        attn = torch.matmul(q, k)
        attn = F.softmax(attn, dim=-1).type_as(attn)
        x = torch.matmul(attn, v).permute(0, 2, 1, 3)
        x = x.reshape(b, -1, n * d)
        # attention_21 end
        return x, attn.detach()


class MultiHeadAttentionSdpa(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
        **kwargs,
    ):
        is_pad_mask = kwargs.get("is_pad_mask", False)

        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk = self.qkv_attention(
            q, k, v, mask, is_pad_mask=is_pad_mask, is_causal=False
        )
        return self.out(wv), qk

    def qkv_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Optional[Tensor] = None,
        **kwargs,
    ):
        is_pad_mask = kwargs.get("is_pad_mask", False)
        is_causal = kwargs.get("is_causal", False)
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.5
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        if mask is not None:
            if not is_pad_mask:
                mask = None
                is_causal = True
            else:
                mask = mask.unsqueeze(1).to(torch.bool)  # (batch, 1, 1, t)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=scale,
        )
        if mask is not None:
            attn_output = attn_output.masked_fill(
                mask.transpose(2, 3).logical_not(), 0.0
            )
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.flatten(start_dim=2)
        return attn_output, None


att_type_dict = {
    "default": MultiHeadAttention,
    "sdpa": MultiHeadAttentionSdpa,
}


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self, n_state: int, n_head: int, cross_attention: bool = False, **kwargs
    ):
        super().__init__()

        att_type = kwargs.get("att_type", "default")
        self.attn = att_type_dict[att_type](
            n_state, n_head
        )  # MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = (
            att_type_dict[att_type](n_state, n_head) if cross_attention else None
        )  # MultiHeadAttention(n_state, n_head) if cross_attention else None
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
        **kwargs,
    ):
        is_pad_mask = kwargs.get("is_pad_mask", False)
        is_pad_memory_mask = kwargs.get("is_pad_memory_mask", False)
        memory_mask = kwargs.get("memory_mask", None)
        x = (
            x
            + self.attn(
                self.attn_ln(x), mask=mask, kv_cache=kv_cache, is_pad_mask=is_pad_mask
            )[0]
        )
        if self.cross_attn:
            x = (
                x
                + self.cross_attn(
                    self.cross_attn_ln(x),
                    xa,
                    mask=memory_mask,
                    kv_cache=kv_cache,
                    is_pad_mask=is_pad_memory_mask,
                )[0]
            )
        x = x + self.mlp(self.mlp_ln(x))
        return x


class AudioEncoder(Qwen2AudioEncoder):
    # def __init__(self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int,
    # **kwargs):
    def __init__(self, config):
        super().__init__(config)
        # self.conv1 = Conv1d(config.num_mel_bins, config.d_model, kernel_size=3, stride=2, padding=1)
        # self.conv2 = Conv1d(config.d_model, config.d_model, kernel_size=3, stride=2, padding=1)
        # self.register_buffer("positional_embedding", sinusoids(config.max_source_positions, config.d_model))

        # self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
        #     [
        #         ResidualAttentionBlock(n_state, n_head, att_type=kwargs.get("att_type", "default"))
        #         for _ in range(n_layer)
        #     ]
        # )
        # self.ln_post = LayerNorm(n_state)

    # def forward(self, x: Tensor):
    #     """
    #     x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
    #         the mel spectrogram of the audio
    #     """
    #     x = F.gelu(self.conv1(x))
    #     x = F.gelu(self.conv2(x))
    #     x = x.permute(0, 2, 1)

    #     # assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
    #     # x = (x + self.positional_embedding).to(x.dtype)
    #     x = (x + self.positional_embedding[: x.size(1), :]).to(x.dtype)

    #     for block in self.blocks:
    #         x = block(x)

    #     x = self.ln_post(x)
    #     return x

    def load_weights(self, weights):
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            if not name.startswith("audio"):
                continue
            name = name.split("audio_tower.")[1]
            if "blocks" in name and "attn.proj.bias" in name:
                continue

            # Note: only used for debug
            if name not in params_dict.keys():
                continue
            default_weight_loader(params_dict[name], loaded_weight)
