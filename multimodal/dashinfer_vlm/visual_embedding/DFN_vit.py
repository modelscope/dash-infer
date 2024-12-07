'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    DFN_vit.py
'''

import math

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import LayerNorm
import torch.utils.checkpoint

try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VisionTransformerPretrainedModel,
)

# from .model_loader import default_weight_loader
dtype = "fp32"


def default_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    """Default weight loader."""
    try:
        if param.numel() == 1 and loaded_weight.numel() == 1:
            # Sometimes scalar values aren't considered tensors with shapes
            # so if both param and loaded_weight are a scalar,
            # "broadcast" instead of copy
            param.data.fill_(loaded_weight.item())
        else:
            assert param.size() == loaded_weight.size(), (
                f"Attempted to load weight ({loaded_weight.size()}) "
                f"into parameter ({param.size()})"
            )

            param.data.copy_(loaded_weight)
    except Exception:
        # NOTE: This exception is added for the purpose of setting breakpoint to
        # debug weight loading issues.
        raise


def quick_gelu(x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    return x * torch.sigmoid(1.702 * x)


class QuickGELU(nn.Module):
    """Applies the Gaussian Error Linear Units function (w/ dummy inplace arg)"""

    def __init__(self, inplace: bool = False) -> None:
        super(QuickGELU, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return quick_gelu(input)


# Copied from transformers.models.qwen2.modeling_qwen2.Qwen2RotaryEmbedding
class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device)
                / self.dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=torch.int64
        ).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_multimodal_rotary_pos_emb(
    q, k, cos, sin, position_ids, mrope_section=None, unsqueeze_dim=1
):
    """Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        mrope_section(`List(int)`):
            Multimodal Sections for t,h,w in Multimodal inputs
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    if mrope_section:
        cos = cos[position_ids]
        sin = sin[position_ids]
        mrope_section = mrope_section * 2
        cos = torch.cat(
            [m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1
        ).unsqueeze(unsqueeze_dim)
        sin = torch.cat(
            [m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1
        ).unsqueeze(unsqueeze_dim)
    else:
        cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb_vision(
    tensor: torch.Tensor, freqs: torch.Tensor
) -> torch.Tensor:
    cos = freqs.cos()
    sin = freqs.sin()
    # rotary_2 interleaved start
    cos = cos.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0)
    sin = sin.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0)
    output = (tensor * cos) + (rotate_half(tensor) * sin)
    # rotary_2 interleaved end
    output = output.type_as(tensor)
    return output


class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._freqs_cached = None

    def forward(self, seqlen: int) -> torch.Tensor:
        seqlen *= 2
        self.inv_freq = 1.0 / (
            self.theta
            ** (
                torch.arange(
                    0, self.dim, 2, dtype=torch.float, device=self.inv_freq.device
                )
                / self.dim
            )
        )
        seq = torch.arange(
            seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )
        freqs = seq.unsqueeze(1) * self.inv_freq.unsqueeze(0)
        # freqs = torch.outer(seq, self.inv_freq)
        return freqs[:seqlen]


class PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_chans: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.embed_dim = embed_dim

        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seqlen = x.shape[0]
        x = x.view(
            seqlen, -1, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        x = self.proj(x).view(seqlen, self.embed_dim)
        return x


class PatchMerger(nn.Module):
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = LayerNorm(context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
        return x


class VisionMlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = QuickGELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class VisionAttention(nn.Module):
    def __init__(
        self, dim: int, num_heads: int = 16, use_flash_attention: bool = False
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_flash_attention = use_flash_attention
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor = None,
        b=1,
    ) -> torch.Tensor:
        if dtype == "fp16":
            return self.flash_forward(x, cu_seqlens, rotary_pos_emb)

        n = self.num_heads
        d = self.head_dim

        N, _ = x.shape
        # torch.save(x, "tmp.pt")
        # print(self.qkv.bias)

        # self.qkv.bias = nn.Parameter(self.qkv.bias.view(n, 3, d).transpose(0, 1).reshape(-1))
        qkv = self.qkv(x)
        # print(qkv.shape)
        # torch.save(qkv, "tmp.pt")

        # qkv = qkv.reshape(N, 3, self.num_heads, -1).permute(1, 0, 2, 3)
        # q, k, v = qkv.unbind(0)
        qkv = qkv.reshape(N, 3, self.num_heads, -1)
        # torch.save(qkv, "tmp.pt")
        q, k, v = qkv.split(1, dim=1)
        # torch.save(q, "tmp.pt")

        q = q.view(1, -1, n, d)
        k = k.view(1, -1, n, d)
        # print(k)
        # print(k.shape)
        # torch.save(v, "tmp.pt")
        # breakpoint()
        if rotary_pos_emb is not None:
            q = apply_rotary_pos_emb_vision(q, rotary_pos_emb)
            k = apply_rotary_pos_emb_vision(k, rotary_pos_emb)
        q = q.view(b, -1, n, d)
        k = k.view(b, -1, n, d)
        v = v.view(b, -1, n, d)

        softmax_scale = math.pow(d, -0.25)
        # attention_21 start
        b = v.size(0)
        q = q.view(b, -1, n, d)
        k = k.view(b, -1, n, d)
        v = v.view(b, -1, n, d)

        q = q.permute(0, 2, 1, 3) * softmax_scale
        k = k.permute(0, 2, 3, 1) * softmax_scale
        v = v.permute(0, 2, 1, 3)

        attn = torch.matmul(q, k)
        attn = F.softmax(attn, dim=-1).type_as(attn)
        x = torch.matmul(attn, v).permute(0, 2, 1, 3)
        x = x.reshape(b, -1, n * d)
        # attention_21 end
        x = self.proj(x.contiguous())
        x = x.view(-1, n * d)
        # print(x)
        return x

    def flash_forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor = None,
    ) -> torch.Tensor:
        L, _ = x.shape
        q, k, v = (
            self.qkv(x).reshape(L, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        if rotary_pos_emb is not None:
            q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
            k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

        if (
            flash_attn_varlen_func is not None
            and q.dtype in [torch.float16, torch.bfloat16]
            and self.use_flash_attention
        ):
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
            x = flash_attn_varlen_func(
                q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen
            )
            x = x.reshape(L, -1)
        else:
            attention_mask = torch.zeros([1, L, L], device=q.device, dtype=torch.bool)
            for i in range(1, len(cu_seqlens)):
                attention_mask[
                    ...,
                    cu_seqlens[i - 1] : cu_seqlens[i],
                    cu_seqlens[i - 1] : cu_seqlens[i],
                ] = True
            q = q.transpose(0, 1)
            k = k.transpose(0, 1)
            v = v.transpose(0, 1)
            x = (
                F.scaled_dot_product_attention(q, k, v, attention_mask, dropout_p=0.0)
                .transpose(0, 1)
                .reshape(L, -1)
            )
        x = self.proj(x)
        return x


class Qwen2VLVisionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        norm_layer: nn.Module = partial(LayerNorm, eps=1e-6),
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.attn = VisionAttention(
            dim, num_heads=num_heads, use_flash_attention=use_flash_attention
        )
        self.mlp = VisionMlp(dim=dim, hidden_dim=mlp_hidden_dim)

    def forward(self, x, cu_seqlens, rotary_pos_emb, b) -> torch.Tensor:
        x = x + self.attn(
            self.norm1(x), cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb, b=b
        )
        x = x + self.mlp(self.norm2(x))
        return x


# class Qwen2VisionTransformer(nn.Module):
class Qwen2VisionTransformer(Qwen2VisionTransformerPretrainedModel):
    def __init__(self, config):
        #     img_size: int = 378,
        #     patch_size: int = 14,
        #     temporal_patch_size: int = 2,
        #     spatial_merge_size: int = 2,
        #     in_chans: int = 3,
        #     hidden_size: int = 1000,
        #     embed_dim: int = 768,
        #     depth: int = 12,
        #     num_heads: int = 16,
        #     mlp_ratio: float = 4.0,
        #     norm_layer: nn.Module = partial(LayerNorm, eps=1e-6),
        #     use_flash_attention: bool = False,
        #     *args,
        #     **kwargs,
        # ) -> None:
        super().__init__(config)
        self.spatial_merge_size = config.spatial_merge_size

        self.patch_embed = PatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_chans=config.in_chans,
            embed_dim=config.embed_dim,
        )

        head_dim = config.embed_dim // config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        norm_layer = partial(LayerNorm, eps=1e-6)
        use_flash_attention = False
        self.blocks = nn.ModuleList(
            [
                Qwen2VLVisionBlock(
                    dim=config.embed_dim,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    norm_layer=norm_layer,
                    use_flash_attention=use_flash_attention,
                )
                for _ in range(config.depth)
            ]
        )
        self.merger = PatchMerger(dim=config.hidden_size, context_dim=config.embed_dim)

    def get_dtype(self) -> torch.dtype:
        return self.blocks[0].mlp.fc2.weight.dtype

    def get_device(self) -> torch.device:
        return self.blocks[0].mlp.fc2.weight.device

    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            hpos_ids = (
                hpos_ids.reshape(
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                )
                .permute(0, 2, 1, 3)
                .flatten()
            )
            wpos_ids = (
                wpos_ids.reshape(
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                )
                .permute(0, 2, 1, 3)
                .flatten()
            )
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def fix_attn_bias(self):
        for blk in self.blocks:
            blk.attn.qkv.bias = nn.Parameter(
                blk.attn.qkv.bias.view(blk.attn.num_heads, 3, -1)
                .transpose(0, 1)
                .reshape(-1)
            )

    def forward(
        self, x: torch.Tensor, grid_thw: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        x = self.patch_embed(x)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        print(grid_thw.shape)
        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        print("batch: ", batch.size(0))
        for blk in self.blocks:
            x = blk(
                x, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb, b=batch.size(0)
            )

        return self.merger(x)

    def load_weights(self, weights):
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            if not name.startswith("visual."):
                continue
            name = name.split("visual.")[1]
            if "blocks" in name and "attn.proj.bias" in name:
                continue

            # Note: only used for debug
            if name not in params_dict.keys():
                continue
            default_weight_loader(params_dict[name], loaded_weight)
