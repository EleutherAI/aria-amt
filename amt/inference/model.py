"""Contains code modified from https://github.com/openai/whisper"""

import math
import torch
import torch.nn.functional as F

from torch import Tensor, nn
from dataclasses import dataclass
from typing import Dict, Iterable, Optional


@dataclass
class ModelConfig:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int
    n_vocab: Optional[int] = None

    def set_vocab_size(self, vocab_size: int):
        self.n_vocab = vocab_size


class KVCache(nn.Module):
    def __init__(
        self,
        max_batch_size: int,
        max_seq_length: int,
        n_heads: int,
        head_dim: int,
        dtype=torch.bfloat16,
    ):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val, v_val: [B, H, L, D]

        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out


def sinusoids(
    length: int, channels: int, max_timescale: float = 10000
) -> torch.Tensor:
    """Returns sinusoids for positional embedding"""
    if channels % 2 != 0:
        raise ValueError(
            f"Number of channels has to be divisible by 2 for sinusoidal positional embeddings, got {channels} channels."
        )
    log_timescale_increment = math.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(
        -log_timescale_increment * torch.arange(channels // 2)
    )
    scaled_time = torch.arange(length).view(-1, 1) * inv_timescales.view(1, -1)
    return torch.cat([scaled_time.sin(), scaled_time.cos()], dim=1)


class EncoderAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        assert n_state % n_head == 0, "n_head does not evenly devide n_state"

        self.n_head = n_head
        self.d_head = n_state // n_head
        self.query = nn.Linear(n_state, n_state, bias=False)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state, bias=False)
        self.out = nn.Linear(n_state, n_state, bias=False)

    def forward(
        self,
        xa: Tensor,
    ):
        q = self.query(xa)
        k = self.key(xa)
        v = self.value(xa)

        # Reshape for correct format
        batch_size, source_seq_len, _ = k.shape
        batch_size, target_seq_len, _ = q.shape
        q = q.view(
            batch_size, target_seq_len, self.n_head, self.d_head
        ).transpose(1, 2)
        k = k.view(
            batch_size, source_seq_len, self.n_head, self.d_head
        ).transpose(1, 2)
        v = v.view(
            batch_size, source_seq_len, self.n_head, self.d_head
        ).transpose(1, 2)

        wv = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            is_causal=False,
        )
        wv = wv.transpose(1, 2).reshape(
            batch_size,
            target_seq_len,
            self.n_head * self.d_head,
        )

        return self.out(wv)


class CrossAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        assert n_state % n_head == 0, "n_head does not evenly devide n_state"

        self.n_head = n_head
        self.d_head = n_state // n_head
        self.query = nn.Linear(n_state, n_state, bias=False)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state, bias=False)
        self.out = nn.Linear(n_state, n_state, bias=False)
        self.kv_cache: KVCache | None = None

    def get_kv(self, xa: torch.Tensor, xa_input_pos: Tensor):
        assert self.kv_cache is not None, "No kv_cache"
        k = self.key(xa[:, xa_input_pos])
        v = self.value(xa[:, xa_input_pos])

        # Reshape for correct format
        batch_size, source_seq_len, _ = k.shape
        k = k.view(
            batch_size, source_seq_len, self.n_head, self.d_head
        ).transpose(1, 2)
        v = v.view(
            batch_size, source_seq_len, self.n_head, self.d_head
        ).transpose(1, 2)

        k, v = self.kv_cache.update(k_val=k, v_val=v, input_pos=xa_input_pos)

        return k, v

    def forward(
        self,
        x: Tensor,
        xa: Tensor,
        xa_input_pos: Tensor,
    ):
        q = self.query(x)
        batch_size, target_seq_len, _ = q.shape
        q = q.view(
            batch_size, target_seq_len, self.n_head, self.d_head
        ).transpose(1, 2)

        k, v = self.get_kv(xa, xa_input_pos)
        wv = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            is_causal=False,
        )
        wv = wv.transpose(1, 2).reshape(
            batch_size,
            target_seq_len,
            self.n_head * self.d_head,
        )

        return self.out(wv)


class CausalSelfAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        assert n_state % n_head == 0, "n_head does not evenly devide n_state"

        self.n_state = n_state
        self.n_head = n_head
        self.d_head = n_state // n_head
        self.out = nn.Linear(n_state, n_state, bias=False)
        self.kv_cache: KVCache | None = None

        # Add this back after
        self.combined_qkv = nn.Linear(n_state, 3 * n_state, bias=False)
        self._register_load_state_dict_pre_hook(self.combined_qkv_hook)

    def get_kv(self, k: Tensor, v: Tensor, input_pos: Tensor):
        k, v = self.kv_cache.update(k_val=k, v_val=v, input_pos=input_pos)

        return k, v

    def combined_qkv_hook(self, state_dict, prefix, *args):
        if prefix + "query.weight" in state_dict:
            wq = state_dict.pop(prefix + "query.weight")
            wk = state_dict.pop(prefix + "key.weight")
            wv = state_dict.pop(prefix + "value.weight")
            state_dict[prefix + "combined_qkv.weight"] = torch.cat([wq, wk, wv])

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        input_pos: Optional[Tensor] = None,
    ):
        q, k, v = self.combined_qkv(x).split(
            [self.n_state, self.n_state, self.n_state], dim=-1
        )

        batch_size, target_seq_len, _ = q.shape
        q = q.view(
            batch_size, target_seq_len, self.n_head, self.d_head
        ).transpose(1, 2)

        batch_size, source_seq_len, _ = k.shape
        k = k.view(
            batch_size, source_seq_len, self.n_head, self.d_head
        ).transpose(1, 2)
        v = v.view(
            batch_size, source_seq_len, self.n_head, self.d_head
        ).transpose(1, 2)

        k, v = self.get_kv(k, v, input_pos=input_pos)
        wv = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=mask,
        )

        # (bz, nh, L, dh) -> (bz, L, nh, dh) -> (bz, L, d)
        wv = wv.transpose(1, 2).reshape(
            batch_size, target_seq_len, self.n_head * self.d_head
        )

        return self.out(wv)


class EncoderAttentionBlock(nn.Module):
    def __init__(
        self, n_state: int, n_head: int, cross_attention: bool = False
    ):
        super().__init__()
        self.attn = EncoderAttention(n_state, n_head)
        self.attn_ln = nn.LayerNorm(n_state)
        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            nn.Linear(n_state, n_mlp, bias=False),
            nn.GELU(),
            nn.Linear(n_mlp, n_state, bias=False),
        )
        self.mlp_ln = nn.LayerNorm(n_state)

    def forward(
        self,
        xa: Tensor,
    ):
        xa = xa + self.attn(
            self.attn_ln(xa),
        )
        xa = xa + self.mlp(self.mlp_ln(xa))

        return xa


class DecoderAttentionBlock(nn.Module):
    def __init__(
        self, n_state: int, n_head: int, cross_attention: bool = False
    ):
        super().__init__()
        self.attn = CausalSelfAttention(n_state, n_head)
        self.attn_ln = nn.LayerNorm(n_state)
        self.cross_attn = (
            CrossAttention(n_state, n_head) if cross_attention else None
        )
        self.cross_attn_ln = nn.LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            nn.Linear(n_state, n_mlp, bias=False),
            nn.GELU(),
            nn.Linear(n_mlp, n_state, bias=False),
        )
        self.mlp_ln = nn.LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Tensor,
        mask: Optional[Tensor] = None,
        x_input_pos: Optional[Tensor] = None,
        xa_input_pos: Optional[Tensor] = None,
    ):
        x = x + self.attn(
            self.attn_ln(x),
            mask=mask,
            input_pos=x_input_pos,
        )
        x = x + self.cross_attn(self.cross_attn_ln(x), xa, xa_input_pos)
        x = x + self.mlp(self.mlp_ln(x))

        return x


class AudioEncoder(nn.Module):
    def __init__(
        self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(
            n_state, n_state, kernel_size=3, stride=2, padding=1
        )
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks: Iterable[EncoderAttentionBlock] = nn.ModuleList(
            [EncoderAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = nn.LayerNorm(n_state)

    def forward(self, xa: Tensor):
        xa = F.gelu(self.conv1(xa))
        xa = F.gelu(self.conv2(xa))
        xa = xa.permute(0, 2, 1)

        assert (
            xa.shape[1:] == self.positional_embedding.shape
        ), f"incorrect audio shape: {xa.shape[1:]} != {self.positional_embedding.shape}"
        xa = (xa + self.positional_embedding).to(xa.dtype)

        for block in self.blocks:
            xa = block(xa)

        xa = self.ln_post(xa)
        return xa


class TextDecoder(nn.Module):
    def __init__(
        self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

        self.blocks: Iterable[DecoderAttentionBlock] = nn.ModuleList(
            [
                DecoderAttentionBlock(n_state, n_head, cross_attention=True)
                for _ in range(n_layer)
            ]
        )
        self.ln = nn.LayerNorm(n_state)
        self.output = nn.Linear(n_state, n_vocab, bias=False)
        self.register_buffer("causal_mask", None, persistent=False)

    def forward(
        self,
        x: Tensor,
        xa: Tensor,
        x_input_pos: Tensor,
        xa_input_pos: Tensor,
    ):
        mask = self.causal_mask[None, None, x_input_pos]
        x = self.token_embedding(x) + self.positional_embedding[x_input_pos]

        for block in self.blocks:
            x = block(
                x=x,
                xa=xa,
                mask=mask,
                x_input_pos=x_input_pos,
                xa_input_pos=xa_input_pos,
            )

        x = self.ln(x)
        logits = self.output(x)

        return logits

    def setup_cache(
        self,
        batch_size,
        max_seq_len=4096,
        max_audio_len=1500,
    ):
        self.causal_mask = torch.tril(
            torch.ones(max_seq_len, max_seq_len, dtype=torch.bool)
        )
        # Init cache
        for b in self.blocks:
            b.attn.kv_cache = KVCache(
                max_batch_size=batch_size,
                max_seq_length=max_seq_len,
                n_heads=8,
                head_dim=64,
            ).cuda()
            b.cross_attn.kv_cache = KVCache(
                max_batch_size=batch_size,
                max_seq_length=max_audio_len,
                n_heads=8,
                head_dim=64,
            ).cuda()


class AmtEncoderDecoder(nn.Module):
    def __init__(self, dims: ModelConfig):
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
        )
        self.decoder = TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
        )

    def forward(self, mel: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        _buff = self.encoder(mel)
        return self.decoder(tokens, _buff)

    @property
    def device(self):
        return next(self.parameters()).device
