"""Contains code modified from https://github.com/openai/whisper"""

import math
import numpy as np
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


class MultiHeadAttention(nn.Module):
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
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ):
        q = self.query(x)

        if xa is not None:
            # Cross att
            k = self.key(xa)
            v = self.value(xa)
        else:
            # Self att in encoder/decoder
            k = self.key(x)
            v = self.value(x)

        # Reshape and transpose for attention calculation
        batch_size, target_seq_len, _ = q.shape
        batch_size, source_seq_len, _ = k.shape
        q = q.view(batch_size, target_seq_len, self.n_head, self.d_head)
        k = k.view(batch_size, source_seq_len, self.n_head, self.d_head)
        v = v.view(batch_size, source_seq_len, self.n_head, self.d_head)

        # (bz, L, nh, dh) -> (bz, nh, L, dh)
        q, k, v = map(lambda t: t.transpose(1, 2), (q, k, v))

        if mask == None:
            _is_causal = False
        else:
            _is_causal = True

        wv = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            is_causal=_is_causal,
        )

        # (bz, nh, L, dh) -> (bz, L, nh, dh) -> (bz, L, d)
        wv = wv.transpose(1, 2)
        wv = wv.view(batch_size, target_seq_len, self.n_head * self.d_head)

        return self.out(wv)


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self, n_state: int, n_head: int, cross_attention: bool = False
    ):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = nn.LayerNorm(n_state)

        self.cross_attn = (
            MultiHeadAttention(n_state, n_head) if cross_attention else None
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
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ):
        x = x + self.attn(self.attn_ln(x), mask=mask)
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa)
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

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = nn.LayerNorm(n_state)

    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        assert (
            x.shape[1:] == self.positional_embedding.shape
        ), f"incorrect audio shape: {x.shape[1:]} != {self.positional_embedding.shape}"
        x = (x + self.positional_embedding).to(x.dtype)

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        return x


class TextDecoder(nn.Module):
    def __init__(
        self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [
                ResidualAttentionBlock(n_state, n_head, cross_attention=True)
                for _ in range(n_layer)
            ]
        )
        self.ln = nn.LayerNorm(n_state)
        self.output = nn.Linear(n_state, n_vocab, bias=False)

        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: Tensor, xa: Tensor):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_audio_ctx, n_audio_state)
            the encoded audio features to be attended on
        """
        x = self.token_embedding(x) + self.positional_embedding[: x.shape[-1]]
        x = x.to(xa.dtype)

        for block in self.blocks:
            x = block(x, xa, mask=self.mask)

        x = self.ln(x)
        logits = self.output(x)

        return logits


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

    def embed_audio(self, mel: torch.Tensor):
        return self.encoder(mel)

    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
        return self.decoder(tokens, audio_features)

    def forward(self, mel: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        _buff = self.encoder(mel)
        return self.decoder(tokens, _buff)

    @property
    def device(self):
        return next(self.parameters()).device
