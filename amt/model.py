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
        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state)
        self.out = nn.Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
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

        # Old ---
        # wv, qk = self.qkv_attention(q, k, v, mask)
        # End ---

        # New code ------
        debug = False
        # Reshape and transpose for attention calculation
        batch_size, target_seq_len, _ = q.shape
        batch_size, source_seq_len, _ = k.shape
        q = q.view(batch_size, target_seq_len, self.n_head, self.d_head)
        k = k.view(batch_size, source_seq_len, self.n_head, self.d_head)
        v = v.view(batch_size, source_seq_len, self.n_head, self.d_head)

        # (bz, L, nh, dh) -> (bz, nh, L, dh)
        q, k, v = map(lambda t: t.transpose(1, 2), (q, k, v))

        if debug is True:
            print(f"q shape: {q.shape}")
            print(f"k shape: {k.shape}")
            print(f"v shape: {v.shape}")
            if mask is not None:
                print(f"mask shape: {mask.shape}")

        if mask == None:
            _is_causal = False
        else:
            _is_causal = True

        qk = None  # Only used during kv-caching?
        wv = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            is_causal=_is_causal,
        )

        # (bz, nh, L, dh) -> (bz, L, nh, dh) -> (bz, L, d)
        wv = wv.transpose(1, 2)
        wv = wv.view(batch_size, target_seq_len, self.n_head * self.d_head)

        if debug is True:
            print(f"att_out shape: {wv.shape}")
            if qk is not None:
                print(f"att_weights shape: {qk.shape}")

        # End new code ------

        return self.out(wv), qk

    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
        qk = qk.float()

        w = F.softmax(qk, dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()


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
            nn.Linear(n_state, n_mlp), nn.GELU(), nn.Linear(n_mlp, n_state)
        )
        self.mlp_ln = nn.LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
        if self.cross_attn:
            x = (
                x
                + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[
                    0
                ]
            )
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

        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_audio_ctx, n_audio_state)
            the encoded audio features to be attended on
        """
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = (
            self.token_embedding(x)
            + self.positional_embedding[offset : offset + x.shape[-1]]
        )
        x = x.to(xa.dtype)

        for block in self.blocks:
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)

        x = self.ln(x)
        logits = (
            x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)
        ).float()

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
        # use the last half among the decoder layers for time alignment by default;
        # to use a specific set of heads, see `set_alignment_heads()` below.
        # all_heads = torch.zeros(
        #     self.dims.n_text_layer, self.dims.n_text_head, dtype=torch.bool
        # )
        # all_heads[self.dims.n_text_layer // 2 :] = True
        # self.register_buffer(
        #     "alignment_heads", all_heads.to_sparse(), persistent=False
        # )

    # def set_alignment_heads(self, dump: bytes):
    #     array = np.frombuffer(
    #         gzip.decompress(base64.b85decode(dump)), dtype=bool
    #     ).copy()
    #     mask = torch.from_numpy(array).reshape(
    #         self.dims.n_text_layer, self.dims.n_text_head
    #     )
    #     self.register_buffer(
    #         "alignment_heads", mask.to_sparse(), persistent=False
    #     )

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

    def install_kv_cache_hooks(self, cache: Optional[dict] = None):
        """
        The `MultiHeadAttention` module optionally accepts `kv_cache` which stores the key and value
        tensors calculated for the previous positions. This method returns a dictionary that stores
        all caches, and the necessary hooks for the key and value projection modules that save the
        intermediate tensors to be reused during later calculations.

        Returns
        -------
        cache : Dict[nn.Module, torch.Tensor]
            A dictionary object mapping the key/value projection modules to its cache
        hooks : List[RemovableHandle]
            List of PyTorch RemovableHandle objects to stop the hooks to be called
        """
        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            if module not in cache or output.shape[1] > self.dims.n_text_ctx:
                # save as-is, for the first token or cross attention
                cache[module] = output
            else:
                cache[module] = torch.cat(
                    [cache[module], output], dim=1
                ).detach()
            return cache[module]

        def install_hooks(layer: nn.Module):
            if isinstance(layer, MultiHeadAttention):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        self.decoder.apply(install_hooks)
        return cache, hooks
