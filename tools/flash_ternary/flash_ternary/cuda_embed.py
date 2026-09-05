"""Full ternlight embed on CUDA with BitLinear weights pack-maintained (Triton).

Loads a `.bin`, keeps every BitLinear as uint8 pack on GPU, runs the same
graph as `training/pack/unpack.UnpackedModel.forward` (LN / RoPE / attn / FFN /
mean-pool / fp32 proj / L2-norm).

BitLinear weights stay uint8 pack. The embedding table stays int4 pack on
GPU too (gather + dequant per lookup) — expanding the 50k×512 table to
fp32 at load was ~98 MiB and the reason pack_cuda VRAM looked like a
"few-MB model using hundreds of MiB" (plus the PyTorch CUDA context).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ternlight_layer import BitLinearTritonFromBin


def _ensure_pack_path():
    repo = Path(os.environ.get("TERNLIGHT_REPO") or Path(__file__).resolve().parents[3])
    pack = repo / "training" / "pack"
    if str(pack) not in sys.path:
        sys.path.insert(0, str(pack))


def _opt_to(t: torch.Tensor | None, device) -> torch.Tensor | None:
    return None if t is None else t.to(device)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _rope_cos_sin(T: int, d_head: int, device, dtype, base: float = 10_000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2, device=device).float() / d_head))
    t = torch.arange(T, device=device, dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype)[None, None], emb.sin().to(dtype)[None, None]


def _apply_rope(q, k, cos, sin):
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


@dataclass
class _CudaLayer:
    ln1_w: torch.Tensor
    ln1_b: torch.Tensor
    w_q: BitLinearTritonFromBin
    w_k: BitLinearTritonFromBin
    w_v: BitLinearTritonFromBin
    subln_attn_w: torch.Tensor | None
    subln_attn_b: torch.Tensor | None
    w_out: BitLinearTritonFromBin
    ln2_w: torch.Tensor
    ln2_b: torch.Tensor
    fc1: BitLinearTritonFromBin
    subln_ffn_w: torch.Tensor | None
    subln_ffn_b: torch.Tensor | None
    fc2: BitLinearTritonFromBin


class PackCudaEmbed(nn.Module):
    """CUDA pack-maintained embedder from a ternlight `.bin`."""

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        output_dim: int,
        pos_rope: bool,
        embedding: torch.Tensor | None,
        layers: list[_CudaLayer],
        ln_final_w: torch.Tensor,
        ln_final_b: torch.Tensor,
        proj_W: torch.Tensor,
        proj_b: torch.Tensor,
        emb_packed: torch.Tensor | None = None,
        emb_scales: torch.Tensor | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.output_dim = output_dim
        self.pos_rope = pos_rope
        if emb_packed is not None and emb_scales is not None:
            self.register_buffer("emb_packed", emb_packed.contiguous())
            self.register_buffer("emb_scales", emb_scales.contiguous())
            # Device sentinel for embed_texts; not used as a lookup table.
            self.register_buffer("embedding", torch.empty(0, device=emb_packed.device))
        else:
            if embedding is None:
                raise ValueError("need embedding or int4 emb_packed+emb_scales")
            self.register_buffer("embedding", embedding.contiguous())
            self.emb_packed = None
            self.emb_scales = None
        self.layers = layers  # not nn.ModuleList of plain dataclass — register children
        self._mods = nn.ModuleList()
        for L in layers:
            self._mods.extend([L.w_q, L.w_k, L.w_v, L.w_out, L.fc1, L.fc2])
        self.register_buffer("ln_final_w", ln_final_w.contiguous())
        self.register_buffer("ln_final_b", ln_final_b.contiguous())
        self.register_buffer("proj_W", proj_W.contiguous())
        self.register_buffer("proj_b", proj_b.contiguous())
        # keep layer non-module tensors as buffers via flat registration
        for i, L in enumerate(layers):
            self.register_buffer(f"L{i}_ln1_w", L.ln1_w)
            self.register_buffer(f"L{i}_ln1_b", L.ln1_b)
            self.register_buffer(f"L{i}_ln2_w", L.ln2_w)
            self.register_buffer(f"L{i}_ln2_b", L.ln2_b)
            if L.subln_attn_w is not None:
                self.register_buffer(f"L{i}_subln_attn_w", L.subln_attn_w)
                self.register_buffer(f"L{i}_subln_attn_b", L.subln_attn_b)
            if L.subln_ffn_w is not None:
                self.register_buffer(f"L{i}_subln_ffn_w", L.subln_ffn_w)
                self.register_buffer(f"L{i}_subln_ffn_b", L.subln_ffn_b)

    @classmethod
    def from_bin(cls, path: str | Path, device: str = "cuda") -> "PackCudaEmbed":
        _ensure_pack_path()
        from unpack import UnpackedModel  # type: ignore

        path = Path(path)
        ref = UnpackedModel.from_bin(path)
        h = ref.header
        dev = torch.device(device)
        emb_packed = emb_scales = embedding = None
        if h.embedding_format == 3:  # EMB_INT4 — keep packed on GPU if available
            try:
                from unpack import (  # type: ignore
                    HEADER_SIZE,
                    SHA256_SIZE,
                    decode_embedding_int4_packed,
                )
            except ImportError:
                # fertileweb/soycaporal's training/pack/unpack.py only has the
                # eager decode_embedding_int4 (dequantizes to fp32 up front) —
                # the packed/lazy-dequant path is a first-fork-only VRAM
                # optimization. Fall back to the already-dequantized embedding
                # UnpackedModel.from_bin() built above (same values, more VRAM).
                embedding = ref.embedding.to(dev)
            else:
                body = path.read_bytes()[:-SHA256_SIZE]
                packed, scales, _ = decode_embedding_int4_packed(
                    body, HEADER_SIZE, h.vocab_size, h.d_model
                )
                emb_packed = packed.to(dev)
                emb_scales = scales.to(dev)
        else:
            embedding = ref.embedding.to(dev)

        layers: list[_CudaLayer] = []
        for li in range(h.n_layers):
            Lref = ref.layers[li]
            def bl(name: str, _li: int = li) -> BitLinearTritonFromBin:
                return BitLinearTritonFromBin.from_bin(
                    path, layer=_li, name=name, device=device
                )

            layers.append(
                _CudaLayer(
                    ln1_w=Lref.ln1_w.to(dev),
                    ln1_b=Lref.ln1_b.to(dev),
                    w_q=bl("w_q"),
                    w_k=bl("w_k"),
                    w_v=bl("w_v"),
                    # getattr: fertileweb/soycaporal's unpack.py::_LayerBlocks
                    # has no subln_* fields (v1 wire format, no sub-LN).
                    subln_attn_w=_opt_to(getattr(Lref, "subln_attn_w", None), dev),
                    subln_attn_b=_opt_to(getattr(Lref, "subln_attn_b", None), dev),
                    w_out=bl("w_out"),
                    ln2_w=Lref.ln2_w.to(dev),
                    ln2_b=Lref.ln2_b.to(dev),
                    fc1=bl("fc1"),
                    subln_ffn_w=_opt_to(getattr(Lref, "subln_ffn_w", None), dev),
                    subln_ffn_b=_opt_to(getattr(Lref, "subln_ffn_b", None), dev),
                    fc2=bl("fc2"),
                )
            )

        model = cls(
            d_model=h.d_model,
            n_heads=h.n_heads,
            output_dim=h.output_dim,
            # fertileweb/soycaporal's Header (v1) has no pos_encoding field —
            # StudentEncoder has no positional encoding by design.
            pos_rope=(getattr(h, "pos_encoding", 0) == 1),
            embedding=embedding,
            layers=layers,
            ln_final_w=ref.ln_final_w.to(dev),
            ln_final_b=ref.ln_final_b.to(dev),
            proj_W=ref.proj_W.to(dev),
            proj_b=ref.proj_b.to(dev),
            emb_packed=emb_packed,
            emb_scales=emb_scales,
        )
        del ref
        return model

    def packed_bitlinear_bytes(self) -> int:
        return sum(m.packed_bytes for m in self._mods)

    def embedding_bytes(self) -> int:
        if getattr(self, "emb_packed", None) is not None and self.emb_packed.numel() > 0:
            return int(self.emb_packed.nbytes + self.emb_scales.nbytes)
        return int(self.embedding.nbytes)

    def _lookup_embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        packed = getattr(self, "emb_packed", None)
        if packed is not None and packed.numel() > 0:
            from unpack import dequant_int4_rows  # type: ignore

            rows = packed[input_ids]
            scales = self.emb_scales[input_ids]
            return dequant_int4_rows(rows, scales, self.d_model)
        return self.embedding[input_ids]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """input_ids [B,T] long on CUDA → [B, output_dim] L2-normalized."""
        x = self._lookup_embedding(input_ids)  # [B,T,d]
        for i, L in enumerate(self.layers):
            x = self._transformer_layer(x, L, attention_mask)
        x = F.layer_norm(x, [self.d_model], weight=self.ln_final_w, bias=self.ln_final_b)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = x.mean(dim=1)
        projected = F.linear(pooled, self.proj_W, self.proj_b)
        return F.normalize(projected, dim=-1)

    def _transformer_layer(self, x, L: _CudaLayer, attention_mask):
        x_norm = F.layer_norm(x, [self.d_model], weight=L.ln1_w, bias=L.ln1_b)
        attn_out = self._attention(x_norm, L, attention_mask)
        x = x + attn_out
        x_norm = F.layer_norm(x, [self.d_model], weight=L.ln2_w, bias=L.ln2_b)
        ff_out = self._feedforward(x_norm, L)
        return x + ff_out

    def _attention(self, x, L: _CudaLayer, attention_mask):
        B, T, _ = x.shape
        Q = L.w_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = L.w_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = L.w_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        if self.pos_rope:
            cos, sin = _rope_cos_sin(T, self.d_head, Q.device, Q.dtype)
            Q, K = _apply_rope(Q, K, cos, sin)
        scale = self.d_head ** 0.5
        scores = (Q @ K.transpose(-2, -1)) / scale
        if attention_mask is not None:
            pad_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(pad_mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, self.d_model)
        if L.subln_attn_w is not None:
            out = F.layer_norm(out, [self.d_model], weight=L.subln_attn_w, bias=L.subln_attn_b)
        return L.w_out(out)

    def _feedforward(self, x, L: _CudaLayer):
        h = L.fc1(x)
        h = F.gelu(h)
        if L.subln_ffn_w is not None:
            h = F.layer_norm(h, [h.shape[-1]], weight=L.subln_ffn_w, bias=L.subln_ffn_b)
        return L.fc2(h)

    def embed_texts(
        self,
        texts: list[str],
        *,
        tokenizer_path: str | Path | None = None,
        tokenizer=None,
        max_len: int = 128,
    ) -> torch.Tensor:
        """Tokenize + forward. Returns [B, output_dim] on CUDA.

        Prefer ``FastSpmTokenizer`` (tokenizers JSON) over transformers.
        """
        if tokenizer is None:
            if tokenizer_path is None:
                raise ValueError("tokenizer or tokenizer_path required")
            if not hasattr(self, "_tokenizer") or self._tokenizer is None:
                from .spm_tok import load_fast_tokenizer

                self._tokenizer = load_fast_tokenizer(tokenizer_path)
            tokenizer = self._tokenizer
        if hasattr(tokenizer, "encode_batch_np"):
            import numpy as np

            ids_np, mask_np = tokenizer.encode_batch_np(texts, max_len=max_len)
            ids = torch.from_numpy(np.ascontiguousarray(ids_np)).to(self.embedding.device)
            mask = torch.from_numpy(np.ascontiguousarray(mask_np)).to(self.embedding.device)
            return self.forward(ids, mask)
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        )
        ids = enc["input_ids"].to(self.embedding.device)
        mask = enc["attention_mask"].to(self.embedding.device)
        return self.forward(ids, mask)
