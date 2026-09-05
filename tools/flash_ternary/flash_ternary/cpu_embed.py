"""Full ternlight embed on CPU with BitLinear weights pack-maintained (FlashTernary).

Mirror of ``cuda_embed.PackCudaEmbed``: same graph as
``training/pack/unpack.UnpackedModel.forward``, but every BitLinear matmul goes
through ``flash_ternary.ternary_matmul`` (default backend: rust-simd when
available, else numpy). Weights stay uint8-packed in the hot path.

**CPU latency note:** the hot path keeps activations as contiguous float32 NumPy
end-to-end (LN / RoPE / attn / BitLinear / pool / proj). BitLinear no longer
does torch↔numpy↔rust↔torch per layer — that round-trip dominated the previous
~8 ms query path.

Use for RAG query encode when CUDA is busy/unavailable:

  RAG_EMBED_IMPL=pack_cpu
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .bin_loader import load_bitlinear
from .matmul import available_backends, ternary_matmul
from .quant import unpack_ternary

_BITLINEAR_EPS = 1e-5
_ACTIVATION_RANGE_MAX = 128.0
_LN_EPS = 1e-5

# Query encode has tiny M (B*T ≈ 8–32). Pre-unpacked f32 + OpenBLAS GEMM can be
# slightly faster than pack-maintained rust for *hybrid-only* embeds, but BLAS
# shares the OMP pool with FlashRank ONNX and inflates e2e retrieve+rerank
# (~2.5× on this host). RAG production default is therefore **rust**.
# Set RAG_FLASH_BACKEND=blas only for hybrid-only microbenches.
_BLAS_BACKENDS = frozenset({"blas", "unpacked", "f32"})
_PACK_BACKENDS = frozenset({"rust", "rust-simd", "rust-scalar", "numpy", "torch"})
# "auto"/"" follow the rust default (not blas) — see _default_cpu_backend.
_AUTO_BACKENDS = frozenset({"auto", ""})


def _default_cpu_backend() -> str:
    """Default ``rust`` (pack path). Override with RAG_FLASH_BACKEND=blas for GEMM."""
    env = (
        os.environ.get("RAG_FLASH_BACKEND")
        or os.environ.get("FLASH_TERNARY_BACKEND")
        or "rust"
    ).strip().lower()
    if env in _AUTO_BACKENDS:
        return "rust"
    if env in _BLAS_BACKENDS:
        return "blas"
    if env in _PACK_BACKENDS:
        return env
    # unknown → keep as-is (ternary_matmul may accept it)
    return env or "rust"


def _ensure_pack_path():
    repo = Path(os.environ.get("TERNLIGHT_REPO") or Path(__file__).resolve().parents[3])
    pack = repo / "training" / "pack"
    if str(pack) not in sys.path:
        sys.path.insert(0, str(pack))


def _as_f32_contig(a) -> np.ndarray:
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    return np.ascontiguousarray(a, dtype=np.float32)


def _layer_norm_np(
    x: np.ndarray,
    weight: np.ndarray | None = None,
    bias: np.ndarray | None = None,
    eps: float = _LN_EPS,
) -> np.ndarray:
    """Match torch.nn.functional.layer_norm (population var, last dim)."""
    mean = x.mean(axis=-1, keepdims=True, dtype=np.float32)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True, dtype=np.float32)
    y = (x - mean) * np.float32(1.0 / np.sqrt(var + eps))
    if weight is not None:
        y = y * weight
    if bias is not None:
        y = y + bias
    return y.astype(np.float32, copy=False)


def _bitlinear_pre_act_np(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Parameterless LN + AbsMax act-quant. x: (M, K) or (..., K)."""
    x_norm = _layer_norm_np(x, None, None, eps=_BITLINEAR_EPS)
    x_max = np.abs(x_norm).max(axis=-1, keepdims=True).clip(min=_BITLINEAR_EPS)
    x_scale = np.float32(_ACTIVATION_RANGE_MAX) / x_max.astype(np.float32)
    x_quant = np.clip(np.rint(x_norm * x_scale), -128, 127).astype(np.float32)
    return x_quant, x_scale.astype(np.float32)


def _erf_np(x: np.ndarray) -> np.ndarray:
    """Elementwise erf (NumPy has no np.erf on older builds; match torch gelu)."""
    try:
        from scipy.special import erf as _scipy_erf  # type: ignore

        return _scipy_erf(x).astype(np.float32, copy=False)
    except Exception:
        pass
    # Abramowitz & Stegun 7.1.26 (max abs error ~1.5e-7)
    ax = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * ax)
    poly = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592) * t
    y = 1.0 - poly * np.exp(-ax * ax)
    return np.where(x >= 0, y, -y).astype(np.float32, copy=False)


def _gelu_np(x: np.ndarray) -> np.ndarray:
    """torch.nn.functional.gelu (erf form) in float32."""
    # 0.5 * x * (1 + erf(x / sqrt(2)))
    return (0.5 * x * (1.0 + _erf_np(x * np.float32(0.7071067811865476)))).astype(
        np.float32, copy=False
    )


def _softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x, dtype=np.float32)
    return (e / e.sum(axis=axis, keepdims=True)).astype(np.float32, copy=False)


def _rotate_half_np(x: np.ndarray) -> np.ndarray:
    x1, x2 = np.split(x, 2, axis=-1)
    return np.concatenate((-x2, x1), axis=-1)


def _rope_cos_sin_np(T: int, d_head: int, base: float = 10_000.0) -> tuple[np.ndarray, np.ndarray]:
    inv_freq = 1.0 / (
        base ** (np.arange(0, d_head, 2, dtype=np.float32) / d_head)
    )
    t = np.arange(T, dtype=np.float32)
    freqs = np.outer(t, inv_freq)  # (T, d_head/2)
    emb = np.concatenate((freqs, freqs), axis=-1)  # (T, d_head)
    cos = np.cos(emb).astype(np.float32, copy=False)
    sin = np.sin(emb).astype(np.float32, copy=False)
    # shape for broadcast: (1, 1, T, d_head)
    return cos[None, None, :, :], sin[None, None, :, :]


def _apply_rope_np(q, k, cos, sin):
    return (q * cos) + (_rotate_half_np(q) * sin), (k * cos) + (_rotate_half_np(k) * sin)


class BitLinearCpuFromBin(nn.Module):
    """CPU: weights as uint8 pack (in//4, out); full BitLinear via FlashTernary matmul.

    Prefer ``forward_np`` on the hot path (no torch). ``forward`` is a thin torch wrapper.
    """

    def __init__(
        self,
        weight_packed_io: np.ndarray,
        w_scale: float,
        bias: torch.Tensor | None,
        in_features: int,
        out_features: int,
        name: str = "bitlinear",
        backend: str = "rust",
    ):
        super().__init__()
        self.name = name
        self.in_features = in_features
        self.out_features = out_features
        self.w_scale = float(w_scale)
        self.backend = backend
        self._has_bias = bias is not None
        self.weight_packed = np.ascontiguousarray(weight_packed_io, dtype=np.uint8)
        # Pre-unpack for BLAS query path: (K, N) in {-1,0,+1} float32.
        # ~16× packed size; still only a few tens of MiB for the full EN model.
        self.use_blas = backend in _BLAS_BACKENDS or backend == "blas"
        self.weight_f32: np.ndarray | None
        if self.use_blas:
            self.weight_f32 = np.ascontiguousarray(
                unpack_ternary(self.weight_packed, in_features), dtype=np.float32
            )
        else:
            self.weight_f32 = None
        self.bias_np: np.ndarray | None = (
            np.ascontiguousarray(bias.detach().cpu().numpy(), dtype=np.float32)
            if bias is not None
            else None
        )
        # keep a torch buffer for nn.Module state_dict / .to() compatibility
        self.register_buffer(
            "_bias",
            bias.float().contiguous() if bias is not None else torch.empty(0),
            persistent=True,
        )

    @classmethod
    def from_bin(
        cls,
        path: str | Path,
        *,
        layer: int = 0,
        name: str = "fc1",
        backend: str | None = None,
    ) -> "BitLinearCpuFromBin":
        bl = load_bitlinear(path, layer=layer, name=name, device="cpu")
        return cls(
            bl.weight_packed_io.cpu().numpy(),
            bl.w_scale,
            bl.bias,
            bl.in_features,
            bl.out_features,
            name=bl.name,
            backend=backend or _default_cpu_backend(),
        )

    @property
    def packed_bytes(self) -> int:
        return int(self.weight_packed.nbytes)

    def forward_np(self, x: np.ndarray) -> np.ndarray:
        """x: (..., in) float32 → (..., out) float32. No torch."""
        orig = x.shape
        x_flat = np.ascontiguousarray(x.reshape(-1, self.in_features), dtype=np.float32)
        x_quant, x_scale = _bitlinear_pre_act_np(x_flat)
        if self.use_blas and self.weight_f32 is not None:
            # (M,K) @ (K,N) — OpenBLAS GEMM; best for RAG-sized M
            y_raw = x_quant @ self.weight_f32
        else:
            # C = A @ dequant(B_pack); B is (in//4, out) — pack-maintained path
            y_raw = ternary_matmul(
                x_quant,
                self.weight_packed,
                backend=self.backend if self.backend in _PACK_BACKENDS else "rust",
                K=self.in_features,
            )
        if self.bias_np is not None:
            y_raw = y_raw + self.bias_np
        y = y_raw / (np.float32(self.w_scale) * x_scale)
        if len(orig) > 2:
            y = y.reshape(*orig[:-1], self.out_features)
        return y.astype(np.float32, copy=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.forward_np(_as_f32_contig(x))
        return torch.from_numpy(np.ascontiguousarray(y))

    def extra_repr(self) -> str:
        mode = "blas-unpacked" if self.use_blas else f"pack:{self.backend}"
        return (
            f"{self.name} in={self.in_features} out={self.out_features} "
            f"scale={self.w_scale:.4g} packed={self.packed_bytes}B "
            f"mode={mode} bias={self._has_bias}"
        )


@dataclass
class _CpuLayer:
    ln1_w: np.ndarray
    ln1_b: np.ndarray
    w_q: BitLinearCpuFromBin
    w_k: BitLinearCpuFromBin
    w_v: BitLinearCpuFromBin
    subln_attn_w: np.ndarray | None
    subln_attn_b: np.ndarray | None
    w_out: BitLinearCpuFromBin
    ln2_w: np.ndarray
    ln2_b: np.ndarray
    fc1: BitLinearCpuFromBin
    subln_ffn_w: np.ndarray | None
    subln_ffn_b: np.ndarray | None
    fc2: BitLinearCpuFromBin


class PackCpuEmbed(nn.Module):
    """CPU pack-maintained embedder from a ternlight `.bin` (FlashTernary BitLinear).

    Activations stay in NumPy for the whole forward; only the final return is a
    torch tensor (for API parity with PackCudaEmbed).
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        output_dim: int,
        pos_rope: bool,
        embedding: torch.Tensor,
        layers: list[_CpuLayer],
        ln_final_w: torch.Tensor,
        ln_final_b: torch.Tensor,
        proj_W: torch.Tensor,
        proj_b: torch.Tensor,
        backend: str,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.output_dim = output_dim
        self.pos_rope = pos_rope
        self.backend = backend
        self.layers = layers
        self._mods = nn.ModuleList()
        for L in layers:
            self._mods.extend([L.w_q, L.w_k, L.w_v, L.w_out, L.fc1, L.fc2])

        # torch buffers (checkpoint-ish) + numpy views for hot path
        self.register_buffer("embedding", embedding.contiguous())
        self.register_buffer("ln_final_w", ln_final_w.contiguous())
        self.register_buffer("ln_final_b", ln_final_b.contiguous())
        self.register_buffer("proj_W", proj_W.contiguous())
        self.register_buffer("proj_b", proj_b.contiguous())
        for i, L in enumerate(layers):
            self.register_buffer(f"L{i}_ln1_w", torch.from_numpy(L.ln1_w.copy()))
            self.register_buffer(f"L{i}_ln1_b", torch.from_numpy(L.ln1_b.copy()))
            self.register_buffer(f"L{i}_ln2_w", torch.from_numpy(L.ln2_w.copy()))
            self.register_buffer(f"L{i}_ln2_b", torch.from_numpy(L.ln2_b.copy()))
            if L.subln_attn_w is not None:
                self.register_buffer(
                    f"L{i}_subln_attn_w", torch.from_numpy(L.subln_attn_w.copy())
                )
                self.register_buffer(
                    f"L{i}_subln_attn_b", torch.from_numpy(L.subln_attn_b.copy())
                )
            if L.subln_ffn_w is not None:
                self.register_buffer(
                    f"L{i}_subln_ffn_w", torch.from_numpy(L.subln_ffn_w.copy())
                )
                self.register_buffer(
                    f"L{i}_subln_ffn_b", torch.from_numpy(L.subln_ffn_b.copy())
                )

        self._emb_np = _as_f32_contig(embedding)
        self._ln_final_w = _as_f32_contig(ln_final_w)
        self._ln_final_b = _as_f32_contig(ln_final_b)
        self._proj_W = _as_f32_contig(proj_W)
        self._proj_b = _as_f32_contig(proj_b)
        # RoPE cache keyed by T
        self._rope_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    @classmethod
    def from_bin(
        cls,
        path: str | Path,
        *,
        backend: str | None = None,
    ) -> "PackCpuEmbed":
        _ensure_pack_path()
        from unpack import UnpackedModel  # type: ignore

        path = Path(path)
        be = backend or _default_cpu_backend()
        ref = UnpackedModel.from_bin(path)
        h = ref.header

        def _np(t: torch.Tensor) -> np.ndarray:
            return _as_f32_contig(t)

        layers: list[_CpuLayer] = []
        for li in range(h.n_layers):
            Lref = ref.layers[li]

            def bl(name: str, _li: int = li) -> BitLinearCpuFromBin:
                return BitLinearCpuFromBin.from_bin(
                    path, layer=_li, name=name, backend=be
                )

            layers.append(
                _CpuLayer(
                    ln1_w=_np(Lref.ln1_w),
                    ln1_b=_np(Lref.ln1_b),
                    w_q=bl("w_q"),
                    w_k=bl("w_k"),
                    w_v=bl("w_v"),
                    # getattr, not attribute access: fertileweb/soycaporal's
                    # training/pack/unpack.py::_LayerBlocks has no subln_* fields
                    # at all (v1 wire format, no sub-LN) — only the first-fork's
                    # v2/v3 _LayerBlocks does. None here means "not present".
                    subln_attn_w=None if getattr(Lref, "subln_attn_w", None) is None else _np(Lref.subln_attn_w),
                    subln_attn_b=None if getattr(Lref, "subln_attn_b", None) is None else _np(Lref.subln_attn_b),
                    w_out=bl("w_out"),
                    ln2_w=_np(Lref.ln2_w),
                    ln2_b=_np(Lref.ln2_b),
                    fc1=bl("fc1"),
                    subln_ffn_w=None if getattr(Lref, "subln_ffn_w", None) is None else _np(Lref.subln_ffn_w),
                    subln_ffn_b=None if getattr(Lref, "subln_ffn_b", None) is None else _np(Lref.subln_ffn_b),
                    fc2=bl("fc2"),
                )
            )

        return cls(
            d_model=h.d_model,
            n_heads=h.n_heads,
            output_dim=h.output_dim,
            # fertileweb/soycaporal's Header (v1) has no pos_encoding field —
            # StudentEncoder has no positional encoding by design (see
            # training/distill/model.py docstring); default to no RoPE.
            pos_rope=(getattr(h, "pos_encoding", 0) == 1),
            embedding=ref.embedding.detach().cpu().float().contiguous(),
            layers=layers,
            ln_final_w=ref.ln_final_w.detach().cpu().float().contiguous(),
            ln_final_b=ref.ln_final_b.detach().cpu().float().contiguous(),
            proj_W=ref.proj_W.detach().cpu().float().contiguous(),
            proj_b=ref.proj_b.detach().cpu().float().contiguous(),
            backend=be,
        )

    def packed_bitlinear_bytes(self) -> int:
        return sum(m.packed_bytes for m in self._mods)

    def _rope(self, T: int) -> tuple[np.ndarray, np.ndarray]:
        hit = self._rope_cache.get(T)
        if hit is None:
            hit = _rope_cos_sin_np(T, self.d_head)
            self._rope_cache[T] = hit
        return hit

    def forward_np(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """input_ids [B,T] int64 → [B, output_dim] L2-normalized float32."""
        ids = np.asarray(input_ids)
        x = self._emb_np[ids]  # (B, T, d)
        for L in self.layers:
            x = self._transformer_layer_np(x, L, attention_mask)
        x = _layer_norm_np(x, self._ln_final_w, self._ln_final_b)
        if attention_mask is not None:
            mask = np.asarray(attention_mask, dtype=np.float32)[..., None]
            pooled = (x * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
        else:
            pooled = x.mean(axis=1)
        # F.linear: y = x @ W.T + b  with W shape (out, in)
        projected = pooled @ self._proj_W.T + self._proj_b
        norms = np.linalg.norm(projected, axis=-1, keepdims=True)
        return (projected / np.clip(norms, 1e-9, None)).astype(np.float32, copy=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """input_ids [B,T] long on CPU → [B, output_dim] L2-normalized torch tensor."""
        ids = input_ids.detach().cpu().numpy()
        mask = None if attention_mask is None else attention_mask.detach().cpu().numpy()
        y = self.forward_np(ids, mask)
        return torch.from_numpy(np.ascontiguousarray(y))

    def _transformer_layer_np(
        self, x: np.ndarray, L: _CpuLayer, attention_mask: np.ndarray | None
    ) -> np.ndarray:
        x_norm = _layer_norm_np(x, L.ln1_w, L.ln1_b)
        attn_out = self._attention_np(x_norm, L, attention_mask)
        x = x + attn_out
        x_norm = _layer_norm_np(x, L.ln2_w, L.ln2_b)
        ff_out = self._feedforward_np(x_norm, L)
        return x + ff_out

    def _attention_np(
        self, x: np.ndarray, L: _CpuLayer, attention_mask: np.ndarray | None
    ) -> np.ndarray:
        B, T, _ = x.shape
        Q = L.w_q.forward_np(x).reshape(B, T, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        K = L.w_k.forward_np(x).reshape(B, T, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        V = L.w_v.forward_np(x).reshape(B, T, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        # (B, H, T, Dh)
        if self.pos_rope:
            cos, sin = self._rope(T)
            Q, K = _apply_rope_np(Q, K, cos, sin)
        scale = np.float32(self.d_head ** 0.5)
        # scores: (B, H, T, T)
        scores = (Q @ np.swapaxes(K, -1, -2)) / scale
        if attention_mask is not None:
            # mask==0 → -inf
            pad = (np.asarray(attention_mask) == 0)[:, None, None, :]  # (B,1,1,T)
            scores = np.where(pad, np.float32(-1e9), scores)
        attn = _softmax_np(scores, axis=-1)
        out = (attn @ V).transpose(0, 2, 1, 3).reshape(B, T, self.d_model)
        if L.subln_attn_w is not None:
            out = _layer_norm_np(out, L.subln_attn_w, L.subln_attn_b)
        return L.w_out.forward_np(out)

    def _feedforward_np(self, x: np.ndarray, L: _CpuLayer) -> np.ndarray:
        h = L.fc1.forward_np(x)
        h = _gelu_np(h)
        if L.subln_ffn_w is not None:
            h = _layer_norm_np(h, L.subln_ffn_w, L.subln_ffn_b)
        return L.fc2.forward_np(h)

    def embed_texts(
        self,
        texts: list[str],
        *,
        tokenizer_path: str | Path | None = None,
        tokenizer=None,
        max_len: int = 128,
    ) -> torch.Tensor:
        """Tokenize + forward. Returns [B, output_dim] on CPU.

        Prefer ``FastSpmTokenizer`` (tokenizers JSON, no transformers). NumPy
        encode → ``forward_np`` avoids a torch detour on the pack_cpu path.
        """
        if tokenizer is None:
            if tokenizer_path is None:
                raise ValueError("tokenizer or tokenizer_path required")
            if not hasattr(self, "_tokenizer") or self._tokenizer is None:
                from .spm_tok import load_fast_tokenizer

                self._tokenizer = load_fast_tokenizer(tokenizer_path)
            tokenizer = self._tokenizer
        # Fast path: numpy ids without torch
        if hasattr(tokenizer, "encode_batch_np"):
            ids, mask = tokenizer.encode_batch_np(texts, max_len=max_len)
            y = self.forward_np(ids, mask)
            return torch.from_numpy(np.ascontiguousarray(y))
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        )
        return self.forward(enc["input_ids"], enc["attention_mask"])
