"""Load BitLinear sections from a ternlight `.bin` without expanding to fp32.

Wire format matches `training/pack/` + `engine/src/model.rs::compute_layout`.

BitLinear storage (row-major out × in, 2-bit codes, 4 weights/byte along `in`):
  [out * (in//4) bytes] packed weights
  [4 bytes]             w_scale f32 LE
  [out * 4 bytes]       bias f32 LE  (optional)

FlashTernary / our Triton kernel expects packed shape (in//4, out) =
transpose of ternlight's (out, in//4). Encoding of each 2-bit nibble is the same.
"""
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path

import torch

# Keep format constants in sync with training/pack/format.py
HEADER_SIZE = 32
SHA256_SIZE = 32
MAGIC = b"TERN"
EMB_FP32, EMB_INT8, EMB_TERNARY, EMB_INT4 = 0, 1, 2, 3


@dataclass
class BinHeader:
    format_version: int
    embedding_format: int
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    ffn_dim: int
    output_dim: int
    max_seq_len: int
    pos_encoding: int
    use_subln: bool


@dataclass
class BitLinearSlice:
    name: str
    layer: int
    in_features: int
    out_features: int
    has_bias: bool
    weights_offset: int
    weights_bytes: int
    scale_offset: int
    bias_offset: int | None


def parse_header(data: bytes) -> BinHeader:
    if len(data) < HEADER_SIZE:
        raise ValueError("short header")
    magic, ver, emb, _w, vocab, d_model, n_layers, n_heads, ffn, out_dim, max_seq, pos, subln, _ = (
        struct.unpack_from("<4sHBBIHBBHHHBB8s", data, 0)
    )
    if magic != MAGIC:
        raise ValueError(f"bad magic {magic!r}")
    if ver < 2:
        pos = 0
    if ver < 3:
        subln = 0
    return BinHeader(
        format_version=ver,
        embedding_format=emb,
        vocab_size=vocab,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        ffn_dim=ffn,
        output_dim=out_dim,
        max_seq_len=max_seq,
        pos_encoding=pos,
        use_subln=bool(subln),
    )


def verify_sha256(data: bytes) -> bytes:
    if len(data) < HEADER_SIZE + SHA256_SIZE:
        raise ValueError("file too short")
    body, digest = data[:-SHA256_SIZE], data[-SHA256_SIZE:]
    got = hashlib.sha256(body).digest()
    if got != digest:
        raise ValueError("sha256 mismatch")
    return body


def _take_bitlinear(
    off: int,
    name: str,
    layer: int,
    in_f: int,
    out_f: int,
    has_bias: bool,
) -> tuple[int, BitLinearSlice]:
    w_bytes = out_f * (in_f // 4)
    w_off = off
    off += w_bytes
    scale_off = off
    off += 4
    bias_off = None
    if has_bias:
        bias_off = off
        off += out_f * 4
    return off, BitLinearSlice(
        name=name,
        layer=layer,
        in_features=in_f,
        out_features=out_f,
        has_bias=has_bias,
        weights_offset=w_off,
        weights_bytes=w_bytes,
        scale_offset=scale_off,
        bias_offset=bias_off,
    )


def list_bitlinears(header: BinHeader, body_len: int) -> list[BitLinearSlice]:
    """Walk wire format; return every BitLinear section in order."""
    d, ffn, n_layers = header.d_model, header.ffn_dim, header.n_layers
    vocab = header.vocab_size
    off = HEADER_SIZE

    # embedding
    if header.embedding_format == EMB_FP32:
        emb_w, emb_s = vocab * d * 4, 0
    elif header.embedding_format == EMB_INT8:
        emb_w, emb_s = vocab * d, vocab * 4
    elif header.embedding_format == EMB_TERNARY:
        emb_w, emb_s = vocab * (d // 4), vocab * 4
    elif header.embedding_format == EMB_INT4:
        emb_w, emb_s = vocab * (d // 2), vocab * 4
    else:
        raise ValueError(f"unknown emb fmt {header.embedding_format}")
    off += emb_w + emb_s

    ln_bytes = d * 4 * 2
    subln_attn = d * 4 * 2
    subln_ffn = ffn * 4 * 2
    out: list[BitLinearSlice] = []

    for li in range(n_layers):
        off += ln_bytes  # ln1
        for name, inf, outf, bias in (
            ("w_q", d, d, False),
            ("w_k", d, d, False),
            ("w_v", d, d, False),
        ):
            off, sl = _take_bitlinear(off, name, li, inf, outf, bias)
            out.append(sl)
        if header.use_subln:
            off += subln_attn
        off, sl = _take_bitlinear(off, "w_out", li, d, d, True)
        out.append(sl)
        off += ln_bytes  # ln2
        off, sl = _take_bitlinear(off, "fc1", li, d, ffn, True)
        out.append(sl)
        if header.use_subln:
            off += subln_ffn
        off, sl = _take_bitlinear(off, "fc2", li, ffn, d, True)
        out.append(sl)

    off += ln_bytes  # ln_final
    off += d * header.output_dim * 4 + header.output_dim * 4  # projection
    if off != body_len:
        raise ValueError(f"layout walk ended at {off}, body is {body_len}")
    return out


@dataclass
class LoadedBitLinear:
    name: str
    layer: int
    in_features: int
    out_features: int
    """ternlight wire pack: (out, in//4) uint8"""
    weight_packed_oi: torch.Tensor
    """FlashTernary / Triton pack: (in//4, out) uint8"""
    weight_packed_io: torch.Tensor
    w_scale: float
    bias: torch.Tensor | None  # fp32 [out] or None


def load_bin(path: str | Path) -> tuple[BinHeader, bytes, list[BitLinearSlice]]:
    data = Path(path).read_bytes()
    body = verify_sha256(data)
    header = parse_header(body)
    slices = list_bitlinears(header, len(body))
    return header, body, slices


def load_bitlinear(
    path: str | Path,
    *,
    layer: int = 0,
    name: str = "fc1",
    device: str | torch.device = "cpu",
) -> LoadedBitLinear:
    header, body, slices = load_bin(path)
    match = [s for s in slices if s.layer == layer and s.name == name]
    if not match:
        avail = sorted({(s.layer, s.name) for s in slices})
        raise KeyError(f"{name} layer={layer} not found; available={avail[:12]}...")
    sl = match[0]
    packed = torch.frombuffer(
        bytearray(body[sl.weights_offset : sl.weights_offset + sl.weights_bytes]),
        dtype=torch.uint8,
    ).clone()
    packed = packed.view(sl.out_features, sl.in_features // 4).contiguous()
    # Triton layout is transpose
    packed_io = packed.t().contiguous()
    w_scale = struct.unpack_from("<f", body, sl.scale_offset)[0]
    bias = None
    if sl.bias_offset is not None:
        bias = torch.frombuffer(
            bytearray(body[sl.bias_offset : sl.bias_offset + sl.out_features * 4]),
            dtype=torch.float32,
        ).clone()
    dev = torch.device(device)
    return LoadedBitLinear(
        name=f"L{layer}.{name}",
        layer=layer,
        in_features=sl.in_features,
        out_features=sl.out_features,
        weight_packed_oi=packed.to(dev),
        weight_packed_io=packed_io.to(dev),
        w_scale=float(w_scale),
        bias=None if bias is None else bias.to(dev),
    )


def unpack_oi(packed_oi: torch.Tensor, in_features: int) -> torch.Tensor:
    """Unpack ternlight (out, in//4) → (out, in) float32 ternary codes."""
    out_f, kp = packed_oi.shape
    assert kp == in_features // 4
    x = packed_oi.to(torch.int32)
    parts = [((x >> s) & 3) for s in (0, 2, 4, 6)]
    enc = torch.stack(parts, dim=2)  # (out, kp, 4)
    W = (enc == 1).to(torch.float32) - (enc == 2).to(torch.float32)
    return W.reshape(out_f, in_features)
