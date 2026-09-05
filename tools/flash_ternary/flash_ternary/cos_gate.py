"""Cosine-similarity gate for pack embed vs reference.

Default reference is ``UnpackedModel.from_bin`` (same `.bin` wire format as the
engine). Optional second arm compares against ``load_for_eval`` pytorch BitLinear
when ``TERNLIGHT_CKPT`` (or ``--ckpt``) is set.

Exit codes:
  0  all required arms pass (or skipped with --allow-skip and nothing to run)
  1  cos below threshold / shape mismatch
  2  missing inputs (bin/tok) and skip not allowed

Env (overridable by CLI):
  TERNLIGHT_EN_BIN / --bin
  TERNLIGHT_TOKENIZER / --tok   (default engine/assets)
  TERNLIGHT_CKPT / --ckpt       optional pytorch arm
  RAG_FLASH_BACKEND             blas|rust|… for pack_cpu arm
  FLASH_COS_MIN                 default 0.999
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(os.environ.get("TERNLIGHT_REPO") or Path(__file__).resolve().parents[3])

_DEFAULT_BIN = Path(
    os.environ.get(
        "TERNLIGHT_EN_BIN",
        str(_REPO_ROOT / "training" / "pack" / "out" / "en-minilm-int4.bin"),
    )
)
_DEFAULT_TOK = Path(
    os.environ.get(
        "TERNLIGHT_TOKENIZER",
        str(_REPO_ROOT / "engine" / "assets"),
    )
)
_DEFAULT_CKPT = Path(
    os.environ.get(
        "TERNLIGHT_CKPT",
        str(
            _REPO_ROOT
            / "training"
            / "distill"
            / "runs"
            / "en-qat-L3w512-rope-minilm-msmarcomix-3e1bb6d"
            / "checkpoint_ep12.pt"
        ),
    )
)

_DEFAULT_QUERIES = (
    "BitLinear ternary 2-bit packed weights RoPE FlashTernary",
    "engine int4 pack format SubLN",
    "lightweight memory reranker hybrid RRF",
    "a",  # short edge case
)


@dataclass
class ArmResult:
    name: str
    cos: float
    max_abs: float
    ok: bool
    detail: str = ""


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(a @ b / (na * nb))


def _ensure_pack_path() -> Path:
    pack = _REPO_ROOT / "training" / "pack"
    if str(pack) not in sys.path:
        sys.path.insert(0, str(pack))
    return pack


def _embed_unpacked(bin_path: Path, ids: np.ndarray, mask: np.ndarray) -> np.ndarray:
    import torch
    from unpack import UnpackedModel  # type: ignore

    ref = UnpackedModel.from_bin(bin_path)
    with torch.inference_mode():
        y = ref.forward(
            torch.from_numpy(ids).long(),
            torch.from_numpy(mask).long(),
        )
    return y.float().cpu().numpy()


def _embed_pack_cpu(
    bin_path: Path,
    ids: np.ndarray,
    mask: np.ndarray,
    *,
    backend: str,
) -> np.ndarray:
    from .cpu_embed import PackCpuEmbed

    model = PackCpuEmbed.from_bin(bin_path, backend=backend)
    return model.forward_np(ids, mask)


def _embed_pack_cuda(bin_path: Path, ids: np.ndarray, mask: np.ndarray) -> np.ndarray:
    import torch
    from .cuda_embed import PackCudaEmbed

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    model = PackCudaEmbed.from_bin(bin_path, device="cuda")
    with torch.inference_mode():
        y = model.forward(
            torch.from_numpy(ids).long().cuda(),
            torch.from_numpy(mask).long().cuda(),
        )
    return y.float().cpu().numpy()


def _embed_pytorch_ckpt(
    ckpt: Path,
    tok_path: Path,
    texts: list[str],
    *,
    max_len: int,
) -> np.ndarray:
    """Reference via load_for_eval (training BitLinear path)."""
    import torch
    from transformers import AutoTokenizer

    distill = _REPO_ROOT / "training" / "distill"
    if str(distill) not in sys.path:
        sys.path.insert(0, str(distill))
    from evaluation import load_for_eval  # type: ignore

    tok = AutoTokenizer.from_pretrained(str(tok_path))
    em = load_for_eval(ckpt, "cpu", embedding_format=os.environ.get("TERNLIGHT_EMB_FORMAT", "int4"))
    with torch.inference_mode():
        enc = tok(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        )
        y = em.model(enc["input_ids"], enc["attention_mask"])
        y = y.float().cpu().numpy()
        y = y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-9)
    return y.astype(np.float32)


def run_gate(
    *,
    bin_path: Path,
    tok_path: Path,
    queries: list[str],
    cos_min: float = 0.999,
    backends: list[str] | None = None,
    ckpt: Path | None = None,
    max_len: int = 128,
    include_cuda: bool = True,
) -> list[ArmResult]:
    """Run all available arms; returns one ArmResult per (query-averaged) arm."""
    from .spm_tok import FastSpmTokenizer

    if not bin_path.is_file():
        raise FileNotFoundError(f"bin missing: {bin_path}")
    if not (tok_path / "tokenizer.json").is_file() and not tok_path.is_file():
        raise FileNotFoundError(f"tokenizer missing: {tok_path}")

    _ensure_pack_path()
    tok = FastSpmTokenizer.from_pretrained(tok_path)
    ids, mask = tok.encode_batch_np(queries, max_len=max_len)

    ref = _embed_unpacked(bin_path, ids, mask)
    results: list[ArmResult] = []

    backends = backends or ["blas", "rust"]
    for be in backends:
        try:
            hyp = _embed_pack_cpu(bin_path, ids, mask, backend=be)
        except Exception as e:
            results.append(
                ArmResult(f"pack_cpu/{be}", 0.0, 1.0, False, detail=f"error: {e}")
            )
            continue
        # per-row cos, report min across queries (gate on worst)
        cos_rows = [_cos(ref[i], hyp[i]) for i in range(len(queries))]
        cos = float(min(cos_rows))
        max_abs = float(np.max(np.abs(ref - hyp)))
        results.append(
            ArmResult(
                f"pack_cpu/{be}",
                cos,
                max_abs,
                cos >= cos_min,
                detail=f"min_cos_row={cos:.6f} mean={float(np.mean(cos_rows)):.6f}",
            )
        )

    if include_cuda:
        try:
            import torch

            if torch.cuda.is_available():
                hyp = _embed_pack_cuda(bin_path, ids, mask)
                cos_rows = [_cos(ref[i], hyp[i]) for i in range(len(queries))]
                cos = float(min(cos_rows))
                max_abs = float(np.max(np.abs(ref - hyp)))
                results.append(
                    ArmResult(
                        "pack_cuda",
                        cos,
                        max_abs,
                        cos >= cos_min,
                        detail=f"min_cos_row={cos:.6f} mean={float(np.mean(cos_rows)):.6f}",
                    )
                )
            else:
                results.append(
                    ArmResult("pack_cuda", 0.0, 0.0, True, detail="skipped (no CUDA)")
                )
        except Exception as e:
            results.append(
                ArmResult("pack_cuda", 0.0, 1.0, False, detail=f"error: {e}")
            )

    if ckpt is not None and Path(ckpt).is_file():
        try:
            # pytorch arm uses its own tokenize — should match FastSpm for these dumps
            hyp = _embed_pytorch_ckpt(Path(ckpt), tok_path, queries, max_len=max_len)
            # ref is UnpackedModel; also fine to compare pack_cpu/blas vs pytorch
            cos_rows = [_cos(ref[i], hyp[i]) for i in range(len(queries))]
            cos = float(min(cos_rows))
            max_abs = float(np.max(np.abs(ref - hyp)))
            results.append(
                ArmResult(
                    "pytorch_ckpt",
                    cos,
                    max_abs,
                    cos >= cos_min,
                    detail=f"vs UnpackedModel min_cos={cos:.6f}",
                )
            )
            # also pack_cpu/blas vs pytorch if we have it
            try:
                pack = _embed_pack_cpu(bin_path, ids, mask, backend="blas")
                cos_rows = [_cos(pack[i], hyp[i]) for i in range(len(queries))]
                cos = float(min(cos_rows))
                max_abs = float(np.max(np.abs(pack - hyp)))
                results.append(
                    ArmResult(
                        "pack_cpu/blas_vs_pytorch",
                        cos,
                        max_abs,
                        cos >= cos_min,
                        detail=f"min_cos={cos:.6f}",
                    )
                )
            except Exception as e:
                results.append(
                    ArmResult(
                        "pack_cpu/blas_vs_pytorch",
                        0.0,
                        1.0,
                        False,
                        detail=f"error: {e}",
                    )
                )
        except Exception as e:
            results.append(
                ArmResult("pytorch_ckpt", 0.0, 1.0, False, detail=f"error: {e}")
            )

    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="FlashTernary pack embed cos gate")
    ap.add_argument("--bin", type=Path, default=_DEFAULT_BIN)
    ap.add_argument("--tok", type=Path, default=_DEFAULT_TOK)
    ap.add_argument("--ckpt", type=Path, default=None, help="optional pytorch .pt arm")
    ap.add_argument(
        "--with-ckpt",
        action="store_true",
        help=f"also compare vs default ckpt if present ({_DEFAULT_CKPT.name})",
    )
    ap.add_argument("--cos-min", type=float, default=float(os.environ.get("FLASH_COS_MIN", "0.999")))
    ap.add_argument(
        "--backend",
        action="append",
        dest="backends",
        help="pack_cpu backend (repeatable). Default: blas + rust",
    )
    ap.add_argument("--no-cuda", action="store_true")
    ap.add_argument(
        "--allow-skip",
        action="store_true",
        help="exit 0 if bin/tok missing (CI without artifacts)",
    )
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("-q", "--query", action="append", dest="queries")
    args = ap.parse_args(argv)

    ckpt = args.ckpt
    if args.with_ckpt and ckpt is None:
        ckpt = _DEFAULT_CKPT if _DEFAULT_CKPT.is_file() else None

    if not args.bin.is_file() or not (
        (args.tok / "tokenizer.json").is_file() or args.tok.is_file()
    ):
        msg = f"missing bin={args.bin} or tok={args.tok}"
        if args.allow_skip:
            print(f"SKIP cos gate: {msg}")
            return 0
        print(f"FAIL cos gate: {msg}", file=sys.stderr)
        return 2

    queries = args.queries or list(_DEFAULT_QUERIES)
    try:
        results = run_gate(
            bin_path=args.bin,
            tok_path=args.tok,
            queries=queries,
            cos_min=args.cos_min,
            backends=args.backends,
            ckpt=ckpt,
            max_len=args.max_len,
            include_cuda=not args.no_cuda,
        )
    except Exception as e:
        print(f"FAIL cos gate: {e}", file=sys.stderr)
        return 1

    print(f"cos_min={args.cos_min}  bin={args.bin.name}  n_queries={len(queries)}")
    all_ok = True
    for r in results:
        # skipped arms that mark ok=True with cos=0 and detail starting with skipped
        if r.detail.startswith("skipped"):
            print(f"  SKIP {r.name:28}  {r.detail}")
            continue
        mark = "OK  " if r.ok else "FAIL"
        if not r.ok:
            all_ok = False
        print(
            f"  {mark} {r.name:28}  cos={r.cos:.6f}  max_abs={r.max_abs:.3e}  {r.detail}"
        )

    if all_ok:
        print("PASS pack embed cos gate")
        return 0
    print("FAIL pack embed cos gate", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
