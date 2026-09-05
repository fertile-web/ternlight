#!/usr/bin/env python3
"""Smoke: library BitLinear vs flash_ternary QATBitLinear on a tiny student.

Protocol (CPU-friendly, no real corpus / teacher):
  1. Build identical StudentEncoder clones (micro dims).
  2. Swap with library BitLinear | QATBitLinear(dense) | QATBitLinear(pack).
  3. Copy the same shadow weights into all three.
  4. Forward parity at lambda=1 (cos / max_abs on embeddings).
  5. N AdamW steps with synthetic distill loss (cosine to fixed targets).
  6. Report loss curves, weight drift vs library, zero-fractions.

Exit 0 if gates pass; non-zero otherwise.

  cd tools/flash_ternary   # from the repo root
  python smoke_qat_train.py   # use the repo's training venv if it has one
  # optional: --steps 40 --device cuda
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
_REPO_ROOT = Path(os.environ.get("TERNLIGHT_REPO") or Path(__file__).resolve().parents[2])
DISTILL = _REPO_ROOT / "training" / "distill"
for p in (str(ROOT), str(DISTILL)):
    if p not in sys.path:
        sys.path.insert(0, p)

from flash_ternary.qat import QATBitLinear  # noqa: E402

try:
    import ternary_qat  # noqa: E402
    from model import StudentEncoder  # noqa: E402
except ImportError:
    # Needs training/distill/{ternary_qat.py,model.py} from the full ternlight
    # repo — not part of this flash_ternary-only checkout. Importing this
    # module (e.g. for test collection) should still succeed; only running
    # it should fail, with a clear message (see main()).
    ternary_qat = None
    StudentEncoder = None


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()


def _clone_student(seed: int = 0) -> StudentEncoder:
    torch.manual_seed(seed)
    return StudentEncoder(
        vocab_size=512,
        d_model=64,
        n_layers=2,
        n_heads=4,
        ffn_dim=128,
        output_dim=32,
        dropout=0.0,
    )


def _copy_shadows(src: torch.nn.Module, dst: torch.nn.Module) -> None:
    """Copy Parameter tensors by name (shadow weights + projection + emb)."""
    src_sd = src.state_dict()
    dst_sd = dst.state_dict()
    missing = [k for k in dst_sd if k not in src_sd]
    if missing:
        raise RuntimeError(f"state_dict key mismatch missing in src: {missing[:5]}")
    with torch.no_grad():
        for k, v in dst_sd.items():
            if k in src_sd and src_sd[k].shape == v.shape:
                v.copy_(src_sd[k])


def _n_qat_layers(model: torch.nn.Module) -> dict[str, int]:
    from bitlinear import BitLinear

    return {
        "BitLinear": sum(1 for m in model.modules() if isinstance(m, BitLinear)),
        "QATBitLinear": sum(1 for m in model.modules() if isinstance(m, QATBitLinear)),
        "Linear": sum(
            1
            for m in model.modules()
            if isinstance(m, torch.nn.Linear)
            and not isinstance(m, (BitLinear, QATBitLinear))
        ),
    }


def _train_steps(
    model: torch.nn.Module,
    *,
    steps: int,
    batch: int,
    seq: int,
    vocab: int,
    device: str,
    lr: float,
    seed: int,
) -> dict:
    torch.manual_seed(seed)
    model = model.to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    # fixed synthetic teacher targets (reused each step for stable curve)
    with torch.no_grad():
        ids0 = torch.randint(1, vocab, (batch, seq), device=device)
        mask0 = torch.ones(batch, seq, device=device)
        # targets from a frozen random projector of bag-of-ids (not real teacher)
        targets = F.normalize(
            torch.randn(batch, model.projection.out_features, device=device), dim=-1
        )

    losses: list[float] = []
    t0 = time.perf_counter()
    for step in range(steps):
        # new ids each step, same targets shape — pure optim smoke
        ids = torch.randint(1, vocab, (batch, seq), device=device)
        mask = torch.ones(batch, seq, device=device)
        # rotate targets slightly so loss is not degenerate
        if step > 0:
            targets = F.normalize(
                targets + 0.01 * torch.randn_like(targets), dim=-1
            )
        opt.zero_grad()
        emb = model(ids, mask)
        loss = (1.0 - F.cosine_similarity(emb, targets, dim=-1)).mean()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    dt = time.perf_counter() - t0
    return {
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_min": min(losses),
        "loss_curve": losses,
        "seconds": dt,
        "steps_per_s": steps / max(dt, 1e-9),
        # keep last batch for optional parity (not used)
        "_ids0": ids0,
        "_mask0": mask0,
    }


@torch.no_grad()
def _forward_batch(
    model: torch.nn.Module, ids: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    model.eval()
    return model(ids, mask)


def main() -> int:
    if ternary_qat is None or StudentEncoder is None:
        raise ImportError(
            "smoke_qat_train.py needs training/distill/{ternary_qat.py,model.py} "
            f"from the full ternlight repo (not found under {DISTILL}); "
            "set TERNLIGHT_REPO to point at a checkout that has them."
        )
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / ".smoke-summary.json",
    )
    ap.add_argument("--cos-gate", type=float, default=0.999)
    ap.add_argument("--max-abs-gate", type=float, default=5e-3)
    ap.add_argument(
        "--loss-rel-gate",
        type=float,
        default=0.15,
        help="|loss_last(flash)-loss_last(lib)| / max(lib,eps) must be < this",
    )
    args = ap.parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to cpu")
        device = "cpu"

    # ── build three arms from the same fp32 init ──────────────────────────
    base = _clone_student(args.seed)
    lib = copy.deepcopy(base)
    dens = copy.deepcopy(base)
    pack = copy.deepcopy(base)

    n_lib = ternary_qat.swap(lib, backend="bitlinear")
    n_d = ternary_qat.swap(dens, backend="flash_qat", matmul="dense")
    n_p = ternary_qat.swap(pack, backend="flash_qat", matmul="pack")
    ternary_qat.set_lambda(lib, 1.0)
    ternary_qat.set_lambda(dens, 1.0)
    ternary_qat.set_lambda(pack, 1.0)

    # re-seed then re-copy: swap preserves weights, but ensure identical
    _copy_shadows(lib, dens)
    _copy_shadows(lib, pack)

    counts = {
        "lib": _n_qat_layers(lib),
        "dense": _n_qat_layers(dens),
        "pack": _n_qat_layers(pack),
        "swapped": {"lib": n_lib, "dense": n_d, "pack": n_p},
    }

    # ── forward parity on shared batch ────────────────────────────────────
    lib = lib.to(device)
    dens = dens.to(device)
    pack = pack.to(device)
    torch.manual_seed(args.seed + 1)
    ids = torch.randint(1, 512, (args.batch, args.seq), device=device)
    mask = torch.ones(args.batch, args.seq, device=device)
    y_lib = _forward_batch(lib, ids, mask).cpu()
    y_d = _forward_batch(dens, ids, mask).cpu()
    y_p = _forward_batch(pack, ids, mask).cpu()

    fwd = {
        "cos_dense_vs_lib": _cos(y_d, y_lib),
        "cos_pack_vs_lib": _cos(y_p, y_lib),
        "cos_pack_vs_dense": _cos(y_p, y_d),
        "max_abs_dense_vs_lib": (y_d - y_lib).abs().max().item(),
        "max_abs_pack_vs_lib": (y_p - y_lib).abs().max().item(),
        "max_abs_pack_vs_dense": (y_p - y_d).abs().max().item(),
    }

    # ── train smoke ───────────────────────────────────────────────────────
    # re-sync shadows before train so each arm starts equal
    lib_t = copy.deepcopy(lib)
    dens_t = copy.deepcopy(lib)
    pack_t = copy.deepcopy(lib)
    # dens/pack are QATBitLinear models — copy from lib BitLinear into them carefully
    dens_t = copy.deepcopy(dens)
    pack_t = copy.deepcopy(pack)
    _copy_shadows(lib, dens_t)
    _copy_shadows(lib, pack_t)
    lib_t = copy.deepcopy(lib)

    r_lib = _train_steps(
        lib_t,
        steps=args.steps,
        batch=args.batch,
        seq=args.seq,
        vocab=512,
        device=device,
        lr=args.lr,
        seed=args.seed + 7,
    )
    r_d = _train_steps(
        dens_t,
        steps=args.steps,
        batch=args.batch,
        seq=args.seq,
        vocab=512,
        device=device,
        lr=args.lr,
        seed=args.seed + 7,
    )
    r_p = _train_steps(
        pack_t,
        steps=args.steps,
        batch=args.batch,
        seq=args.seq,
        vocab=512,
        device=device,
        lr=args.lr,
        seed=args.seed + 7,
    )

    # strip non-json tensors
    for r in (r_lib, r_d, r_p):
        r.pop("_ids0", None)
        r.pop("_mask0", None)

    # weight drift: mean abs param delta vs lib after train
    def _param_l1(a: torch.nn.Module, b: torch.nn.Module) -> float:
        tot = 0.0
        n = 0
        sa, sb = a.state_dict(), b.state_dict()
        for k, va in sa.items():
            if k not in sb or va.shape != sb[k].shape:
                continue
            if not va.is_floating_point():
                continue
            tot += (va.cpu() - sb[k].cpu()).abs().mean().item()
            n += 1
        return tot / max(n, 1)

    train = {
        "lib": {k: v for k, v in r_lib.items() if k != "loss_curve"},
        "dense": {k: v for k, v in r_d.items() if k != "loss_curve"},
        "pack": {k: v for k, v in r_p.items() if k != "loss_curve"},
        "loss_curve_lib": r_lib["loss_curve"],
        "loss_curve_dense": r_d["loss_curve"],
        "loss_curve_pack": r_p["loss_curve"],
        "param_l1_dense_vs_lib": _param_l1(dens_t, lib_t),
        "param_l1_pack_vs_lib": _param_l1(pack_t, lib_t),
        "param_l1_pack_vs_dense": _param_l1(pack_t, dens_t),
        "zero_frac_mean_lib": (
            sum(ternary_qat.zero_fractions(lib_t).values())
            / max(len(ternary_qat.zero_fractions(lib_t)), 1)
        ),
        "zero_frac_mean_dense": (
            sum(ternary_qat.zero_fractions(dens_t).values())
            / max(len(ternary_qat.zero_fractions(dens_t)), 1)
        ),
        "zero_frac_mean_pack": (
            sum(ternary_qat.zero_fractions(pack_t).values())
            / max(len(ternary_qat.zero_fractions(pack_t)), 1)
        ),
    }

    # ── gates ─────────────────────────────────────────────────────────────
    eps = 1e-8
    loss_rel_d = abs(r_d["loss_last"] - r_lib["loss_last"]) / max(
        abs(r_lib["loss_last"]), eps
    )
    loss_rel_p = abs(r_p["loss_last"] - r_lib["loss_last"]) / max(
        abs(r_lib["loss_last"]), eps
    )
    gates = {
        "fwd_cos_dense": fwd["cos_dense_vs_lib"] >= args.cos_gate,
        "fwd_cos_pack": fwd["cos_pack_vs_lib"] >= args.cos_gate,
        "fwd_cos_pack_dense": fwd["cos_pack_vs_dense"] >= args.cos_gate,
        "fwd_max_abs_dense": fwd["max_abs_dense_vs_lib"] <= args.max_abs_gate,
        "fwd_max_abs_pack": fwd["max_abs_pack_vs_lib"] <= args.max_abs_gate,
        "train_loss_decreased_lib": r_lib["loss_last"] < r_lib["loss_first"],
        "train_loss_decreased_dense": r_d["loss_last"] < r_d["loss_first"],
        "train_loss_decreased_pack": r_p["loss_last"] < r_p["loss_first"],
        "train_loss_rel_dense": loss_rel_d <= args.loss_rel_gate,
        "train_loss_rel_pack": loss_rel_p <= args.loss_rel_gate,
        "swapped_equal": n_lib == n_d == n_p and n_lib > 0,
    }
    ok = all(gates.values())

    report = {
        "ok": ok,
        "device": device,
        "steps": args.steps,
        "batch": args.batch,
        "seq": args.seq,
        "lr": args.lr,
        "seed": args.seed,
        "counts": counts,
        "forward": fwd,
        "train": train,
        "loss_rel_dense": loss_rel_d,
        "loss_rel_pack": loss_rel_p,
        "gates": gates,
        "gates_failed": [k for k, v in gates.items() if not v],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    # also timestamped copy
    stamp = time.strftime("%Y%m%d-%H%M%S")
    stamp_path = args.out.parent / f"summary-{stamp}.json"
    stamp_path.write_text(json.dumps(report, indent=2))

    print(json.dumps({k: report[k] for k in (
        "ok", "device", "steps", "forward", "loss_rel_dense", "loss_rel_pack",
        "gates_failed", "counts",
    )}, indent=2))
    print(
        f"train loss first→last  lib={r_lib['loss_first']:.4f}→{r_lib['loss_last']:.4f}  "
        f"dense={r_d['loss_first']:.4f}→{r_d['loss_last']:.4f}  "
        f"pack={r_p['loss_first']:.4f}→{r_p['loss_last']:.4f}"
    )
    print(f"wrote {args.out} and {stamp_path}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
