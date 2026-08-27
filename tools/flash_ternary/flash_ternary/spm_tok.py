"""Fast tokenizer for ternlight SPM-style Unigram dumps (no transformers).

Loads ``tokenizer.json`` via HuggingFace ``tokenizers`` (Rust). Matches
``AutoTokenizer.from_pretrained(spm-*-*k)`` for:

  - CLS / SEP special tokens (ids from the model; EN: CLS=2, SEP=3, PAD=0)
  - right padding, truncation to ``max_len``
  - dynamic batch pad (pad to longest in batch, not always max_len)

Cold load is ~5× faster than ``transformers.AutoTokenizer`` and avoids pulling
the transformers stack into the RAG query hot path.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class FastSpmTokenizer:
    """Minimal encode API compatible with PackCpuEmbed / PackCudaEmbed."""

    def __init__(self, tokenizer_dir: str | Path):
        from tokenizers import Tokenizer

        root = Path(tokenizer_dir)
        json_path = root / "tokenizer.json"
        if not json_path.is_file():
            # allow passing the json file itself
            if root.is_file() and root.name.endswith(".json"):
                json_path = root
                root = root.parent
            else:
                raise FileNotFoundError(f"tokenizer.json not found under {tokenizer_dir}")

        self.root = root
        self._tok = Tokenizer.from_file(str(json_path))
        self.pad_token_id = 0
        self.cls_token_id: int | None = None
        self.sep_token_id: int | None = None
        cfg_path = root / "tokenizer_config.json"
        if cfg_path.is_file():
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            except Exception:
                cfg = {}
            # resolve special ids from the tokenizer model when possible
            for name, attr in (
                ("pad_token", "pad_token_id"),
                ("cls_token", "cls_token_id"),
                ("sep_token", "sep_token_id"),
            ):
                tok_str = cfg.get(name)
                if not tok_str:
                    continue
                tid = self._tok.token_to_id(tok_str)
                if tid is not None:
                    setattr(self, attr, int(tid))
        # defaults used by train_spm_en/ja Unigram dumps
        if self.cls_token_id is None:
            self.cls_token_id = self._tok.token_to_id("[CLS]")
        if self.sep_token_id is None:
            self.sep_token_id = self._tok.token_to_id("[SEP]")
        if self.pad_token_id is None or self.pad_token_id < 0:
            pid = self._tok.token_to_id("[PAD]")
            self.pad_token_id = 0 if pid is None else int(pid)

        self.model_max_length = 128
        if cfg_path.is_file():
            try:
                self.model_max_length = int(
                    json.loads(cfg_path.read_text(encoding="utf-8")).get(
                        "model_max_length", 128
                    )
                )
            except Exception:
                pass

    @classmethod
    def from_pretrained(cls, tokenizer_dir: str | Path) -> "FastSpmTokenizer":
        return cls(tokenizer_dir)

    def encode_batch_np(
        self,
        texts: list[str],
        *,
        max_len: int | None = None,
        pad_to_max_len: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (input_ids, attention_mask) as int64 arrays [B, T].

        Dynamic pad to longest sequence in the batch (capped at max_len),
        matching ``AutoTokenizer(..., padding=True, truncation=True)``.
        Set ``pad_to_max_len=True`` to always pad to max_len (rarely needed).
        """
        if not texts:
            T = max_len or self.model_max_length
            return (
                np.zeros((0, T), dtype=np.int64),
                np.zeros((0, T), dtype=np.int64),
            )
        max_len = int(max_len or self.model_max_length)
        # tokenizers truncation is set per-call via encode args when possible;
        # enable_truncation mutates the tokenizer — set and leave (max_len stable).
        self._tok.enable_truncation(max_length=max_len)
        # disable fixed-length padding; we pad ourselves to batch max
        self._tok.no_padding()

        encs = self._tok.encode_batch(list(texts))
        lengths = [min(len(e.ids), max_len) for e in encs]
        T = max_len if pad_to_max_len else max(lengths)
        T = max(T, 1)
        B = len(texts)
        ids = np.full((B, T), self.pad_token_id, dtype=np.int64)
        mask = np.zeros((B, T), dtype=np.int64)
        for i, e in enumerate(encs):
            L = lengths[i]
            if L == 0:
                continue
            row = e.ids[:L]
            ids[i, :L] = row
            mask[i, :L] = 1
        return ids, mask

    def __call__(
        self,
        texts: str | list[str],
        *,
        return_tensors: str | None = None,
        padding: bool | str = True,
        truncation: bool = True,
        max_length: int | None = None,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        """HF-ish call surface for drop-in use in embed_texts."""
        if isinstance(texts, str):
            batch = [texts]
        else:
            batch = list(texts)
        max_len = max_length or self.model_max_length
        pad_to_max = padding == "max_length"
        ids, mask = self.encode_batch_np(
            batch, max_len=max_len, pad_to_max_len=bool(pad_to_max)
        )
        if return_tensors in ("pt", "torch"):
            import torch

            return {
                "input_ids": torch.from_numpy(ids),
                "attention_mask": torch.from_numpy(mask),
            }
        if return_tensors in ("np", "numpy"):
            return {"input_ids": ids, "attention_mask": mask}
        # list form
        return {
            "input_ids": ids.tolist(),
            "attention_mask": mask.tolist(),
        }


def load_fast_tokenizer(tokenizer_dir: str | Path) -> FastSpmTokenizer:
    """Load FastSpmTokenizer; fall back to AutoTokenizer only if json missing."""
    root = Path(tokenizer_dir)
    if (root / "tokenizer.json").is_file() or (
        root.is_file() and root.suffix == ".json"
    ):
        return FastSpmTokenizer.from_pretrained(root)
    # legacy path
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(root))  # type: ignore[return-value]
