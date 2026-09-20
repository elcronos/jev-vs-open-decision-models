"""Laya (local) backend, implemented exactly against PROTOCOL.md §3.2.

The general-English checkpoint at the repo root of ``convaiinnovations/laya`` is downloaded at
the pinned revision with root-only ``allow_patterns`` (so the ``multilingual/`` and
``typed-decisions/`` sub-checkpoints are never fetched), loaded with ``laya.load`` and queried one
example per ``predict`` call with the frozen instruction and the six labels as ``choice`` criteria.

Variants (PROTOCOL.md §4):

* ``plain``   - ``criteria[label] = None``  -> ``laya.common.render_options`` renders the bare label.
* ``defined`` - ``criteria[label] = DEFINITIONS[label]`` -> renders ``"label: definition"``.

The library's shipped per-option-count temperature (``choice:6-10``) is applied inside
``Agent.predict`` and is neither fitted nor overridden here.

Run ``python -m models.laya --smoke`` (from the project root) for the 8-row validation smoke test.
"""
from __future__ import annotations

import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

# Allow both ``python -m models.laya`` and ``python models/laya.py``.
try:
    from models import common
except ModuleNotFoundError:  # pragma: no cover - script-style invocation
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models import common

from models.common import (
    DEFINITIONS,
    INSTRUCTION,
    LABELS,
    LAYA_REPO,
    LAYA_REVISION,
    VARIANTS,
    LoadInfo,
    Prediction,
    renormalise,
)

QUESTION_ID = "emotion"

# Allow patterns exactly as enumerated in PROTOCOL.md §3.2. ``huggingface_hub`` matches them with
# ``fnmatch`` against the full repo path, where ``*`` also spans ``/``, so on their own ``*.json``
# and ``*.py`` would also pull the small config files under ``multilingual/`` and
# ``typed-decisions/`` (never their ``model.safetensors``, which has no wildcard). The protocol's
# "[root files only: ...]" bracket is therefore enforced with ``IGNORE_PATTERNS`` as well.
ALLOW_PATTERNS: Tuple[str, ...] = (
    "*.json",
    "model.safetensors",
    "tokenizer/*",
    "encoder/*",
    "*.py",
    "README.md",
)
FORBIDDEN_SUBFOLDERS: Tuple[str, ...] = ("multilingual", "typed-decisions")
IGNORE_PATTERNS: Tuple[str, ...] = tuple(f"{d}/*" for d in FORBIDDEN_SUBFOLDERS)


def select_device() -> str:
    """Device string per PROTOCOL.md §3.2: ``mps`` on Apple silicon, else ``cuda``, else ``cpu``."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_questions(variant: str) -> Dict[str, Dict[str, Any]]:
    """The question dict passed to ``agent.predict`` for a variant (PROTOCOL.md §3.2 / §4)."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")
    if variant == "plain":
        criteria: Dict[str, Optional[str]] = {label: None for label in LABELS}
    else:
        criteria = {label: DEFINITIONS[label] for label in LABELS}
    return {
        QUESTION_ID: {
            "type": "choice",
            "instructions": INSTRUCTION,
            "criteria": criteria,
        }
    }


def rendered_options(variant: str) -> List[str]:
    """Exactly the option strings the model sees, via ``laya.common.render_options`` applied to the
    library's internal question representation (``Agent._to_internal``)."""
    from laya.agent import Agent
    from laya.common import render_options

    internal = Agent._to_internal(build_questions(variant)[QUESTION_ID])
    return render_options(internal)


def _sync_device(device: str) -> None:
    """Block until all queued kernels on ``device`` have finished (for honest latency timing)."""
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


class LayaBackend:
    """``Backend`` implementation for Laya (``name == 'laya'``)."""

    name: str = "laya"

    def __init__(self, device: Optional[str] = None, local_dir: Optional[str] = None) -> None:
        self.device: str = device or select_device()
        self._local_dir: Optional[str] = local_dir
        self.agent: Any = None
        self.load_info: Optional[LoadInfo] = None
        self._peak_mps_bytes: int = 0

    # ------------------------------------------------------------------ loading

    def download(self) -> str:
        """Snapshot-download the root checkpoint at the pinned revision; returns the local dir.

        Raises ``RuntimeError`` if a forbidden sub-checkpoint folder was fetched.
        """
        from huggingface_hub import snapshot_download

        if self._local_dir is None:
            self._local_dir = snapshot_download(
                LAYA_REPO,
                revision=LAYA_REVISION,
                allow_patterns=list(ALLOW_PATTERNS),
                ignore_patterns=list(IGNORE_PATTERNS),
            )
        present = [d for d in FORBIDDEN_SUBFOLDERS if os.path.isdir(os.path.join(self._local_dir, d))]
        if present:
            raise RuntimeError(
                f"snapshot at {self._local_dir} contains sub-checkpoint folders {present}; "
                f"PROTOCOL.md §3.2 requires the repo-root (general English) checkpoint only"
            )
        return self._local_dir

    def load(self) -> LoadInfo:
        """Download (if needed) and load the agent; returns ``LoadInfo`` (PROTOCOL.md §6)."""
        import laya

        t0 = time.perf_counter()
        local_dir = self.download()
        self.agent = laya.load(local_dir, device=self.device)
        self.agent.model.eval()
        # The library may have fallen back to CPU; record what actually happened.
        self.device = self.agent.device.type
        _sync_device(self.device)
        load_time_s = time.perf_counter() - t0

        weights_path = os.path.join(local_dir, "model.safetensors")
        weight_bytes = os.path.getsize(weights_path) if os.path.exists(weights_path) else None
        param_count = int(sum(p.numel() for p in self.agent.model.parameters()))
        cfg = self.agent.cfg

        self._sample_peak_memory()
        self.load_info = LoadInfo(
            name=self.name,
            load_time_s=load_time_s,
            device=self.device,
            dtype=str(self.agent.dtype),
            revision=LAYA_REVISION,
            param_count=param_count,
            weight_bytes=weight_bytes,
            extra={
                "repo": LAYA_REPO,
                "local_dir": local_dir,
                "laya_version": getattr(laya, "__version__", None),
                "temperature_by_options": dict(self.agent.temperature_by_options),
                "temperature": list(self.agent.temperature),
                "head_max_len": cfg.get("head_max_len", 192),
                "max_len": cfg.get("max_len", 512),
                "encoder": cfg.get("encoder"),
                "head_layers": cfg.get("head_layers"),
                "batch_size": 1,
            },
        )
        return self.load_info

    def _require_agent(self) -> Any:
        if self.agent is None:
            raise RuntimeError("LayaBackend.load() must be called before predicting")
        return self.agent

    # ---------------------------------------------------------------- inference

    def warmup(self, texts: List[str], variant: str = "plain") -> None:
        """Run untimed predictions (PROTOCOL.md §6: warm-up calls on validation rows)."""
        for text in texts:
            self.predict_one(-1, text, variant)

    def predict_one(self, dataset_index: int, text: str, variant: str = "plain") -> Prediction:
        """Score one example (batch size 1). ``latency_ms`` covers only ``agent.predict``."""
        agent = self._require_agent()
        questions = build_questions(variant)
        try:
            _sync_device(self.device)
            t0 = time.perf_counter()
            out = agent.predict(text, questions)
            _sync_device(self.device)
            latency_ms = (time.perf_counter() - t0) * 1000.0
        except Exception as exc:  # noqa: BLE001 - recorded per example, never raised
            return Prediction(
                dataset_index=dataset_index,
                probs=[math.nan] * len(LABELS),
                pred=-1,
                confidence=math.nan,
                latency_ms=math.nan,
                error=f"{type(exc).__name__}: {exc}",
                retries=0,
                extra={"variant": variant},
            )
        self._sample_peak_memory()
        return self._to_prediction(dataset_index, variant, out, latency_ms)

    def predict_many(self, items: List[tuple], variant: str = "plain") -> List[Prediction]:
        """Sequential batch-size-1 scoring of ``(dataset_index, text)`` pairs."""
        return [self.predict_one(int(idx), str(text), variant) for idx, text in items]

    def _to_prediction(self, dataset_index: int, variant: str, out: Dict[str, Any], latency_ms: float) -> Prediction:
        answer = out["answers"][QUESTION_ID]
        raw_probs: Dict[str, float] = {str(k): float(v) for k, v in answer["probabilities"].items()}
        api_choice: str = str(answer["choice"])
        missing = [label for label in LABELS if label not in raw_probs]
        if missing:
            raise KeyError(f"laya response is missing probabilities for labels {missing}: {raw_probs}")

        fallback = LABELS.index(api_choice) if api_choice in LABELS else None
        probs = renormalise([raw_probs[label] for label in LABELS], fallback_index=fallback)
        pred = int(max(range(len(LABELS)), key=probs.__getitem__))
        confidence = float(probs[pred])

        action = answer.get("action") or {}
        usage = out.get("usage") or {}
        extra: Dict[str, Any] = {
            "variant": variant,
            "api_choice": api_choice,
            "api_confidence": float(answer["confidence"]),
            "act_probability": (
                float(action["act_probability"]) if "act_probability" in action else None
            ),
            "raw_probs": raw_probs,
            "input_tokens": int(usage["input_tokens"]) if "input_tokens" in usage else None,
            "pred_mismatch": LABELS[pred] != api_choice,
            "model": out.get("model"),
        }
        return Prediction(
            dataset_index=dataset_index,
            probs=probs,
            pred=pred,
            confidence=confidence,
            latency_ms=latency_ms,
            error=None,
            retries=0,
            extra=extra,
        )

    # ------------------------------------------------------------------- memory

    def _sample_peak_memory(self) -> None:
        if self.device == "mps":
            self._peak_mps_bytes = max(self._peak_mps_bytes, int(torch.mps.driver_allocated_memory()))

    def peak_memory_bytes(self) -> Optional[int]:
        """Peak device memory: running max of ``torch.mps.driver_allocated_memory()`` on MPS
        (sampled after load and after every predict), ``torch.cuda.max_memory_allocated()`` on
        CUDA, ``None`` on CPU."""
        if self.device == "mps":
            self._sample_peak_memory()
            return self._peak_mps_bytes
        if self.device == "cuda":
            return int(torch.cuda.max_memory_allocated())
        return None


# ---------------------------------------------------------------------------- smoke

def _smoke(n_rows: int = 8) -> None:
    """Score the first ``n_rows`` validation rows for both variants and print everything."""
    import json

    from datasets import load_dataset

    ds = load_dataset(
        common.DATASET_ID,
        common.DATASET_CONFIG,
        split=f"validation[:{n_rows}]",
        revision=common.DATASET_REVISION,
    )
    rows = [(i, str(r["text"]), int(r["label"])) for i, r in enumerate(ds)]

    backend = LayaBackend()
    info = backend.load()
    print("== load info ==")
    print(json.dumps(info.to_dict(), indent=2))
    local_dir = info.extra["local_dir"]
    print("== downloaded snapshot contents ==")
    for root, dirs, files in os.walk(local_dir):
        rel = os.path.relpath(root, local_dir)
        print(f"  [{rel}] dirs={sorted(dirs)} files={sorted(files)}")
    print("forbidden subfolders present:",
          [d for d in FORBIDDEN_SUBFOLDERS if os.path.isdir(os.path.join(local_dir, d))])

    for variant in VARIANTS:
        print(f"\n== variant: {variant} ==")
        print("rendered options (laya.common.render_options):")
        for opt in rendered_options(variant):
            print(f"  - {opt!r}")
        preds = backend.predict_many([(i, t) for i, t, _ in rows], variant)
        n_correct = 0
        for (i, text, gold), p in zip(rows, preds):
            ok = p.pred == gold
            n_correct += int(ok)
            print(
                f"[{i}] gold={LABELS[gold]:<8} pred={LABELS[p.pred] if p.pred >= 0 else 'ERR':<8} "
                f"{'OK ' if ok else 'X  '} conf={p.confidence:.4f} api_conf={p.extra.get('api_confidence')} "
                f"act={p.extra.get('act_probability')} tok={p.extra.get('input_tokens')} "
                f"lat={p.latency_ms:.1f}ms mismatch={p.extra.get('pred_mismatch')} err={p.error}"
            )
            print(f"      probs={[round(x, 4) for x in p.probs]}")
            print(f"      text={text[:100]!r}")
        print(f"accuracy on {len(rows)} smoke rows: {n_correct}/{len(rows)}")

    peak = backend.peak_memory_bytes()
    print(f"\npeak_memory_bytes={peak} ({peak / 2**20:.1f} MiB)" if peak is not None else "\npeak_memory_bytes=None")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Laya backend (PROTOCOL.md §3.2)")
    parser.add_argument("--smoke", action="store_true", help="run the 8-row validation smoke test")
    args = parser.parse_args()
    if args.smoke:
        _smoke()
    else:
        parser.print_help()
