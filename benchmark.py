"""End-to-end benchmark driver implementing PROTOCOL.md (frozen) §2, §6, §7 and §8.

Flow
----
1. Seed ``random``, ``numpy`` and ``torch`` with ``models.common.SEED`` and enable deterministic
   algorithms where supported (§8).
2. Load ``dair-ai/emotion`` (config ``split``) at the pinned revision (§2). The evaluated split is
   ``--split`` (``test`` for the real run, ``validation`` for smoke tests); warm-up texts are always
   the first 10 rows of ``validation`` (§6), never of the evaluated split.
3. For every requested backend: ``load()`` (timed), ``warmup()`` per variant, ``predict_many()`` over
   the rows for every requested variant, then ``peak_memory_bytes()``.
4. Assemble the wide per-(dataset_index, variant) DataFrame of §7. When
   ``results/raw_predictions.parquet`` already exists for the same split, only the columns of the
   models and variants just run are overwritten, so ``plain`` / ``defined`` and per-model runs can be
   executed in separate invocations.
5. Write ``raw_predictions.parquet``, ``env.json``, ``summary.json``, ``summary.csv`` and the plots.

Per-example failures never abort the run: such rows carry ``{prefix}_error``, ``pred == -1`` and NaN
probabilities and their count is reported as ``n_errors``. PROTOCOL.md §5 requires every metric to be
computed from identical rows per system, so all headline metrics, bootstrap CIs and pairwise tests of
one variant are computed on the *common* subset of rows every present model scored (``n_common``);
with zero errors that is the full split. Whenever ``n_common < n_total`` the affected rows are listed
under ``per_variant[<variant>]["deviation"]`` in ``summary.json``, the per-model numbers on each
model's own valid rows are kept under ``metrics_own_rows`` for reference, and on the test split the
deviation is appended to ``results/deviations.md``. A resumed run re-requests only the failed rows
(the Jev cache never stores failures), so clearing the errors restores the full 2000-row set.

Usage::

    python benchmark.py --split validation --limit 8 --models jev,prismnli,laya --variants plain,defined
    python benchmark.py --split test --models jev,prismnli,laya --variants plain

The OpenRouter key is read by ``models.jev`` from ``OPENROUTER_API_KEY`` and never written anywhere.
"""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import metrics  # noqa: E402
import plots  # noqa: E402
from models.common import (  # noqa: E402
    DATASET_CONFIG,
    DATASET_ID,
    DATASET_REVISION,
    JEV_MODEL_ID,
    LABELS,
    LAYA_REVISION,
    PRISMNLI_REVISION,
    SEED,
    VARIANTS,
    Backend,
    LoadInfo,
    Prediction,
)

# ------------------------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------------------------

RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_ORDER: Tuple[str, ...] = ("jev", "prismnli", "laya")
DISPLAY_NAMES: Dict[str, str] = {"jev": "Jev", "prismnli": "PrismNLI-0.4B", "laya": "Laya"}
REMOTE_MODELS = frozenset({"jev"})
LATENCY_KIND_REMOTE = "remote end-to-end"
LATENCY_KIND_LOCAL = "local compute"
WARMUP_N = 10
SPLITS = ("validation", "test")

BASE_COLUMNS: Tuple[str, ...] = ("dataset_index", "variant", "split", "text", "gold_id", "gold_label")
PER_MODEL_COLUMNS: Tuple[str, ...] = (
    "pred",
    *(f"p_{label}" for label in LABELS),
    "confidence",
    "latency_ms",
    "error",
    "retries",
)
MODEL_SPECIFIC_COLUMNS: Dict[str, Tuple[str, ...]] = {
    "jev": (
        "jev_pred_mismatch",
        "jev_cost_usd",
        "jev_input_tokens",
        "jev_output_tokens",
        "jev_response_id",
        "jev_model",
        "jev_api_confidence",
        "jev_raw_probs_json",
    ),
    "laya": ("laya_pred_mismatch", "laya_api_confidence", "laya_act_probability"),
    "prismnli": ("prismnli_indep_probs_json", "prismnli_indep_probs_l1_json"),
}
SUMMARY_CSV_COLUMNS: Tuple[str, ...] = (
    "model",
    "variant",
    "split",
    "n",
    "n_valid",
    "n_common",
    "accuracy",
    "acc_ci_lo",
    "acc_ci_hi",
    "macro_f1",
    "f1_ci_lo",
    "f1_ci_hi",
    "brier",
    "ece15",
    "ece_adaptive10",
    "nll",
    "mean_confidence",
    "acc_at_cov50",
    "acc_at_cov80",
    "acc_at_cov90",
    "acc_at_cov100",
    "frac_gold_prob_zero",
    "p50_ms",
    "p95_ms",
    "mean_ms",
    "examples_per_sec",
    "total_cost_usd",
    "n_errors",
    "n_retries",
    "latency_kind",
)
PACKAGES_FOR_ENV: Tuple[str, ...] = (
    "torch",
    "transformers",
    "laya",
    "datasets",
    "huggingface_hub",
    "safetensors",
    "numpy",
    "pandas",
    "pyarrow",
    "scipy",
    "scikit-learn",
    "statsmodels",
    "matplotlib",
    "httpx",
)


# ------------------------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------------------------


def log(msg: str) -> None:
    """Timestamped progress line on stdout (flushed so it interleaves with library output)."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def seed_everything(seed: int = SEED) -> Dict[str, Any]:
    """Seed all RNGs and request deterministic kernels (PROTOCOL.md §8). Returns what was set."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    deterministic = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        deterministic = True
    except (RuntimeError, TypeError):
        deterministic = False
    return {"seed": seed, "torch_deterministic_algorithms": deterministic}


def json_safe(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays and non-finite floats so ``json.dumps`` is strict."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def parse_csv_list(value: str, allowed: Sequence[str], what: str) -> List[str]:
    """Parse ``a,b,c`` into a de-duplicated list validated against ``allowed`` (order preserved)."""
    items = [v.strip() for v in value.split(",") if v.strip()]
    if not items:
        raise argparse.ArgumentTypeError(f"at least one {what} is required")
    unknown = [v for v in items if v not in allowed]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown {what}(s) {unknown}; allowed: {list(allowed)}")
    return list(dict.fromkeys(items))


def package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def sysctl(key: str) -> Optional[str]:
    """Read one ``sysctl`` value on macOS; None elsewhere or on failure."""
    if platform.system() != "Darwin":
        return None
    try:
        out = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def hardware_info() -> Dict[str, Any]:
    mem = sysctl("hw.memsize")
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "chip": sysctl("machdep.cpu.brand_string"),
        "memory_bytes": int(mem) if mem and mem.isdigit() else None,
        "cpu_count": os.cpu_count(),
        "torch_mps_available": bool(torch.backends.mps.is_available()),
        "torch_cuda_available": bool(torch.cuda.is_available()),
    }


# ------------------------------------------------------------------------------------------------
# Data
# ------------------------------------------------------------------------------------------------


def load_split(split: str, limit: Optional[int] = None) -> pd.DataFrame:
    """Rows of the pinned dataset split as ``dataset_index, text, gold_id, gold_label``."""
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID, DATASET_CONFIG, split=split, revision=DATASET_REVISION)
    n = len(ds) if limit is None else min(limit, len(ds))
    rows = ds.select(range(n))
    labels = np.asarray(rows["label"], dtype=np.int64)
    return pd.DataFrame(
        {
            "dataset_index": np.arange(n, dtype=np.int64),
            "text": [str(t) for t in rows["text"]],
            "gold_id": labels,
            "gold_label": [LABELS[int(i)] for i in labels],
        }
    )


# ------------------------------------------------------------------------------------------------
# Backends
# ------------------------------------------------------------------------------------------------


def make_backend(name: str, split: str) -> Backend:
    """Instantiate a backend. The production Jev cache is used for the test split only."""
    if name == "jev":
        from models.jev import DEFAULT_CACHE_DIR, SMOKE_CACHE_DIR, JevBackend

        cache_dir = DEFAULT_CACHE_DIR if split == "test" else SMOKE_CACHE_DIR
        return JevBackend(cache_dir=cache_dir, show_progress=True)
    if name == "prismnli":
        from models.prismnli import PrismNLIBackend

        return PrismNLIBackend()
    if name == "laya":
        from models.laya import LayaBackend

        return LayaBackend()
    raise ValueError(f"unknown model {name!r}")


def release_backend(backend: Backend) -> None:
    """Close remote clients / drop local weights so the next model starts from a clean device."""
    close = getattr(backend, "close", None)
    if callable(close):
        close()
    for attr in ("model", "agent", "tokenizer"):
        if hasattr(backend, attr):
            try:
                setattr(backend, attr, None)
            except AttributeError:
                pass
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class ModelRun:
    """Everything one backend produced: LoadInfo, per-variant predictions and timing."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.load_info: Optional[LoadInfo] = None
        self.load_wall_s: float = math.nan
        self.predictions: Dict[str, List[Prediction]] = {}
        self.wall_s: Dict[str, float] = {}
        self.peak_memory_bytes: Optional[int] = None
        self.http_calls: Optional[int] = None


def run_model(name: str, rows: pd.DataFrame, warm_texts: Sequence[str], variants: Sequence[str],
              split: str) -> ModelRun:
    """Load one backend, warm it up per variant, score every row for every variant."""
    run = ModelRun(name)
    backend = make_backend(name, split)
    log(f"[{name}] loading ...")
    t0 = time.perf_counter()
    run.load_info = backend.load()
    run.load_wall_s = time.perf_counter() - t0
    log(f"[{name}] loaded in {run.load_wall_s:.1f}s (device={run.load_info.device}, dtype={run.load_info.dtype})")

    items = list(zip(rows["dataset_index"].astype(int).tolist(), rows["text"].tolist()))
    try:
        for variant in variants:
            log(f"[{name}] warm-up on {len(warm_texts)} validation rows (variant={variant})")
            backend.warmup(list(warm_texts), variant=variant)
            log(f"[{name}] predicting {len(items)} rows (variant={variant})")
            t1 = time.perf_counter()
            preds = backend.predict_many(items, variant=variant)
            run.wall_s[variant] = time.perf_counter() - t1
            if len(preds) != len(items):
                raise RuntimeError(f"{name}.predict_many returned {len(preds)} predictions for {len(items)} items")
            run.predictions[variant] = preds
            n_err = sum(p.error is not None for p in preds)
            log(f"[{name}] variant={variant}: {len(preds) - n_err} scored, {n_err} errors, "
                f"{run.wall_s[variant]:.1f}s wall")
        run.peak_memory_bytes = backend.peak_memory_bytes()
        http_calls = getattr(backend, "http_calls", None)
        run.http_calls = int(http_calls) if isinstance(http_calls, int) else None
    finally:
        release_backend(backend)
    return run


# ------------------------------------------------------------------------------------------------
# DataFrame assembly (§7)
# ------------------------------------------------------------------------------------------------


def _extra(pred: Prediction, key: str) -> Any:
    value = pred.extra.get(key) if pred.extra else None
    return value


def _json_or_none(value: Any) -> Optional[str]:
    return None if value is None else json.dumps(value)


def predictions_frame(name: str, variant: str, preds: Sequence[Prediction]) -> pd.DataFrame:
    """Per-model columns of §7 for one variant, keyed by ``dataset_index`` + ``variant``."""
    p = name
    records: List[Dict[str, Any]] = []
    for pr in preds:
        probs = list(pr.probs) if pr.error is None else [math.nan] * len(LABELS)
        rec: Dict[str, Any] = {
            "dataset_index": int(pr.dataset_index),
            "variant": variant,
            f"{p}_pred": int(pr.pred) if pr.error is None else -1,
            f"{p}_confidence": float(pr.confidence) if pr.error is None else math.nan,
            f"{p}_latency_ms": float(pr.latency_ms) if pr.error is None else math.nan,
            f"{p}_error": pr.error,
            f"{p}_retries": int(pr.retries),
        }
        for label, value in zip(LABELS, probs):
            rec[f"{p}_p_{label}"] = float(value)
        if name == "jev":
            rec.update(
                {
                    "jev_pred_mismatch": _extra(pr, "pred_mismatch"),
                    "jev_cost_usd": _extra(pr, "cost_usd"),
                    "jev_input_tokens": _extra(pr, "input_tokens"),
                    "jev_output_tokens": _extra(pr, "output_tokens"),
                    "jev_response_id": _extra(pr, "response_id"),
                    "jev_model": _extra(pr, "model"),
                    "jev_api_confidence": _extra(pr, "api_confidence"),
                    "jev_raw_probs_json": _json_or_none(_extra(pr, "raw_probs")),
                }
            )
        elif name == "laya":
            rec.update(
                {
                    "laya_pred_mismatch": _extra(pr, "pred_mismatch"),
                    "laya_api_confidence": _extra(pr, "api_confidence"),
                    "laya_act_probability": _extra(pr, "act_probability"),
                }
            )
        elif name == "prismnli":
            rec.update(
                {
                    "prismnli_indep_probs_json": _json_or_none(_extra(pr, "indep_probs")),
                    "prismnli_indep_probs_l1_json": _json_or_none(_extra(pr, "indep_probs_l1")),
                }
            )
        records.append(rec)
    frame = pd.DataFrame.from_records(records)
    # Numeric extras arrive as Python objects (None for errors); coerce to float columns.
    for col in ("jev_cost_usd", "jev_input_tokens", "jev_output_tokens", "jev_api_confidence",
                "laya_api_confidence", "laya_act_probability"):
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float64")
    for col in ("jev_pred_mismatch", "laya_pred_mismatch"):
        if col in frame.columns:
            frame[col] = frame[col].astype("boolean")
    for col in (f"{p}_error", "jev_response_id", "jev_model", "jev_raw_probs_json",
                "prismnli_indep_probs_json", "prismnli_indep_probs_l1_json"):
        if col in frame.columns:
            frame[col] = frame[col].astype("object").where(frame[col].notna(), None)
    return frame


def assemble_frame(rows: pd.DataFrame, split: str, runs: Sequence[ModelRun], variants: Sequence[str]) -> pd.DataFrame:
    """Wide frame for the rows and variants just run: base columns + every model's columns."""
    parts: List[pd.DataFrame] = []
    for variant in variants:
        base = rows.copy()
        base.insert(1, "variant", variant)
        base.insert(2, "split", split)
        for run in runs:
            preds = run.predictions.get(variant)
            if preds is None:
                continue
            base = base.merge(predictions_frame(run.name, variant, preds), on=["dataset_index", "variant"], how="left")
        parts.append(base)
    return pd.concat(parts, ignore_index=True)


def column_order(columns: Iterable[str]) -> List[str]:
    """Canonical §7 column order; unknown columns are appended alphabetically."""
    cols = set(columns)
    ordered: List[str] = [c for c in BASE_COLUMNS if c in cols]
    for name in MODEL_ORDER:
        ordered += [f"{name}_{c}" for c in PER_MODEL_COLUMNS if f"{name}_{c}" in cols]
    for name in MODEL_ORDER:
        ordered += [c for c in MODEL_SPECIFIC_COLUMNS[name] if c in cols]
    ordered += sorted(cols - set(ordered))
    return ordered


def merge_with_existing(new: pd.DataFrame, path: Path, split: str) -> pd.DataFrame:
    """Overlay ``new`` on the parquet at ``path``: columns of ``new`` win for the rows it covers, every
    other row/column of the old file is kept. An old file from a *different* split is moved aside
    instead of merged, so validation smoke rows can never leak into test-split results."""
    key = ["dataset_index", "variant"]
    if not path.exists():
        return new[column_order(new.columns)]
    old = pd.read_parquet(path)
    old_splits = set(old["split"].dropna().unique().tolist()) if "split" in old.columns else set()
    if old_splits and old_splits != {split}:
        backup = path.with_name(f"{path.stem}.{'_'.join(sorted(old_splits))}.bak{path.suffix}")
        shutil.move(str(path), str(backup))
        log(f"existing {path.name} holds split(s) {sorted(old_splits)} != {split!r}; moved to {backup.name}")
        return new[column_order(new.columns)]

    old_i = old.set_index(key)
    new_i = new.set_index(key)
    in_new = old_i.index.isin(new_i.index)
    kept_rows = old_i.loc[~in_new]
    overlapping = old_i.loc[in_new].drop(columns=[c for c in new_i.columns if c in old_i.columns])
    updated = new_i.join(overlapping, how="left")
    merged = pd.concat([updated, kept_rows]).sort_index().reset_index()
    return merged[column_order(merged.columns)]


# ------------------------------------------------------------------------------------------------
# Metrics / summary
# ------------------------------------------------------------------------------------------------


def latency_stats(values: np.ndarray) -> Dict[str, Optional[float]]:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return {"n": 0, "p50_ms": None, "p95_ms": None, "mean_ms": None, "examples_per_sec": None}
    mean = float(np.mean(v))
    return {
        "n": int(v.size),
        "p50_ms": float(np.percentile(v, 50)),
        "p95_ms": float(np.percentile(v, 95)),
        "mean_ms": mean,
        "examples_per_sec": (1000.0 / mean) if mean > 0 else None,
    }


def valid_mask(df: pd.DataFrame, name: str) -> np.ndarray:
    """Rows where ``name`` produced a usable prediction (no error, pred in range, finite probs)."""
    pred_col, err_col = f"{name}_pred", f"{name}_error"
    if pred_col not in df.columns:
        return np.zeros(len(df), dtype=bool)
    pred = pd.to_numeric(df[pred_col], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(pred) & (pred >= 0) & (pred < len(LABELS))
    if err_col in df.columns:
        ok &= df[err_col].isna().to_numpy()
    probs = df[[f"{name}_p_{label}" for label in LABELS]].to_numpy(dtype=float)
    ok &= np.all(np.isfinite(probs), axis=1)
    return ok


def common_mask(masks: Dict[str, np.ndarray], n_rows: int) -> np.ndarray:
    """Rows scored by *every* model in ``masks`` (§5 "identical rows per system"). All rows if empty."""
    if not masks:
        return np.ones(n_rows, dtype=bool)
    return np.logical_and.reduce(list(masks.values()))


def deviation_record(variant: str, split: Optional[str], masks: Dict[str, np.ndarray], common: np.ndarray,
                     dataset_index: np.ndarray) -> Optional[Dict[str, Any]]:
    """Describe a §5 deviation (metrics on fewer than all rows) or None when every row is common."""
    excluded = dataset_index[~common]
    if excluded.size == 0:
        return None
    per_model_excluded = {
        name: [int(i) for i in dataset_index[~mask]] for name, mask in masks.items() if not bool(mask.all())
    }
    return {
        "protocol_section": "§5 identical rows per system",
        "variant": variant,
        "split": split,
        "n_total": int(common.size),
        "n_common": int(common.sum()),
        "n_excluded": int(excluded.size),
        "excluded_dataset_index": [int(i) for i in excluded],
        "excluded_by_model": per_model_excluded,
        "reason": "rows without a usable prediction from every model were dropped from all headline "
                  "metrics, CIs and pairwise tests so every system is scored on the same rows",
    }


def format_deviation_md(record: Dict[str, Any], timestamp_utc: str) -> str:
    """Markdown block appended to ``results/deviations.md`` for one variant's deviation."""
    lines = [
        f"## {timestamp_utc} - split `{record['split']}`, variant `{record['variant']}`: "
        f"metrics on {record['n_common']}/{record['n_total']} rows",
        "",
        f"Protocol: {record['protocol_section']}.",
        f"Reason: {record['reason']}.",
        f"Excluded dataset_index ({record['n_excluded']}): {record['excluded_dataset_index']}",
    ]
    for name, idx in record["excluded_by_model"].items():
        lines.append(f"- `{name}` had no usable prediction for {len(idx)} row(s): {idx}")
    lines.append("")
    return "\n".join(lines) + "\n"


def append_deviation(path: Path, record: Dict[str, Any], timestamp_utc: str) -> None:
    """Append one deviation block to ``path`` (created with a header on first use)."""
    header = "# Deviations from PROTOCOL.md\n\nRecorded automatically by benchmark.py; see §5.\n\n"
    text = format_deviation_md(record, timestamp_utc)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        if fh.tell() == 0:
            fh.write(header)
        fh.write(text)


def summarize_variant(df_v: pd.DataFrame, variant: str, models: Sequence[str], runs: Dict[str, ModelRun],
                      n_bootstrap: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """All §5/§6 numbers for one variant: per-model summaries, pairwise tests, CSV rows.

    Headline ``metrics`` (and their bootstrap CIs) and every pairwise test are computed on the rows
    scored by *all* present models (``common``), so per §5 every system is evaluated on identical
    rows and every bootstrap draws the same ``(n_bootstrap, n_common)`` index matrix. When a model
    has additional valid rows outside the common set its own-rows numbers are kept under
    ``metrics_own_rows`` for reference only. Latency, cost and token totals use each model's own
    successful calls (§6 is per system, not paired).
    """
    per_model: Dict[str, Any] = {}
    csv_rows: List[Dict[str, Any]] = []
    gold = df_v["gold_id"].to_numpy(dtype=np.int64)
    dataset_index = df_v["dataset_index"].to_numpy(dtype=np.int64)
    split = str(df_v["split"].iloc[0]) if "split" in df_v.columns and len(df_v) else None
    n_total = int(len(df_v))

    present = [name for name in models if f"{name}_pred" in df_v.columns]
    masks: Dict[str, np.ndarray] = {name: valid_mask(df_v, name) for name in present}
    common = common_mask(masks, n_total)
    n_common = int(common.sum())
    deviation = deviation_record(variant, split, masks, common, dataset_index)

    for name in present:
        mask = masks[name]
        n_valid = int(mask.sum())
        n_errors = n_total - n_valid
        latency = pd.to_numeric(df_v[f"{name}_latency_ms"], errors="coerce").to_numpy(dtype=float)[mask]
        retries = pd.to_numeric(df_v[f"{name}_retries"], errors="coerce").fillna(0).to_numpy(dtype=float)
        kind = LATENCY_KIND_REMOTE if name in REMOTE_MODELS else LATENCY_KIND_LOCAL
        entry: Dict[str, Any] = {
            "display_name": DISPLAY_NAMES[name],
            "n_total": n_total,
            "n_valid": n_valid,
            "n_common": n_common,
            "n_errors": n_errors,
            "n_retries": int(retries.sum()),
            "latency_kind": kind,
            "latency": latency_stats(latency),
        }
        run = runs.get(name)
        if run is not None and variant in run.wall_s:
            wall = run.wall_s[variant]
            entry["wall_time_s"] = wall
            entry["wall_examples_per_sec"] = (n_total / wall) if wall > 0 else None
        mismatch_col, api_conf_col = f"{name}_pred_mismatch", f"{name}_api_confidence"
        if mismatch_col in df_v.columns:
            entry["n_pred_mismatch"] = int(df_v[mismatch_col].fillna(False).astype(bool).sum())
        if api_conf_col in df_v.columns:
            api_conf = pd.to_numeric(df_v[api_conf_col], errors="coerce")
            entry["mean_api_confidence"] = float(api_conf.mean()) if api_conf.notna().any() else None
        if name == "jev":
            cost = pd.to_numeric(df_v["jev_cost_usd"], errors="coerce")
            entry["total_cost_usd"] = float(cost.sum(skipna=True))
            entry["total_input_tokens"] = int(pd.to_numeric(df_v["jev_input_tokens"], errors="coerce").sum(skipna=True))
            entry["total_output_tokens"] = int(pd.to_numeric(df_v["jev_output_tokens"], errors="coerce").sum(skipna=True))
            models_seen = df_v["jev_model"].dropna().unique().tolist() if "jev_model" in df_v.columns else []
            entry["api_model_strings"] = [str(m) for m in models_seen]
        else:
            entry["total_cost_usd"] = None

        prob_cols = [f"{name}_p_{label}" for label in LABELS]
        if n_common > 0:
            probs = df_v.loc[common, prob_cols].to_numpy(dtype=float)
            entry["metrics"] = metrics.summarize_model(gold[common], probs, dataset_index[common],
                                                       n_bootstrap=n_bootstrap, seed=SEED)
        else:
            entry["metrics"] = None
        entry["metrics_rows"] = "common" if n_common > 0 else None
        if n_valid > n_common:
            probs_own = df_v.loc[mask, prob_cols].to_numpy(dtype=float)
            entry["metrics_own_rows"] = metrics.summarize_model(gold[mask], probs_own, dataset_index[mask],
                                                                n_bootstrap=n_bootstrap, seed=SEED)
        else:
            entry["metrics_own_rows"] = None
        per_model[name] = entry

        m = entry["metrics"] or {}
        lat = entry["latency"]
        acc_cov = m.get("accuracy_at_coverage", {})
        csv_rows.append(
            {
                "model": DISPLAY_NAMES[name],
                "variant": variant,
                "split": split,
                "n": n_total,
                "n_valid": n_valid,
                "n_common": n_common,
                "accuracy": m.get("accuracy"),
                "acc_ci_lo": (m.get("accuracy_ci") or {}).get("low"),
                "acc_ci_hi": (m.get("accuracy_ci") or {}).get("high"),
                "macro_f1": m.get("macro_f1"),
                "f1_ci_lo": (m.get("macro_f1_ci") or {}).get("low"),
                "f1_ci_hi": (m.get("macro_f1_ci") or {}).get("high"),
                "brier": m.get("brier"),
                "ece15": m.get("ece_15"),
                "ece_adaptive10": m.get("ece_adaptive_10"),
                "nll": m.get("nll"),
                "mean_confidence": m.get("mean_confidence"),
                "acc_at_cov50": acc_cov.get("0.5"),
                "acc_at_cov80": acc_cov.get("0.8"),
                "acc_at_cov90": acc_cov.get("0.9"),
                "acc_at_cov100": acc_cov.get("1.0"),
                "frac_gold_prob_zero": m.get("frac_gold_prob_zero"),
                "p50_ms": lat["p50_ms"],
                "p95_ms": lat["p95_ms"],
                "mean_ms": lat["mean_ms"],
                "examples_per_sec": lat["examples_per_sec"],
                "total_cost_usd": entry["total_cost_usd"],
                "n_errors": n_errors,
                "n_retries": entry["n_retries"],
                "latency_kind": kind,
            }
        )

    # Pairwise tests on the same common rows as the single-model metrics, so the paired bootstrap
    # uses exactly the same (n_bootstrap, n_common) index matrix as every per-model CI (§5).
    pairwise: Dict[str, Any] = {}
    for a, b in combinations(present, 2):
        pair_key = f"{a}_vs_{b}"
        if n_common == 0:
            pairwise[pair_key] = {"n": 0, "error": "no rows scored by all models"}
            continue
        pred_a = pd.to_numeric(df_v[f"{a}_pred"], errors="coerce").to_numpy(dtype=float)[common].astype(np.int64)
        pred_b = pd.to_numeric(df_v[f"{b}_pred"], errors="coerce").to_numpy(dtype=float)[common].astype(np.int64)
        try:
            result = metrics.compare_models(gold[common], pred_a, pred_b, n_bootstrap=n_bootstrap, seed=SEED)
        except Exception as exc:  # noqa: BLE001 - a failed test must not abort the run
            result = {"error": f"{type(exc).__name__}: {exc}"}
        result["n"] = n_common
        result["a"] = a
        result["b"] = b
        pairwise[pair_key] = result

    return {
        "models": per_model,
        "pairwise": pairwise,
        "n_total": n_total,
        "n_common": n_common,
        "metrics_rows": "rows with a usable prediction from every present model",
        "deviation": deviation,
    }, csv_rows


# ------------------------------------------------------------------------------------------------
# env.json
# ------------------------------------------------------------------------------------------------


def build_env(args: argparse.Namespace, runs: Dict[str, ModelRun], seed_info: Dict[str, Any],
              n_rows: int, started_utc: str) -> Dict[str, Any]:
    load_infos = {name: run.load_info.to_dict() for name, run in runs.items() if run.load_info is not None}
    return {
        "timestamp_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "split": args.split,
        "limit": args.limit,
        "n_rows": n_rows,
        "models": list(runs),
        "variants": args.variants,
        "python": sys.version,
        "python_executable": sys.executable,
        "packages": {p: package_version(p) for p in PACKAGES_FOR_ENV},
        "hardware": hardware_info(),
        "dataset": {"id": DATASET_ID, "config": DATASET_CONFIG, "revision": DATASET_REVISION,
                    "warmup_split": "validation", "warmup_rows": WARMUP_N},
        "model_revisions": {"jev": JEV_MODEL_ID, "prismnli": PRISMNLI_REVISION, "laya": LAYA_REVISION},
        "seeds": seed_info,
        "per_model": {
            name: {
                "load_info": load_infos.get(name),
                "load_wall_s": run.load_wall_s,
                "device": run.load_info.device if run.load_info else None,
                "dtype": run.load_info.dtype if run.load_info else None,
                "peak_memory_bytes": run.peak_memory_bytes,
                "peak_memory_gib": (run.peak_memory_bytes / 2**30) if run.peak_memory_bytes else None,
                "http_calls": run.http_calls,
                "predict_wall_s": run.wall_s,
            }
            for name, run in runs.items()
        },
    }


# ------------------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen emotion-classification benchmark (PROTOCOL.md).")
    parser.add_argument("--split", choices=SPLITS, default="validation",
                        help="dataset split to score; 'test' is the real run, 'validation' is for smoke tests")
    parser.add_argument("--limit", type=int, default=None, help="score only the first N rows of the split")
    parser.add_argument("--models", type=lambda v: parse_csv_list(v, MODEL_ORDER, "model"),
                        default=list(MODEL_ORDER), help="comma-separated subset of jev,prismnli,laya")
    parser.add_argument("--variants", type=lambda v: parse_csv_list(v, VARIANTS, "variant"),
                        default=["plain"], help="comma-separated subset of plain,defined")
    parser.add_argument("--skip-plots", action="store_true", help="do not render figures")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR,
                        help="where raw_predictions.parquet, summary.*, env.json and plots/ are written")
    parser.add_argument("--n-bootstrap", type=int, default=metrics.BOOTSTRAP_N,
                        help="bootstrap resamples (protocol value 10000; lower only for quick checks)")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    started_utc = datetime.now(timezone.utc).isoformat()
    seed_info = seed_everything(SEED)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    models: List[str] = [m for m in MODEL_ORDER if m in args.models]
    variants: List[str] = [v for v in VARIANTS if v in args.variants]

    log(f"split={args.split} limit={args.limit} models={models} variants={variants} out_dir={out_dir}")
    rows = load_split(args.split, args.limit)
    warm_texts = load_split("validation", WARMUP_N)["text"].tolist()
    log(f"loaded {len(rows)} rows of {args.split!r} (revision {DATASET_REVISION[:8]}); "
        f"{len(warm_texts)} validation warm-up texts")

    runs: Dict[str, ModelRun] = {}
    for name in models:
        runs[name] = run_model(name, rows, warm_texts, variants, args.split)

    new_frame = assemble_frame(rows, args.split, list(runs.values()), variants)
    parquet_path = out_dir / "raw_predictions.parquet"
    df = merge_with_existing(new_frame, parquet_path, args.split)
    df.to_parquet(parquet_path, index=False)
    log(f"wrote {parquet_path} ({len(df)} rows x {len(df.columns)} columns)")

    env = build_env(args, runs, seed_info, len(rows), started_utc)
    (out_dir / "env.json").write_text(json.dumps(json_safe(env), indent=2))
    log(f"wrote {out_dir / 'env.json'}")

    # Metrics over every model x variant present in the (merged) frame, not only those just run.
    present_models = [m for m in MODEL_ORDER if f"{m}_pred" in df.columns]
    present_variants = [v for v in VARIANTS if (df["variant"] == v).any()]
    summary: Dict[str, Any] = {
        "protocol": "PROTOCOL.md v1 (frozen 2026-09-20)",
        "timestamp_utc": started_utc,
        "split": args.split,
        "n_rows": int(df["dataset_index"].nunique()),
        "models": present_models,
        "variants": present_variants,
        "n_bootstrap": args.n_bootstrap,
        "seed": SEED,
        "nll_eps": metrics.NLL_EPS,
        "latency_kinds": {m: (LATENCY_KIND_REMOTE if m in REMOTE_MODELS else LATENCY_KIND_LOCAL) for m in present_models},
        "per_variant": {},
    }
    csv_rows: List[Dict[str, Any]] = []
    for variant in present_variants:
        df_v = df[df["variant"] == variant].sort_values("dataset_index").reset_index(drop=True)
        per_variant, rows_v = summarize_variant(df_v, variant, present_models, runs, args.n_bootstrap)
        summary["per_variant"][variant] = per_variant
        csv_rows += rows_v
        for name, entry in per_variant["models"].items():
            m = entry["metrics"] or {}
            log(f"{variant:8s} {DISPLAY_NAMES[name]:14s} n_valid={entry['n_valid']:5d} "
                f"n_common={entry['n_common']:5d} errors={entry['n_errors']} "
                f"acc={m.get('accuracy', float('nan')):.4f} macroF1={m.get('macro_f1', float('nan')):.4f} "
                f"ece15={m.get('ece_15', float('nan')):.4f} p50={entry['latency']['p50_ms']}")
        deviation = per_variant["deviation"]
        if deviation is not None:
            log(f"DEVIATION (§5): variant={variant} metrics computed on {deviation['n_common']}/"
                f"{deviation['n_total']} common rows; excluded dataset_index={deviation['excluded_dataset_index']}")
            if args.split == "test":
                dev_path = out_dir / "deviations.md"
                append_deviation(dev_path, deviation, started_utc)
                log(f"appended deviation to {dev_path}; rerun to re-request the failed rows")

    (out_dir / "summary.json").write_text(json.dumps(json_safe(summary), indent=2))
    summary_csv = pd.DataFrame(csv_rows, columns=list(SUMMARY_CSV_COLUMNS))
    summary_csv.to_csv(out_dir / "summary.csv", index=False)
    log(f"wrote {out_dir / 'summary.json'} and {out_dir / 'summary.csv'}")

    if not args.skip_plots:
        for variant in present_variants:
            plot_dir = out_dir / "plots" if variant == "plain" else out_dir / "plots" / variant
            per_variant = summary["per_variant"][variant]
            summaries = {DISPLAY_NAMES[n]: e["metrics"] for n, e in per_variant["models"].items() if e["metrics"]}
            if not summaries:
                log(f"no scored rows for variant={variant}; skipping plots")
                continue
            prefixes = {DISPLAY_NAMES[n]: n for n in per_variant["models"]}
            df_v = df[df["variant"] == variant]
            if per_variant["deviation"] is not None:  # plot the same common rows as the metrics
                df_v = df_v[~df_v["dataset_index"].isin(per_variant["deviation"]["excluded_dataset_index"])]
            written = plots.make_all_plots(df_v, summaries, None, plot_dir, model_prefixes=prefixes, variant=variant)
            log(f"wrote {len(written)} figure files to {plot_dir}")

    total_errors = sum(e["n_errors"] for v in summary["per_variant"].values() for e in v["models"].values())
    log(f"done. total per-example errors across all models/variants: {total_errors}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
