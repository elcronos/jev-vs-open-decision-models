"""Publication-quality figures for the emotion-classification benchmark (PROTOCOL.md §7).

Every public plotting function writes one figure as PNG (200 dpi) **and** PDF into
``out_dir`` and returns the list of written paths. ``make_all_plots`` runs them all.

Inputs
------
``summaries`` : ``dict[str, dict]`` keyed by *model display name*. Each value is the
dict produced by ``metrics.summarize_model``. Because ``metrics.py`` is developed
concurrently, this module consumes exactly the following shape and reads every key
defensively with ``.get`` (a missing key degrades the corresponding figure gracefully
instead of raising)::

    {
      "accuracy":        float,
      "macro_f1":        float,
      "accuracy_ci":     [lo, hi] | {"low": lo, "high": hi, ...},   # bootstrap 95% CI
      "macro_f1_ci":     [lo, hi] | {"low": lo, "high": hi, ...},
      "ece" | "ece_15":  float,                   # optional; recomputed from bins if absent
      "per_class":       {label: {"precision": f, "recall": f, "f1": f, "support": int}}
                         | [{"label": str, "precision": f, "recall": f, "f1": f, "support": int}, ...],
      "confusion_matrix": [[int]*6]*6,            # rows = gold, cols = predicted, LABELS order
      "reliability_bins" | "reliability_15":
                         [{"lower": f, "upper": f, "count": int, "mean_conf": f, "acc": f}, ...],
      "risk_coverage":   {"coverage": [f, ...], "risk": [f, ...]},
    }

Both spellings are accepted so the native output of ``metrics.summarize_model`` (dict CIs,
list-form ``per_class``, ``reliability_15`` / ``ece_15``) can be passed through unchanged.

``df`` : the raw predictions DataFrame (``results/raw_predictions.parquet``) with columns
``gold_id``, ``variant`` and, per model prefix ``p`` in {``jev``, ``laya``, ``prismnli``},
``{p}_pred``, ``{p}_confidence``, ``{p}_latency_ms``, ``{p}_error``.

``latencies`` : ``dict[str, Sequence[float] | dict]`` keyed by display name; a value is
either a sequence of per-example latencies in ms, or ``{"values": [...], "remote": bool}``.
Jev is always labelled "remote API end-to-end"; the others "local compute latency"
(PROTOCOL.md §6: never compare the two as if equivalent).

Design notes
------------
* Colour is assigned to a model *once* (fixed order of the Okabe-Ito colour-blind-safe
  palette) and reused across every figure; a per-model hatch / marker / line style acts as
  secondary encoding so identity never relies on colour alone.
* Sans-serif fonts only, recessive grid, no top/right spines, tight layout.
* Only ``matplotlib`` + ``numpy`` (+ ``pandas`` for the DataFrame inputs) are required.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from models.common import LABELS  # noqa: E402

__all__ = [
    "SUMMARY_KEYS",
    "MODEL_PREFIXES",
    "model_colors",
    "plot_accuracy_f1_ci",
    "plot_per_class_f1",
    "plot_confusion_matrices",
    "plot_reliability",
    "plot_risk_coverage",
    "plot_confidence_hist",
    "plot_latency",
    "make_all_plots",
]

# --------------------------------------------------------------------------------------
# Contract with metrics.summarize_model (documented above; consumed via .get everywhere)
# --------------------------------------------------------------------------------------
SUMMARY_KEYS: Tuple[str, ...] = (
    "accuracy",
    "macro_f1",
    "accuracy_ci",
    "macro_f1_ci",
    "ece",
    "per_class",
    "confusion_matrix",
    "reliability_bins",
    "risk_coverage",
)

#: Column prefixes in the raw predictions DataFrame (PROTOCOL.md §7), keyed by a lowercase
#: token that is searched for in the display name to associate names with columns.
MODEL_PREFIXES: Dict[str, str] = {"jev": "jev", "laya": "laya", "prism": "prismnli"}

#: Display-name tokens (lowercase) that identify the remote system (PROTOCOL.md §3.1, §6).
REMOTE_TOKENS: Tuple[str, ...] = ("jev",)

# --------------------------------------------------------------------------------------
# Style
# --------------------------------------------------------------------------------------
#: Okabe-Ito palette, ordered so that adjacent pairs are maximally separable under
#: deutan/protan/tritan simulation (validated; first three slots serve the three systems).
OKABE_ITO: Tuple[str, ...] = ("#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#000000")
HATCHES: Tuple[str, ...] = ("", "///", "...", "xx", "\\\\\\", "++", "oo")
MARKERS: Tuple[str, ...] = ("o", "s", "D", "^", "v", "P", "X")
LINESTYLES: Tuple[str, ...] = ("-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1)))

INK = "#1f1f1f"
MUTED = "#6b6b6b"
GRID = "#e3e3e3"
SURFACE = "#ffffff"
SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list("blues_single_hue", ["#f7fbff", "#0072B2"])

PNG_DPI = 200

_RC: Dict[str, Any] = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans", "Liberation Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "legend.frameon": False,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "pdf.fonttype": 42,  # embed TrueType so text stays editable/searchable in the PDF
    "ps.fonttype": 42,
    "hatch.linewidth": 0.6,
}



def _fnum(x) -> float:
    """float() that maps None (JSON null from an empty bin) to NaN."""
    return float("nan") if x is None else float(x)

def model_colors(names: Iterable[str]) -> Dict[str, str]:
    """Assign one fixed palette colour per model display name (order of first appearance)."""
    out: Dict[str, str] = {}
    for i, n in enumerate(names):
        out[n] = OKABE_ITO[i % len(OKABE_ITO)]
    return out


def _style_index(names: Sequence[str], name: str) -> int:
    return list(names).index(name) % len(HATCHES)


def _is_remote(name: str, spec: Any = None) -> bool:
    if isinstance(spec, Mapping) and "remote" in spec:
        return bool(spec["remote"])
    lname = name.lower()
    return any(tok in lname for tok in REMOTE_TOKENS)


def _prefix_for(name: str, overrides: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """Map a display name to its DataFrame column prefix (explicit override wins)."""
    if overrides and name in overrides:
        return overrides[name]
    lname = name.lower()
    for token, prefix in MODEL_PREFIXES.items():
        if token in lname:
            return prefix
    return None


def _save(fig: plt.Figure, out_dir: Union[str, Path], stem: str) -> List[Path]:
    """Save ``fig`` as ``<stem>.png`` (200 dpi) and ``<stem>.pdf`` in ``out_dir``; close it."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f"{stem}.png", out / f"{stem}.pdf"]
    fig.savefig(paths[0], dpi=PNG_DPI, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def _ci(summary: Mapping[str, Any], key: str, point: Optional[float]) -> Tuple[float, float]:
    """Return (lower_err, upper_err) for an errorbar call; zeros if the CI is missing.

    Accepts either ``[lo, hi]`` or the ``metrics.bootstrap_ci`` dict ``{"low": lo, "high": hi, ...}``.
    """
    ci = summary.get(key)
    if point is None:
        return 0.0, 0.0
    if isinstance(ci, Mapping) and "low" in ci and "high" in ci:
        lo, hi = float(ci["low"]), float(ci["high"])
    elif isinstance(ci, (list, tuple)) and len(ci) == 2:
        lo, hi = float(ci[0]), float(ci[1])
    else:
        return 0.0, 0.0
    return max(point - lo, 0.0), max(hi - point, 0.0)


def _per_class_map(summary: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    """``per_class`` as ``{label: {...}}``; also accepts the ``metrics.per_class_report`` list form."""
    per_class = summary.get("per_class")
    if isinstance(per_class, Mapping):
        return dict(per_class)
    if isinstance(per_class, (list, tuple)):
        return {str(row.get("label")): row for row in per_class if isinstance(row, Mapping)}
    return {}


def _reliability_bins(summary: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    """Reliability table under ``reliability_bins`` or, failing that, ``reliability_15``."""
    bins = summary.get("reliability_bins")
    if bins is None:
        bins = summary.get("reliability_15")
    return [b for b in (bins or []) if isinstance(b, Mapping)]


def _ece_value(summary: Mapping[str, Any], bins: Sequence[Mapping[str, Any]]) -> Optional[float]:
    """ECE under ``ece`` or ``ece_15``; recomputed from the bins when neither is present."""
    for key in ("ece", "ece_15"):
        v = summary.get(key)
        if v is not None:
            return float(v)
    return _ece_from_bins(bins)


def _finite(values: Any) -> np.ndarray:
    arr = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float).ravel()
    return arr[np.isfinite(arr)]


def _ece_from_bins(bins: Sequence[Mapping[str, Any]]) -> Optional[float]:
    total = sum(int(b.get("count", 0) or 0) for b in bins)
    if total <= 0:
        return None
    ece = 0.0
    for b in bins:
        n = int(b.get("count", 0) or 0)
        if n == 0:
            continue
        acc = _fnum(b.get("acc"))
        conf = _fnum(b.get("mean_conf"))
        if math.isfinite(acc) and math.isfinite(conf):
            ece += (n / total) * abs(acc - conf)
    return ece


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------
def plot_accuracy_f1_ci(
    summaries: Mapping[str, Mapping[str, Any]], out_dir: Union[str, Path], stem: str = "accuracy_macro_f1_ci"
) -> List[Path]:
    """Accuracy and macro-F1 per model with bootstrap 95% CI error bars (two panels)."""
    names = list(summaries)
    colors = model_colors(names)
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=False)
        for ax, (metric, ci_key, title) in zip(
            axes,
            [("accuracy", "accuracy_ci", "Accuracy"), ("macro_f1", "macro_f1_ci", "Macro-F1")],
        ):
            y = np.arange(len(names))[::-1]
            for yi, n in zip(y, names):
                s = summaries[n]
                v = s.get(metric)
                if v is None or not math.isfinite(float(v)):
                    ax.text(0.5, yi, "n/a", ha="center", va="center", color=MUTED)
                    continue
                v = float(v)
                lo_e, hi_e = _ci(s, ci_key, v)
                ax.errorbar(
                    v, yi, xerr=[[lo_e], [hi_e]], fmt=MARKERS[_style_index(names, n)], color=colors[n],
                    ecolor=colors[n], elinewidth=1.4, capsize=3, markersize=6, zorder=3,
                )
                ax.text(v, yi + 0.28, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, color=INK)
            ax.set_yticks(y)
            ax.set_yticklabels(names)
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.6, len(names) - 0.4 + 0.3)
            ax.set_xlabel(f"{title} (95% bootstrap CI)")
            ax.set_title(title, loc="left")
            ax.grid(axis="y", visible=False)
        fig.tight_layout()
        return _save(fig, out_dir, stem)


def plot_per_class_f1(
    summaries: Mapping[str, Mapping[str, Any]], out_dir: Union[str, Path], stem: str = "per_class_f1"
) -> List[Path]:
    """Grouped bars: per-class F1 for every model, classes in canonical LABELS order."""
    names = list(summaries)
    colors = model_colors(names)
    k = max(len(names), 1)
    width = 0.8 / k
    x = np.arange(len(LABELS))
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(7.0, 3.2))
        for i, n in enumerate(names):
            per_class = _per_class_map(summaries[n])
            vals = []
            for lab in LABELS:
                f1 = (per_class.get(lab) or {}).get("f1")
                vals.append(float(f1) if f1 is not None else np.nan)
            offs = x - 0.4 + width * (i + 0.5)
            bars = ax.bar(
                offs, np.nan_to_num(vals, nan=0.0), width=width * 0.92, color=colors[n], label=n,
                hatch=HATCHES[_style_index(names, n)], edgecolor=SURFACE, linewidth=0.8, zorder=3,
            )
            for b, v in zip(bars, vals):
                if math.isfinite(v):
                    ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom",
                            fontsize=6.5, color=INK, rotation=90)
        ax.set_xticks(x)
        ax.set_xticklabels(LABELS)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("F1")
        ax.set_title("Per-class F1", loc="left")
        ax.grid(axis="x", visible=False)
        ax.legend(ncol=min(k, 3), loc="upper right")
        fig.tight_layout()
        return _save(fig, out_dir, stem)


def plot_confusion_matrices(
    summaries: Mapping[str, Mapping[str, Any]], out_dir: Union[str, Path], stem: str = "confusion_matrices"
) -> List[Path]:
    """One row-normalised confusion-matrix heatmap per model (shared 0..1 colour scale),
    raw counts annotated in each cell (rows = gold, columns = predicted)."""
    names = list(summaries)
    k = max(len(names), 1)
    n_lab = len(LABELS)
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(1, k, figsize=(3.1 * k + 0.8, 3.4), squeeze=False)
        axes = axes[0]
        im = None
        for ax, n in zip(axes, names):
            cm = summaries[n].get("confusion_matrix")
            ax.grid(False)
            ax.set_title(n, loc="left")
            if cm is None or np.asarray(cm).shape != (n_lab, n_lab):
                ax.text(0.5, 0.5, "confusion matrix\nunavailable", ha="center", va="center",
                        transform=ax.transAxes, color=MUTED)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            counts = np.asarray(cm, dtype=float)
            row_sums = counts.sum(axis=1, keepdims=True)
            norm = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums > 0)
            im = ax.imshow(norm, cmap=SEQUENTIAL_CMAP, vmin=0.0, vmax=1.0, aspect="equal")
            for i in range(n_lab):
                for j in range(n_lab):
                    ax.text(j, i, f"{int(round(counts[i, j]))}", ha="center", va="center", fontsize=6.5,
                            color=SURFACE if norm[i, j] > 0.55 else INK)
            ax.set_xticks(range(n_lab))
            ax.set_yticks(range(n_lab))
            ax.set_xticklabels(LABELS, rotation=45, ha="right")
            ax.set_yticklabels(LABELS if ax is axes[0] else [])
            ax.set_xlabel("predicted")
            if ax is axes[0]:
                ax.set_ylabel("gold")
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(length=0)
        if im is not None:
            cbar = fig.colorbar(im, ax=list(axes), fraction=0.025, pad=0.02)
            cbar.set_label("row-normalised fraction")
            cbar.outline.set_visible(False)
        return _save(fig, out_dir, stem)


def plot_reliability(
    summaries: Mapping[str, Mapping[str, Any]], out_dir: Union[str, Path], stem: str = "reliability"
) -> List[Path]:
    """Reliability diagram per model: accuracy per confidence bin as bars vs. the diagonal.
    Bar alpha encodes the bin count (also printed above each bar); ECE in the panel title."""
    names = list(summaries)
    colors = model_colors(names)
    k = max(len(names), 1)
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(1, k, figsize=(3.0 * k + 0.4, 3.2), squeeze=False, sharey=True)
        axes = axes[0]
        for ax, n in zip(axes, names):
            s = summaries[n]
            bins = _reliability_bins(s)
            ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1.0, linestyle="--", zorder=2, label="perfect calibration")
            ece = _ece_value(s, bins)
            title = n if ece is None else f"{n}  (ECE = {float(ece):.3f})"
            ax.set_title(title, loc="left")
            counts = np.array([int(b.get("count", 0) or 0) for b in bins], dtype=float)
            total = counts.sum()
            max_count = counts.max() if counts.size else 0.0
            for b, c in zip(bins, counts):
                lower, upper = float(b.get("lower", np.nan)), float(b.get("upper", np.nan))
                acc = _fnum(b.get("acc"))
                if not (math.isfinite(lower) and math.isfinite(upper)) or c == 0 or not math.isfinite(acc):
                    continue
                w = upper - lower
                alpha = 0.25 + 0.75 * (c / max_count if max_count > 0 else 1.0)
                ax.bar(lower + w / 2, acc, width=w * 0.94, color=colors[n], alpha=alpha,
                       edgecolor=colors[n], linewidth=0.6, zorder=3)
                ax.text(lower + w / 2, acc + 0.015, f"{int(c)}", ha="center", va="bottom", fontsize=5.5,
                        color=MUTED, rotation=90)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.12)
            ax.set_xlabel("confidence (max probability)")
            if ax is axes[0]:
                ax.set_ylabel("accuracy in bin")
            ax.grid(axis="x", visible=False)
            if total > 0:
                ax.text(0.02, 0.97, f"N = {int(total)}\nbar alpha scales with bin count", transform=ax.transAxes,
                        ha="left", va="top", fontsize=7, color=MUTED)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=1, bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        return _save(fig, out_dir, stem)


def plot_risk_coverage(
    summaries: Mapping[str, Mapping[str, Any]], out_dir: Union[str, Path], stem: str = "risk_coverage"
) -> List[Path]:
    """Selective-classification risk-coverage curves for all models on one axes
    (x = coverage, y = error rate among the covered examples)."""
    names = list(summaries)
    colors = model_colors(names)
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        any_curve = False
        end_labels = []
        for n in names:
            rc = summaries[n].get("risk_coverage") or {}
            cov = np.asarray(rc.get("coverage") or [], dtype=float)
            risk = np.asarray(rc.get("risk") or [], dtype=float)
            if cov.size == 0 or cov.size != risk.size:
                continue
            order = np.argsort(cov)
            i = _style_index(names, n)
            ax.plot(cov[order], risk[order], color=colors[n], linestyle=LINESTYLES[i], linewidth=1.8,
                    label=n, zorder=3)
            # direct label at the right end; stagger vertically when end points nearly coincide
            end_y = risk[order][-1]
            taken = [e for e in end_labels if abs(e - end_y) < 0.02]
            offset_pts = 6 * len(taken) if taken else 0
            end_labels.append(end_y + offset_pts * 0.004)
            ax.annotate(n, (cov[order][-1], end_y), xytext=(4, offset_pts - 3 * len(taken)),
                        textcoords="offset points", va="center", fontsize=7.5, color=INK)
            any_curve = True
        for c in (0.5, 0.8, 0.9):
            ax.axvline(c, color=GRID, linewidth=0.8, zorder=1)
        ax.set_xlim(0, 1.12)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("coverage (fraction of examples answered, most confident first)")
        ax.set_ylabel("error rate on covered examples")
        ax.set_title("Risk-coverage curve", loc="left")
        if any_curve:
            ax.legend(loc="upper left")
        fig.tight_layout()
        return _save(fig, out_dir, stem)


def plot_confidence_hist(
    df: pd.DataFrame,
    model_prefixes: Mapping[str, str],
    out_dir: Union[str, Path],
    variant: Optional[str] = "plain",
    stem: str = "confidence_hist",
    bins: int = 25,
) -> List[Path]:
    """Per model: histogram of confidence for correct vs. incorrect predictions, overlaid.

    ``model_prefixes`` maps display name -> DataFrame column prefix (e.g. ``{"Jev": "jev"}``).
    Rows with a non-null ``{prefix}_error`` are excluded. If ``variant`` is given and the
    DataFrame has a ``variant`` column, only that variant is used.
    """
    names = list(model_prefixes)
    colors = model_colors(names)
    k = max(len(names), 1)
    if variant is not None and "variant" in df.columns:
        df = df[df["variant"] == variant]
    edges = np.linspace(0.0, 1.0, bins + 1)
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(1, k, figsize=(3.0 * k + 0.4, 3.0), squeeze=False)
        axes = axes[0]
        for ax, n in zip(axes, names):
            p = model_prefixes[n]
            conf_col, pred_col, err_col = f"{p}_confidence", f"{p}_pred", f"{p}_error"
            if conf_col not in df.columns or pred_col not in df.columns or "gold_id" not in df.columns:
                ax.text(0.5, 0.5, "columns unavailable", ha="center", va="center", transform=ax.transAxes,
                        color=MUTED)
                ax.set_title(n, loc="left")
                continue
            sub = df
            if err_col in df.columns:
                sub = sub[sub[err_col].isna()]
            conf = pd.to_numeric(sub[conf_col], errors="coerce").to_numpy(dtype=float)
            correct = (sub[pred_col].to_numpy() == sub["gold_id"].to_numpy())
            ok = np.isfinite(conf)
            c_ok, c_bad = conf[ok & correct], conf[ok & ~correct]
            ax.hist(c_ok, bins=edges, color=colors[n], alpha=0.85, label=f"correct (n={c_ok.size})", zorder=3)
            ax.hist(c_bad, bins=edges, color=INK, alpha=0.45, hatch="///", edgecolor=INK, linewidth=0.4,
                    label=f"incorrect (n={c_bad.size})", zorder=4)
            ax.set_title(n, loc="left")
            ax.set_xlim(0, 1)
            ax.set_xlabel("confidence (max probability)")
            if ax is axes[0]:
                ax.set_ylabel("examples")
            ax.grid(axis="x", visible=False)
            ax.set_ylim(top=ax.get_ylim()[1] * 1.35)  # headroom so the legend never covers bars
            ax.legend(loc="upper left", ncol=2)
        fig.tight_layout()
        return _save(fig, out_dir, stem)


def plot_latency(
    latencies: Mapping[str, Union[Sequence[float], Mapping[str, Any]]],
    out_dir: Union[str, Path],
    stem: str = "latency",
) -> List[Path]:
    """Per-example latency per model (horizontal box plots, log-scale x) with p50/p95 annotated.

    Remote systems (Jev) are labelled "remote API end-to-end" and drawn hatched; local systems
    are labelled "local compute latency" (PROTOCOL.md §6). The two are not comparable and the
    figure says so.
    """
    names = list(latencies)
    colors = model_colors(names)
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(7.0, 0.9 * max(len(names), 1) + 1.6))
        y_positions = np.arange(len(names))[::-1]
        data: List[np.ndarray] = []
        labels: List[str] = []
        for n in names:
            spec = latencies[n]
            values = spec.get("values", []) if isinstance(spec, Mapping) else spec
            arr = _finite(values)
            arr = arr[arr > 0]  # log axis
            data.append(arr)
            kind = "remote API end-to-end" if _is_remote(n, spec) else "local compute latency"
            labels.append(f"{n}\n({kind})")
        bp = ax.boxplot(
            [d if d.size else np.array([np.nan]) for d in data], positions=y_positions, orientation="horizontal",
            widths=0.55, showfliers=True, patch_artist=True, whis=(5, 95),
            flierprops={"marker": ".", "markersize": 2, "alpha": 0.35, "markeredgecolor": MUTED},
            medianprops={"color": INK, "linewidth": 1.4},
            whiskerprops={"color": MUTED}, capprops={"color": MUTED},
        )
        for patch, n in zip(bp["boxes"], names):
            spec = latencies[n]
            patch.set_facecolor(colors[n])
            patch.set_alpha(0.75)
            patch.set_edgecolor(INK if _is_remote(n, spec) else colors[n])
            patch.set_hatch("///" if _is_remote(n, spec) else "")
        x_max = max((d.max() for d in data if d.size), default=1.0)
        for yi, n, d in zip(y_positions, names, data):
            if d.size == 0:
                ax.text(1.0, yi, "no data", va="center", color=MUTED)
                continue
            p50, p95 = np.percentile(d, 50), np.percentile(d, 95)
            ax.text(x_max * 1.6, yi, f"p50 = {p50:,.0f} ms   p95 = {p95:,.0f} ms   n = {d.size}",
                    va="center", ha="left", fontsize=7.5, color=INK)
        ax.set_xscale("log")
        ax.set_xlim(right=x_max * 30)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels)
        ax.set_xlabel("per-example latency, ms (log scale; whiskers 5–95th percentile)")
        ax.set_title("Latency — remote end-to-end vs. local compute are NOT comparable", loc="left")
        ax.grid(axis="y", visible=False)
        handles = [
            Patch(facecolor="#bbbbbb", edgecolor=INK, hatch="///", label="remote API end-to-end (network + queue + compute)"),
            Patch(facecolor="#bbbbbb", edgecolor="#bbbbbb", label="local compute latency (batch size 1, warm)"),
        ]
        ax.legend(handles=handles, loc="lower right", fontsize=7)
        fig.tight_layout()
        return _save(fig, out_dir, stem)


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------
def make_all_plots(
    df: Optional[pd.DataFrame],
    summaries: Mapping[str, Mapping[str, Any]],
    latencies: Optional[Mapping[str, Union[Sequence[float], Mapping[str, Any]]]],
    out_dir: Union[str, Path],
    model_prefixes: Optional[Mapping[str, str]] = None,
    variant: Optional[str] = "plain",
) -> List[Path]:
    """Produce every figure and return the list of written files.

    ``model_prefixes`` maps display name -> DataFrame column prefix; when omitted it is
    inferred from the display names via ``MODEL_PREFIXES`` (``"Jev"`` -> ``"jev"``, ...).
    ``latencies`` may be ``None``, in which case per-example latencies are pulled from
    ``df["{prefix}_latency_ms"]`` for the chosen ``variant``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    written += plot_accuracy_f1_ci(summaries, out_dir)
    written += plot_per_class_f1(summaries, out_dir)
    written += plot_confusion_matrices(summaries, out_dir)
    written += plot_reliability(summaries, out_dir)
    written += plot_risk_coverage(summaries, out_dir)

    prefixes: Dict[str, str] = {}
    for n in summaries:
        p = _prefix_for(n, model_prefixes)
        if p is not None:
            prefixes[n] = p

    if df is not None and prefixes:
        written += plot_confidence_hist(df, prefixes, out_dir, variant=variant)
        if latencies is None:
            sub = df[df["variant"] == variant] if (variant is not None and "variant" in df.columns) else df
            latencies = {}
            for n, p in prefixes.items():
                col = f"{p}_latency_ms"
                if col in sub.columns:
                    latencies[n] = {"values": _finite(pd.to_numeric(sub[col], errors="coerce")),
                                    "remote": _is_remote(n)}
    if latencies:
        written += plot_latency(latencies, out_dir)
    return written


# --------------------------------------------------------------------------------------
# Self-test on synthetic data (no real model or dataset involved)
# --------------------------------------------------------------------------------------
def _synthetic_summary(gold: np.ndarray, probs: np.ndarray) -> Dict[str, Any]:
    """Minimal stand-in for metrics.summarize_model producing the documented shape."""
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    correct = pred == gold
    n_lab = len(LABELS)
    cm = np.zeros((n_lab, n_lab), dtype=int)
    for g, p in zip(gold, pred):
        cm[g, p] += 1
    per_class: Dict[str, Dict[str, float]] = {}
    f1s = []
    for i, lab in enumerate(LABELS):
        tp = cm[i, i]
        prec = tp / cm[:, i].sum() if cm[:, i].sum() else 0.0
        rec = tp / cm[i].sum() if cm[i].sum() else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
        per_class[lab] = {"precision": prec, "recall": rec, "f1": f1, "support": int(cm[i].sum())}
    acc = float(correct.mean())
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(gold), size=(500, len(gold)))
    boot = correct[idx].mean(axis=1)
    edges = np.linspace(0, 1, 16)
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & ((conf < hi) if hi < 1 else (conf <= hi))
        bins.append({"lower": float(lo), "upper": float(hi), "count": int(m.sum()),
                     "mean_conf": float(conf[m].mean()) if m.any() else float("nan"),
                     "acc": float(correct[m].mean()) if m.any() else float("nan")})
    order = np.lexsort((np.arange(len(gold)), -conf))
    cum_err = np.cumsum(~correct[order])
    coverage = np.arange(1, len(gold) + 1) / len(gold)
    risk = cum_err / np.arange(1, len(gold) + 1)
    return {
        "accuracy": acc,
        "macro_f1": float(np.mean(f1s)),
        "accuracy_ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "macro_f1_ci": [float(np.mean(f1s)) - 0.02, float(np.mean(f1s)) + 0.02],
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "reliability_bins": bins,
        "risk_coverage": {"coverage": coverage.tolist(), "risk": risk.tolist()},
    }


def _selftest(out_dir: Union[str, Path], n: int = 2000) -> List[Path]:
    """Build N synthetic rows for three fake models, run make_all_plots, assert every file exists."""
    rng = np.random.default_rng(0)
    n_lab = len(LABELS)
    gold = rng.integers(0, n_lab, size=n)
    fake = {"Jev (fake)": ("jev", 2.2, 900.0), "Laya (fake)": ("laya", 1.4, 12.0), "PrismNLI-0.4B (fake)": ("prismnli", 1.0, 45.0)}
    df = pd.DataFrame({"dataset_index": np.arange(n), "variant": "plain", "gold_id": gold,
                       "gold_label": [LABELS[g] for g in gold]})
    summaries: Dict[str, Any] = {}
    for name, (prefix, sharpness, lat_scale) in fake.items():
        logits = rng.normal(size=(n, n_lab))
        logits[np.arange(n), gold] += sharpness
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        summaries[name] = _synthetic_summary(gold, probs)
        df[f"{prefix}_pred"] = probs.argmax(axis=1)
        for i, lab in enumerate(LABELS):
            df[f"{prefix}_p_{lab}"] = probs[:, i]
        df[f"{prefix}_confidence"] = probs.max(axis=1)
        df[f"{prefix}_latency_ms"] = rng.lognormal(mean=np.log(lat_scale), sigma=0.35, size=n)
        df[f"{prefix}_error"] = None
        df[f"{prefix}_retries"] = 0
    written = make_all_plots(df, summaries, None, out_dir)
    expected_stems = ["accuracy_macro_f1_ci", "per_class_f1", "confusion_matrices", "reliability",
                      "risk_coverage", "confidence_hist", "latency"]
    for stem in expected_stems:
        for ext in ("png", "pdf"):
            path = Path(out_dir) / f"{stem}.{ext}"
            if not path.exists() or path.stat().st_size == 0:
                raise AssertionError(f"missing or empty figure: {path}")
    return written


if __name__ == "__main__":
    import sys

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/plots/_selftest")
    for p in _selftest(target):
        print(f"{p}  {p.stat().st_size:,} bytes")
    print(f"OK: {len(list(Path(target).iterdir()))} files in {target}")
