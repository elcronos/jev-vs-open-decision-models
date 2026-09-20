"""Evaluation metrics implementing PROTOCOL.md §5 (frozen).

All functions are pure and operate on numpy arrays:

* ``y_true``  : int array of shape (N,), gold label ids in canonical ``LABELS`` order (0..5).
* ``probs``   : float array of shape (N, 6), one *renormalised* categorical distribution per row
                (``Prediction.probs``). Predictions are ``argmax(probs, axis=1)``.
* ``dataset_index`` : int array of shape (N,), row position in the evaluated split; used only as
                a deterministic tie-breaker when sorting by confidence.

Conventions (documented here because they affect numbers):

* NLL epsilon is ``1e-6``: ``-mean(log(clip(p_gold, 1e-6, 1)))``. No renormalisation after clipping.
* Multiclass Brier is ``mean(sum_k (p_k - onehot_k)^2)``, range ``[0, 2]``.
* Confidence is ``max(probs, axis=1)``.
* Equal-width ECE uses 15 bins on ``[0, 1]`` with edges ``k/15``; a confidence ``c`` falls in bin
  ``b`` when ``edge[b] < c <= edge[b+1]``, except ``c == 0`` which goes in bin 0 (so every value in
  ``[0, 1]`` is binned exactly once). ECE ``= sum_b (n_b / N) * |acc_b - conf_b|`` over non-empty bins.
* Adaptive ECE uses 10 equal-mass bins: rows sorted by confidence (ascending, ties by
  ``dataset_index``) and split into 10 contiguous groups of (near-)equal size via
  ``numpy.array_split``.
* Selective classification sorts by confidence descending, ties broken by ascending
  ``dataset_index``. Accuracy at coverage ``c`` uses the top ``ceil(c * N)`` rows.
* Bootstrap resamples use ``numpy.random.default_rng(seed)`` and ``rng.integers(0, N, size=(n, N))``;
  the paired bootstrap reuses exactly this index matrix for both models.

Nothing in this module reads the dataset, so it can never touch the test split on its own.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import confusion_matrix as _sk_confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from statsmodels.stats.contingency_tables import mcnemar as _sm_mcnemar

from models.common import LABELS, SEED

N_CLASSES: int = len(LABELS)
NLL_EPS: float = 1e-6
ECE_EQUAL_WIDTH_BINS: int = 15
ECE_ADAPTIVE_BINS: int = 10
COVERAGE_LEVELS: Tuple[float, ...] = (0.5, 0.8, 0.9, 1.0)
BOOTSTRAP_N: int = 10_000

MetricFn = Callable[[np.ndarray, np.ndarray], float]


# --------------------------------------------------------------------------------------------
# Input validation helpers
# --------------------------------------------------------------------------------------------


def _as_labels(y: Any, name: str = "y") -> np.ndarray:
    """Coerce to a 1-D int64 array of label ids and validate the range."""
    arr = np.asarray(y)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.issubdtype(arr.dtype, np.integer):
        if not np.all(np.equal(np.mod(arr, 1), 0)):
            raise ValueError(f"{name} must contain integer label ids")
        arr = arr.astype(np.int64)
    arr = arr.astype(np.int64, copy=False)
    if arr.min() < 0 or arr.max() >= N_CLASSES:
        raise ValueError(f"{name} must be in [0, {N_CLASSES - 1}]")
    return arr


def _as_probs(probs: Any, n: Optional[int] = None) -> np.ndarray:
    """Coerce to a float64 (N, 6) array and validate shape."""
    arr = np.asarray(probs, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != N_CLASSES:
        raise ValueError(f"probs must have shape (N, {N_CLASSES}), got {arr.shape}")
    if n is not None and arr.shape[0] != n:
        raise ValueError(f"probs has {arr.shape[0]} rows but y_true has {n}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("probs must be finite")
    return arr


def _check_same_length(*arrays: np.ndarray) -> None:
    lengths = {a.shape[0] for a in arrays}
    if len(lengths) != 1:
        raise ValueError(f"arrays must have the same length, got {sorted(lengths)}")


def predictions(probs: Any) -> np.ndarray:
    """Argmax prediction per row (first index wins ties, matching ``numpy.argmax``)."""
    return np.argmax(_as_probs(probs), axis=1).astype(np.int64)


# --------------------------------------------------------------------------------------------
# Classification metrics
# --------------------------------------------------------------------------------------------


def accuracy(y_true: Any, y_pred: Any) -> float:
    """Fraction of rows where ``y_pred == y_true``."""
    yt, yp = _as_labels(y_true, "y_true"), _as_labels(y_pred, "y_pred")
    _check_same_length(yt, yp)
    return float(np.mean(yt == yp))


def macro_f1(y_true: Any, y_pred: Any) -> float:
    """Unweighted mean of per-class F1 over all six labels.

    Classes with zero predicted *and* zero gold occurrences contribute F1 = 0 (sklearn's
    ``zero_division=0`` convention), so results match
    ``sklearn.metrics.f1_score(average="macro", labels=range(6), zero_division=0)``.
    Implemented with ``numpy.bincount`` so it is fast enough for 10 000 bootstrap resamples.
    """
    yt, yp = _as_labels(y_true, "y_true"), _as_labels(y_pred, "y_pred")
    _check_same_length(yt, yp)
    tp = np.bincount(yt[yt == yp], minlength=N_CLASSES).astype(np.float64)
    pred_count = np.bincount(yp, minlength=N_CLASSES).astype(np.float64)
    gold_count = np.bincount(yt, minlength=N_CLASSES).astype(np.float64)
    denom = pred_count + gold_count  # 2*TP + FP + FN
    f1 = np.divide(2.0 * tp, denom, out=np.zeros(N_CLASSES), where=denom > 0)
    return float(np.mean(f1))


def per_class_report(y_true: Any, y_pred: Any) -> List[Dict[str, Any]]:
    """Per-label precision / recall / F1 / support (gold count), in canonical label order."""
    yt, yp = _as_labels(y_true, "y_true"), _as_labels(y_pred, "y_pred")
    _check_same_length(yt, yp)
    p, r, f, s = precision_recall_fscore_support(
        yt, yp, labels=list(range(N_CLASSES)), zero_division=0
    )
    return [
        {
            "label_id": i,
            "label": LABELS[i],
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f[i]),
            "support": int(s[i]),
        }
        for i in range(N_CLASSES)
    ]


def confusion_matrix(y_true: Any, y_pred: Any) -> np.ndarray:
    """6x6 confusion matrix, rows = gold label, columns = predicted label."""
    yt, yp = _as_labels(y_true, "y_true"), _as_labels(y_pred, "y_pred")
    _check_same_length(yt, yp)
    return _sk_confusion_matrix(yt, yp, labels=list(range(N_CLASSES))).astype(np.int64)


# --------------------------------------------------------------------------------------------
# Probabilistic / calibration metrics
# --------------------------------------------------------------------------------------------


def gold_probs(y_true: Any, probs: Any) -> np.ndarray:
    """Probability assigned to the gold label, shape (N,)."""
    yt = _as_labels(y_true, "y_true")
    pr = _as_probs(probs, yt.shape[0])
    return pr[np.arange(yt.shape[0]), yt]


def nll(y_true: Any, probs: Any, eps: float = NLL_EPS) -> float:
    """Negative log-likelihood ``-mean(log(clip(p_gold, eps, 1)))``.

    ``eps`` defaults to ``1e-6`` per PROTOCOL.md §5. The clip is applied to ``p_gold`` only and
    the distribution is *not* renormalised afterwards.
    """
    pg = gold_probs(y_true, probs)
    return float(-np.mean(np.log(np.clip(pg, eps, 1.0))))


def brier_multiclass(y_true: Any, probs: Any) -> float:
    """Multiclass Brier score ``mean(sum_k (p_k - onehot_k)^2)``; range ``[0, 2]``."""
    yt = _as_labels(y_true, "y_true")
    pr = _as_probs(probs, yt.shape[0])
    onehot = np.zeros_like(pr)
    onehot[np.arange(yt.shape[0]), yt] = 1.0
    return float(np.mean(np.sum((pr - onehot) ** 2, axis=1)))


def confidences(probs: Any) -> np.ndarray:
    """Confidence = ``max(probs, axis=1)``."""
    return np.max(_as_probs(probs), axis=1)


def mean_confidence(probs: Any) -> float:
    """Mean of the per-row maximum probability."""
    return float(np.mean(confidences(probs)))


def _bin_table(
    conf: np.ndarray, correct: np.ndarray, bin_ids: np.ndarray, lowers: np.ndarray, uppers: np.ndarray
) -> Tuple[float, List[Dict[str, Any]]]:
    """Shared reliability-table builder. ``bin_ids[i]`` is the bin of row ``i``."""
    n = conf.shape[0]
    n_bins = lowers.shape[0]
    counts = np.bincount(bin_ids, minlength=n_bins).astype(np.int64)
    conf_sum = np.bincount(bin_ids, weights=conf, minlength=n_bins)
    acc_sum = np.bincount(bin_ids, weights=correct.astype(np.float64), minlength=n_bins)
    ece = 0.0
    table: List[Dict[str, Any]] = []
    for b in range(n_bins):
        c = int(counts[b])
        if c > 0:
            mean_conf = float(conf_sum[b] / c)
            acc = float(acc_sum[b] / c)
            ece += (c / n) * abs(acc - mean_conf)
        else:
            mean_conf, acc = float("nan"), float("nan")
        table.append(
            {
                "bin": b,
                "lower": float(lowers[b]),
                "upper": float(uppers[b]),
                "count": c,
                "mean_conf": mean_conf,
                "acc": acc,
            }
        )
    return float(ece), table


def ece_equal_width(
    y_true: Any, probs: Any, n_bins: int = ECE_EQUAL_WIDTH_BINS
) -> Tuple[float, List[Dict[str, Any]]]:
    """Expected calibration error with ``n_bins`` equal-width bins on ``[0, 1]``.

    Bin ``b`` covers ``(b/n_bins, (b+1)/n_bins]``; confidence exactly 0 is placed in bin 0.
    Returns ``(ece, table)`` where ``table`` has one dict per bin (including empty bins, whose
    ``mean_conf``/``acc`` are NaN) with keys ``bin, lower, upper, count, mean_conf, acc``.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    yt = _as_labels(y_true, "y_true")
    pr = _as_probs(probs, yt.shape[0])
    conf = np.max(pr, axis=1)
    if conf.min() < 0.0 or conf.max() > 1.0:
        raise ValueError("confidences must lie in [0, 1]")
    correct = np.argmax(pr, axis=1) == yt
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # right=True: bins[i-1] < x <= bins[i]; with interior edges only, x == 0 -> 0, x == 1 -> n_bins-1.
    bin_ids = np.digitize(conf, edges[1:-1], right=True)
    return _bin_table(conf, correct, bin_ids, edges[:-1], edges[1:])


def ece_adaptive(
    y_true: Any,
    probs: Any,
    n_bins: int = ECE_ADAPTIVE_BINS,
    dataset_index: Optional[Any] = None,
) -> Tuple[float, List[Dict[str, Any]]]:
    """Adaptive (equal-mass) ECE with ``n_bins`` bins.

    Rows are sorted by confidence ascending (ties by ``dataset_index`` if given, else by row
    position) and split into ``n_bins`` contiguous groups with ``numpy.array_split`` (sizes differ by
    at most one). Each bin's ``lower``/``upper`` are the min/max confidence it contains.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    yt = _as_labels(y_true, "y_true")
    pr = _as_probs(probs, yt.shape[0])
    n = yt.shape[0]
    conf = np.max(pr, axis=1)
    correct = np.argmax(pr, axis=1) == yt
    tiebreak = np.arange(n) if dataset_index is None else np.asarray(dataset_index)
    _check_same_length(yt, tiebreak)
    order = np.lexsort((tiebreak, conf))  # primary key conf asc, secondary tiebreak asc
    bin_ids = np.empty(n, dtype=np.int64)
    lowers = np.full(n_bins, np.nan)
    uppers = np.full(n_bins, np.nan)
    for b, idx in enumerate(np.array_split(order, n_bins)):
        bin_ids[idx] = b
        if idx.size > 0:
            lowers[b] = conf[idx].min()
            uppers[b] = conf[idx].max()
    return _bin_table(conf, correct, bin_ids, lowers, uppers)


# --------------------------------------------------------------------------------------------
# Selective classification
# --------------------------------------------------------------------------------------------


def selective_order(probs: Any, dataset_index: Any) -> np.ndarray:
    """Row order for selective classification: confidence descending, ties by ``dataset_index`` asc."""
    conf = confidences(probs)
    di = np.asarray(dataset_index)
    _check_same_length(conf, di)
    return np.lexsort((di, -conf))


def risk_coverage_curve(
    y_true: Any, probs: Any, dataset_index: Any
) -> Tuple[np.ndarray, np.ndarray]:
    """Risk-coverage curve.

    Returns ``(coverage, risk)``, each of shape (N,): after accepting the ``k`` most confident rows
    (``k = 1..N``), ``coverage[k-1] = k / N`` and ``risk[k-1]`` = error rate among those ``k`` rows.
    Ordering is confidence descending with ``dataset_index`` ascending as tie-breaker.
    """
    yt = _as_labels(y_true, "y_true")
    pr = _as_probs(probs, yt.shape[0])
    order = selective_order(pr, dataset_index)
    wrong = (np.argmax(pr, axis=1) != yt)[order].astype(np.float64)
    k = np.arange(1, yt.shape[0] + 1, dtype=np.float64)
    coverage = k / yt.shape[0]
    risk = np.cumsum(wrong) / k
    return coverage, risk


def accuracy_at_coverage(
    y_true: Any,
    probs: Any,
    dataset_index: Any,
    coverages: Sequence[float] = COVERAGE_LEVELS,
) -> Dict[str, float]:
    """Accuracy on the ``ceil(c * N)`` most confident rows for each coverage ``c``.

    Keys are the coverage levels formatted as strings (e.g. ``"0.5"``) so the result is JSON-safe.
    """
    yt = _as_labels(y_true, "y_true")
    pr = _as_probs(probs, yt.shape[0])
    n = yt.shape[0]
    order = selective_order(pr, dataset_index)
    correct = (np.argmax(pr, axis=1) == yt)[order]
    out: Dict[str, float] = {}
    for c in coverages:
        if not 0.0 < c <= 1.0:
            raise ValueError(f"coverage must be in (0, 1], got {c}")
        k = max(1, int(math.ceil(c * n - 1e-12)))
        out[str(c)] = float(np.mean(correct[:k]))
    return out


# --------------------------------------------------------------------------------------------
# Uncertainty: bootstrap and paired tests
# --------------------------------------------------------------------------------------------


def _bootstrap_indices(n_rows: int, n: int, seed: int) -> np.ndarray:
    """The (n, n_rows) resample index matrix shared by all bootstrap functions."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_rows, size=(n, n_rows))


def bootstrap_ci(
    metric_fn: MetricFn,
    y_true: Any,
    y_pred: Any,
    n: int = BOOTSTRAP_N,
    seed: int = SEED,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Percentile bootstrap CI for ``metric_fn(y_true, y_pred)``.

    Returns ``{"point", "low", "high", "n", "seed", "alpha"}`` where ``low``/``high`` are the
    ``100*alpha/2`` and ``100*(1-alpha/2)`` percentiles of the resampled metric.
    """
    yt, yp = _as_labels(y_true, "y_true"), _as_labels(y_pred, "y_pred")
    _check_same_length(yt, yp)
    idx = _bootstrap_indices(yt.shape[0], n, seed)
    stats = np.fromiter((metric_fn(yt[i], yp[i]) for i in idx), dtype=np.float64, count=n)
    low, high = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "point": float(metric_fn(yt, yp)),
        "low": float(low),
        "high": float(high),
        "n": int(n),
        "seed": int(seed),
        "alpha": float(alpha),
    }


def paired_bootstrap_accuracy_diff(
    y_true: Any,
    pred_a: Any,
    pred_b: Any,
    n: int = BOOTSTRAP_N,
    seed: int = SEED,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Paired bootstrap of ``accuracy(a) - accuracy(b)`` using the SAME resample indices for both.

    Returns point difference, percentile CI, and ``p_value``: the two-sided bootstrap
    p-value ``2 * min(P(diff <= 0), P(diff >= 0))`` clipped to ``[0, 1]``.
    """
    yt = _as_labels(y_true, "y_true")
    pa, pb = _as_labels(pred_a, "pred_a"), _as_labels(pred_b, "pred_b")
    _check_same_length(yt, pa, pb)
    ca = (pa == yt).astype(np.float64)
    cb = (pb == yt).astype(np.float64)
    idx = _bootstrap_indices(yt.shape[0], n, seed)
    diffs = ca[idx].mean(axis=1) - cb[idx].mean(axis=1)
    low, high = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p_le = float(np.mean(diffs <= 0.0))
    p_ge = float(np.mean(diffs >= 0.0))
    return {
        "point": float(ca.mean() - cb.mean()),
        "low": float(low),
        "high": float(high),
        "p_value": float(min(1.0, 2.0 * min(p_le, p_ge))),
        "n": int(n),
        "seed": int(seed),
        "alpha": float(alpha),
    }


def mcnemar_exact(y_true: Any, pred_a: Any, pred_b: Any) -> Dict[str, Any]:
    """Exact McNemar test on correct/incorrect indicators of two models.

    Contingency table ``[[both correct, a correct & b wrong], [a wrong & b correct, both wrong]]``
    is passed to ``statsmodels.stats.contingency_tables.mcnemar(exact=True)``.
    Returns ``{"statistic", "pvalue", "b", "c"}`` with ``b`` = #(a correct, b wrong) and
    ``c`` = #(a wrong, b correct). The exact statistic is ``min(b, c)``.
    """
    yt = _as_labels(y_true, "y_true")
    pa, pb = _as_labels(pred_a, "pred_a"), _as_labels(pred_b, "pred_b")
    _check_same_length(yt, pa, pb)
    ca, cb = pa == yt, pb == yt
    both = int(np.sum(ca & cb))
    b = int(np.sum(ca & ~cb))
    c = int(np.sum(~ca & cb))
    neither = int(np.sum(~ca & ~cb))
    table = np.array([[both, b], [c, neither]], dtype=np.int64)
    res = _sm_mcnemar(table, exact=True)
    return {
        "statistic": float(res.statistic),
        "pvalue": float(res.pvalue),
        "b": b,
        "c": c,
        "both_correct": both,
        "both_wrong": neither,
    }


def frac_gold_prob_zero(probs: Any, y_true: Any) -> float:
    """Fraction of rows where the (renormalised) probability of the gold label is exactly 0.0."""
    return float(np.mean(gold_probs(y_true, probs) == 0.0))


# --------------------------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------------------------


def summarize_model(
    y_true: Any,
    probs: Any,
    dataset_index: Any,
    n_bootstrap: int = BOOTSTRAP_N,
    seed: int = SEED,
) -> Dict[str, Any]:
    """All PROTOCOL.md §5 single-model metrics as a JSON-serialisable dict.

    Scalar keys: ``n, accuracy, macro_f1, nll, nll_eps, brier, ece_15, ece_adaptive_10,
    mean_confidence, frac_gold_prob_zero``. Tables: ``per_class`` (list of dicts),
    ``confusion_matrix`` (6x6 nested list, rows gold), ``reliability_15``/``reliability_adaptive_10``
    (bin tables), ``accuracy_at_coverage`` (dict), ``risk_coverage`` (``{"coverage": [...], "risk": [...]}``),
    ``accuracy_ci``/``macro_f1_ci`` (bootstrap dicts).
    """
    yt = _as_labels(y_true, "y_true")
    pr = _as_probs(probs, yt.shape[0])
    di = np.asarray(dataset_index)
    _check_same_length(yt, di)
    yp = np.argmax(pr, axis=1)

    ece15, table15 = ece_equal_width(yt, pr, ECE_EQUAL_WIDTH_BINS)
    ece_ad, table_ad = ece_adaptive(yt, pr, ECE_ADAPTIVE_BINS, dataset_index=di)
    coverage, risk = risk_coverage_curve(yt, pr, di)

    return {
        "n": int(yt.shape[0]),
        "accuracy": accuracy(yt, yp),
        "macro_f1": macro_f1(yt, yp),
        "nll": nll(yt, pr),
        "nll_eps": NLL_EPS,
        "brier": brier_multiclass(yt, pr),
        "ece_15": ece15,
        "ece_adaptive_10": ece_ad,
        "mean_confidence": mean_confidence(pr),
        "frac_gold_prob_zero": frac_gold_prob_zero(pr, yt),
        "accuracy_ci": bootstrap_ci(accuracy, yt, yp, n=n_bootstrap, seed=seed),
        "macro_f1_ci": bootstrap_ci(macro_f1, yt, yp, n=n_bootstrap, seed=seed),
        "per_class": per_class_report(yt, yp),
        "confusion_matrix": confusion_matrix(yt, yp).tolist(),
        "reliability_15": table15,
        "reliability_adaptive_10": table_ad,
        "accuracy_at_coverage": accuracy_at_coverage(yt, pr, di, COVERAGE_LEVELS),
        "risk_coverage": {"coverage": coverage.tolist(), "risk": risk.tolist()},
    }


def compare_models(y_true: Any, pred_a: Any, pred_b: Any, n_bootstrap: int = BOOTSTRAP_N, seed: int = SEED) -> Dict[str, Any]:
    """Both PROTOCOL.md §5 paired tests for one model pair, JSON-serialisable."""
    return {
        "mcnemar_exact": mcnemar_exact(y_true, pred_a, pred_b),
        "paired_bootstrap_accuracy_diff": paired_bootstrap_accuracy_diff(
            y_true, pred_a, pred_b, n=n_bootstrap, seed=seed
        ),
    }


__all__ = [
    "N_CLASSES",
    "NLL_EPS",
    "ECE_EQUAL_WIDTH_BINS",
    "ECE_ADAPTIVE_BINS",
    "COVERAGE_LEVELS",
    "BOOTSTRAP_N",
    "predictions",
    "accuracy",
    "macro_f1",
    "per_class_report",
    "confusion_matrix",
    "gold_probs",
    "nll",
    "brier_multiclass",
    "confidences",
    "mean_confidence",
    "ece_equal_width",
    "ece_adaptive",
    "selective_order",
    "risk_coverage_curve",
    "accuracy_at_coverage",
    "bootstrap_ci",
    "paired_bootstrap_accuracy_diff",
    "mcnemar_exact",
    "frac_gold_prob_zero",
    "summarize_model",
    "compare_models",
]
