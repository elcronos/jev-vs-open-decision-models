"""metrics.py with 7 and 20 classes (PROTOCOL_ADDENDUM_v2.md §5) plus majority_class_accuracy."""
from __future__ import annotations

import math

import numpy as np
import pytest
from sklearn.metrics import f1_score

import metrics as M


def _onehot(y: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros((y.shape[0], k))
    out[np.arange(y.shape[0]), y] = 1.0
    return out


@pytest.mark.parametrize("k", [7, 20])
def test_perfect_predictor(k: int) -> None:
    y = np.tile(np.arange(k), 3)
    probs = _onehot(y, k)
    y_pred = M.predictions(probs)
    assert M.accuracy(y, y_pred, n_classes=k) == 1.0
    assert M.macro_f1(y, y_pred, n_classes=k) == 1.0
    assert M.nll(y, probs) == pytest.approx(0.0, abs=1e-12)
    assert M.brier_multiclass(y, probs) == 0.0
    ece, table = M.ece_equal_width(y, probs)
    assert ece == 0.0 and len(table) == 15 and table[-1]["count"] == y.shape[0]
    assert M.ece_adaptive(y, probs)[0] == 0.0
    assert M.mean_confidence(probs) == 1.0
    assert M.frac_gold_prob_zero(probs, y) == 0.0
    cm = M.confusion_matrix(y, y_pred, n_classes=k)
    assert cm.shape == (k, k) and np.trace(cm) == y.shape[0]
    rep = M.per_class_report(y, y_pred, n_classes=k)
    assert len(rep) == k and all(r["f1"] == 1.0 and r["support"] == 3 for r in rep)
    assert [r["label"] for r in rep] == [str(i) for i in range(k)]  # generic names beyond six classes
    assert M.majority_class_accuracy(y, k) == pytest.approx(1 / k)
    assert M.accuracy_at_coverage(y, probs, np.arange(y.shape[0]))["1.0"] == 1.0


@pytest.mark.parametrize("k", [7, 20])
def test_uniform_predictor(k: int) -> None:
    y = np.tile(np.arange(k), 5)
    n = y.shape[0]
    probs = np.full((n, k), 1.0 / k)
    assert M.nll(y, probs) == pytest.approx(math.log(k), rel=1e-12)
    assert M.brier_multiclass(y, probs) == pytest.approx(1.0 - 1.0 / k, rel=1e-12)
    assert M.mean_confidence(probs) == pytest.approx(1.0 / k)
    y_pred = M.predictions(probs)
    assert (y_pred == 0).all()  # argmax of a uniform row is label 0
    assert M.accuracy(y, y_pred, n_classes=k) == pytest.approx(1 / k)
    # only class 0 is ever predicted: F1_0 = 2*(1/k)*1/((1/k)+1), others 0
    p0 = 1 / k
    assert M.macro_f1(y, y_pred, n_classes=k) == pytest.approx((2 * p0 / (p0 + 1)) / k, rel=1e-12)
    ece, _ = M.ece_equal_width(y, probs)
    assert ece == pytest.approx(abs(1 / k - 1 / k), abs=1e-12)  # acc = conf = 1/k -> perfectly calibrated
    s = M.summarize_model(y, probs, np.arange(n), n_bootstrap=50, seed=0)
    assert s["n_classes"] == k and len(s["confusion_matrix"]) == k and len(s["per_class"]) == k
    assert s["majority_class_accuracy"] == pytest.approx(1 / k) and s["majority_class_id"] == 0


@pytest.mark.parametrize("k", [7, 20])
def test_macro_f1_matches_sklearn_k_classes(k: int) -> None:
    rng = np.random.default_rng(k)
    y = rng.integers(0, k, size=500)
    p = rng.integers(0, k, size=500)
    p[p == k - 1] = 0  # last label never predicted
    ref = f1_score(y, p, average="macro", labels=list(range(k)), zero_division=0)
    assert M.macro_f1(y, p, n_classes=k) == pytest.approx(ref, abs=1e-12)
    # with the default six classes, ids >= 6 are rejected
    with pytest.raises(ValueError):
        M.macro_f1(y, p)


def test_majority_class_accuracy() -> None:
    y = np.array([0, 0, 0, 1, 1, 2, 5, 5, 5, 5])
    assert M.majority_class_accuracy(y) == 0.4 and M.majority_class_id(y) == 5
    assert M.majority_class_accuracy(np.array([3, 3, 6]), n_classes=7) == pytest.approx(2 / 3)
    assert M.majority_class_id(np.array([1, 2, 1, 2]), n_classes=3) == 1  # ties -> lowest id
    with pytest.raises(ValueError):
        M.majority_class_accuracy(np.array([0, 6]))


def test_label_names_in_summary() -> None:
    k = 7
    y = np.tile(np.arange(k), 2)
    probs = _onehot(y, k)
    names = ["no emotion", "anger", "disgust", "fear", "happiness", "sadness", "surprise"]
    s = M.summarize_model(y, probs, np.arange(y.shape[0]), n_bootstrap=20, label_names=names)
    assert [r["label"] for r in s["per_class"]] == names
    with pytest.raises(ValueError):
        M.per_class_report(y, y, n_classes=k, label_names=names[:-1])


def test_compare_models_k_classes() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 20, size=300)
    a = y.copy()
    a[rng.random(300) < 0.3] = 19
    b = y.copy()
    b[rng.random(300) < 0.5] = 19
    res = M.compare_models(y, a, b, n_bootstrap=200, seed=0, n_classes=20)
    assert res["mcnemar_exact"]["b"] > res["mcnemar_exact"]["c"]
    assert res["paired_bootstrap_accuracy_diff"]["point"] > 0
