"""Unit tests for metrics.py (PROTOCOL.md §5). Synthetic arrays only; no dataset access."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
from sklearn.metrics import f1_score

import metrics as M

K = M.N_CLASSES


def _onehot(y: np.ndarray) -> np.ndarray:
    out = np.zeros((y.shape[0], K))
    out[np.arange(y.shape[0]), y] = 1.0
    return out


@pytest.fixture
def y_true() -> np.ndarray:
    return np.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5], dtype=np.int64)


# ------------------------------------------------------------------ perfect / uniform predictors


def test_perfect_predictor(y_true: np.ndarray) -> None:
    probs = _onehot(y_true)
    y_pred = M.predictions(probs)
    assert M.accuracy(y_true, y_pred) == 1.0
    assert M.macro_f1(y_true, y_pred) == 1.0
    assert M.nll(y_true, probs) == pytest.approx(0.0, abs=1e-12)
    assert M.brier_multiclass(y_true, probs) == 0.0
    ece, table = M.ece_equal_width(y_true, probs)
    assert ece == 0.0
    assert len(table) == 15
    assert table[-1]["count"] == y_true.shape[0]  # confidence 1.0 lands in the last bin
    ece_ad, _ = M.ece_adaptive(y_true, probs)
    assert ece_ad == 0.0
    assert M.mean_confidence(probs) == 1.0
    assert M.frac_gold_prob_zero(probs, y_true) == 0.0


def test_uniform_predictor(y_true: np.ndarray) -> None:
    probs = np.full((y_true.shape[0], K), 1.0 / K)
    assert M.nll(y_true, probs) == pytest.approx(math.log(K), rel=1e-12)
    assert M.brier_multiclass(y_true, probs) == pytest.approx(1.0 - 1.0 / K, rel=1e-12)
    assert M.brier_multiclass(y_true, probs) == pytest.approx(5.0 / 6.0, rel=1e-12)
    assert M.mean_confidence(probs) == pytest.approx(1.0 / K)
    # argmax of a uniform row is label 0 -> accuracy equals the base rate of label 0
    assert M.accuracy(y_true, M.predictions(probs)) == pytest.approx(2 / 12)


def test_nll_epsilon_clip() -> None:
    y = np.array([0])
    probs = np.array([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
    assert M.nll(y, probs) == pytest.approx(-math.log(1e-6))
    assert M.nll(y, probs, eps=1e-3) == pytest.approx(-math.log(1e-3))
    assert M.frac_gold_prob_zero(probs, y) == 1.0


# ------------------------------------------------------------------ ECE


def test_ece_equal_width_hand_computed() -> None:
    # Two rows with confidence 0.9 (bin 13: (13/15, 14/15]), one correct one wrong -> |0.5 - 0.9| = 0.4
    # Two rows with confidence 0.5 (bin 7: (7/15, 8/15]), both correct           -> |1.0 - 0.5| = 0.5
    # ECE = (2/4)*0.4 + (2/4)*0.5 = 0.45
    y = np.array([0, 0, 1, 1])
    probs = np.array(
        [
            [0.9, 0.1, 0.0, 0.0, 0.0, 0.0],  # pred 0, correct
            [0.1, 0.9, 0.0, 0.0, 0.0, 0.0],  # pred 1, wrong
            [0.1, 0.5, 0.1, 0.1, 0.1, 0.1],  # pred 1, correct
            [0.2, 0.5, 0.3, 0.0, 0.0, 0.0],  # pred 1, correct
        ]
    )
    ece, table = M.ece_equal_width(y, probs, n_bins=15)
    assert ece == pytest.approx(0.45, abs=1e-12)
    assert sum(row["count"] for row in table) == 4
    b13, b7 = table[13], table[7]
    assert b13["count"] == 2 and b13["acc"] == pytest.approx(0.5) and b13["mean_conf"] == pytest.approx(0.9)
    assert b7["count"] == 2 and b7["acc"] == pytest.approx(1.0) and b7["mean_conf"] == pytest.approx(0.5)
    assert b13["lower"] == pytest.approx(13 / 15) and b13["upper"] == pytest.approx(14 / 15)
    # empty bins are reported with NaN stats
    assert table[0]["count"] == 0 and math.isnan(table[0]["acc"])


def test_ece_bin_edges_inclusive_right() -> None:
    # confidence exactly on an edge k/15 goes to the lower bin (edge[b] < c <= edge[b+1])
    y = np.array([0, 0])
    probs = np.array([[0.4, 0.3, 0.3, 0.0, 0.0, 0.0]] * 2)  # 0.4 = 6/15 exactly -> bin 5, not 6
    _, table = M.ece_equal_width(y, probs)
    assert table[5]["count"] == 2
    assert table[6]["count"] == 0


def test_ece_adaptive_equal_mass() -> None:
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(100, K))
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    y = rng.integers(0, K, size=100)
    ece, table = M.ece_adaptive(y, probs, n_bins=10)
    assert len(table) == 10
    assert all(row["count"] == 10 for row in table)
    assert 0.0 <= ece <= 1.0
    # bins are ordered by confidence
    uppers = [row["upper"] for row in table]
    lowers = [row["lower"] for row in table]
    assert all(lowers[i + 1] >= uppers[i] for i in range(9))


# ------------------------------------------------------------------ classification tables


def test_macro_f1_matches_sklearn() -> None:
    rng = np.random.default_rng(2)
    y = rng.integers(0, K, size=200)
    p = rng.integers(0, K, size=200)
    p[y == 5] = 0  # make one label never predicted to exercise zero_division
    ref = f1_score(y, p, average="macro", labels=list(range(K)), zero_division=0)
    assert M.macro_f1(y, p) == pytest.approx(ref, abs=1e-12)


def test_confusion_and_per_class() -> None:
    y = np.array([0, 0, 1, 2])
    p = np.array([0, 1, 1, 2])
    cm = M.confusion_matrix(y, p)
    assert cm.shape == (K, K)
    assert cm[0, 0] == 1 and cm[0, 1] == 1 and cm[1, 1] == 1 and cm[2, 2] == 1
    assert cm.sum() == 4
    rep = M.per_class_report(y, p)
    assert [r["label"] for r in rep] == M.LABELS
    assert rep[0]["precision"] == 1.0 and rep[0]["recall"] == 0.5 and rep[0]["support"] == 2
    assert rep[1]["precision"] == 0.5 and rep[1]["recall"] == 1.0
    assert rep[5]["support"] == 0 and rep[5]["f1"] == 0.0


# ------------------------------------------------------------------ selective classification


def test_risk_coverage_and_accuracy_at_coverage() -> None:
    y = np.array([0, 0, 0, 0])
    # confidences: 0.9 (correct), 0.8 (wrong), 0.7 (correct), 0.6 (wrong)
    probs = np.array(
        [
            [0.9, 0.1, 0, 0, 0, 0],
            [0.2, 0.8, 0, 0, 0, 0],
            [0.7, 0.3, 0, 0, 0, 0],
            [0.4, 0.6, 0, 0, 0, 0],
        ]
    )
    di = np.arange(4)
    cov, risk = M.risk_coverage_curve(y, probs, di)
    assert cov.tolist() == [0.25, 0.5, 0.75, 1.0]
    assert risk.tolist() == pytest.approx([0.0, 0.5, 1 / 3, 0.5])
    acc = M.accuracy_at_coverage(y, probs, di, [0.5, 0.8, 0.9, 1.0])
    assert acc["0.5"] == 0.5  # top 2
    assert acc["0.8"] == pytest.approx(2 / 4)  # ceil(3.2) = 4
    assert acc["0.9"] == pytest.approx(2 / 4)
    assert acc["1.0"] == 0.5


def test_selective_tiebreak_by_dataset_index() -> None:
    y = np.array([0, 0])
    probs = np.array([[0.9, 0.1, 0, 0, 0, 0], [0.1, 0.9, 0, 0, 0, 0]])  # equal confidence
    # row 0 correct, row 1 wrong. With dataset_index [5, 3], row 1 (index 3) comes first.
    order = M.selective_order(probs, np.array([5, 3]))
    assert order.tolist() == [1, 0]
    _, risk = M.risk_coverage_curve(y, probs, np.array([5, 3]))
    assert risk[0] == 1.0
    _, risk2 = M.risk_coverage_curve(y, probs, np.array([3, 5]))
    assert risk2[0] == 0.0


# ------------------------------------------------------------------ McNemar


def test_mcnemar_counts_and_pvalue() -> None:
    y = np.zeros(10, dtype=np.int64)
    # a correct on rows 0..6, b correct on rows 0..3 and 7..8
    pred_a = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
    pred_b = np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 1])
    res = M.mcnemar_exact(y, pred_a, pred_b)
    assert res["b"] == 3  # a correct, b wrong: rows 4,5,6
    assert res["c"] == 2  # a wrong, b correct: rows 7,8
    assert res["both_correct"] == 4 and res["both_wrong"] == 1
    assert res["statistic"] == 2.0  # min(b, c)
    # exact two-sided binomial p-value with n=5, k=2, p=0.5
    from scipy.stats import binomtest

    assert res["pvalue"] == pytest.approx(binomtest(2, 5, 0.5).pvalue, rel=1e-9)
    assert res["pvalue"] == pytest.approx(1.0)


# ------------------------------------------------------------------ bootstrap


def test_bootstrap_ci_contains_point_and_is_reproducible() -> None:
    rng = np.random.default_rng(3)
    y = rng.integers(0, K, size=300)
    p = y.copy()
    flip = rng.random(300) < 0.3
    p[flip] = (p[flip] + 1) % K
    r1 = M.bootstrap_ci(M.accuracy, y, p, n=2000, seed=0)
    r2 = M.bootstrap_ci(M.accuracy, y, p, n=2000, seed=0)
    assert r1 == r2
    assert r1["low"] <= r1["point"] <= r1["high"]
    assert r1["low"] < r1["high"]
    assert r1["point"] == pytest.approx(M.accuracy(y, p))
    r3 = M.bootstrap_ci(M.accuracy, y, p, n=2000, seed=1)
    assert (r3["low"], r3["high"]) != (r1["low"], r1["high"])
    f = M.bootstrap_ci(M.macro_f1, y, p, n=500, seed=0)
    assert f["low"] <= f["point"] <= f["high"]


def test_paired_bootstrap_uses_same_indices() -> None:
    rng = np.random.default_rng(4)
    y = rng.integers(0, K, size=200)
    pa = y.copy()
    pb = y.copy()
    pa[rng.random(200) < 0.2] = 5
    pb[rng.random(200) < 0.4] = 5
    res = M.paired_bootstrap_accuracy_diff(y, pa, pb, n=2000, seed=0)
    assert res["point"] == pytest.approx(M.accuracy(y, pa) - M.accuracy(y, pb))
    assert res["low"] <= res["point"] <= res["high"]
    assert 0.0 <= res["p_value"] <= 1.0
    # identical models -> zero difference in every resample
    same = M.paired_bootstrap_accuracy_diff(y, pa, pa, n=200, seed=0)
    assert same["point"] == 0.0 and same["low"] == 0.0 and same["high"] == 0.0
    # reproducible
    assert res == M.paired_bootstrap_accuracy_diff(y, pa, pb, n=2000, seed=0)


# ------------------------------------------------------------------ summary


def test_summarize_model_is_json_serialisable() -> None:
    rng = np.random.default_rng(5)
    n = 60
    y = rng.integers(0, K, size=n)
    logits = rng.normal(size=(n, K))
    logits[np.arange(n), y] += 1.5
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    di = np.arange(n)
    s = M.summarize_model(y, probs, di, n_bootstrap=300, seed=0)
    text = json.dumps(s)  # raises if not serialisable (NaN allowed by default encoder)
    assert isinstance(text, str)
    for key in [
        "n", "accuracy", "macro_f1", "nll", "nll_eps", "brier", "ece_15", "ece_adaptive_10",
        "mean_confidence", "frac_gold_prob_zero", "accuracy_ci", "macro_f1_ci", "per_class",
        "confusion_matrix", "reliability_15", "reliability_adaptive_10", "accuracy_at_coverage",
        "risk_coverage",
    ]:
        assert key in s
    assert s["n"] == n
    assert len(s["confusion_matrix"]) == K and sum(map(sum, s["confusion_matrix"])) == n
    assert len(s["reliability_15"]) == 15 and len(s["reliability_adaptive_10"]) == 10
    assert set(s["accuracy_at_coverage"]) == {"0.5", "0.8", "0.9", "1.0"}
    assert s["accuracy_at_coverage"]["1.0"] == pytest.approx(s["accuracy"])
    assert len(s["risk_coverage"]["coverage"]) == n
    assert s["nll_eps"] == 1e-6


def test_input_validation() -> None:
    with pytest.raises(ValueError):
        M.accuracy(np.array([0, 1]), np.array([0]))
    # class count is dynamic (probs.shape[1]): 5 columns are fine, but the gold id must fit ...
    assert M.nll(np.array([0]), np.full((1, 5), 0.2)) == pytest.approx(-math.log(0.2))
    with pytest.raises(ValueError):
        M.nll(np.array([5]), np.ones((1, 5)))
    with pytest.raises(ValueError):  # ... and a single column is not a categorical distribution
        M.nll(np.array([0]), np.ones((1, 1)))
    with pytest.raises(ValueError):  # label-only functions default to the six primary classes
        M.accuracy(np.array([0, 6]), np.array([0, 0]))
    assert M.accuracy(np.array([0, 6]), np.array([0, 6]), n_classes=7) == 1.0
