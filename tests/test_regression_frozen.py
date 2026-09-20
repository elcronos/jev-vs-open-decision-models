"""REGRESSION: the refactored (dynamic-K) metrics must reproduce the frozen primary results exactly.

Reads ``results/frozen_primary_plain/raw_predictions.parquet`` and compares recomputed accuracy,
macro-F1, Brier, ECE-15, NLL and accuracy-at-coverage (plus the bootstrap CIs, adaptive ECE and
confusion matrices) with ``results/frozen_primary_plain/summary.json`` to 1e-9. Nothing is written."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import benchmark as B
import metrics as M
from models.common import LABELS

FROZEN = Path(__file__).resolve().parents[1] / "results" / "frozen_primary_plain"
TOL = 1e-9
SCALARS = ("accuracy", "macro_f1", "brier", "ece_15", "nll", "ece_adaptive_10", "mean_confidence", "frac_gold_prob_zero")


@pytest.fixture(scope="module")
def frozen():
    if not FROZEN.exists():
        pytest.skip("frozen primary snapshot not present")
    df = pd.read_parquet(FROZEN / "raw_predictions.parquet")
    summary = json.loads((FROZEN / "summary.json").read_text())
    return df, summary


def test_frozen_snapshot_shape(frozen) -> None:
    df, summary = frozen
    assert len(df) == 2000 and set(df["variant"]) == {"plain"} and set(df["split"]) == {"test"}
    assert summary["split"] == "test" and summary["per_variant"]["plain"]["n_common"] == 2000


@pytest.mark.parametrize("model", ["jev", "prismnli", "laya"])
def test_metrics_reproduce_frozen_summary(frozen, model: str) -> None:
    df, summary = frozen
    ref = summary["per_variant"]["plain"]["models"][model]["metrics"]
    gold = df["gold_id"].to_numpy(dtype=np.int64)
    di = df["dataset_index"].to_numpy(dtype=np.int64)
    probs = df[[f"{model}_p_{l}" for l in LABELS]].to_numpy(dtype=float)
    assert B.valid_mask(df, model, LABELS).all()

    got = M.summarize_model(gold, probs, di, n_bootstrap=summary["n_bootstrap"], seed=summary["seed"])
    assert got["n"] == ref["n"] == 2000 and got["n_classes"] == 6
    for key in SCALARS:
        assert abs(got[key] - ref[key]) <= TOL, key
    for cov, value in ref["accuracy_at_coverage"].items():
        assert abs(got["accuracy_at_coverage"][cov] - value) <= TOL, cov
    for ci in ("accuracy_ci", "macro_f1_ci"):
        for kk in ("point", "low", "high"):
            assert abs(got[ci][kk] - ref[ci][kk]) <= TOL, (ci, kk)
    assert got["confusion_matrix"] == ref["confusion_matrix"]
    assert [r["label"] for r in got["per_class"]] == LABELS
    for g, r in zip(got["per_class"], ref["per_class"]):
        assert abs(g["f1"] - r["f1"]) <= TOL and g["support"] == r["support"]
    # the standalone functions agree with the summary
    y_pred = M.predictions(probs)
    assert abs(M.accuracy(gold, y_pred) - ref["accuracy"]) <= TOL
    assert abs(M.macro_f1(gold, y_pred) - ref["macro_f1"]) <= TOL
    assert abs(M.brier_multiclass(gold, probs) - ref["brier"]) <= TOL
    assert abs(M.ece_equal_width(gold, probs)[0] - ref["ece_15"]) <= TOL
    assert abs(M.nll(gold, probs) - ref["nll"]) <= TOL
    assert abs(M.majority_class_accuracy(gold) - 0.3475) <= TOL  # joy = 695/2000


def test_pairwise_reproduce_frozen_summary(frozen) -> None:
    df, summary = frozen
    gold = df["gold_id"].to_numpy(dtype=np.int64)
    for key, ref in summary["per_variant"]["plain"]["pairwise"].items():
        a, b = ref["a"], ref["b"]
        got = M.compare_models(gold, df[f"{a}_pred"].to_numpy(), df[f"{b}_pred"].to_numpy(),
                               n_bootstrap=summary["n_bootstrap"], seed=summary["seed"], n_classes=6)
        assert got["mcnemar_exact"]["b"] == ref["mcnemar_exact"]["b"]
        assert abs(got["mcnemar_exact"]["pvalue"] - ref["mcnemar_exact"]["pvalue"]) <= TOL
        for kk in ("point", "low", "high", "p_value"):
            assert abs(got["paired_bootstrap_accuracy_diff"][kk] - ref["paired_bootstrap_accuracy_diff"][kk]) <= TOL


def test_summarize_variant_reproduces_frozen_csv(frozen) -> None:
    """benchmark.summarize_variant on the frozen frame reproduces summary.csv (default six labels)."""
    df, summary = frozen
    csv = pd.read_csv(FROZEN / "summary.csv")
    per_variant, rows = B.summarize_variant(df, "plain", ["jev", "prismnli", "laya"], {}, summary["n_bootstrap"])
    assert per_variant["n_classes"] == 6
    for row in rows:
        ref = csv[csv["model"] == row["model"]].iloc[0]
        for col in ("accuracy", "macro_f1", "brier", "ece15", "nll", "acc_at_cov50", "acc_ci_lo", "f1_ci_hi"):
            assert abs(row[col] - float(ref[col])) <= TOL, (row["model"], col)
        assert row["n_classes"] == 6 and abs(row["majority_class_accuracy"] - 0.3475) <= TOL
