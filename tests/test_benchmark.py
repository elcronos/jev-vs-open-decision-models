"""Tests for benchmark.summarize_variant: PROTOCOL.md §5 identical-rows guarantee. Synthetic frame only."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import benchmark as B
from models.common import LABELS

N_BOOT = 200


def _frame(jev_bad: List[int]) -> pd.DataFrame:
    """Six validation-like rows; Jev has an error row and a pred=-1 row, PrismNLI scores everything."""
    gold = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
    df = pd.DataFrame(
        {
            "dataset_index": np.arange(6, dtype=np.int64),
            "variant": "plain",
            "split": "validation",
            "text": [f"t{i}" for i in range(6)],
            "gold_id": gold,
            "gold_label": [LABELS[g] for g in gold],
        }
    )
    for name, preds in (("jev", [0, 1, 2, 3, 5, 5]), ("prismnli", [0, 1, 2, 3, 4, 5])):
        probs = np.full((6, 6), 0.02)
        probs[np.arange(6), preds] = 0.9
        df[f"{name}_pred"] = preds
        for k, label in enumerate(LABELS):
            df[f"{name}_p_{label}"] = probs[:, k]
        df[f"{name}_confidence"] = 0.9
        df[f"{name}_latency_ms"] = 10.0
        df[f"{name}_error"] = None
        df[f"{name}_retries"] = 0
    # Jev-specific §7 columns that summarize_variant reads for cost/token totals.
    df["jev_cost_usd"] = 1e-6
    df["jev_input_tokens"] = 12.0
    df["jev_output_tokens"] = 1.0
    df["jev_model"] = "typesafe/jev-1.13-20260917"
    # One hard error and one pred=-1 row for Jev.
    df.loc[jev_bad[0], "jev_error"] = "HTTP 502: upstream"
    df.loc[jev_bad[0], [f"jev_p_{l}" for l in LABELS]] = math.nan
    df.loc[jev_bad[0], "jev_pred"] = -1
    df.loc[jev_bad[1], "jev_pred"] = -1
    return df


def test_headline_metrics_use_identical_rows() -> None:
    df = _frame(jev_bad=[1, 4])
    per_variant, csv_rows = B.summarize_variant(df, "plain", ["jev", "prismnli", "laya"], {}, N_BOOT)
    models: Dict[str, dict] = per_variant["models"]
    assert set(models) == {"jev", "prismnli"}  # laya absent from the frame

    assert per_variant["n_common"] == 4
    for name in ("jev", "prismnli"):
        assert models[name]["n_common"] == 4
        assert models[name]["metrics"]["n"] == 4  # identical row set for every model
    assert models["jev"]["n_valid"] == 4 and models["jev"]["n_errors"] == 2
    assert models["prismnli"]["n_valid"] == 6 and models["prismnli"]["n_errors"] == 0

    # Same rows -> same CI resample matrix; all remaining Jev/Prism predictions are correct.
    assert models["jev"]["metrics"]["accuracy"] == 1.0
    assert models["prismnli"]["metrics"]["accuracy"] == 1.0
    assert models["prismnli"]["metrics_own_rows"]["n"] == 6  # reference block on own rows
    assert models["jev"]["metrics_own_rows"] is None  # own rows == common rows

    pair = per_variant["pairwise"]["jev_vs_prismnli"]
    assert pair["n"] == 4

    dev = per_variant["deviation"]
    assert dev is not None
    assert dev["excluded_dataset_index"] == [1, 4]
    assert dev["excluded_by_model"] == {"jev": [1, 4]}
    assert dev["n_common"] == 4 and dev["n_total"] == 6

    by_model = {r["model"]: r for r in csv_rows}
    assert by_model["Jev"]["n_common"] == by_model["PrismNLI-0.4B"]["n_common"] == 4
    assert "n_common" in B.SUMMARY_CSV_COLUMNS


def test_no_errors_means_no_deviation() -> None:
    df = _frame(jev_bad=[1, 4])
    df["jev_error"] = None
    df["jev_pred"] = [0, 1, 2, 3, 5, 5]
    for label in LABELS:
        df[f"jev_p_{label}"] = df[f"prismnli_p_{label}"]
    per_variant, _ = B.summarize_variant(df, "plain", ["jev", "prismnli"], {}, N_BOOT)
    assert per_variant["deviation"] is None
    assert per_variant["n_common"] == 6
    assert all(e["metrics_own_rows"] is None for e in per_variant["models"].values())


def test_append_deviation_writes_markdown(tmp_path: Path) -> None:
    df = _frame(jev_bad=[1, 4])
    per_variant, _ = B.summarize_variant(df, "plain", ["jev", "prismnli"], {}, N_BOOT)
    path = tmp_path / "deviations.md"
    B.append_deviation(path, per_variant["deviation"], "2026-09-20T00:00:00+00:00")
    B.append_deviation(path, per_variant["deviation"], "2026-09-20T00:00:01+00:00")
    text = path.read_text()
    assert text.startswith("# Deviations from PROTOCOL.md")
    assert text.count("## 2026-09-20") == 2
    assert "metrics on 4/6 rows" in text
    assert "`jev` had no usable prediction for 2 row(s): [1, 4]" in text
