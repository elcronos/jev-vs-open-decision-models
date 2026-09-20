"""plots.py must render every figure for 6, 7 and 20 classes (synthetic data, no models)."""
from __future__ import annotations

from pathlib import Path

import pytest

import plots
from datasets_registry import DAILY_DIALOG, EMOTION, FIN_TOPIC

STEMS = ["accuracy_macro_f1_ci", "per_class_f1", "confusion_matrices", "reliability",
         "risk_coverage", "confidence_hist", "latency"]


@pytest.mark.parametrize("spec", [EMOTION, DAILY_DIALOG, FIN_TOPIC], ids=lambda s: s.key)
def test_selftest_renders_all_figures(tmp_path: Path, spec) -> None:
    written = plots._selftest(tmp_path / spec.key, n=300, labels=spec.labels)
    assert len(written) == 2 * len(STEMS)
    for stem in STEMS:
        for ext in ("png", "pdf"):
            assert (tmp_path / spec.key / f"{stem}.{ext}").stat().st_size > 0


def test_infer_labels_from_summary() -> None:
    summ = {"m": {"per_class": [{"label_id": i, "label": l, "f1": 0.5} for i, l in enumerate(FIN_TOPIC.labels)]}}
    assert plots.infer_labels(summ) == FIN_TOPIC.labels
    assert plots.infer_labels({"m": {}}) == EMOTION.labels
    assert plots.infer_labels({}, ["a", "b"]) == ["a", "b"]
