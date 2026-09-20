"""Figures for the README that combine several experiments (no model calls).

- results/plots/supervised_vs_zeroshot.png : accuracy and macro-F1 per dataset for the three zero-shot
  systems and the four supervised configurations (temperature-scaled), with majority-class markers.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASETS = ["emotion", "tweet_topic", "fin_topic", "daily_dialog"]
DLABEL = {"emotion": "emotion\n(6 cls, in PrismNLI\nlineage)", "tweet_topic": "tweet_topic\n(6 cls)", "fin_topic": "fin_topic\n(20 cls)", "daily_dialog": "daily_dialog\n(7 cls)"}
ZS = [("Jev 1.13", "#0072B2"), ("PrismNLI-0.4B", "#E69F00"), ("Laya", "#009E73")]
SUP = [("logreg_tfidf", "LR + TF-IDF", "#999999"), ("lgbm_tfidf", "LightGBM + TF-IDF", "#666666"),
       ("logreg_minilm", "LR + MiniLM", "#CC79A7"), ("lgbm_minilm", "LightGBM + MiniLM", "#882255")]


def main() -> None:
    cross = json.loads((ROOT / "results" / "cross_dataset_summary.json").read_text())["models"]
    sup = json.loads((ROOT / "results" / "supervised_models.json").read_text())["datasets"]
    zs = {(r["dataset"], r["model"]): r for r in cross}

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    series = [(name, col, lambda d, n=name: zs[(d, n)], "zs") for name, col in ZS] + \
             [(lab, col, lambda d, c=cfg: sup[d]["configs"][c]["temperature_scaled"], "sup") for cfg, lab, col in SUP]
    width = 0.11
    for ax, metric, title in zip(axes, ["accuracy", "macro_f1", "ece15"], ["Accuracy", "Macro-F1", "ECE (15 bins, lower is better)"]):
        for j, (lab, col, get, kind) in enumerate(series):
            xs = [i + (j - 3) * width for i in range(len(DATASETS))]
            ys = [get(d)[metric] for d in DATASETS]
            ax.bar(xs, ys, width, color=col, label=lab, hatch="//" if kind == "sup" else None, edgecolor="white", linewidth=0.5)
        if metric == "accuracy":
            for i, d in enumerate(DATASETS):
                ax.hlines(sup[d]["majority_class_accuracy"], i - 3.5 * width, i + 3.5 * width, colors="black", linestyles=":", linewidth=1)
            ax.plot([], [], ":", color="black", label="majority class")
        ax.set_xticks(range(len(DATASETS)))
        ax.set_xticklabels([DLABEL[d] for d in DATASETS], fontsize=8)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].legend(fontsize=7, frameon=False, ncol=2, loc="upper right")
    fig.suptitle("Zero-shot systems (solid) vs supervised models trained on each dataset's own labels (hatched), same evaluation rows", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "results" / "plots" / f"supervised_vs_zeroshot.{ext}", dpi=200)
    print("wrote results/plots/supervised_vs_zeroshot.png")


if __name__ == "__main__":
    main()
