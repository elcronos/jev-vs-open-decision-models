"""Single social-media-ready chart: accuracy per dataset for the three zero-shot systems plus the
supervised TF-IDF + logistic-regression reference (trained on each dataset's own labels).
Writes results/plots/linkedin_accuracy.png (1600x900 px). No model calls."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
cross = {(r["dataset"], r["model"]): r for r in json.loads((ROOT / "results/cross_dataset_summary.json").read_text())["models"]}
sup = json.loads((ROOT / "results/supervised_models.json").read_text())["datasets"]
DS = ["emotion", "tweet_topic", "fin_topic", "daily_dialog"]
XL = {"emotion": "Emotion\n(6 classes)*", "tweet_topic": "Tweet topics\n(6 classes)", "fin_topic": "Finance topics\n(20 classes)", "daily_dialog": "Dialogue emotion\n(7 classes)"}
SERIES = [("Jev 1.13 (zero-shot)", "#0072B2", lambda d: cross[(d, "Jev 1.13")]["accuracy"], None),
          ("PrismNLI-0.4B (zero-shot)", "#E69F00", lambda d: cross[(d, "PrismNLI-0.4B")]["accuracy"], None),
          ("Laya (zero-shot)", "#009E73", lambda d: cross[(d, "Laya")]["accuracy"], None),
          ("TF-IDF + logistic regression (trained on labels)", "#444444", lambda d: sup[d]["configs"]["logreg_tfidf"]["temperature_scaled"]["accuracy"], "//")]

plt.rcParams.update({"font.size": 13, "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
w = 0.2
for j, (lab, col, get, hatch) in enumerate(SERIES):
    xs = [i + (j - 1.5) * w for i in range(len(DS))]
    ys = [get(d) for d in DS]
    ax.bar(xs, ys, w, color=col, label=lab, hatch=hatch, edgecolor="white", linewidth=1)
    for x, y in zip(xs, ys):
        ax.text(x, y + 0.012, f"{y:.2f}", ha="center", va="bottom", fontsize=12, color="#222222")
for i, d in enumerate(DS):
    m = sup[d]["majority_class_accuracy"]
    ax.hlines(m, i - 2 * w, i + 2 * w, colors="black", linestyles=":", linewidth=1.5)
ax.plot([], [], ":", color="black", label="always predict the most common class")
ax.set_xticks(range(len(DS)))
ax.set_xticklabels([XL[d] for d in DS], fontsize=14)
ax.set_ylim(0, 1.0)
ax.set_ylabel("Accuracy (same evaluation rows for every model)", fontsize=14)
ax.set_title("Zero-shot decision models vs the most boring supervised baseline", fontsize=20, loc="left", pad=18, weight="bold")
fig.text(0.01, 0.01, "n = 2000 / 1693 / 4117 / 7740 test rows. Zero-shot systems see only the text and the label names; logistic regression is trained on each dataset's own training split.\n"
         "* PrismNLI's initialisation checkpoint was trained on this dataset's train/validation splits.   Source: github.com/elcronos/jev-vs-open-decision-models",
         fontsize=10.5, color="#555555", va="bottom")
ax.grid(axis="y", alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(fontsize=12, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=5, columnspacing=1.2, handlelength=1.6)
ax.set_ylim(0, 1.12)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(ROOT / "results/plots/linkedin_accuracy.png", dpi=100)
print("wrote results/plots/linkedin_accuracy.png")
