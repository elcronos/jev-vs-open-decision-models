"""Label-efficiency experiment (supervised, NOT zero-shot): how many labelled examples does a simple
supervised model need before it matches or beats the best zero-shot system on the same evaluation rows?

For each dataset and each training-set size n in N_GRID (plus the full split), draw n labelled rows
from the dataset's own training split (3 seeds), fit logistic regression on (a) MiniLM sentence
embeddings and (b) word 1-2 gram TF-IDF, and score accuracy / macro-F1 on exactly the rows the
zero-shot systems scored. No calibration step (accuracy and macro-F1 only). Also reports, for
daily_dialog, the same metrics on the eval rows whose text does not occur in the training split.

Outputs: results/learning_curve.json, results/plots/learning_curve.{png,pdf}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import metrics
from supervised_models import ZS_MODELS, load_eval, load_splits, minilm

ROOT = Path(__file__).resolve().parent
N_GRID = [50, 100, 200, 500, 1000, 2000, 5000]
SEEDS = [0, 1, 2]
DATASETS = ["emotion", "tweet_topic", "fin_topic", "daily_dialog"]
ZS_NAMES = {"jev": "Jev 1.13", "prismnli": "PrismNLI-0.4B", "laya": "Laya"}
COLORS = {"jev": "#0072B2", "prismnli": "#E69F00", "laya": "#009E73", "minilm": "#CC79A7", "tfidf": "#555555"}


def fit_predict(feat: str, Xtr, ytr, Xev, n_classes: int) -> np.ndarray:
    """Return (N_eval, K) probabilities; classes absent from the sample get probability 0."""
    clf = LogisticRegression(max_iter=3000, C=4.0, random_state=0).fit(Xtr, ytr)
    P = np.zeros((Xev.shape[0], n_classes))
    P[:, clf.classes_] = clf.predict_proba(Xev)
    return P


def main() -> None:
    out = {"n_grid": N_GRID, "seeds": SEEDS, "datasets": {}}
    for key in DATASETS:
        print("===", key, flush=True)
        trx, try_, _, _ = load_splits(key)
        ytr_all = np.asarray(try_)
        di, evx, yev, zs = load_eval(key)
        K = int(max(ytr_all.max(), yev.max())) + 1
        train_texts = set(trx)
        keep = ~np.array([t in train_texts for t in evx])
        emb_tr, emb_ev = minilm(trx), minilm(evx)
        grid = [n for n in N_GRID if n < len(trx)] + [len(trx)]
        res = {"n_train_full": len(trx), "n_eval": int(len(yev)), "n_classes": K, "frac_eval_text_in_train": float((~keep).mean()),
               "zero_shot": {m: {"accuracy": float((zs[m].argmax(1) == yev).mean()),
                                 "macro_f1": float(metrics.macro_f1(yev, zs[m].argmax(1), n_classes=K)),
                                 "accuracy_non_overlap": float((zs[m].argmax(1)[keep] == yev[keep]).mean())} for m in ZS_MODELS},
               "points": []}
        for n in grid:
            for feat in ("minilm", "tfidf"):
                accs, f1s, accs_no = [], [], []
                for seed in (SEEDS if n < len(trx) else [0]):
                    rng = np.random.default_rng(seed)
                    idx = rng.choice(len(trx), size=n, replace=False) if n < len(trx) else np.arange(len(trx))
                    if len(np.unique(ytr_all[idx])) < 2:  # degenerate sample; resample deterministically
                        idx = rng.choice(len(trx), size=n, replace=False)
                    if feat == "minilm":
                        P = fit_predict(feat, emb_tr[idx], ytr_all[idx], emb_ev, K)
                    else:
                        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, max_features=50000)
                        P = fit_predict(feat, vec.fit_transform([trx[i] for i in idx]), ytr_all[idx], vec.transform(evx), K)
                    pred = P.argmax(1)
                    accs.append(float((pred == yev).mean()))
                    f1s.append(float(metrics.macro_f1(yev, pred, n_classes=K)))
                    accs_no.append(float((pred[keep] == yev[keep]).mean()))
                res["points"].append({"n": int(n), "features": feat, "accuracy_mean": float(np.mean(accs)), "accuracy_min": float(np.min(accs)),
                                      "accuracy_max": float(np.max(accs)), "macro_f1_mean": float(np.mean(f1s)), "macro_f1_min": float(np.min(f1s)),
                                      "macro_f1_max": float(np.max(f1s)), "accuracy_non_overlap_mean": float(np.mean(accs_no)), "n_seeds": len(accs)})
                print(f"  n={n:6d} {feat:6s} acc={np.mean(accs):.3f} [{np.min(accs):.3f},{np.max(accs):.3f}] f1={np.mean(f1s):.3f}", flush=True)
        # crossover: smallest n at which the mean accuracy of each feature set reaches the best zero-shot accuracy
        best_zs_acc = max(v["accuracy"] for v in res["zero_shot"].values())
        res["best_zero_shot_accuracy"] = best_zs_acc
        res["crossover_n"] = {}
        for feat in ("minilm", "tfidf"):
            pts = sorted([p for p in res["points"] if p["features"] == feat], key=lambda p: p["n"])
            hit = [p["n"] for p in pts if p["accuracy_mean"] >= best_zs_acc]
            res["crossover_n"][feat] = hit[0] if hit else None
        out["datasets"][key] = res
        (ROOT / "results" / "learning_curve.json").write_text(json.dumps(out, indent=1))

    # figure: 2 rows (accuracy, macro-F1) x 4 datasets
    fig, axes = plt.subplots(2, len(DATASETS), figsize=(16, 7), sharex="col")
    for j, key in enumerate(DATASETS):
        res = out["datasets"][key]
        for i, metric in enumerate(["accuracy", "macro_f1"]):
            ax = axes[i, j]
            for feat in ("minilm", "tfidf"):
                pts = sorted([p for p in res["points"] if p["features"] == feat], key=lambda p: p["n"])
                xs = [p["n"] for p in pts]
                ax.plot(xs, [p[f"{metric}_mean"] for p in pts], "-o", ms=3, color=COLORS[feat],
                        label=f"LR on {'MiniLM embeddings' if feat == 'minilm' else 'TF-IDF'} (supervised)")
                ax.fill_between(xs, [p[f"{metric}_min"] for p in pts], [p[f"{metric}_max"] for p in pts], color=COLORS[feat], alpha=0.15)
            for m in ZS_MODELS:
                ax.axhline(res["zero_shot"][m][metric], ls="--", lw=1.2, color=COLORS[m], label=f"{ZS_NAMES[m]} (zero-shot)")
            ax.set_xscale("log")
            ax.grid(alpha=0.3)
            ax.spines[["top", "right"]].set_visible(False)
            if i == 0:
                ax.set_title(f"{key}  ({res['n_classes']} classes)", fontsize=10, loc="left")
            if i == 1:
                ax.set_xlabel("labelled training examples (log scale)")
            if j == 0:
                ax.set_ylabel(metric.replace("_", "-"))
    axes[0, 0].legend(fontsize=7, frameon=False, loc="lower right")
    fig.suptitle("How many labels does a simple supervised model need to match the best zero-shot system? (mean of 3 seeds, band = min-max)", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "results" / "plots" / f"learning_curve.{ext}", dpi=200)
    print("wrote results/plots/learning_curve.png; crossovers:", {k: v["crossover_n"] for k, v in out["datasets"].items()})


if __name__ == "__main__":
    main()
