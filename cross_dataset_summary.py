"""Aggregate the primary run and the three follow-up datasets into one table + figure.

Reads results/summary.json (emotion, variant plain) and results/<key>/summary.json for the follow-up
datasets, writes results/cross_dataset_summary.{json,csv} and results/plots/cross_dataset_accuracy.{png,pdf}.
No model calls; pure post-processing of frozen outputs.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import metrics

ROOT = Path(__file__).resolve().parent
DATASETS = [
    ("emotion", ROOT / "results" / "summary.json", "emotion\n(6 cls, in PrismNLI\nlineage)"),
    ("tweet_topic", ROOT / "results" / "tweet_topic" / "summary.json", "tweet_topic\n(6 cls)"),
    ("fin_topic", ROOT / "results" / "fin_topic" / "summary.json", "fin_topic\n(20 cls)"),
    ("daily_dialog", ROOT / "results" / "daily_dialog" / "summary.json", "daily_dialog\n(7 cls, utterances)"),
]
MODELS = [("jev", "Jev 1.13"), ("prismnli", "PrismNLI-0.4B"), ("laya", "Laya")]
COLORS = {"jev": "#0072B2", "prismnli": "#E69F00", "laya": "#009E73"}


def _macro_f1_rows(yt: np.ndarray, yp: np.ndarray, k: int) -> float:
    return metrics.macro_f1(yt, yp, n_classes=k)


def _ece_rows(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    b = np.digitize(conf, edges[1:-1], right=True)
    cnt = np.bincount(b, minlength=n_bins).astype(np.float64)
    sc = np.bincount(b, weights=conf, minlength=n_bins)
    sa = np.bincount(b, weights=correct.astype(np.float64), minlength=n_bins)
    m = cnt > 0
    return float(np.sum(np.abs(sc[m] - sa[m])) / conf.shape[0])


def parquet_extras(key: str, parquet: Path, n_classes: int) -> tuple[list, dict]:
    """Paired bootstrap of macro-F1 / Brier / ECE-15 differences (same 10 000 seed-0 resample indices
    as metrics.paired_bootstrap_accuracy_diff) for every model pair, plus PrismNLI's independent
    entailment aggregates (from prismnli_indep_probs_json). Pure post-processing of the frozen parquet."""
    df = pd.read_parquet(parquet)
    df = df[df["variant"] == "plain"].sort_values("dataset_index").reset_index(drop=True)
    yt = df["gold_id"].to_numpy().astype(int)
    labels = [c[len("prismnli_p_"):] for c in df.columns if c.startswith("prismnli_p_")]
    assert len(labels) == n_classes, (key, labels, n_classes)
    per = {}
    for mkey, _ in MODELS:
        pr = df[[f"{mkey}_p_{l}" for l in labels]].to_numpy(dtype=np.float64)
        yp = df[f"{mkey}_pred"].to_numpy().astype(int)
        onehot = np.zeros_like(pr)
        onehot[np.arange(len(yt)), yt] = 1.0
        per[mkey] = {"pred": yp, "brier_row": np.sum((pr - onehot) ** 2, axis=1),
                     "conf": np.max(pr, axis=1), "correct": (np.argmax(pr, axis=1) == yt)}
    idx = metrics._bootstrap_indices(len(yt), metrics.BOOTSTRAP_N, 0)
    pair_rows = []
    for a, b in [("jev", "prismnli"), ("jev", "laya"), ("prismnli", "laya")]:
        A, B = per[a], per[b]
        d_f1 = np.empty(idx.shape[0]); d_ece = np.empty(idx.shape[0])
        for i in range(idx.shape[0]):
            ii = idx[i]
            d_f1[i] = _macro_f1_rows(yt[ii], A["pred"][ii], n_classes) - _macro_f1_rows(yt[ii], B["pred"][ii], n_classes)
            d_ece[i] = _ece_rows(A["conf"][ii], A["correct"][ii]) - _ece_rows(B["conf"][ii], B["correct"][ii])
        d_brier = A["brier_row"][idx].mean(axis=1) - B["brier_row"][idx].mean(axis=1)
        row = {"dataset": key, "pair": f"{a}_vs_{b}", "n_bootstrap": int(idx.shape[0]), "seed": 0}
        for name, diffs, point in [
            ("macro_f1_diff", d_f1, _macro_f1_rows(yt, A["pred"], n_classes) - _macro_f1_rows(yt, B["pred"], n_classes)),
            ("brier_diff", d_brier, float(A["brier_row"].mean() - B["brier_row"].mean())),
            ("ece15_diff", d_ece, _ece_rows(A["conf"], A["correct"]) - _ece_rows(B["conf"], B["correct"])),
        ]:
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            p = min(1.0, 2.0 * min(float(np.mean(diffs <= 0)), float(np.mean(diffs >= 0))))
            row[name] = {"point": float(point), "low": float(lo), "high": float(hi), "p_value": float(p)}
        pair_rows.append(row)
    indep = np.array([json.loads(x) for x in df["prismnli_indep_probs_json"]], dtype=np.float64)
    pred = per["prismnli"]["pred"]
    top = int(np.bincount(pred, minlength=n_classes).argmax())
    ent = {
        "dataset": key, "n": int(len(yt)),
        "frac_rows_with_ge2_labels_p_entail_over_0.5": float(np.mean((indep > 0.5).sum(axis=1) >= 2)),
        "mean_max_p_entail": float(indep.max(axis=1).mean()),
        "mean_n_labels_p_entail_over_0.5": float((indep > 0.5).sum(axis=1).mean()),
        "top_predicted_label": labels[top],
        "top_predicted_share": float(np.mean(pred == top)),
        "top_predicted_label_p_entail_over_0.5_rate": float(np.mean(indep[:, top] > 0.5)),
        "top_predicted_label_gold_share": float(np.mean(yt == top)),
    }
    return pair_rows, ent


def main() -> None:
    rows = []
    pair_rows = []
    extra_pairs = []
    entail = []
    for key, path, _ in DATASETS:
        if not path.exists():
            print("missing", path)
            continue
        s = json.loads(path.read_text())
        pv = s["per_variant"]["plain"]
        n_classes = s.get("n_classes") or pv.get("n_classes") or 6
        import numpy as np  # majority-class accuracy: from summary if present, else from the gold labels in the parquet
        maj = pv.get("majority_class_accuracy")
        if maj is None:
            g = pd.read_parquet(path.with_name("raw_predictions.parquet"))
            g = g[g["variant"] == "plain"]["gold_id"].to_numpy()
            maj = float(np.bincount(g).max() / len(g))
        for mkey, mname in MODELS:
            e = pv["models"][mkey]
            m = e["metrics"]
            rows.append({
                "dataset": key, "model": mname, "n": m["n"], "n_classes": n_classes,
                "majority_class_accuracy": maj,
                "accuracy": m["accuracy"], "acc_ci_lo": m["accuracy_ci"]["low"], "acc_ci_hi": m["accuracy_ci"]["high"],
                "macro_f1": m["macro_f1"], "f1_ci_lo": m["macro_f1_ci"]["low"], "f1_ci_hi": m["macro_f1_ci"]["high"],
                "brier": m["brier"], "ece15": m["ece_15"], "nll": m["nll"], "mean_confidence": m["mean_confidence"],
                "frac_gold_prob_zero": m["frac_gold_prob_zero"],
                "acc_at_cov50": m["accuracy_at_coverage"]["0.5"], "acc_at_cov80": m["accuracy_at_coverage"]["0.8"],
                "p50_ms": e["latency"]["p50_ms"], "p95_ms": e["latency"]["p95_ms"], "latency_kind": e["latency_kind"],
                "total_cost_usd": e.get("total_cost_usd"), "n_errors": e["n_errors"],
            })
        for pk, p in pv["pairwise"].items():
            pair_rows.append({"dataset": key, "pair": pk, "mcnemar_p": p["mcnemar_exact"]["pvalue"],
                              "b": p["mcnemar_exact"]["b"], "c": p["mcnemar_exact"]["c"],
                              "acc_diff": p["paired_bootstrap_accuracy_diff"]["point"],
                              "ci_lo": p["paired_bootstrap_accuracy_diff"]["low"],
                              "ci_hi": p["paired_bootstrap_accuracy_diff"]["high"]})
        parquet = path.with_name("raw_predictions.parquet")
        if key == "emotion":
            parquet = ROOT / "results" / "frozen_primary_plain" / "raw_predictions.parquet"
        ep, en = parquet_extras(key, parquet, n_classes)
        extra_pairs.extend(ep)
        entail.append(en)
    df = pd.DataFrame(rows)
    pairs = pd.DataFrame(pair_rows)
    df.to_csv(ROOT / "results" / "cross_dataset_summary.csv", index=False)
    (ROOT / "results" / "cross_dataset_summary.json").write_text(json.dumps({
        "models": rows, "pairwise": pair_rows,
        "pairwise_paired_bootstrap_extra": extra_pairs,
        "prismnli_independent_entailment": entail,
    }, indent=1))
    for e in entail:
        print(e)
    for r in extra_pairs:
        print(r["dataset"], r["pair"], {k: (round(v["point"], 4), round(v["low"], 4), round(v["high"], 4)) for k, v in r.items() if isinstance(v, dict)})
    print(df[["dataset", "model", "n", "n_classes", "majority_class_accuracy", "accuracy", "acc_ci_lo", "acc_ci_hi", "macro_f1", "brier", "ece15", "nll", "acc_at_cov50", "p50_ms"]].round(3).to_string(index=False))
    print(pairs.round(4).to_string(index=False))

    # figure: accuracy with CI (left) and ECE (right), grouped by dataset
    present = [(k, lbl) for k, _, lbl in DATASETS if k in set(df.dataset)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    width = 0.26
    for ax, metric, title in zip(axes, ["accuracy", "macro_f1", "ece15"], ["Accuracy (95% bootstrap CI)", "Macro-F1 (95% bootstrap CI)", "ECE (15 bins, lower is better)"]):
        for j, (mkey, mname) in enumerate(MODELS):
            sub = df[df.model == mname].set_index("dataset")
            xs = [i + (j - 1) * width for i in range(len(present))]
            ys = [sub.loc[k, metric] for k, _ in present]
            if metric == "accuracy":
                err = [[sub.loc[k, "accuracy"] - sub.loc[k, "acc_ci_lo"] for k, _ in present], [sub.loc[k, "acc_ci_hi"] - sub.loc[k, "accuracy"] for k, _ in present]]
            elif metric == "macro_f1":
                err = [[sub.loc[k, "macro_f1"] - sub.loc[k, "f1_ci_lo"] for k, _ in present], [sub.loc[k, "f1_ci_hi"] - sub.loc[k, "macro_f1"] for k, _ in present]]
            else:
                err = None
            ax.bar(xs, ys, width, color=COLORS[mkey], label=mname, yerr=err, capsize=2, error_kw={"linewidth": 0.8})
        if metric == "accuracy":
            for i, (k, _) in enumerate(present):
                maj = df[df.dataset == k].majority_class_accuracy.iloc[0]
                ax.hlines(maj, i - 1.6 * width, i + 1.6 * width, colors="black", linestyles=":", linewidth=1)
            ax.plot([], [], ":", color="black", label="majority class")
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels([lbl for _, lbl in present], fontsize=8)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_ylim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Zero-shot results across datasets (plain variant). Only dair-ai/emotion is in PrismNLI's training lineage.", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "results" / "plots" / f"cross_dataset_accuracy.{ext}", dpi=200)
    print("wrote results/plots/cross_dataset_accuracy.png")


if __name__ == "__main__":
    main()
