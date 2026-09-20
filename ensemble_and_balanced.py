"""Post-hoc analysis (no model calls): majority-vote and probability-average ensembles of the three
zero-shot systems, the any-correct oracle, and balanced accuracy (macro recall) per dataset.
Writes results/ensemble_and_balanced.json."""
import json, numpy as np, pandas as pd
M = ["jev", "prismnli", "laya"]
out = {}
for key, path in [("emotion", "results/raw_predictions.parquet"), ("tweet_topic", "results/tweet_topic/raw_predictions.parquet"),
                  ("fin_topic", "results/fin_topic/raw_predictions.parquet"), ("daily_dialog", "results/daily_dialog/raw_predictions.parquet")]:
    df = pd.read_parquet(path); df = df[df.variant == "plain"]
    y = df.gold_id.values; P = {m: df[f"{m}_pred"].values for m in M}; C = {m: df[f"{m}_confidence"].values for m in M}
    K = int(y.max()) + 1
    votes = np.stack([P[m] for m in M], 1); conf = np.stack([C[m] for m in M], 1)
    mv = np.empty(len(y), int)
    for i in range(len(y)):  # majority vote; three-way tie -> prediction of the most confident system
        vals, cnt = np.unique(votes[i], return_counts=True)
        mv[i] = vals[cnt.argmax()] if cnt.max() >= 2 else votes[i][conf[i].argmax()]
    labels = [c[len("jev_p_"):] for c in df.columns if c.startswith("jev_p_")]
    pa = (sum(df[[f"{m}_p_{l}" for l in labels]].values for m in M) / 3).argmax(1)
    bal = lambda p: float(np.mean([(p[y == k] == k).mean() for k in range(K) if (y == k).any()]))
    out[key] = {"majority_class": float(np.bincount(y).max() / len(y)), **{f"acc_{m}": float((P[m] == y).mean()) for m in M},
                "acc_majority_vote": float((mv == y).mean()), "acc_prob_avg": float((pa == y).mean()),
                "acc_oracle_any": float(np.logical_or.reduce([P[m] == y for m in M]).mean()), **{f"balanced_acc_{m}": bal(P[m]) for m in M}}
    print(key, {k: round(v, 3) for k, v in out[key].items()})
json.dump(out, open("results/ensemble_and_balanced.json", "w"), indent=1)
