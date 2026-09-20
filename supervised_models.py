"""Supervised reference models (NOT zero-shot, NOT part of the frozen comparison).

Four configurations per dataset, all trained on the dataset's own training split and
temperature-scaled on a held-out calibration split, then evaluated on exactly the same rows the
zero-shot systems scored (same evaluated split, same label ids):

    features x model
    ------------------------------------------------------------------
    tfidf   : word 1-2 gram TF-IDF (min_df=2, sublinear tf, <=50k feats)
    minilm  : sentence-transformers/all-MiniLM-L6-v2 mean-pooled, L2-normalised (384-d)
    logreg  : multinomial logistic regression (C=4)
    lgbm    : LightGBM gradient-boosted trees (multiclass, lr 0.1, 63 leaves, early stopping on the
              calibration split, max 1000 rounds)

Splits (identical to the earlier TF-IDF + LR baseline):
    emotion      train / validation (calibration) / test (eval)
    tweet_topic  train_2020+train_2021 / validation_2021 / test_2021
    fin_topic    90% of train / 10% of train (seed 0) / validation
    daily_dialog train utterances / validation utterances / test utterances

Outputs: results/supervised_models.json (metrics, paired tests, timings) and
results/supervised/<dataset>_predictions.parquet (per-row probabilities for every configuration).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from lightgbm import LGBMClassifier, early_stopping
from scipy.optimize import minimize_scalar
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import metrics

SEED = 0
ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "results" / "supervised"
ZS_PARQUET = {
    "emotion": ROOT / "results" / "raw_predictions.parquet",
    "tweet_topic": ROOT / "results" / "tweet_topic" / "raw_predictions.parquet",
    "fin_topic": ROOT / "results" / "fin_topic" / "raw_predictions.parquet",
    "daily_dialog": ROOT / "results" / "daily_dialog" / "raw_predictions.parquet",
}
ZS_MODELS = ["jev", "prismnli", "laya"]


# ----------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------
def load_splits(key: str):
    """Return (train_texts, train_y, cal_texts, cal_y)."""
    if key == "emotion":
        ds = load_dataset("dair-ai/emotion", "split", revision="cab853a1dbdf4c42c2b3ef2173804746df8825fe")
        return ds["train"]["text"], ds["train"]["label"], ds["validation"]["text"], ds["validation"]["label"]
    if key == "tweet_topic":
        rev = "87b7a0d1c402dbb481db649569c556d9aa27ac05"

        def tt(f):
            return [json.loads(l) for l in open(hf_hub_download("cardiffnlp/tweet_topic_single", f, repo_type="dataset", revision=rev))]

        tr = tt("dataset/split_temporal/train_2020.single.json") + tt("dataset/split_temporal/train_2021.single.json")
        va = tt("dataset/split_temporal/validation_2021.single.json")
        return [r["text"] for r in tr], [r["label"] for r in tr], [r["text"] for r in va], [r["label"] for r in va]
    if key == "fin_topic":
        ds = load_dataset("zeroshot/twitter-financial-news-topic", revision="acbc8af2a35ccf0916124efcbe9e6cf25f191012")["train"]
        rng = np.random.default_rng(SEED)
        idx = rng.permutation(len(ds))
        ncal = len(ds) // 10
        cal, trn = idx[:ncal], idx[ncal:]
        txt, lab = ds["text"], ds["label"]
        return [txt[i] for i in trn], [lab[i] for i in trn], [txt[i] for i in cal], [lab[i] for i in cal]
    if key == "daily_dialog":
        ds = load_dataset("OpenRL/daily_dialog", revision="1668faf0c0dc44664f108c489fd0666128db2c48")

        def flat(split):
            return [u for d in ds[split]["dialog"] for u in d], [e for es in ds[split]["emotion"] for e in es]

        trx, try_ = flat("train")
        cax, cay = flat("validation")
        return trx, try_, cax, cay
    raise KeyError(key)


def load_eval(key: str):
    df = pd.read_parquet(ZS_PARQUET[key])
    df = df[df["variant"] == "plain"].sort_values("dataset_index").reset_index(drop=True)
    labels = [c[len("jev_p_"):] for c in df.columns if c.startswith("jev_p_")]
    zs_probs = {m: df[[f"{m}_p_{l}" for l in labels]].to_numpy() for m in ZS_MODELS}
    return df["dataset_index"].to_numpy(), df["text"].tolist(), df["gold_id"].to_numpy(), zs_probs


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
def softmax(z: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = z / T
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    def nll(T):
        p = softmax(logits, T)
        return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()

    return float(minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x)


def report(y: np.ndarray, P: np.ndarray) -> dict:
    pred = P.argmax(1)
    return {
        "n": int(len(y)),
        "accuracy": float((pred == y).mean()),
        "macro_f1": float(metrics.macro_f1(y, pred, n_classes=P.shape[1])),
        "balanced_accuracy": float(np.mean([(pred[y == k] == k).mean() for k in np.unique(y)])),
        "ece15": float(metrics.ece_equal_width(y, P, n_bins=15)[0]),
        "brier": float(metrics.brier_multiclass(y, P)),
        "nll": float(metrics.nll(y, P)),
        "mean_confidence": float(P.max(1).mean()),
    }


_encoder = None


def minilm(texts):
    global _encoder
    if _encoder is None:
        import torch
        from sentence_transformers import SentenceTransformer

        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        _encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=dev)
    return _encoder.encode(list(texts), batch_size=256, normalize_embeddings=True, show_progress_bar=False)


# ----------------------------------------------------------------------------------------------
# one configuration
# ----------------------------------------------------------------------------------------------
def run_config(feat: str, model: str, Xtr, ytr, Xcal, ycal, Xev, n_classes: int):
    t0 = time.perf_counter()
    if model == "logreg":
        clf = LogisticRegression(max_iter=3000, C=4.0, random_state=SEED)
        clf.fit(Xtr, ytr)
        assert list(clf.classes_) == list(range(n_classes)), "class ids must be 0..K-1 and all present in train"
        margin = lambda X: clf.decision_function(X)
        best_iter = None
    elif model == "lgbm":
        clf = LGBMClassifier(
            objective="multiclass", n_estimators=1000, learning_rate=0.1, num_leaves=63,
            min_child_samples=20, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.3 if feat == "tfidf" else 0.8, reg_lambda=1.0,
            random_state=SEED, n_jobs=8, verbose=-1,
        )
        clf.fit(Xtr, ytr, eval_set=[(Xcal, ycal)], callbacks=[early_stopping(50, verbose=False)])
        assert list(clf.classes_) == list(range(n_classes)), "class ids must be 0..K-1 and all present in train"
        margin = lambda X: clf.predict(X, raw_score=True)
        best_iter = int(clf.best_iteration_) if clf.best_iteration_ else None
    else:
        raise KeyError(model)
    train_s = time.perf_counter() - t0

    z_cal = margin(Xcal)
    if z_cal.ndim == 1:  # binary edge case; not expected here
        z_cal = np.stack([-z_cal, z_cal], 1)
    T = fit_temperature(z_cal, np.asarray(ycal))
    t1 = time.perf_counter()
    z_ev = margin(Xev)
    predict_s = time.perf_counter() - t1
    return {
        "P_raw": softmax(z_ev), "P": softmax(z_ev, T), "temperature": T, "train_s": train_s,
        "predict_ms_per_example": 1000.0 * predict_s / Xev.shape[0], "best_iteration": best_iter,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="emotion,tweet_topic,fin_topic,daily_dialog")
    ap.add_argument("--limit-train", type=int, default=None, help="smoke: subsample training rows")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    out = {"seed": SEED, "configs": {}, "datasets": {}}
    for key in [k for k in args.datasets.split(",") if k]:
        print(f"=== {key}", flush=True)
        trx, try_, cax, cay = load_splits(key)
        if args.limit_train:
            rng = np.random.default_rng(SEED)
            sel = rng.choice(len(trx), size=min(args.limit_train, len(trx)), replace=False)
            trx, try_ = [trx[i] for i in sel], [try_[i] for i in sel]
        ytr, ycal = np.asarray(try_), np.asarray(cay)
        di, evx, yev, zs = load_eval(key)
        n_classes = int(max(ytr.max(), yev.max())) + 1

        feats = {}
        t0 = time.perf_counter()
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=50000)
        feats["tfidf"] = (vec.fit_transform(trx), vec.transform(cax), vec.transform(evx), time.perf_counter() - t0)
        t0 = time.perf_counter()
        emb_tr, emb_cal = minilm(trx), minilm(cax)
        t1 = time.perf_counter()
        emb_ev = minilm(evx)
        minilm_encode_ms_per_eval_example = 1000.0 * (time.perf_counter() - t1) / len(evx)
        feats["minilm"] = (emb_tr, emb_cal, emb_ev, time.perf_counter() - t0)

        train_texts = set(trx)
        overlap = np.array([t in train_texts for t in evx])  # eval rows whose exact text also occurs in train
        keep = ~overlap
        ds_out = {"n_train": len(trx), "n_cal": len(cax), "n_eval": len(yev), "n_classes": n_classes,
                  "majority_class_accuracy": float(np.bincount(yev).max() / len(yev)),
                  "n_eval_text_in_train": int(overlap.sum()), "frac_eval_text_in_train": float(overlap.mean()),
                  "majority_class_accuracy_non_overlap": float(np.bincount(yev[keep]).max() / keep.sum()),
                  "feature_time_s": {f: feats[f][3] for f in feats},
                  "minilm_encode_ms_per_eval_example": minilm_encode_ms_per_eval_example,
                  "zero_shot_non_overlap": {m: report(yev[keep], zs[m][keep]) for m in ZS_MODELS},
                  "configs": {}}
        preds = pd.DataFrame({"dataset_index": di, "gold_id": yev})
        P_by_cfg = {}
        for feat in ("tfidf", "minilm"):
            for model in ("logreg", "lgbm"):
                cfg = f"{model}_{feat}"
                Xtr, Xcal, Xev, _ = feats[feat]
                r = run_config(feat, model, Xtr, ytr, Xcal, ycal, Xev, n_classes)
                P_by_cfg[cfg] = r["P"]
                ds_out["configs"][cfg] = {
                    "temperature": r["temperature"], "train_s": r["train_s"], "best_iteration": r["best_iteration"],
                    "predict_ms_per_example": r["predict_ms_per_example"],
                    "uncalibrated": report(yev, r["P_raw"]), "temperature_scaled": report(yev, r["P"]),
                    "temperature_scaled_non_overlap": report(yev[keep], r["P"][keep]),
                }
                for k in range(n_classes):
                    preds[f"{cfg}_p{k}"] = r["P"][:, k]
                ts = ds_out["configs"][cfg]["temperature_scaled"]
                print(f"  {cfg:14s} acc={ts['accuracy']:.3f} f1={ts['macro_f1']:.3f} ece={ts['ece15']:.3f} "
                      f"T={r['temperature']:.2f} train={r['train_s']:.0f}s iters={r['best_iteration']}", flush=True)

        # paired tests on the identical eval rows
        def paired(pa, pb):
            mc = metrics.mcnemar_exact(yev, pa, pb, n_classes=n_classes)
            pb_ = metrics.paired_bootstrap_accuracy_diff(yev, pa, pb, n=10000, seed=SEED, n_classes=n_classes)
            return {"mcnemar_p": float(mc["pvalue"]), "b": int(mc["b"]), "c": int(mc["c"]),
                    "acc_diff": float(pb_["point"]), "ci_lo": float(pb_["low"]), "ci_hi": float(pb_["high"])}

        pairs = {}
        pred_cfg = {c: P.argmax(1) for c, P in P_by_cfg.items()}
        for a, b in [("lgbm_tfidf", "logreg_tfidf"), ("lgbm_minilm", "logreg_minilm"),
                     ("lgbm_minilm", "logreg_tfidf"), ("logreg_minilm", "logreg_tfidf")]:
            pairs[f"{a}_vs_{b}"] = paired(pred_cfg[a], pred_cfg[b])
        best_sup = max(pred_cfg, key=lambda c: (pred_cfg[c] == yev).mean())
        zs_pred = {m: zs[m].argmax(1) for m in ZS_MODELS}
        best_zs = max(zs_pred, key=lambda m: (zs_pred[m] == yev).mean())
        pairs[f"{best_sup}_vs_zeroshot_{best_zs}"] = paired(pred_cfg[best_sup], zs_pred[best_zs])
        ds_out["pairwise"] = pairs
        ds_out["best_supervised"] = best_sup
        ds_out["best_zero_shot"] = best_zs
        preds["text_in_train"] = overlap
        out["datasets"][key] = ds_out
        preds.to_parquet(OUT_DIR / f"{key}_predictions.parquet", index=False)
        (ROOT / "results" / "supervised_models.json").write_text(json.dumps(out, indent=1))  # incremental save

    out["configs"] = {
        "tfidf": "TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True, max_features=50000)",
        "minilm": "sentence-transformers/all-MiniLM-L6-v2, normalize_embeddings=True (384-d)",
        "logreg": "LogisticRegression(C=4.0, max_iter=3000)",
        "lgbm": "LGBMClassifier(multiclass, n_estimators<=1000, learning_rate=0.1, num_leaves=63, min_child_samples=20, subsample=0.8, colsample_bytree=0.3 (tfidf) / 0.8 (minilm), reg_lambda=1.0, early_stopping(50) on the calibration split)",
        "calibration": "single temperature fitted by minimising NLL on the calibration split (bounded scalar search); for LightGBM the same split also drives early stopping, so its reported calibration is slightly optimistic",
        "latency_note": "predict_ms_per_example covers the classifier only; add minilm_encode_ms_per_eval_example for the *_minilm configs",
        "non_overlap_note": "temperature_scaled_non_overlap / zero_shot_non_overlap restrict evaluation to eval rows whose exact text does not occur in the training split (matters for daily_dialog, where whole dialogs are duplicated across splits)",
    }
    (ROOT / "results" / "supervised_models.json").write_text(json.dumps(out, indent=1))
    print("wrote results/supervised_models.json")


if __name__ == "__main__":
    main()
