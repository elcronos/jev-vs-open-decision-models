"""Supervised reference point (NOT zero-shot, NOT part of the frozen comparison).

TF-IDF (word 1-2 grams) + multinomial logistic regression trained on each dataset's own training split,
then temperature-scaled on a held-out calibration split. Evaluated on exactly the same rows as the
zero-shot systems (same evaluated split, same label ids). Purpose: show what a small, well-calibrated
classical ML model reaches when labelled data exist, to contextualise the zero-shot numbers.
Writes results/supervised_baseline.json.
"""
import json, numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import metrics

SEED = 0
def temp_scale(logits, y):
    def nll(T): 
        z = logits / T; z = z - z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
        return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()
    return minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x
def softmax(z, T=1.0):
    z = z / T; z = z - z.max(1, keepdims=True); p = np.exp(z); return p / p.sum(1, keepdims=True)

def run(key, train_txt, train_y, cal_txt, cal_y, eval_parquet):
    df = pd.read_parquet(eval_parquet); df = df[df.variant == "plain"].sort_values("dataset_index")
    ev_txt, ev_y = df.text.tolist(), df.gold_id.values
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    Xtr = vec.fit_transform(train_txt); clf = LogisticRegression(max_iter=2000, C=4.0, random_state=SEED).fit(Xtr, train_y)
    T = temp_scale(clf.decision_function(vec.transform(cal_txt)), np.asarray(cal_y))
    P_raw = softmax(clf.decision_function(vec.transform(ev_txt))); P = softmax(clf.decision_function(vec.transform(ev_txt)), T)
    def rep(P):
        pred = P.argmax(1)
        return {"accuracy": float((pred == ev_y).mean()), "macro_f1": float(f1_score(ev_y, pred, average="macro")),
                "ece15": float(metrics.ece_equal_width(ev_y, P, n_bins=15)[0]), "brier": float(metrics.brier_multiclass(ev_y, P)),
                "nll": float(metrics.nll(ev_y, P)), "mean_confidence": float(P.max(1).mean())}
    r = {"n_train": len(train_txt), "n_cal": len(cal_txt), "n_eval": len(ev_y), "temperature": float(T), "uncalibrated": rep(P_raw), "temperature_scaled": rep(P)}
    print(key, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r["temperature_scaled"].items()}), "T=%.2f" % T)
    return r

out = {}
# emotion: train / validation(cal) / test(eval)
ds = load_dataset("dair-ai/emotion", "split", revision="cab853a1dbdf4c42c2b3ef2173804746df8825fe")
out["emotion"] = run("emotion", ds["train"]["text"], ds["train"]["label"], ds["validation"]["text"], ds["validation"]["label"], "results/raw_predictions.parquet")
# tweet_topic: train_all / validation_2021(cal) / test_2021(eval)
rev = "87b7a0d1c402dbb481db649569c556d9aa27ac05"
def tt(f): return [json.loads(l) for l in open(hf_hub_download("cardiffnlp/tweet_topic_single", f, repo_type="dataset", revision=rev))]
tr = tt("dataset/split_temporal/train_2020.single.json") + tt("dataset/split_temporal/train_2021.single.json"); va = tt("dataset/split_temporal/validation_2021.single.json")
out["tweet_topic"] = run("tweet_topic", [r["text"] for r in tr], [r["label"] for r in tr], [r["text"] for r in va], [r["label"] for r in va], "results/tweet_topic/raw_predictions.parquet")
# fin_topic: train split minus a seeded 10% calibration slice / validation(eval)
ds = load_dataset("zeroshot/twitter-financial-news-topic", revision="acbc8af2a35ccf0916124efcbe9e6cf25f191012")["train"]
rng = np.random.default_rng(SEED); idx = rng.permutation(len(ds)); ncal = len(ds) // 10
cal, trn = idx[:ncal], idx[ncal:]
out["fin_topic"] = run("fin_topic", [ds[int(i)]["text"] for i in trn], [ds[int(i)]["label"] for i in trn], [ds[int(i)]["text"] for i in cal], [ds[int(i)]["label"] for i in cal], "results/fin_topic/raw_predictions.parquet")
# daily_dialog: train utterances / validation utterances(cal) / test utterances(eval)
ds = load_dataset("OpenRL/daily_dialog", revision="1668faf0c0dc44664f108c489fd0666128db2c48")
def flat(split): return [u for d in ds[split]["dialog"] for u in d], [e for es in ds[split]["emotion"] for e in es]
trx, try_ = flat("train"); cax, cay = flat("validation")
out["daily_dialog"] = run("daily_dialog", trx, try_, cax, cay, "results/daily_dialog/raw_predictions.parquet")
json.dump(out, open("results/supervised_baseline.json", "w"), indent=1)
