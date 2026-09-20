"""Supplementary analyses computed from the frozen predictions (no model calls). Writes results/supplementary.json."""
import json, numpy as np, pandas as pd, metrics
from statsmodels.stats.contingency_tables import mcnemar
L = ["sadness","joy","love","anger","fear","surprise"]; M = ["jev","prismnli","laya"]
df = pd.read_parquet("results/raw_predictions.parquet")
out = {}
def _ece_scalar(correct, conf, n_bins=15):
    edges = np.linspace(0,1,n_bins+1); idx = np.clip(np.searchsorted(edges, conf, side="left")-1, 0, n_bins-1); e=0.0
    for b in range(n_bins):
        s = idx==b
        if s.any(): e += s.mean()*abs(correct[s].mean()-conf[s].mean())
    return float(e)
pl = df[df.variant=="plain"].sort_values("dataset_index").reset_index(drop=True)
de = df[df.variant=="defined"].sort_values("dataset_index").reset_index(drop=True)
assert (pl.dataset_index.values==de.dataset_index.values).all() and len(pl)==2000
y = pl.gold_id.values
# 1. plain vs defined per model: McNemar + paired bootstrap
out["plain_vs_defined"] = {}
for m in M:
    a = (pl[f"{m}_pred"].values==y); b = (de[f"{m}_pred"].values==y)
    tab = [[int((a&b).sum()), int((a&~b).sum())],[int((~a&b).sum()), int((~a&~b).sum())]]
    r = mcnemar(tab, exact=True)
    pb = metrics.paired_bootstrap_accuracy_diff(y, de[f"{m}_pred"].values, pl[f"{m}_pred"].values, n=10000, seed=0)
    out["plain_vs_defined"][m] = {"acc_plain": float(a.mean()), "acc_defined": float(b.mean()), "mcnemar_p": float(r.pvalue), "b_plain_only": tab[0][1], "c_defined_only": tab[1][0], "paired_bootstrap_diff_defined_minus_plain": {k: (float(v) if isinstance(v,(int,float,np.floating)) else v) for k,v in pb.items()}}
# 2. top confusions per model (plain)
out["top_confusions_plain"] = {}
for m in M:
    p = pl[f"{m}_pred"].values; pairs = {}
    for g,q in zip(y,p):
        if g!=q: pairs[(L[g],L[q])] = pairs.get((L[g],L[q]),0)+1
    top = sorted(pairs.items(), key=lambda kv:-kv[1])[:6]
    out["top_confusions_plain"][m] = [{"gold":g,"pred":q,"count":c,"frac_of_errors":c/ (p!=y).sum()} for (g,q),c in top]
# 3. agreement / oracle
ok = {m: pl[f"{m}_pred"].values==y for m in M}
out["agreement_plain"] = {"all_three_same_pred": float((pl.jev_pred.eq(pl.prismnli_pred)&pl.jev_pred.eq(pl.laya_pred)).mean()),
  "jev_laya_same_pred": float(pl.jev_pred.eq(pl.laya_pred).mean()), "jev_prismnli_same_pred": float(pl.jev_pred.eq(pl.prismnli_pred).mean()), "prismnli_laya_same_pred": float(pl.prismnli_pred.eq(pl.laya_pred).mean()),
  "all_correct": int((ok["jev"]&ok["prismnli"]&ok["laya"]).sum()), "all_wrong": int((~ok["jev"]&~ok["prismnli"]&~ok["laya"]).sum()),
  "only_correct": {m:int((ok[m]&~np.logical_or.reduce([ok[o] for o in M if o!=m])).sum()) for m in M}, "oracle_any_correct_acc": float(np.logical_or.reduce([ok[m] for m in M]).mean()),
  "majority_class_acc": float((y==np.bincount(y).argmax()).mean()), "majority_class": L[int(np.bincount(y).argmax())], "class_support": {L[i]:int(c) for i,c in enumerate(np.bincount(y))}}
# 4. Jev rounding: NLL sensitivity to eps
P = {m: pl[[f"{m}_p_{l}" for l in L]].values for m in M}
out["nll_eps_sensitivity_plain"] = {m: {f"eps={e}": float(-np.log(np.clip(P[m][np.arange(2000),y], e, 1)).mean()) for e in [1e-2,1e-3,1e-6]} for m in M}
out["frac_gold_prob_below_0.005_plain"] = {m: float((P[m][np.arange(2000),y]<0.005).mean()) for m in M}
# 5. confidence->accuracy: adaptive 10-bin table and accuracy among conf>=0.9, >=0.99
out["conf_slices_plain"] = {}
for m in M:
    c = P[m].max(1); d = {}
    for thr in [0.5,0.7,0.9,0.95,0.99]:
        sel = c>=thr; d[f"conf>={thr}"] = {"coverage": float(sel.mean()), "accuracy": float(ok[m][sel].mean()) if sel.any() else None}
    out["conf_slices_plain"][m] = d
# 6. errors at coverage (count of wrong decisions among top-k confident)
out["errors_at_coverage_plain"] = {}
for m in M:
    c = P[m].max(1); order = np.lexsort((pl.dataset_index.values, -c))
    d = {}
    for cov in [0.5,0.8,0.9,1.0]:
        k = int(np.ceil(cov*2000)); d[str(cov)] = {"n": k, "errors": int((~ok[m][order[:k]]).sum()), "accuracy": float(ok[m][order[:k]].mean())}
    out["errors_at_coverage_plain"][m] = d
# 7. API-native confidence vs max-prob for jev/laya
for m in ["jev","laya"]:
    ac = pl[f"{m}_api_confidence"].values.astype(float); c = P[m].max(1)
    out[f"{m}_api_confidence_plain"] = {"mean_api_conf": float(np.nanmean(ac)), "mean_maxprob": float(c.mean()), "corr": float(np.corrcoef(ac,c)[0,1]), "ece15_using_api_conf_as_confidence": _ece_scalar(ok[m], ac)}
# 8. length effect
pl["nw"] = pl.text.str.split().str.len(); q = pd.qcut(pl.nw, 3, labels=["short","mid","long"])
out["accuracy_by_length_tercile_plain"] = {m: {str(k): float(ok[m][(q==k).values].mean()) for k in ["short","mid","long"]} for m in M}
out["length_terciles_words"] = {str(k): [int(pl.nw[q==k].min()), int(pl.nw[q==k].max())] for k in ["short","mid","long"]}
# 9. latency: jev sequential
out["jev_sequential_latency"] = json.load(open("results/jev_sequential_latency.json"))
out["jev_cost"] = {v: {"total_cost_usd": float(df[df.variant==v].jev_cost_usd.sum()), "input_tokens": int(df[df.variant==v].jev_input_tokens.sum()), "output_tokens": int(df[df.variant==v].jev_output_tokens.sum()), "mean_input_tokens": float(df[df.variant==v].jev_input_tokens.mean())} for v in ["plain","defined"]}
json.dump(out, open("results/supplementary.json","w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float)[:6000])

# ---------------------------------------------------------------------------------------------
# 10. Paired bootstrap CIs for Brier / ECE-15 / errors-at-coverage differences (a minus b),
#     accuracy on the first 600 / 400 rows, and the Jev confidence == 1.00 group.
#     Same resample matrix as metrics.py (default_rng(0), integers(0, N, (10000, N))).
# ---------------------------------------------------------------------------------------------
NB = 10000
idxmat = metrics._bootstrap_indices(2000, NB, 0)
di = pl.dataset_index.values
onehot = np.eye(6)[y]
def _brier_rows(p): return ((p - onehot) ** 2).sum(1)
def _ece_rows(correct, conf, rows, n_bins=15):
    c = conf[rows]; k = correct[rows]
    edges = np.linspace(0, 1, n_bins + 1); b = np.clip(np.searchsorted(edges, c, side="left") - 1, 0, n_bins - 1)
    e = 0.0
    for j in range(n_bins):
        s = b == j
        if s.any(): e += s.mean() * abs(k[s].mean() - c[s].mean())
    return e
def _errs_at_cov(correct, conf, rows, cov):
    order = np.lexsort((di[rows], -conf[rows])); k = int(np.ceil(cov * len(rows)))
    return int((~correct[rows][order[:k]]).sum())
pairs = {}
for a, b in [("jev", "laya"), ("jev", "prismnli"), ("prismnli", "laya")]:
    ca, cb = ok[a], ok[b]; fa, fb = P[a].max(1), P[b].max(1); ba, bb = _brier_rows(P[a]), _brier_rows(P[b])
    d = {"brier": [], "ece15": [], "errors_at_coverage_0.5": [], "errors_at_coverage_0.8": [], "errors_at_coverage_0.9": []}
    for r in idxmat:
        d["brier"].append(ba[r].mean() - bb[r].mean())
        d["ece15"].append(_ece_rows(ca, fa, r) - _ece_rows(cb, fb, r))
        for cov in (0.5, 0.8, 0.9):
            d[f"errors_at_coverage_{cov}"].append(_errs_at_cov(ca, fa, r, cov) - _errs_at_cov(cb, fb, r, cov))
    full = np.arange(2000)
    point = {"brier": ba.mean() - bb.mean(), "ece15": _ece_rows(ca, fa, full) - _ece_rows(cb, fb, full),
             **{f"errors_at_coverage_{cov}": _errs_at_cov(ca, fa, full, cov) - _errs_at_cov(cb, fb, full, cov) for cov in (0.5, 0.8, 0.9)}}
    pairs[f"{a}_vs_{b}"] = {k: {"point": float(point[k]), "low": float(np.percentile(v, 2.5)), "high": float(np.percentile(v, 97.5))} for k, v in d.items()}
out["paired_bootstrap_calibration_and_coverage_plain"] = {
    "method": "paired percentile bootstrap, 10000 resamples, numpy.random.default_rng(0), one shared index matrix; a minus b; computed from the frozen plain predictions (Brier per row, 15-bin equal-width ECE on max prob, errors among the ceil(c*N) most confident rows with dataset_index tie-break)",
    "n": 2000, "seed": 0, "n_bootstrap": NB, "pairs": pairs}
out["accuracy_on_first_rows_plain"] = {"note": "accuracy restricted to dataset_index < k, for comparison with the Laya authors' 600-row / 400-row runs",
    "subsets": {f"first_{k}_rows": {m: {"n": k, "accuracy": float(ok[m][di < k].mean()), "n_correct": int(ok[m][di < k].sum())} for m in M} for k in (600, 400)}}
sel = P["jev"].max(1) >= 1.0
out["jev_confidence_exactly_one_plain"] = {"note": "rows whose renormalised max probability is 1.00 (API rounds to 2 decimals)", "n": int(sel.sum()), "share": float(sel.mean()), "n_correct": int(ok["jev"][sel].sum()), "accuracy": float(ok["jev"][sel].mean())}
json.dump(out, open("results/supplementary.json", "w"), indent=1, default=float)
print("added keys 10; totals:", len(out))
