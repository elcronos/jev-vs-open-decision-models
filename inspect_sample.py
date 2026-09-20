"""Select >=20 test examples for qualitative error analysis (PROTOCOL: inspection only, never alters benchmark).

Categories: every gold class; all-agree-correct; all-disagree (three distinct predictions); only Jev correct;
only PrismNLI correct; only Laya correct; high-confidence wrong per model. Deterministic (seed 0).
Writes results/inspection_sample.md and .json from the FROZEN primary parquet.
"""
import json, numpy as np, pandas as pd
L = ["sadness", "joy", "love", "anger", "fear", "surprise"]
df = pd.read_parquet("results/frozen_primary_plain/raw_predictions.parquet")
rng = np.random.default_rng(0)
M = ["jev", "prismnli", "laya"]
for m in M:
    df[f"{m}_ok"] = df[f"{m}_pred"] == df.gold_id
def pick(mask, k, name):
    ids = df.index[mask].to_numpy()
    if len(ids) == 0: return []
    ch = rng.choice(ids, size=min(k, len(ids)), replace=False)
    return [(int(i), name) for i in ch]
sel = []
allok = df.jev_ok & df.prismnli_ok & df.laya_ok
sel += pick(allok, 3, "all_correct")
sel += pick(~df.jev_ok & ~df.prismnli_ok & ~df.laya_ok & (df[[f"{m}_pred" for m in M]].nunique(axis=1) == 3), 3, "all_wrong_three_distinct")
sel += pick(df.jev_ok & ~df.prismnli_ok & ~df.laya_ok, 3, "only_jev_correct")
sel += pick(~df.jev_ok & df.prismnli_ok & ~df.laya_ok, 3, "only_prismnli_correct")
sel += pick(~df.jev_ok & ~df.prismnli_ok & df.laya_ok, 3, "only_laya_correct")
for m in M:
    sel += pick(~df[f"{m}_ok"] & (df[f"{m}_confidence"] >= 0.95), 2, f"{m}_highconf_wrong")
# ensure every gold class covered
covered = {int(df.gold_id[i]) for i, _ in sel}
for g in range(6):
    if g not in covered:
        sel += pick(df.gold_id == g, 1, f"class_fill_{L[g]}")
seen = set(); rows = []
for i, cat in sel:
    if i in seen: continue
    seen.add(i); r = df.loc[i]
    rows.append({"category": cat, "dataset_index": int(r.dataset_index), "text": r.text, "gold": L[int(r.gold_id)],
                 **{f"{m}_pred": L[int(r[f"{m}_pred"])] for m in M},
                 **{f"{m}_probs": {l: round(float(r[f"{m}_p_{l}"]), 3) for l in L} for m in M}})
json.dump(rows, open("results/inspection_sample.json", "w"), indent=1)
with open("results/inspection_sample.md", "w") as f:
    f.write(f"# Qualitative inspection sample (n={len(rows)}), primary `plain` run, frozen\n\n")
    for r in rows:
        f.write(f"## [{r['category']}] idx {r['dataset_index']} gold={r['gold']}\n> {r['text']}\n\n")
        for m in M:
            f.write(f"- **{m}** -> {r[f'{m}_pred']}  {r[f'{m}_probs']}\n")
        f.write("\n")
print(len(rows), "rows;", "classes:", sorted({r['gold'] for r in rows}))
# agreement stats
print("all three agree:", (df.jev_pred.eq(df.prismnli_pred) & df.jev_pred.eq(df.laya_pred)).mean())
print("jev==laya:", df.jev_pred.eq(df.laya_pred).mean(), "jev==prism:", df.jev_pred.eq(df.prismnli_pred).mean(), "prism==laya:", df.prismnli_pred.eq(df.laya_pred).mean())
print("counts only_x_correct:", {m: int((df[f"{m}_ok"] & ~df[[f"{o}_ok" for o in M if o != m]].any(axis=1)).sum()) for m in M}, "all wrong:", int((~df[[f"{m}_ok" for m in M]].any(axis=1)).sum()), "all right:", int(allok.sum()))
# oracle
print("oracle any-correct acc:", df[[f"{m}_ok" for m in M]].any(axis=1).mean())
