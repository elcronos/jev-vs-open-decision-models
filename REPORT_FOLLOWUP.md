# Follow-up: does the emotion result survive on datasets absent from every disclosed training list?

Companion to [`REPORT.md`](REPORT.md). Three additional datasets were run under the frozen
[`PROTOCOL_ADDENDUM_v2.md`](PROTOCOL_ADDENDUM_v2.md) on 2026-09-20 (04:03 to 05:06 UTC), `plain`
variant only, full evaluated splits. Every number below comes from
`results/cross_dataset_summary.{csv,json}` (built by `cross_dataset_summary.py` from
`results/summary.json` for the primary emotion run, `results/<key>/summary.json` for the
follow-ups, and the frozen `raw_predictions.parquet` files for the paired bootstrap of macro-F1 /
Brier / ECE differences and PrismNLI's independent-entailment aggregates, keys
`pairwise_paired_bootstrap_extra` and `prismnli_independent_entailment`), the per-dataset
`summary.json` / `env.json`, and, for the primary dataset's lineage only,
`results/contamination_research.json`. The training-list rows for the three follow-up datasets were
checked against the same pinned `zeroshot-v2.0` CSV but are not recorded in that JSON. No
third-party published numbers appear in any table.

Terminology: "clean" below means *absent from every disclosed training list of the three systems*,
not *verified unseen* (Section 2 and 6 give the residual risk).

Rounding: accuracy, F1, Brier, ECE, NLL to 3 decimals; latency to whole ms; cost to 4 decimals USD.

---

## 1. Executive conclusion

The question was whether the primary finding, PrismNLI-0.4B far ahead of Jev 1.13 and Laya on
`dair-ai/emotion` (0.725 vs 0.587 vs 0.587), holds on datasets that appear in none of the three
systems' disclosed training lists. It does not. On both topic sets Jev is the strongest system on
every quality metric; on the clean emotion set (`daily_dialog`) the evidence is mixed:

- `tweet_topic` (n = 1693, 6 classes): Jev 0.793 vs PrismNLI 0.633 vs Laya 0.632; Jev minus PrismNLI
  +0.161 [+0.139, +0.182], McNemar p = 3.3e-48.
- `fin_topic` (n = 4117, 20 classes): Jev 0.670 vs PrismNLI 0.352 vs Laya 0.342; Jev minus PrismNLI
  +0.317 [+0.300, +0.335], p = 1.7e-239.
- `daily_dialog` (n = 7740, 7 classes, 82% `no emotion`): PrismNLI is significantly more accurate
  (0.765 vs Jev 0.710; Jev minus PrismNLI -0.055 [-0.066, -0.045], McNemar p = 1.4e-24) and has the
  lower Brier (0.409 vs 0.460) and NLL (1.250 vs 1.451); every system is below the 0.817 majority
  baseline. On macro-F1 Jev leads (0.385 [0.362, 0.406] vs 0.345 [0.322, 0.369] vs 0.275
  [0.256, 0.294]); the Jev and PrismNLI marginal CIs overlap slightly, but the paired bootstrap of
  the macro-F1 difference is +0.039 [+0.019, +0.060] (bootstrap p = 0.0002).

Jev has the lowest ECE on all three clean sets (0.063 / 0.166 / 0.156; paired ECE difference to
PrismNLI -0.118 [-0.136, -0.093], -0.032 [-0.048, -0.015], -0.019 [-0.030, -0.008]), whereas on
emotion it was second worst (0.281). On the two topic sets Jev also has the lower Brier and NLL; on
`daily_dialog` Brier (paired difference +0.051 [+0.035, +0.068]) and NLL favour PrismNLI.

The clean-set reversal is therefore established for topic classification and only partial for
emotion. Task type is confounded with lineage status in this design: the two datasets on which
PrismNLI has the higher accuracy are the two emotion tasks (one in its lineage, one not), and the
two on which Jev wins are the two topic tasks. PrismNLI's emotion lead on `dair-ai/emotion` is
consistent with the verified exposure of its initialisation checkpoint to that dataset's train and
validation splits, and an equally consistent alternative is task-type specialisation (a declarative
NLI hypothesis suits emotion labels; a native `choice` primitive suits topic taxonomies). Hedges:
three datasets, all English Twitter or scripted dialogue, one frozen prompt per dataset, and
"absent from disclosed lists" is weaker than "never seen".

---

## 2. Motivation and dataset constraints

`REPORT.md` Section 5 established that PrismNLI-0.4B is initialised from
`deberta-v3-large-zeroshot-v2.0`, whose pinned training CSV lists `dair-ai/emotion`
(`emotion6_twitter`, `used_in_v1.1 = TRUE`, train + validation, with a declarative
emotion-hypothesis template; the card states up to 500 rows per class, which is not verifiable
against the public notebook, whose saved output shows a 10,344-row `emotiondair` NLI pool). That checkpoint's own card reports macro-F1 on the
emotion set of 0.484 zero-shot versus 0.688 once emotion data was included (context, not our
measurement). The primary result therefore could not distinguish capability from inherited
familiarity. The follow-up datasets were chosen so that this ambiguity is removed as far as public
disclosures allow. The full evidence table is in
[README.md, "Dataset constraints"](README.md#dataset-constraints-follow-up-datasets-protocol_addendum_v2md-6);
the summary, with the primary dataset added for contrast:

| dataset (pinned) | split, n, classes | `zeroshot-v2.0` training CSV (PrismNLI lineage) | PrismNLI synthetic seeds (WANLI) | Laya disclosed mix / eval harness | Jev corpus | verdict |
|---|---|---|---|---|---|---|
| `dair-ai/emotion` @ `cab853a1` (primary) | `test`, 2000, 6 | **`used_in_v1.1 = TRUE`** (train + validation) | not a seed | card: held out | undisclosed | **inherited exposure verified** |
| `cardiffnlp/tweet_topic_single` @ `87b7a0d1` | `test_2021`, 1693, 6 | not in the CSV (sister multi-label set `FALSE`, `excluded`) | not a seed | not in table, not in harness | undisclosed | absent from all disclosed lists |
| `zeroshot/twitter-financial-news-topic` @ `acbc8af2` | `validation`, 4117, 20 | `FALSE`, `future_use = excluded` | not a seed | not in table, not in harness | undisclosed | absent; explicitly excluded by the zeroshot-v2.0 authors |
| `OpenRL/daily_dialog` @ `1668faf0` | `test` flattened to utterances, 7740, 7 | `FALSE`, `future_use = later` | not a seed | not in table, not in harness; Laya has an unnamed "emotion and tone" task family | undisclosed | absent; residual risk via Laya's unnamed emotion family |

"Absent from disclosed lists" is not "never seen". Jev's corpus is self-made and undisclosed;
Laya's prior checkpoint is undisclosed. Dataset age is a residual-risk factor for such corpora.
Measured: the `date` column of the evaluated `tweet_topic` split runs from 2020-09-06 to 2021-08-29
(`results/tweet_topic/raw_predictions.parquet`). External context, not our measurement: DailyDialog
is a 2017 paper, the financial-news set appeared on the Hub in 2022 (its card gives no tweet dates),
and the primary emotion set is 2018 Twitter (EMNLP 2018, quoted in `contamination_research.json`).
All predate every model here, so any of them could sit in an undisclosed pre-training crawl. What the constraint buys is narrower and specific: none of the
three clean sets is in the one training list we could verify, the one that did contain the primary
dataset.

---

## 3. Protocol

`PROTOCOL_ADDENDUM_v2.md` was frozen before any inference on the new datasets and inherits
everything from `PROTOCOL.md` that it does not override: model revisions (Jev
`typesafe/jev-1.13-20260917`, echoed in every response; PrismNLI `02b9102b`; Laya `c5d78730`),
Jev concurrency 8 and retry policy, Laya shipped temperature, PrismNLI softmax over entailment
logits with `only_first` truncation and all hypotheses of one example in one forward pass,
metrics, 10 000-resample bootstrap with seed 0, exact McNemar, paired bootstrap.

Every system saw identical strings per dataset (Jev and Laya as instruction plus bare criteria
keys; PrismNLI inside the hypothesis template), verbatim from the addendum:

| dataset | instruction | labels (dataset id order) | PrismNLI hypothesis |
|---|---|---|---|
| `tweet_topic` | `Which single topic does this tweet belong to?` | `arts & culture`, `business & entrepreneurs`, `pop culture`, `daily life`, `sports & gaming`, `science & technology` | `The topic of this tweet is {label}.` |
| `fin_topic` | `Which single topic does this financial news tweet belong to?` | `Analyst Update`, `Fed \| Central Banks`, `Company \| Product News`, `Treasuries \| Corporate Debt`, `Dividend`, `Earnings`, `Energy \| Oil`, `Financials`, `Currencies`, `General News \| Opinion`, `Gold \| Metals \| Materials`, `IPO`, `Legal \| Regulation`, `M&A \| Investments`, `Macro`, `Markets`, `Politics`, `Personnel Change`, `Stock Commentary`, `Stock Movement` | `The topic of this tweet is {label}.` |
| `daily_dialog` | `Which single emotion is expressed in this utterance?` | `no emotion`, `anger`, `disgust`, `fear`, `happiness`, `sadness`, `surprise` | `The emotion expressed in this utterance is {label}.` (used verbatim for `no emotion`) |

Full evaluated splits, no sampling, no rebalancing. Only the `plain` variant exists (no
definitions were written, so no `defined` variant could be chosen after the fact). `fin_topic` has
no test split; its `validation` split is used purely as an evaluation set and was never used for
tuning. `daily_dialog` utterances are scored without dialogue context. Smoke and warm-up rows come
from a different split and were never evaluated. Metrics are unchanged from `PROTOCOL.md` §5 with
the class count taken from the dataset. All runs finished with zero per-example errors and
`deviation = null`; Jev needed one retry in total (on `fin_topic`). Jev's returned `choice`
disagreed with the argmax of its 2-decimal probabilities on 2 / 9 / 12 rows (`tweet_topic` /
`fin_topic` / `daily_dialog`; `n_pred_mismatch` in each `summary.json`); the argmax was scored, as
in the primary run (4 such rows there).

---

## 4. Results

### 4.1 Cross-dataset headline table

Brackets are 95% percentile bootstrap CIs (10 000 resamples, seed 0). ECE is the 15-bin
equal-width value on max probability. Jev p50 is remote end-to-end (Perth client, 8 concurrent
streams); PrismNLI and Laya p50 are on-device compute (M1 Max, MPS, fp32, batch size 1) and are
not comparable to it. `maj` = majority-class accuracy.

| dataset | n | classes | maj | model | accuracy [CI] | macro-F1 [CI] | Brier | ECE | NLL | p50 latency |
|---|---|---|---|---|---|---|---|---|---|---|
| emotion (in PrismNLI lineage) | 2000 | 6 | 0.348 | Jev 1.13 | 0.587 [0.565, 0.608] | 0.500 [0.471, 0.528] | 0.667 | 0.281 | 2.845 | 349 ms (remote e2e) |
| | | | | PrismNLI-0.4B | **0.725** [0.705, 0.744] | **0.647** [0.620, 0.673] | 0.441 | 0.174 | 1.174 | 58 ms (local) |
| | | | | Laya | 0.587 [0.565, 0.609] | 0.493 [0.463, 0.522] | 0.707 | 0.307 | 2.032 | 31 ms (local) |
| tweet_topic | 1693 | 6 | 0.396 | Jev 1.13 | **0.793** [0.774, 0.812] | **0.694** [0.667, 0.718] | 0.294 | 0.063 | 0.703 | 348 ms (remote e2e) |
| | | | | PrismNLI-0.4B | 0.633 [0.609, 0.655] | 0.544 [0.516, 0.571] | 0.537 | 0.181 | 1.197 | 85 ms (local) |
| | | | | Laya | 0.632 [0.609, 0.656] | 0.461 [0.434, 0.487] | 0.505 | 0.129 | 1.091 | 35 ms (local) |
| fin_topic | 4117 | 20 | 0.207 | Jev 1.13 | **0.670** [0.656, 0.684] | **0.630** [0.610, 0.647] | 0.509 | 0.166 | 1.788 | 338 ms (remote e2e) |
| | | | | PrismNLI-0.4B | 0.352 [0.338, 0.367] | 0.256 [0.240, 0.272] | 0.806 | 0.199 | 2.170 | 195 ms (local) |
| | | | | Laya | 0.342 [0.327, 0.357] | 0.362 [0.341, 0.381] | 1.249 | 0.610 | 7.841 | 44 ms (local) |
| daily_dialog | 7740 | 7 | 0.817 | Jev 1.13 | 0.710 [0.700, 0.720] | **0.385** [0.362, 0.406] | 0.460 | 0.156 | 1.451 | 330 ms (remote e2e) |
| | | | | PrismNLI-0.4B | **0.765** [0.756, 0.775] | 0.345 [0.322, 0.369] | 0.409 | 0.176 | 1.250 | 149 ms (local) |
| | | | | Laya | 0.614 [0.603, 0.625] | 0.275 [0.256, 0.294] | 0.601 | 0.208 | 1.253 | 53 ms (local) |

Figure: `results/plots/cross_dataset_accuracy.png` (accuracy and macro-F1 with CIs, ECE; dotted
line = majority class).

### 4.2 Pairwise tests (all 12 pairs)

Exact McNemar on discordant rows (`b` = first model right and second wrong, `c` = the reverse);
paired bootstrap difference is first minus second.

| dataset | pair | McNemar p | b / c | accuracy diff [95% CI] |
|---|---|---|---|---|
| emotion | Jev vs PrismNLI | 1.5e-41 | 85 / 361 | -0.138 [-0.158, -0.118] |
| emotion | Jev vs Laya | 1.000 | 218 / 219 | -0.001 [-0.022, +0.020] |
| emotion | PrismNLI vs Laya | 4.1e-35 | 396 / 121 | +0.138 [+0.116, +0.159] |
| tweet_topic | Jev vs PrismNLI | 3.3e-48 | 327 / 55 | +0.161 [+0.139, +0.182] |
| tweet_topic | Jev vs Laya | 1.0e-37 | 375 / 102 | +0.161 [+0.137, +0.185] |
| tweet_topic | PrismNLI vs Laya | 1.000 | 216 / 215 | +0.001 [-0.024, +0.024] |
| fin_topic | Jev vs PrismNLI | 1.7e-239 | 1530 / 223 | +0.317 [+0.300, +0.335] |
| fin_topic | Jev vs Laya | 6.1e-272 | 1518 / 168 | +0.328 [+0.311, +0.345] |
| fin_topic | PrismNLI vs Laya | 0.284 | 791 / 748 | +0.010 [-0.008, +0.029] |
| daily_dialog | Jev vs PrismNLI | 1.4e-24 | 665 / 1093 | -0.055 [-0.066, -0.045] |
| daily_dialog | Jev vs Laya | 4.0e-55 | 1507 / 767 | +0.096 [+0.084, +0.107] |
| daily_dialog | PrismNLI vs Laya | 2.0e-135 | 1746 / 578 | +0.151 [+0.139, +0.163] |

On emotion Jev and Laya are tied and PrismNLI is 14 points ahead; on both topic sets PrismNLI and
Laya are tied (p = 1.000 and 0.284) and Jev is 16 to 33 points ahead (16 over both on `tweet_topic`;
32 over PrismNLI and 33 over Laya on `fin_topic`). On `daily_dialog` all three pairs differ.

Paired bootstrap of the other metrics (same resample indices; `pairwise_paired_bootstrap_extra` in
`cross_dataset_summary.json`), Jev minus PrismNLI: macro-F1 +0.149 [+0.121, +0.177] on
`tweet_topic`, +0.374 [+0.350, +0.396] on `fin_topic`, +0.039 [+0.019, +0.060] on `daily_dialog`;
Brier -0.244 [-0.271, -0.217], -0.297 [-0.319, -0.276], +0.051 [+0.035, +0.068]; ECE -0.118
[-0.136, -0.093], -0.032 [-0.048, -0.015], -0.019 [-0.030, -0.008]. Every interval excludes zero;
on `daily_dialog` the Brier difference favours PrismNLI and the macro-F1 and ECE differences favour
Jev.

### 4.3 tweet_topic (6 topics, test_2021)

*Per-class F1.* Jev: `sports & gaming` 0.964, `science & technology` 0.814, `pop culture` 0.775,
`business & entrepreneurs` 0.701, `daily life` 0.682, `arts & culture` 0.227 (support 48).
PrismNLI: 0.920 / 0.502 / 0.518 / 0.613 / 0.541 / 0.173. Laya: 0.885 / 0.323 / 0.673 / 0.365 /
0.379 / 0.142. Jev leads on every class; the smallest class, `arts & culture`, is poor for all
three because all three over-predict it (Jev 199, PrismNLI 334, Laya 163 predictions against 48
gold rows).

*Top confusions (gold to predicted).* All three: `pop culture` to `arts & culture` (Jev 163,
PrismNLI 267, Laya 131). PrismNLI additionally pushes `pop culture` to `daily life` (66) and
`science & technology` (57); its `pop culture` recall is 0.359 against Jev's 0.662. Laya
over-predicts `science & technology` (284 vs 88 gold) from `pop culture` (74), `sports & gaming`
(55) and `daily life` (53).

*Calibration.* Jev ECE 0.063 (adaptive 10-bin 0.060), Brier 0.294, NLL 0.703, mean confidence 0.850
against accuracy 0.793, gold-probability exactly zero on 1.8% of rows. PrismNLI ECE 0.181, mean
confidence 0.791 against 0.633. Laya ECE 0.129, mean confidence 0.761 against 0.632. Accuracy at
50% / 80% coverage: Jev 0.954 / 0.866, PrismNLI 0.837 / 0.700, Laya 0.835 / 0.708.

### 4.4 fin_topic (20 topics, validation)

*Measurement, PrismNLI.* PrismNLI predicts `Markets` for 1567 of 4117 rows (38.1%; 125 gold rows),
giving `Markets` precision 0.074 at recall 0.928. Its largest confusions are all into `Markets`:
`Stock Commentary` 291, `Macro` 236, `Company | Product News` 171, `Fed | Central Banks` 134. Three
classes get F1 0.000 (`Analyst Update`, `Treasuries | Corporate Debt` with zero predictions, `Gold |
Metals | Materials`) and three more are near it (`Stock Movement` 0.010, `Dividend` 0.020, `Fed |
Central Banks` 0.045). In the independent entailment view (`prismnli_indep_probs_json`, aggregated
in `cross_dataset_summary.json`), P(entail) exceeds 0.5 for at least two labels on 83.9% of rows
(mean 4.1 labels per row), the mean maximum P(entail) is 0.954, and the `Markets` hypothesis alone
exceeds 0.5 on 71.3% of rows. The same statistic is 4.7% on `tweet_topic`, 11.6% on `daily_dialog`
and 5.0% on emotion, so the multi-entailment is specific to this label set. Its mean max-softmax
confidence is 0.551 (accuracy 0.352), ECE 0.199, NLL 2.170.

*Interpretation.* The hypothesis `The topic of this tweet is Markets.` is entailed by almost any
financial-news tweet, so the binary NLI head accepts it, and the many near-synonymous labels
(`Markets`, `Stock Commentary`, `Stock Movement`, `Macro`, `Financials`) are not mutually exclusive
as hypotheses. The softmax over 20 entailment logits then hands the most generic label the argmax.
This is a property of the generic template plus a label set designed as a taxonomy rather than as
contrastive statements, not necessarily a lack of financial knowledge; whether a contrastive
template removes the collapse was not tested.

*Measurement, Laya.* Laya's shipped `temperature_by_options` bucket for 11 or more options,
`choice:11+` = 0.1006, was applied (recorded in `summary.json` and `env.json`); the 6/7-class sets
use `choice:6-10` = 1.00002. Laya's mean confidence is 0.952 against accuracy 0.342, ECE 0.610, Brier
1.249, NLL 7.841, and the gold label receives probability exactly zero (4-decimal rounding) on 51.0%
of rows. It predicts `Financials` for 1006 rows (160 gold; from `Company | Product News` 273, `Stock
Commentary` 197, `General News | Opinion` 113, `Macro` 93) and `Stock Commentary` only 7 times
(528 gold rows, F1 0.004). Accuracy at 50% / 80% coverage: 0.453 / 0.374, barely above its full
coverage 0.342.

*Interpretation.* The near-one-hot outputs (mean confidence 0.952 against accuracy 0.342) are
consistent with the 0.1 temperature sharpening an already wrong distribution; no run at another
temperature exists, so the mechanism is inferred, not ablated. This is out-of-the-box behaviour and
was not altered. Temperature does not change the argmax, so Laya's accuracy and macro-F1 on
`fin_topic` are unaffected by the bucket; only the probability-based metrics are. Laya's macro-F1
(0.362) is above PrismNLI's (0.256) because its errors are spread across classes rather than
concentrated on one, although it too collapses onto a single label (`Financials`, 24.4% of its
predictions against 3.9% gold).

*Jev.* Macro-F1 0.630 with 16 of 20 classes above 0.5 F1 (`Dividend` 0.900, `Personnel Change` 0.888,
`Politics` 0.832, `Fed | Central Banks` 0.791, `Company | Product News` 0.775). Its one collapse is
`Financials` (F1 0.010): 146 of 160 rows go to `Earnings`, a label-taxonomy ambiguity the dataset
card does not resolve. Other confusions are semantically adjacent pairs (`Stock Commentary` to
`Stock Movement` 67 and to `Markets` 51, `Company | Product News` to `M&A | Investments` 77).
Jev's ECE is 0.166, mean confidence 0.835 against 0.670, 7.8% exact-zero gold probabilities;
accuracy at 50% / 80% coverage 0.828 / 0.737.

### 4.5 daily_dialog (7 emotions, utterances without context)

*Measurement.* Gold is 81.7% `no emotion`. Every system is below the majority baseline: PrismNLI
0.765, Jev 0.710, Laya 0.614. Predicted `no emotion` rate: PrismNLI 0.807, Jev 0.622, Laya 0.589,
against gold 0.817. Macro-F1: Jev 0.385 [0.362, 0.406], PrismNLI 0.345 [0.322, 0.369], Laya 0.275
[0.256, 0.294]; the PrismNLI and Laya intervals are disjoint, the Jev and PrismNLI intervals overlap
on [0.362, 0.369], and the paired bootstrap of the Jev minus PrismNLI macro-F1 difference is +0.039
[+0.019, +0.060] (bootstrap p = 0.0002). `no emotion` F1: PrismNLI 0.868, Jev 0.811, Laya 0.743. `happiness` (1019 rows) F1:
Jev 0.560, Laya 0.394, PrismNLI 0.389 (PrismNLI sends 653 `happiness` rows to `no emotion`; recall
0.283). The five small classes (17 to 118 rows each) are below 0.38 F1 for everyone; `fear`
(17 rows) is 0.142 / 0.143 / 0.066.

*Top confusions.* Jev and Laya over-read emotion into neutral utterances (`no emotion` to
`happiness` 816 and 1272, to `sadness` 305 and 438, to `anger` 270 and 177). PrismNLI's dominant
error is the opposite direction (`happiness` to `no emotion` 653) plus `no emotion` to `sadness` 236.

*Calibration.* Jev ECE 0.156, Brier 0.460, NLL 1.451, mean confidence 0.865, 5.9% exact zeros;
PrismNLI ECE 0.176, Brier 0.409, NLL 1.250, mean confidence 0.940; Laya ECE 0.208, Brier 0.601, NLL
1.253, mean confidence 0.821. Accuracy at 50% / 80% coverage: PrismNLI 0.877 / 0.825, Jev 0.828 /
0.763, Laya 0.658 / 0.641.

*Interpretation.* PrismNLI's accuracy lead is the majority class: the hypothesis `The emotion
expressed in this utterance is no emotion.` is the argmax for 80.7% of rows and its independent
P(entail) exceeds 0.5 on 79.6% of rows, close to the 81.7% base rate, so accuracy follows. Only
11.6% of rows have two or more independently entailed labels, so this is a different mechanism
from the `fin_topic` collapse. When the class prior is removed (macro-F1) Jev is ahead, by a paired
margin that excludes zero but is small (about 4 points). Because an utterance
is judged without its dialogue, a fraction of the "errors" are unresolvable from the text alone.

### 4.6 Latency, throughput and cost

Local and remote numbers are different quantities and are kept apart. Local wall time is device
wall time on one M1 Max (MPS, fp32, batch size 1) and is labelled an estimate; it scales with
hypothesis count for PrismNLI (one forward pass with 6, 7 or 20 hypotheses).

| dataset | n | Jev p50 / p95 (remote e2e) | Jev wall, conc. 8 | Jev input tokens | Jev cost (per 1000 decisions) | PrismNLI p50 / p95 (local) | PrismNLI wall (est.) | PrismNLI peak MPS | Laya p50 / p95 (local) | Laya wall (est.) |
|---|---|---|---|---|---|---|---|---|---|---|
| emotion | 2000 | 349 / 593 ms | 97 s | 676 323 | $0.0284 ($0.0142) | 58 / 83 ms | 123 s | 2.05 GiB | 31 / 35 ms | 63 s |
| tweet_topic | 1693 | 348 / 608 ms | 76 s (1595 calls, see note) | 628 743 | $0.0264 ($0.0156) | 85 / 125 ms | 150 s | 2.05 GiB | 35 / 40 ms | 60 s |
| fin_topic | 4117 | 338 / 446 ms | 187 s | 1 985 374 | $0.0834 ($0.0203) | 195 / 318 ms | 856 s | 3.08 GiB | 44 / 46 ms | 180 s |
| daily_dialog | 7740 | 330 / 449 ms | 338 s | 2 675 430 | $0.1124 ($0.0145) | 149 / 312 ms | 1168 s | 2.14 GiB | 53 / 197 ms | 566 s |

Note on `tweet_topic`: 98 of the 1693 Jev responses were produced by an aborted invocation started at
04:03:19Z (response ids `gen-dec-1789876999-*` onwards, 16 s before the logged run) and were reused
from `results/tweet_topic/cache/jev_plain.jsonl`; the logged run made 1595 remote calls plus one
warm-up (`env.json` `http_calls = 1596`, progress bar `0/1595`). The 76 s wall and any throughput
derived from it cover those 1595 calls, not 1693. The per-row latency, cost and token figures are
unaffected (every row's response carries its own values). `fin_topic` (4117 + 1 warm-up + 1 retry =
4119 calls) and `daily_dialog` (7741) have no such gap. The addendum freeze (02:59Z) precedes the
aborted invocation.

Laya peak MPS memory is about 2.02 GiB on every dataset; load time is 27 to 29 s for Laya and 1 to
4 s for PrismNLI (warm cache). Jev's remote p50 is flat across datasets (330 to 349 ms) and its
cost per decision moves with prompt length (20 labels on `fin_topic` cost 43% more per decision
than the 6-label emotion set). Jev totals: $0.2222 for 13 550 follow-up decisions plus $0.0284 for the primary run.

---

## 5. Required-analysis questions, revisited with four datasets

**Capability versus NLI.** `REPORT.md` 6(b) answered "no clear advantage for Jev" on one dataset.
With four datasets the picture is the reverse on the two topic sets: Jev is ahead by 16 accuracy
points on `tweet_topic` and 32 on `fin_topic` (McNemar p = 3.3e-48 and 1.7e-239), and by 15 and 37
macro-F1 points (paired CIs exclude zero). On `daily_dialog` PrismNLI is significantly more accurate
(p = 1.4e-24; -0.055 [-0.066, -0.045]) while Jev's macro-F1 is higher by +0.039 [+0.019, +0.060]
with overlapping marginal CIs. The one dataset where the NLI classifier wins on both accuracy and
macro-F1 is the one its lineage trained on; the one clean emotion set splits the two metrics.
On `fin_topic` all three systems collapse onto a single label with this generic template, in
different ways: PrismNLI's independent hypotheses onto the most generic label (`Markets`, 38.1% of
predictions, 3.0% gold), Laya onto `Financials` (24.4% of predictions, 3.9% gold) and Jev, more
narrowly, `Financials` to `Earnings` (146 of 160 rows). Whether a contrastive template removes the
PrismNLI collapse was not tested (Section 4.4, 6).

**Calibration.** Jev has the lowest ECE on the three clean sets, 0.063 (`tweet_topic`), 0.166
(`fin_topic`), 0.156 (`daily_dialog`), with paired differences to PrismNLI that exclude zero (Section
4.2), versus 0.281 on emotion. ECE is not the whole picture: on `daily_dialog` PrismNLI has the lower
Brier (0.409 vs 0.460, paired difference +0.051 [+0.035, +0.068]) and NLL (1.250 vs 1.451), so
"best calibrated" there depends on the metric. Jev's ECE level is dataset-dependent (0.281 on
emotion, where its accuracy was lowest and 15.1% of rows had exact-zero gold probability) and
tracks its accuracy: when it is right often, its high stated confidence (mean 0.835 to 0.867 on
every set) is justified; when it is wrong often, the same confidence is over-confidence. Its
2-decimal rounding still inflates NLL (1.8% to 7.8% exact zeros on the clean sets). PrismNLI's
ECE is flat at 0.174 to 0.199 on all four datasets, but its probabilities remain adapter-derived
(softmax over entailment logits from a binary head), and the `fin_topic` result shows the
construction can be confidently wrong in a systematic way (mean confidence 0.551, accuracy 0.352,
83.9% of rows with multiple entailed labels). Laya's calibration is consistent with its shipped
temperature buckets being the dominant factor: 0.129 to 0.307 ECE under `choice:6-10` (temperature
1.00002), 0.610 under `choice:11+` (0.1006); no temperature ablation was run.

**Selective classification.** Jev has the highest accuracy at 50% and 80% coverage on
`tweet_topic` (0.954 / 0.866) and `fin_topic` (0.828 / 0.737); PrismNLI has it on `daily_dialog`
(0.877 / 0.825, again driven by confident `no emotion`) and on emotion. Laya's risk-coverage curve
is the flattest everywhere (on `fin_topic` 50% coverage buys only 11 points).

**Latency and economics.** Unchanged in kind from `REPORT.md` 6(i). Jev's remote p50 is
330 to 349 ms regardless of dataset; local p50 is 31 to 53 ms for Laya and 58 to 195 ms for
PrismNLI, growing with the number of hypotheses. Jev's price per 1000 decisions is $0.0142 to
$0.0203 depending on prompt length. These remain different quantities and neither is a proxy for
the other.

**Which of the four candidate explanations does the whole evidence support?** `REPORT.md` 6(j)
rejected "Jev is a substantial capability improvement over NLI" and partly supported "different
strengths". With four datasets:

- *Jev is a substantial capability improvement over NLI:* supported on the two topic sets (16 and
  32 accuracy points, 15 and 37 macro-F1 points); mixed on `daily_dialog` (PrismNLI more accurate,
  Jev higher macro-F1 with overlapping marginal CIs); not supported on emotion. The contamination
  evidence is one way to reconcile these, and the emotion lead is consistent with inherited
  exposure; but task type is confounded with lineage status in this design, so neither a
  template-sensitivity account (Section 6) nor a task-type account (NLI stronger on emotion,
  choice models stronger on topic) can be excluded. Only a run of PrismNLI on an emotion set outside
  its lineage with matched difficulty, or an ablated checkpoint, would separate them.
- *Jev has similar raw ability but better calibration/efficiency/ergonomics:* the calibration half
  is supported on ECE on all three clean sets and on Brier/NLL on the two topic sets, not on
  Brier/NLL on `daily_dialog`; the "similar raw ability" half is not, because the ability gap on
  the topic sets is large.
- *Laya is an open-weight approximation of Jev's abstraction:* no longer supported. Laya matched Jev
  on emotion only; on the clean sets it trails Jev by 16, 33 and 10 accuracy points and has the
  lowest macro-F1 on two of three. It behaves like PrismNLI on the topic sets (tied twice) rather
  than like Jev.
- *The three approaches have different strengths:* still true for latency and hosting (Laya
  fastest locally, PrismNLI no temperature quirks, Jev no weights), and on quality the topic-set
  evidence points one way while the two emotion sets split by metric.

What changed relative to `REPORT.md`: the emotion result stands as measured, but it no longer
generalises: on the two topic sets the ranking reverses on every quality metric, and the single
dataset on which PrismNLI wins on both accuracy and macro-F1 is the single dataset in its training
lineage. Because task type and lineage are confounded, the reversal is established for topic
classification and only partially for emotion.

---

## 6. Limitations

- Three follow-up datasets, all English, all Twitter or scripted two-person dialogue; no long
  documents, no non-English text, no ordinal or multi-label tasks.
- Label noise. `daily_dialog` is 82% `no emotion` and utterances are scored without context, so
  part of every system's error is unresolvable from the input; `fin_topic`'s taxonomy has
  overlapping labels (`Financials` / `Earnings`, `Markets` / `Stock Commentary` / `Stock Movement`)
  and the card gives no definitions; `tweet_topic`'s `pop culture` / `arts & culture` boundary
  defeats all three systems.
- `fin_topic` has no test split; `validation` was used purely as an evaluation set, but it is the
  split the dataset authors intended for model selection.
- One frozen prompt per dataset and no `defined` variant, so PrismNLI's `fin_topic` collapse may
  be template sensitivity (a contrastive template or label definitions could change it) rather than
  a capability ceiling. Equally, PrismNLI's emotion lead may be template familiarity rather than
  dataset familiarity; the two are confounded and neither was ablated.
- Jev returns probabilities rounded to 2 decimals (exact-zero gold probability on 1.8% to 7.8% of
  clean-set rows), which inflates NLL by construction; Laya rounds to 4 decimals and its
  `choice:11+` temperature produces 51.0% exact zeros on `fin_topic`.
- Local latency and memory are Apple MPS fp32 at batch size 1, not CUDA; the PrismNLI cost of 20
  hypotheses per example would differ with batching.
- Undisclosed corpora: Jev's training data and Laya's prior checkpoint are unknown, so "absent
  from disclosed lists" cannot be strengthened. All four datasets predate all three models.
- Laya's undated "emotion and tone" training family could overlap DailyDialog-like data; it did
  not help Laya there (lowest accuracy and macro-F1).
- Task type is confounded with lineage status: the two emotion sets are the two on which PrismNLI
  has the higher accuracy (one in its lineage, one not) and the two topic sets are the two on
  which Jev wins. The design cannot separate "PrismNLI saw `dair-ai/emotion`" from "NLI with a
  declarative template is better at emotion".
- The follow-up runs happened after the primary result was known; the addendum was frozen before
  any inference on the new data, but the choice of datasets was made knowing which direction
  would be "interesting".

---

## 7. Reproducibility

Commands (repo root, venv Python 3.11, `OPENROUTER_API_KEY` set for Jev only):

```
python benchmark.py --dataset tweet_topic  --split eval --models jev,prismnli,laya --variants plain   # 04:03:36Z to 04:08:58Z; 98 Jev rows reused from an aborted invocation at 04:03:19Z (Section 4.6)
python benchmark.py --dataset fin_topic    --split eval --models jev,prismnli,laya --variants plain   # 04:09:05Z to 04:30:11Z
python benchmark.py --dataset daily_dialog --split eval --models jev,prismnli,laya --variants plain   # 04:30:21Z to 05:05:41Z
python cross_dataset_summary.py    # -> results/cross_dataset_summary.{csv,json} (incl. paired macro-F1/Brier/ECE bootstraps and PrismNLI independent-entailment aggregates), results/plots/cross_dataset_accuracy.{png,pdf}
for d in tweet_topic fin_topic daily_dialog; do (cd results/$d && shasum -a 256 -c SHA256SUMS); done
```

Revisions: `cardiffnlp/tweet_topic_single` @ `87b7a0d1c402dbb481db649569c556d9aa27ac05` (raw file
`dataset/split_temporal/test_2021.single.json`); `zeroshot/twitter-financial-news-topic` @
`acbc8af2a35ccf0916124efcbe9e6cf25f191012`; `OpenRL/daily_dialog` @
`1668faf0c0dc44664f108c489fd0666128db2c48`; `Jaehun/PrismNLI-0.4B` @
`02b9102b34d5dce1bf29c4e6903fea33e1a0ddeb`; `convaiinnovations/laya` @
`c5d78730f3493e4fe16d61507ef4b78eef7318cf`; Jev `typesafe/jev-1.13-20260917` (echoed by every
response). Packages as in `REPORT.md` Section 8 (torch 2.14.0, transformers 4.57.6, laya 0.3.3,
datasets 5.0.1, ...); recorded per run in `results/<key>/env.json`. Seeds 0 throughout.

Checksums (`results/<key>/SHA256SUMS`, SHA-256 prefixes): `tweet_topic` summary.json `9f8501cf`,
raw_predictions.parquet `2e8db820`; `fin_topic` summary.json `36612940`, parquet `671649fc`;
`daily_dialog` summary.json `9ab0c83f`, parquet `88e60a1d`.

Files:
- `PROTOCOL_ADDENDUM_v2.md` - frozen addendum
- `datasets_registry.py` - DatasetSpec per dataset (source, revision, split, labels, instruction, template, smoke rows)
- `cross_dataset_summary.py` - aggregation script; `results/cross_dataset_summary.csv`, `.json`
- `results/<key>/{summary.json,summary.csv,raw_predictions.parquet,env.json,SHA256SUMS,run_plain.log}` for `tweet_topic`, `fin_topic`, `daily_dialog`
- `results/<key>/cache/jev_plain.jsonl` - Jev responses keyed by `dataset_index`
- `results/<key>/plots/` - per-dataset figures; `results/plots/cross_dataset_accuracy.{png,pdf}`
- `results/summary.json`, `results/frozen_primary_plain/` - the unchanged primary emotion run

Figures:
- `results/plots/cross_dataset_accuracy.png` - accuracy and macro-F1 with 95% CIs and ECE for all three models on all four datasets; dotted line = majority-class accuracy.
- `results/<key>/plots/accuracy_macro_f1_ci.png` - per-dataset accuracy and macro-F1 with CIs.
- `results/<key>/plots/per_class_f1.png` - per-class F1 grouped by model (6, 20 or 7 classes).
- `results/<key>/plots/confusion_matrices.png` - confusion matrices, rows = gold (the `fin_topic` `Markets` column and `Financials` column show the PrismNLI and Laya collapses).
- `results/<key>/plots/reliability.png` - 15-bin reliability diagrams.
- `results/<key>/plots/risk_coverage.png` - selective-classification risk-coverage curves.
- `results/<key>/plots/confidence_hist.png` - max-probability histograms (Laya's `fin_topic` mass at 1.0 under temperature 0.10).
- `results/<key>/plots/latency.png` - latency distributions, remote and local plotted separately.
