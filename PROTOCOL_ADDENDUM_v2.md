# Protocol addendum v2: follow-up datasets (frozen 2026-09-20, before any inference on them)

Motivation: the primary benchmark (`PROTOCOL.md`, dair-ai/emotion) turned out to be inside the
training lineage of PrismNLI-0.4B (via `deberta-v3-large-zeroshot-v2.0`). This addendum adds three
datasets chosen because they are **absent from every disclosed training list** of the three systems
(see the constraints table in README.md). Everything in `PROTOCOL.md` still applies unless
overridden here. Nothing may change after the first test inference on a dataset.

## 1. Datasets (all evaluated on the FULL split, no sampling)

| key | source (pinned revision) | evaluated split | n | classes | smoke/warm-up rows (never evaluated) |
|---|---|---|---|---|---|
| `tweet_topic` | `cardiffnlp/tweet_topic_single` @ `87b7a0d1c402dbb481db649569c556d9aa27ac05`, raw file `dataset/split_temporal/test_2021.single.json` (the repo's loading script is no longer runnable under `datasets` 5) | `test_2021` | 1693 | 6 | `dataset/split_temporal/validation_2021.single.json`, first 10 rows |
| `fin_topic` | `zeroshot/twitter-financial-news-topic` @ `acbc8af2a35ccf0916124efcbe9e6cf25f191012` | `validation` (the dataset has no test split; validation is used purely as an evaluation set, never for tuning) | 4117 | 20 | `train`, first 10 rows |
| `daily_dialog` | `OpenRL/daily_dialog` @ `1668faf0c0dc44664f108c489fd0666128db2c48` (parquet mirror of `li2017dailydialog/daily_dialog`; identical schema and split sizes 11118/1000/1000; the original zip at yanran.li is no longer served, so the mirror is the only pinned reproducible source) | `test`, flattened to one row per utterance | 7740 | 7 | `validation`, first 10 utterances |

`dataset_index` = row position in the evaluated split file (for `daily_dialog`: running utterance
index over the flattened test split; `dialog_id` and `turn_id` are also stored). Each utterance is
scored **without dialogue context** (single-text classification, same as the other datasets); the
dataset's own labels are per utterance.

Class imbalance is kept as-is (no rebalancing). `daily_dialog` is 82% `no emotion`, so macro-F1 and
per-class results carry the information there; the majority-class accuracy is reported next to every
accuracy.

## 2. Labels, instruction, hypothesis template (identical strings for all three systems)

Label strings below are what every system sees as criteria keys (Jev, Laya) and inside the NLI
hypothesis (PrismNLI). Order = dataset label id order.

### `tweet_topic`
Labels: `arts & culture`, `business & entrepreneurs`, `pop culture`, `daily life`,
`sports & gaming`, `science & technology` (dataset ids 0..5; underscores in the raw names replaced by
spaces, `&` kept).
Instruction: `Which single topic does this tweet belong to?`
Hypothesis: `The topic of this tweet is {label}.`

### `fin_topic`
Labels (verbatim from the dataset card, ids 0..19): `Analyst Update`, `Fed | Central Banks`,
`Company | Product News`, `Treasuries | Corporate Debt`, `Dividend`, `Earnings`, `Energy | Oil`,
`Financials`, `Currencies`, `General News | Opinion`, `Gold | Metals | Materials`, `IPO`,
`Legal | Regulation`, `M&A | Investments`, `Macro`, `Markets`, `Politics`, `Personnel Change`,
`Stock Commentary`, `Stock Movement`.
Instruction: `Which single topic does this financial news tweet belong to?`
Hypothesis: `The topic of this tweet is {label}.`

### `daily_dialog`
Labels (ids 0..6): `no emotion`, `anger`, `disgust`, `fear`, `happiness`, `sadness`, `surprise`.
Instruction: `Which single emotion is expressed in this utterance?`
Hypothesis: `The emotion expressed in this utterance is {label}.` (for id 0 this reads
`... is no emotion.`; it is used verbatim, no special-casing).

## 3. Variants
Only `plain` (bare labels, empty-string / `None` criteria descriptions). No `defined` variant for the
follow-up datasets (no definitions were written, so none can be selected post hoc).

## 4. Systems
Unchanged from `PROTOCOL.md` §3: Jev pinned `typesafe/jev-1.13-20260917`, concurrency 8, same retry
policy, cache per dataset (`results/<key>/cache/jev_plain.jsonl`); Laya general English checkpoint,
shipped temperature (note: Laya's `temperature_by_options` bucket `choice:11+` = 0.1006 applies to
`fin_topic`; `choice:6-10` = 1.00002 to the 6/7-class sets; this is out-of-the-box behaviour and is
recorded, not altered); PrismNLI softmax over entailment logits, `only_first` truncation, fp32 MPS,
batch = all hypotheses of one example in one forward pass (20 for `fin_topic`).

## 5. Metrics, latency, outputs
Unchanged (`PROTOCOL.md` §5-§6) with the class count taken from the dataset (`N_CLASSES` per run).
ECE 15 equal-width bins; adaptive 10-bin secondary; bootstrap 10 000 / seed 0; exact McNemar; paired
bootstrap. Outputs under `results/<key>/` with the same file names as the primary run
(`raw_predictions.parquet`, `summary.json`, `summary.csv`, `env.json`, `plots/`). Latency of local
models is warm, batch size 1 (one example = one forward pass of all its hypotheses/options), after 10
warm-up calls on the smoke rows. Jev latency is remote end-to-end under concurrency 8.

## 6. Constraints table (why these datasets)
Recorded in README.md ("Dataset constraints"): for each dataset, its status against the
`deberta-v3-large-zeroshot-v2.0` training CSV (PrismNLI lineage), the PrismNLI synthetic data seeds
(WANLI), Laya's disclosed training mix and eval harness flags, and Jev's (undisclosed) corpus.
"Absent from disclosed lists" is the strongest statement available; it is not "never seen".
