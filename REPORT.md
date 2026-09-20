# Zero-shot emotion classification: Jev 1.13 vs PrismNLI-0.4B vs Laya

Apples-to-apples zero-shot benchmark on `dair-ai/emotion` (test split, n = 2000), run under the frozen protocol in `PROTOCOL.md` on 2026-09-20. All numbers in this report come from `results/summary.json` / `summary.csv` (checksummed copy of the primary run in `results/frozen_primary_plain/`), `results/supplementary.json`, `results/env.json`, `results/prismnli_pipeline_check.json`, `results/contamination_research.json`, `results/error_analysis.md` and `results/inspection_sample.md`. Third-party published figures appear only in Sections 5 and 6(c), never in the headline tables.

Rounding: accuracy and F1 to 3 decimals, Brier / ECE / NLL to 3 decimals, latency to whole milliseconds, cost to 4 decimals USD.

---

## 1. Executive conclusion

The central question was whether Jev's "decision model" architecture gives a clear advantage over a strong open NLI zero-shot classifier, and whether Laya reproduces Jev's behaviour with open weights. On this benchmark the answer is no on the first count and yes-in-accuracy-only on the second.

PrismNLI-0.4B is the best system on every headline quality metric (and on every per-class F1 point estimate; see 4.2 for the one small class where the margin is not established): accuracy 0.725 vs 0.587 (Jev) and 0.587 (Laya); the 0.138 gap is significant (exact McNemar p = 1.5e-41; paired bootstrap 95% CI [-0.158, -0.118] for Jev minus PrismNLI). Jev and Laya are statistically indistinguishable (McNemar p = 1.000; paired difference CI [-0.022, 0.020]). Jev's calibration is not better than PrismNLI's here (ECE 0.281 vs 0.174; Brier 0.667 vs 0.441; NLL 2.845 vs 1.174, where Jev's NLL is inflated by its 2-decimal API rounding: 1.454 at eps = 0.01, still above PrismNLI's 0.935; see 4.4). On the primary variant Laya is the worst calibrated (ECE 0.307 vs Jev 0.281; Brier 0.707 vs 0.667; the paired bootstrap CIs for these Jev-Laya differences exclude 0, Section 4.4); the Jev-Laya ordering reverses under `defined` (Laya ECE 0.230 vs Jev 0.276) and is much smaller than either model's gap to PrismNLI. PrismNLI also makes the fewest errors at every partial coverage level.

Two caveats bound this. PrismNLI's initialisation checkpoint has verified training exposure to the train and validation splits of this dataset, so part of its lead cannot be called zero-shot. And PrismNLI's probabilities are adapter-derived (softmax over entailment logits), so accuracy/F1 comparisons are more direct than calibration comparisons.

---

## 2. Headline tables

### 2.1 Primary variant (`plain`, bare label names), test split, n = 2000

| Model | Accuracy | Macro-F1 | Brier ↓ | ECE ↓ | NLL ↓ | p50 latency | p95 latency | Cost |
|---|---|---|---|---|---|---|---|---|
| Jev 1.13 (pinned `typesafe/jev-1.13-20260917`) | 0.587 [0.565, 0.608] | 0.500 [0.471, 0.528] | 0.667 | 0.281 | 2.845 | 349 ms (remote e2e) | 593 ms (remote e2e) | $0.0284 (sum of the per-response `usage.cost` field; equals list price $0.042/M input x 676 323 tokens), 2000 calls |
| PrismNLI-0.4B | 0.725 [0.705, 0.744] | 0.647 [0.620, 0.673] | 0.441 | 0.174 | 1.174 | 58 ms (local, MPS fp32 bs=1) | 83 ms (local, MPS fp32 bs=1) | $0 API (self-hosted); *estimate*: 123 s device wall time for 2000 rows on M1 Max (about 0.034 device-hours) |
| Laya | 0.587 [0.565, 0.609] | 0.493 [0.463, 0.522] | 0.707 | 0.307 | 2.032 | 31 ms (local, MPS fp32 bs=1) | 35 ms (local, MPS fp32 bs=1) | $0 API (self-hosted); *estimate*: 63 s device wall time for 2000 rows on M1 Max (about 0.018 device-hours) |

Brackets are 95% percentile bootstrap CIs (10 000 resamples, seed 0). ECE is the 15-bin equal-width value on max probability. Jev latency was measured under 8 concurrent requests from Perth, Australia (a sequential run is reported in Section 4.9). The device-hour figures are derived from the measured `predict_wall_s` in `results/frozen_primary_plain/env.json` (plain) and `results/env.json` (defined) and are estimates, not billed costs; no dollar value is attached because no accelerator price was measured.

### 2.2 Secondary variant (`defined`, each label carries a frozen one-line definition), n = 2000

| Model | Accuracy | Macro-F1 | Brier ↓ | ECE ↓ | NLL ↓ | p50 latency | p95 latency | Cost |
|---|---|---|---|---|---|---|---|---|
| Jev 1.13 | 0.599 [0.577, 0.621] | 0.519 [0.490, 0.548] | 0.653 | 0.276 | 2.898 | 340 ms (remote e2e) | 512 ms (remote e2e) | $0.0378 actual, 2000 calls |
| PrismNLI-0.4B | 0.702 [0.681, 0.722] | 0.639 [0.609, 0.667] | 0.472 | 0.158 | 1.213 | 72 ms (local, MPS fp32 bs=1) | 98 ms (local, MPS fp32 bs=1) | $0 API (self-hosted); *estimate*: 149 s device wall time |
| Laya | 0.598 [0.576, 0.620] | 0.481 [0.451, 0.509] | 0.632 | 0.230 | 1.493 | 40 ms (local, MPS fp32 bs=1) | 45 ms (local, MPS fp32 bs=1) | $0 API (self-hosted); *estimate*: 81 s device wall time |

### 2.3 Plain vs defined, paired tests per model (`supplementary.json: plain_vs_defined`)

| Model | Acc plain | Acc defined | Delta (defined - plain) | Paired bootstrap 95% CI | Exact McNemar p | plain-only correct / defined-only correct |
|---|---|---|---|---|---|---|
| Jev 1.13 | 0.587 | 0.599 | +0.013 | [+0.003, +0.023] | 0.015 | 37 / 62 |
| PrismNLI-0.4B | 0.725 | 0.702 | -0.023 | [-0.036, -0.010] | 0.0007 | 111 / 65 |
| Laya | 0.587 | 0.598 | +0.011 | [-0.006, +0.027] | 0.243 | 136 / 157 |

---

## 3. Methodology

### 3.1 Data
`dair-ai/emotion`, config `split`, split `test`, all 2000 rows, pinned to revision `cab853a1dbdf4c42c2b3ef2173804746df8825fe`. Class support: sadness 581, joy 695, love 159, anger 275, fear 224, surprise 66; majority-class (joy) accuracy is 0.348. Only the first rows of `validation` were used for smoke tests and warm-up; no code or prompt decision touched the test split.

### 3.2 Frozen protocol
`PROTOCOL.md` v1 was frozen before any test inference and fixes the instruction, label order, request bodies, hypothesis template, model revisions, retry policy, metric definitions, latency methodology, output schema and seeds. The `plain` variant is primary; `defined` was run in a separate invocation after `plain` results were written. The primary (`plain`) and secondary (`defined`) designations were fixed in `PROTOCOL.md` v1 before any test inference; `defined` is reported alongside `plain` in Sections 2.2/2.3 and was not selected as the headline after the fact. No deviations were recorded (`results/deviations.md` does not exist; `summary.json: deviation = null` for both variants). Every system scored all 2000 rows with zero errors and zero retries.

### 3.3 What each system saw
Instruction (identical for all): `Which single primary emotion is expressed in this text?`. Labels in dataset order: sadness, joy, love, anger, fear, surprise. The instruction, label wording, definitions and hypothesis template were not tuned per model; all three systems received the same strings, so no system was given a prompt optimised for it.

- **Jev** (remote, OpenRouter Decisions API, `POST /api/alpha/decisions`, model pinned to `typesafe/jev-1.13-20260917`): `state` = raw text; one `choice` question with the instruction and `criteria` = the six labels with empty-string descriptions (`plain`) or the frozen definitions (`defined`). The API returns six probabilities rounded to 2 decimals plus a `choice` and a `confidence` field. Metrics use the renormalised probabilities; argmax disagreed with the API `choice` on 4 rows (`plain`) and 1 row (`defined`). Every response echoed the pinned model string.
- **Laya** (local, `laya` 0.3.3, `convaiinnovations/laya` root English checkpoint, revision `c5d78730f3493e4fe16d61507ef4b78eef7318cf`, ModernBERT-large encoder): the same question dict via `agent.predict(text, questions)`; `criteria[label] = None` renders the bare label (`plain`) or `label: definition` (`defined`), via `laya.common.render_options`. The library's shipped per-option-count temperature (`choice:6-10` = 1.0000158) is applied by the library and not overridden. Probabilities are returned to 4 decimals.
- **PrismNLI-0.4B** (local, `Jaehun/PrismNLI-0.4B`, revision `02b9102b34d5dce1bf29c4e6903fea33e1a0ddeb`, DeBERTa-v3-large binary NLI): premise = text; hypothesis `The primary emotion expressed in this text is {label}.` (`plain`) or `The primary emotion expressed in this text is {label} ({definition}).` (`defined`); six pairs in one forward pass, truncation `only_first`, max length 512.

Frozen definitions (`defined` only): sadness "feeling unhappy, down, grief, loss, disappointment or loneliness"; joy "feeling happy, pleased, cheerful, content or excited"; love "feeling affection, warmth, tenderness, caring or romantic attachment"; anger "feeling mad, irritated, annoyed, resentful or hostile"; fear "feeling afraid, scared, anxious, nervous or worried"; surprise "feeling amazed, shocked, startled or caught off guard by something unexpected".

### 3.4 PrismNLI conversion and verification
Primary conversion (HF `zero-shot-classification`, `multi_label=False` semantics): softmax over the six entailment logits. Secondary conversion: independent P(entail) = softmax over each pair's binary logits, L1-normalised across labels. On the 8 validation smoke rows x 2 variants the primary conversion matched `transformers.pipeline("zero-shot-classification")` with the same template to max abs diff 9.5e-07 (tolerance 1e-4; argmax agreed on all 16 rows).

### 3.5 Metrics (`metrics.py`)
Accuracy; macro-F1; per-class precision/recall/F1; 6x6 confusion matrix (rows = gold). NLL = -mean(log(clip(p_gold, 1e-6, 1))), no renormalisation after clipping. Multiclass Brier = mean(sum_k (p_k - onehot_k)^2), range 0-2. ECE: confidence = max probability, 15 equal-width bins on [0, 1], sum_b (n_b/N)|acc_b - conf_b|; secondary adaptive ECE with 10 equal-mass bins. Selective classification: rows sorted by confidence descending, ties broken by ascending `dataset_index`; accuracy at coverage c uses the top ceil(c*N) rows. Bootstrap 95% percentile CIs and paired bootstrap accuracy differences: 10 000 resamples, `numpy.random.default_rng(0)`, one shared index matrix. Exact McNemar (`statsmodels`, `exact=True`) on correct/incorrect indicators. Fraction of rows with p_gold exactly 0 is reported for the rounding artefact.

### 3.6 Latency
Remote (Jev): client wall-clock around the successful HTTP attempt, 8 concurrent requests, timeout 60 s, from Perth, Australia; labelled "remote end-to-end". A supplementary sequential run (concurrency 1, 100 validation rows) is reported separately. Local (Laya, PrismNLI): model load time measured separately; 10 warm-up calls on validation rows; then per-example warm latency at batch size 1 for all 2000 rows; labelled "local compute". The two kinds are never compared as equivalent.

### 3.7 Hardware, software, seeds
Apple M1 Max, 64 GB, macOS 15.6.1, torch 2.14.0 with MPS backend (no CUDA), fp32, batch size 1; transformers 4.57.6, laya 0.3.3, datasets 5.0.1, Python 3.11.6. `random`, `numpy`, `torch` seeded 0; `torch.use_deterministic_algorithms(True, warn_only=True)`; eval mode; no sampling.

---

## 4. Results

### 4.1 Accuracy and macro-F1 with pairwise significance (`plain`)

*Measurement.* PrismNLI 0.725 [0.705, 0.744]; Jev 0.587 [0.565, 0.608]; Laya 0.587 [0.565, 0.609]. Macro-F1: PrismNLI 0.647 [0.620, 0.673]; Jev 0.500 [0.471, 0.528]; Laya 0.493 [0.463, 0.522].

| Pair (a vs b) | Exact McNemar p | a-only correct / b-only correct | Paired bootstrap delta acc (a - b) | 95% CI |
|---|---|---|---|---|
| Jev vs PrismNLI | 1.5e-41 | 85 / 361 | -0.138 | [-0.158, -0.118] |
| Jev vs Laya | 1.000 | 218 / 219 | -0.001 | [-0.022, +0.020] |
| PrismNLI vs Laya | 4.1e-35 | 396 / 121 | +0.138 | [+0.116, +0.159] |

Under `defined` the ordering is the same: Jev vs PrismNLI -0.103 [-0.122, -0.083] (p = 1.9e-24); Jev vs Laya +0.002 [-0.022, +0.025] (p = 0.931); PrismNLI vs Laya +0.104 [+0.081, +0.127] (p = 2.1e-18).

*Interpretation.* The Jev-Laya difference is a single row out of 2000 (218 vs 219 discordant rows). The PrismNLI gap is roughly 14 accuracy points, several times the width of any CI.

### 4.2 Per-class precision, recall and F1 (`plain`)

| Class (support) | Jev P / R / F1 | PrismNLI P / R / F1 | Laya P / R / F1 |
|---|---|---|---|
| sadness (581) | 0.583 / 0.692 / 0.633 | 0.782 / 0.759 / 0.770 | 0.566 / 0.687 / 0.621 |
| joy (695) | 0.728 / 0.640 / 0.681 | 0.821 / 0.753 / 0.785 | 0.697 / 0.709 / 0.703 |
| love (159) | 0.316 / 0.270 / 0.292 | 0.473 / 0.497 / 0.485 | 0.322 / 0.365 / 0.342 |
| anger (275) | 0.563 / 0.487 / 0.522 | 0.717 / 0.738 / 0.728 | 0.561 / 0.469 / 0.511 |
| fear (224) | 0.467 / 0.571 / 0.514 | 0.649 / 0.768 / 0.703 | 0.547 / 0.339 / 0.419 |
| surprise (66) | 0.412 / 0.318 / 0.359 | 0.369 / 0.470 / 0.413 | 0.487 / 0.288 / 0.362 |

*Interpretation.* PrismNLI leads on every class point estimate; per-class F1 carries no CI here and the surprise margin (n = 66; 0.413 vs 0.359 / 0.362, i.e. 31 vs 21 / 19 correct rows) should not be read as established. Its largest margins are anger (+0.21 over Jev) and fear (+0.19 over Jev, +0.28 over Laya). love and surprise are hard for all three; Laya's weakest class is love (0.342), followed by surprise (0.362), and its largest shortfall versus PrismNLI is on fear (-0.28); Jev's weakest class is love (0.292). Jev and Laya both show recall above precision on sadness (0.692 / 0.687 recall vs 0.583 / 0.566 precision) and the reverse on fear for Laya (recall 0.339), the numeric face of the sadness over-production in 4.3.

### 4.3 Confusion patterns (`supplementary.json: top_confusions_plain`)

*Measurement.* Top confusions (gold -> predicted, count, share of that model's errors):

- Jev (827 errors): joy->sadness 110 (13.3%), anger->sadness 73 (8.8%), sadness->fear 65 (7.9%), joy->love 59 (7.1%), love->joy 53 (6.4%), sadness->joy 51 (6.2%).
- PrismNLI (551 errors): joy->love 64 (11.6%), joy->sadness 51 (9.3%), love->joy 45 (8.2%), sadness->anger 41 (7.4%), sadness->fear 39 (7.1%), sadness->joy 34 (6.2%).
- Laya (826 errors): fear->sadness 91 (11.0%), anger->sadness 86 (10.4%), joy->sadness 83 (10.0%), joy->love 78 (9.4%), sadness->joy 72 (8.7%), sadness->anger 57 (6.9%).

Full 6x6 confusion matrices (`summary.json: metrics.confusion_matrix`; rows = gold, columns = predicted, dataset label order):

Jev (row sums = support):

| gold \ pred | sadness | joy | love | anger | fear | surprise |
|---|---|---|---|---|---|---|
| sadness | 402 | 51 | 14 | 46 | 65 | 3 |
| joy | 110 | 445 | 59 | 37 | 33 | 11 |
| love | 44 | 53 | 43 | 6 | 11 | 2 |
| anger | 73 | 24 | 13 | 134 | 27 | 4 |
| fear | 46 | 21 | 5 | 14 | 128 | 10 |
| surprise | 15 | 17 | 2 | 1 | 10 | 21 |

PrismNLI (row sums = support):

| gold \ pred | sadness | joy | love | anger | fear | surprise |
|---|---|---|---|---|---|---|
| sadness | 441 | 34 | 15 | 41 | 39 | 11 |
| joy | 51 | 523 | 64 | 19 | 20 | 18 |
| love | 22 | 45 | 79 | 7 | 6 | 0 |
| anger | 25 | 11 | 7 | 203 | 16 | 13 |
| fear | 21 | 9 | 0 | 11 | 172 | 11 |
| surprise | 4 | 15 | 2 | 2 | 12 | 31 |

Laya (row sums = support):

| gold \ pred | sadness | joy | love | anger | fear | surprise |
|---|---|---|---|---|---|---|
| sadness | 399 | 72 | 20 | 57 | 30 | 3 |
| joy | 83 | 493 | 78 | 22 | 13 | 6 |
| love | 35 | 57 | 58 | 5 | 3 | 1 |
| anger | 86 | 30 | 13 | 129 | 16 | 1 |
| fear | 91 | 26 | 9 | 13 | 76 | 9 |
| surprise | 11 | 29 | 2 | 4 | 1 | 19 |

The same matrices are plotted in `results/plots/confusion_matrices.png`.

*Interpretation.* Jev and Laya both over-produce sadness on anger, fear and joy text; the sadness column absorbs 110 + 73 (Jev) and 91 + 86 + 83 (Laya) of their errors. PrismNLI's residual errors are dominated by the joy/love boundary, which `error_analysis.md` argues is largely a label-definition problem in the gold data.

### 4.4 Calibration

*Measurement (plain).*

| Model | ECE-15 | ECE adaptive-10 | Brier | NLL (eps 1e-6) | Mean confidence | p_gold = 0 exactly | p_gold < 0.005 |
|---|---|---|---|---|---|---|---|
| Jev | 0.281 | 0.280 | 0.667 | 2.845 | 0.867 | 0.151 (302 rows) | 0.151 |
| PrismNLI | 0.174 | 0.170 | 0.441 | 1.174 | 0.894 | 0.000 | 0.081 |
| Laya | 0.307 | 0.307 | 0.707 | 2.032 | 0.894 | 0.013 (25 rows) | 0.162 |

Per-bin reliability (15 equal-width bins on max probability, `summary.json: metrics.reliability_15`; n / mean confidence / accuracy; empty bins below 0.2 omitted):

| Bin | Jev | PrismNLI | Laya |
|---|---|---|---|
| (0.200, 0.267] | 1 / 0.250 / 1.000 | 3 / 0.243 / 0.667 | - |
| (0.267, 0.333] | 3 / 0.318 / 0.000 | 24 / 0.305 / 0.458 | 2 / 0.329 / 0.000 |
| (0.333, 0.400] | 15 / 0.378 / 0.400 | 24 / 0.366 / 0.208 | 13 / 0.369 / 0.308 |
| (0.400, 0.467] | 34 / 0.439 / 0.206 | 44 / 0.438 / 0.295 | 21 / 0.428 / 0.333 |
| (0.467, 0.533] | 71 / 0.507 / 0.282 | 57 / 0.505 / 0.368 | 49 / 0.506 / 0.367 |
| (0.533, 0.600] | 98 / 0.570 / 0.367 | 64 / 0.568 / 0.438 | 69 / 0.564 / 0.449 |
| (0.600, 0.667] | 99 / 0.633 / 0.364 | 65 / 0.631 / 0.492 | 86 / 0.632 / 0.465 |
| (0.667, 0.733] | 127 / 0.699 / 0.370 | 63 / 0.700 / 0.476 | 94 / 0.703 / 0.426 |
| (0.733, 0.800] | 136 / 0.770 / 0.419 | 81 / 0.769 / 0.506 | 110 / 0.768 / 0.364 |
| (0.800, 0.867] | 119 / 0.835 / 0.487 | 83 / 0.836 / 0.542 | 109 / 0.837 / 0.468 |
| (0.867, 0.933] | 211 / 0.904 / 0.550 | 114 / 0.900 / 0.570 | 163 / 0.903 / 0.460 |
| (0.933, 1.000] | 1086 / 0.988 / 0.727 | 1378 / 0.992 / 0.839 | 1284 / 0.986 / 0.676 |

Reliability summary (15-bin, top bin [0.933, 1.0]): Jev places 1086 rows (54%) there at mean confidence 0.988 with accuracy 0.727; PrismNLI 1378 rows (69%) at 0.992 with accuracy 0.839; Laya 1284 rows (64%) at 0.986 with accuracy 0.676. In the adaptive deciles, PrismNLI's top decile (confidence at least 0.9998) reaches accuracy 0.970, while Jev's 623 rows at confidence 1.00 (31% of the test set, spanning three adaptive deciles whose split is only the `dataset_index` tie-break) have accuracy 0.806 (`supplementary.json: jev_confidence_exactly_one_plain`). Every model is over-confident in every equal-width bin above 0.40 confidence and in every adaptive decile; the sparsely populated low-confidence bins are under-confident for Jev (n = 1 at 0.250 -> acc 1.000; n = 15 at 0.378 -> 0.400) and PrismNLI (n = 3 at 0.243 -> 0.667; n = 24 at 0.305 -> 0.458), carrying 19 and 27 rows respectively; Laya is over-confident in every populated bin (figure `reliability.png`).

NLL sensitivity to the clipping epsilon:

| Model | eps = 0.01 | eps = 0.001 | eps = 1e-6 (protocol) |
|---|---|---|---|
| Jev | 1.454 | 1.802 | 2.845 |
| PrismNLI | 0.935 | 1.097 | 1.174 |
| Laya | 1.530 | 1.850 | 2.032 |

API-native confidence field vs max probability: Jev `confidence` mean 0.837 vs max-prob mean 0.867, Pearson r = 0.9995, ECE-15 using the API field 0.251 (vs 0.281). Laya `confidence` mean 0.834 vs 0.894, r = 0.956, ECE-15 using it 0.255 (vs 0.307).

*Interpretation.* The eps table shows the NLL ordering PrismNLI < Jev, Laya is stable, but the Jev-Laya NLL gap is mostly an artefact of Jev's 2-decimal rounding: 15.1% of Jev rows put exactly 0 on gold (all errors by construction), and the clip at 1e-6 charges each of those ln(1e6), about 13.8 nats. At eps = 0.01 Jev's NLL (1.454) is below Laya's (1.530). Brier and ECE are not affected by the clip and rank PrismNLI first, Jev second, Laya third on `plain`; the Jev-minus-Laya paired bootstrap differences (10 000 resamples, seed 0; `supplementary.json: paired_bootstrap_calibration_and_coverage_plain`) are Brier -0.040 [-0.072, -0.009] and ECE-15 -0.026 [-0.047, -0.004], so the Jev-over-Laya ordering is supported on this variant but is small, and it reverses under `defined` (Section 4.7). Both gaps to PrismNLI are far larger (Jev minus PrismNLI: Brier +0.226 [+0.196, +0.255], ECE +0.107 [+0.086, +0.128]). The API-native confidence fields are slightly less over-confident than max-prob for both Jev and Laya but do not change the ranking.

### 4.5 Selective classification (`plain`)

*Measurement.* Number of errors (and accuracy) among the top-confidence rows at each coverage:

| Coverage (rows) | Jev | PrismNLI | Laya | Fewest errors |
|---|---|---|---|---|
| 50% (1000) | 261 (0.739) | 105 (0.895) | 292 (0.708) | PrismNLI |
| 80% (1600) | 561 (0.649) | 322 (0.799) | 592 (0.630) | PrismNLI |
| 90% (1800) | 687 (0.618) | 422 (0.766) | 706 (0.608) | PrismNLI |
| 100% (2000) | 827 (0.587) | 551 (0.725) | 826 (0.587) | PrismNLI (Laya edges Jev by one row) |

Accuracy above fixed confidence thresholds: at confidence at least 0.99, Jev covers 38.6% of rows at accuracy 0.783, PrismNLI 54.3% at 0.888, Laya 38.7% at 0.745.

*Interpretation.* Confidence does rank errors for all three systems (accuracy rises monotonically as coverage falls), but PrismNLI makes about 2.5x fewer errors than Jev at 50% coverage and 1.7x fewer at 80%. Jev has fewer errors than Laya at 50% coverage (261 vs 292; paired bootstrap difference -31 [-58, -4], CI excludes 0); at 80% and 90% the gaps (561 vs 592, -31 [-66, +6]; 687 vs 706, -19 [-57, +20]) are not distinguishable from zero (`supplementary.json: paired_bootstrap_calibration_and_coverage_plain`). Jev's confidence is therefore somewhat more informative than Laya's only at the high-confidence end.

### 4.6 Effect of the PrismNLI conversion

*Measurement.* Primary (softmax over entailment logits): accuracy 0.725, macro-F1 0.647, Brier 0.441, ECE 0.174, NLL 1.174. Secondary (L1-normalised independent P(entail)): accuracy 0.724, macro-F1 0.647, Brier 0.459, ECE 0.199, NLL 1.444; argmax agreement with the primary conversion 0.999. Under the independent view, 22.85% of examples have no label with P(entail) > 0.5 and 5.0% have two or more labels above 0.5.

*Interpretation.* The conversion has no material effect on accuracy or F1 (about 2 rows in 2000 change argmax) and a modest effect on calibration (ECE +0.025, NLL +0.27). PrismNLI's accuracy lead is not a conversion artefact; its calibration numbers are somewhat conversion-dependent.

### 4.7 Prompt robustness (`defined` variant)

*Measurement.* Section 2.3: definitions raise Jev by +0.013 (CI excludes 0, p = 0.015), lower PrismNLI by -0.023 (p = 0.0007) and leave Laya unchanged within noise (+0.011, CI includes 0). Calibration under `defined`: Jev ECE 0.276 / Brier 0.653 / NLL 2.898; PrismNLI ECE 0.158 / Brier 0.472 / NLL 1.213; Laya ECE 0.230 / Brier 0.632 / NLL 1.493, with Laya's mean confidence falling from 0.894 to 0.825 and its p_gold = 0 rate from 0.013 to 0.001. Local latency rises with the longer inputs (PrismNLI p50 58 -> 72 ms, Laya 31 -> 40 ms); Jev input tokens rise from 676 323 to 900 323 and cost from $0.0284 to $0.0378.

*Interpretation.* Both rankings survive the prompt change. Laya's calibration improves noticeably with definitions while its accuracy does not, consistent with the definitions diluting its saturated outputs. PrismNLI's drop is consistent with the `plain` hypothesis being closer to the template its initialisation checkpoint was trained with (Section 5). An alternative explanation is that long parenthetical definitions are unusual NLI hypotheses for any model and lengthen the inputs; the two cannot be separated here.

### 4.8 Agreement and oracle (`plain`)

*Measurement.* All three agree on 58.0% of rows; Jev-PrismNLI agree on 72.2%, Jev-Laya 69.4%, PrismNLI-Laya 68.2%. All three correct: 911 rows; all wrong: 389. Uniquely correct: PrismNLI 219, Laya 77, Jev 41. Oracle (any model correct) accuracy 0.806.

*Interpretation.* The three systems are complementary to the extent of 8 points of accuracy above PrismNLI alone, but 389 rows (19.5%) defeat all of them; the qualitative sample suggests a large share of these are gold-label noise.

### 4.9 Latency, efficiency and cost

| | Jev 1.13 (remote e2e) | PrismNLI-0.4B (local) | Laya (local) |
|---|---|---|---|
| p50 / p95 / mean, `plain` | 349 / 593 / 385 ms (8 concurrent) | 58 / 83 / 61 ms | 31 / 35 / 32 ms |
| Sequential supplementary run | 362 / 483 / 373 ms (n = 100 validation rows, concurrency 1, min 284, max 616) | - | - |
| p50 / p95 / mean, `defined`; examples per second | 340 / 512 / 364 ms; 2.7 per stream (2000 rows in 91.4 s wall, 8 streams) | 72 / 98 / 74 ms; 13.5 | 40 / 45 / 41 ms; 24.6 |
| Examples per second (1 / mean per-request) | 2.6 per stream; 2000 rows in 96.6 s wall with 8 streams | 16.4 | 31.6 |
| Model load time | n/a (remote) | 1.1 s | 27.3 s |
| Parameters | undisclosed | 435 063 810 | 421 293 827 |
| Weight bytes on disk | undisclosed | 870 176 356 (fp16 checkpoint) | 842 609 210 |
| Peak MPS allocated memory | n/a | 2.05 GiB | 2.02 GiB |
| Precision / batch / device | n/a / 1 request / OpenRouter (TypeSafe endpoint) | fp32 / 1 / MPS | fp32 / 1 / MPS |
| Tokens (`plain`) | 676 323 input, 122 741 output (mean 338 input per call) | - | - |
| Cost, 2000 decisions (`plain`) | $0.0284 (sum of the per-response `usage.cost` field; equals list price $0.042 per M input tokens x 676 323 tokens, output free) | $0 API; *estimate* 0.034 device-hours | $0 API; *estimate* 0.018 device-hours |
| Cost per 1000 decisions | $0.0142 (`plain`), $0.0189 (`defined`) | $0 API | $0 API |

*Interpretation.* Local per-example compute latency on an M1 Max is 6-11x lower than Jev's remote end-to-end latency, but the two are different quantities (network round trip from Perth versus on-device compute) and the local figures would change on other hardware. Jev's per-request cost is small in absolute terms. Laya's 27 s load time cannot be explained by weight size (842 MB vs PrismNLI's 870 MB, which loads in 1.1 s); the load path was not profiled.

### 4.10 Length effect

*Measurement.* Accuracy by word-count tercile (short 3-12, mid 13-22, long 23-61 words): Jev 0.696 / 0.580 / 0.480; PrismNLI 0.812 / 0.735 / 0.624; Laya 0.671 / 0.582 / 0.505.

*Interpretation.* All three lose 0.17-0.22 from the shortest to the longest tercile (Jev 0.217, PrismNLI 0.188, Laya 0.166); PrismNLI's margin over Jev (0.12-0.15) and over Laya (0.12-0.15) is present in every tercile. Tercile-level CIs were not computed (about 667 rows each).

### 4.11 Manual inspection (22 examples, `plain` run)

*Measurement.* `results/inspection_sample.md` freezes n = 22 test rows in nine categories: all_correct (3), all_wrong_three_distinct (3), only_jev_correct (3), only_prismnli_correct (3), only_laya_correct (3), jev_highconf_wrong (2), prismnli_highconf_wrong (2), laya_highconf_wrong (2), class_fill_fear (1). Per-category observations (from `results/error_analysis.md`, Section 1; probabilities are the models' outputs on those rows):

- **all_correct** (idx 1284, 1019, 1724): single-keyword texts ("ashamed", "enjoyed", "divine intervention"). Confidence differs (PrismNLI hedges joy/surprise 0.53/0.47 on idx 1019 where Laya gives 0.99) although all three are right.
- **all_wrong_three_distinct** (idx 1979, 1588, 62): idx 1979, gold = anger, "no strong feelings for this book neither hated nor loved it": Jev sadness 0.78, PrismNLI joy 0.29 (flat), Laya love 0.91; the text asserts neutrality and the gold reads as noise. idx 62, gold = joy, a "dazed ... in hiding" text: Jev sadness 0.95, Laya fear 0.77, PrismNLI surprise 0.77. idx 1588, gold = love: Jev fear 0.72 on "hurt/betray".
- **only_jev_correct** (idx 985, 711, 1627): Jev love 0.71 on "how treasured my london flatmates are" (others joy 0.9-0.99); Jev anger 0.77 on a complaint about a class leader where PrismNLI gives surprise 0.56 and Laya love 0.89 on the keyword "friends".
- **only_prismnli_correct** (idx 1376, 1049, 1179): PrismNLI anger 1.0 on "completely rude" (others sadness); surprise 0.65 on "impatience ... and impressed"; and idx 1179, gold = joy, "i didn t and still don t feel lucky though" -> PrismNLI joy 0.97, Jev sadness 1.0, Laya sadness 0.95 (discussed in 6(f)).
- **only_laya_correct** (idx 1746, 1404, 604): a 0.37/0.37 near-tie landing on joy; sadness 0.93 on a numbness text where Jev gives fear 0.9 and PrismNLI surprise 0.45; anger 0.96 on a profanity-laden text where Jev gives fear 0.99 and PrismNLI sadness 0.97.
- **jev_highconf_wrong** (idx 1183, 1725): idx 1183, gold = joy: Jev sadness 0.99 with joy exactly 0.0 while PrismNLI gives joy 0.99; idx 1725 (fabric review, gold = anger): all three joy, gold looks like noise.
- **prismnli_highconf_wrong** (idx 1537, 1475): love 0.98 on "sweet he gave me" (gold = joy; Laya agrees); fear about 1.0 from all three on "repressed fear and anxiety" (gold = sadness). Boundary or noise cases.
- **laya_highconf_wrong** (idx 341, 215): joy 0.95+ from all three on "brave and excited" (gold = love); Laya anger 0.98 and PrismNLI anger 0.94 on "i feel dirty talking to people for my personal gain" (gold = sadness) where Jev gives sadness 0.88.
- **class_fill_fear** (idx 28): "i do feel insecure sometimes but who doesnt": Jev and PrismNLI fear; Laya sadness 0.86, an instance of Laya's largest confusion (fear->sadness, 91 rows).

*Interpretation.* Of the 22 rows, the all-wrong and high-confidence-wrong categories are dominated by gold-label ambiguity or noise rather than clear model failures; the model-specific patterns that survive are Jev/Laya collapsing low-arousal negative text to sadness, Laya's keyword capture, and PrismNLI's confident joy on a negated joy word (one row; see 6(f) for why it cannot be attributed to exposure on its own). Category counts here are by construction and are not estimates of prevalence.

---

## 5. Fairness and contamination (`results/contamination_research.json`)

All quotes below were re-fetched and verified verbatim against the cited sources on 2026-09-20. For every system, "no contamination" cannot be inferred from undisclosed data; absence of a published training manifest is absence of evidence, not evidence of absence.

### PrismNLI-0.4B: explicit evidence of training exposure (inherited)
The model card states: "Instead of starting from scratch, we start from deberta-v3-large-zeroshot-v2.0, a checkpoint of deberta-v3-lage trained on diverse classification data." That checkpoint's card says non-`-c` models "also included a broader mix of training data with a broader mix of licenses: ANLI, WANLI, LingNLI, and all datasets in [this list] where `used_in_v1.1==True`", and the pinned CSV lists `emotion6_twitter` (link `https://huggingface.co/datasets/dair-ai/emotion`) as `used_in_v1.0=TRUE, used_in_v1.1=TRUE`. The same card states the released model is "the final run including up to 500 training data points per class from each of the 28 datasets ... No model was trained on test data." The harmonisation notebook pools `dataset_emotion_dair["train"]` and `["validation"]` for training and keeps `["test"]` as the eval set, using the hypothesis template "This example tweet expresses the emotion: {label}". PrismNLI's own fine-tuning stage is "515K NLI datapoints from PrismNLI, a synthetic dataset to improve generalization of NLI models ... generated by Qwen2.5-72B-Instruct via our algorithm, Prismatic Synthesis", seeded from WANLI; its decontamination covers HANS, WNLI, ANLI, Diagnostics, BigBench and Control, not emotion. The scale of exposure is bounded by the card's "up to 500 training data points per class" statement (at most about 3,000 texts of the 18,000 train + validation rows) and by the 10,344-row `emotiondair` NLI subset visible in the training-pool listing (several hypotheses per text, so the unique-text count cannot be inferred); the cap itself is asserted on the card and is not verifiable in the published code. Verdict: exposure to dair-ai/emotion train + validation (not test) is verified through the initialisation checkpoint; the final fine-tuning data is NLI/zero-shot-oriented synthetic NLI. For context only (not our measurement): the starting checkpoint's card reports emotiondair macro-F1 0.484 zero-shot / 0.688 in the released emotion-trained run.

### Laya: author claims held out (unverifiable)
The model card table lists "DAIR Emotion | **0.573** | 0.513 | held out"; the eval harness hard-codes "sst5, emotion, prompt_injections, banking77 were held out" as a string in a caveats dict, not derived from a training manifest. No training manifest, dataset list or training script for the English checkpoint is published; `rl_agent_config.json` records `fine_tuned_from_checkpoint: true` with an undisclosed prior checkpoint. The training-side `eval/results.md` shows an in-task family "emotion and tone | 1825 | 0.906 | 0.018 | 0.238" alongside a 600-question "zero-shot" emotion family at 0.583, so an "emotion and tone" training family exists whose composition is undisclosed; the dair-ai `unsplit` config has 416 809 rows, so a held-out claim scoped to the 2000-row test split does not exclude the rest of the pool.

### Jev 1.13: corpus not public; author claims all data self-made (unverifiable)
TypeSafe's launch FAQ: "We make all the data ourselves. We wouldn't train on your data even if you asked us to (no offense)." TechCrunch relays that Jev "is trained exclusively on synthetic data" and that "outside observers suspect [it] is built on top of an open-weight LLM." TypeSafe "deliberately chose not to publish performance against public benchmarks" and has released no model card, data card or paper. The docs state "Jev is not fine-tuned or LoRA-adapted with customer data." Nothing names or excludes dair-ai/emotion; if there is a pretrained base, its corpus is unknown.

---

## 6. Required-analysis answers

### (a) How different are the three in accuracy?
Jev and Laya: not different (0.587 vs 0.587; 218 vs 219 discordant rows; McNemar p = 1.000; delta CI [-0.022, +0.020]). PrismNLI vs either: 0.138 higher, CI width about 0.04, p < 1e-34. Macro-F1 tells the same story (0.647 vs 0.500 vs 0.493).

### (b) Does Jev's architecture give a clear advantage over a strong NLI classifier?
No. On accuracy, F1, every per-class F1 point estimate (the surprise margin on n = 66 is not established), Brier, ECE, NLL and errors at every coverage level, the 0.4B NLI classifier is ahead. Jev's remaining advantages over PrismNLI are operational properties (no weights to host, fixed per-call price) that this benchmark did not measure. The contamination caveat weakens PrismNLI's claim to be zero-shot but does not create an advantage for Jev.

### (c) Does Laya reproduce its claimed performance and the claimed gap vs Jev?
Our Laya accuracy is 0.587 on all 2000 test rows with the frozen prompt. The authors report 0.573 (first 600 rows) and 0.595 (first 400 rows) with a different instruction ("Which emotion is most strongly expressed in `text`?"); on the same first 600 / 400 test rows our Laya accuracy is 0.565 / 0.585 (`supplementary.json: accuracy_on_first_rows_plain`) versus the authors' 0.573 / 0.595, so the author-reported level is consistent with ours despite the different instruction. The claimed gap versus Jev (0.595 vs 0.480, "+0.115") is not reproduced: measured on identical rows with identical prompts, Jev scores 0.587 and the gap is -0.001 with a CI spanning zero. The authors' Jev figure came from a third-party run (AbdelStark/jev-benchmarks) on 100 class-balanced rows with a different question ("Which single label best describes the input text?") and opaque option ids (`label_000` ...), and the Laya card itself notes "sample sizes and prompts differ". Under a matched protocol the Laya-over-Jev claim does not hold.

### (d) Best calibrated before post-hoc calibration?
PrismNLI on every metric (ECE 0.174, Brier 0.441, NLL 1.174), then Jev (0.281 / 0.667 / 2.845), then Laya (0.307 / 0.707 / 2.032) on the primary variant. Jev's NLL is inflated by its 2-decimal API rounding (1.454 at eps = 0.01, still above PrismNLI's 0.935; see 4.4), and the Jev-Laya NLL order flips at eps = 0.01 because of Jev's exact zeros. The Jev-over-Laya ordering on Brier and ECE is supported by paired bootstrap on `plain` (Jev minus Laya: Brier -0.040 [-0.072, -0.009], ECE -0.026 [-0.047, -0.004]) but reverses under `defined` (Laya ECE 0.230 / Brier 0.632 vs Jev 0.276 / 0.653) and is small next to both models' gap to PrismNLI. Caveat: PrismNLI's probabilities are adapter-derived (a softmax over six entailment logits from a binary NLI head, a construction it was never trained to calibrate), whereas Jev and Laya output categorical choice probabilities natively and are trained to do so. Accuracy and F1 are therefore more directly comparable across the three than calibration is, and the conversion choice moves PrismNLI's ECE by 0.025 (Section 4.6). Even under the secondary conversion PrismNLI's ECE (0.199) remains below Jev's (0.281).

### (e) How much of PrismNLI's result comes from the conversion?
Essentially none of the accuracy (0.725 vs 0.724; argmax agreement 0.999) and a minority of the calibration advantage (ECE 0.174 -> 0.199, NLL 1.174 -> 1.444 under the secondary conversion, still better than Jev and Laya). The independent view also shows the binary head is often undecided: 22.85% of rows have no label with P(entail) > 0.5.

### (f) Common semantic confusions
Shared by all: joy/love (a label-definition issue where affection words are hashtagged as love inconsistently) and sadness/fear on threat-tinged text. Jev- and Laya-specific: low-arousal anger (envy, irritation, dissatisfaction) and keyword-free anxiety collapse into sadness (anger->sadness 73 / 86; fear->sadness 91 for Laya). PrismNLI-specific: surprise used as an uncertainty sink and confident joy on negated joy words (idx 1179, "i didn t and still don t feel lucky though" -> joy 0.97), which `error_analysis.md` offers as one hypothesis (inherited dataset-specific lexical mapping, its H1); negation failures are also common in NLI models without any exposure, so a single row cannot distinguish the two. The 22-row inspection sample is summarised in 4.11. Surprise (66 rows) has the noisiest gold and the lowest F1 for all three.

### (g) Does confidence track accuracy?
Yes, directionally, for all three: accuracy rises with confidence in both binning schemes and with coverage reduction. But all three are over-confident in every bin that carries meaningful mass (every equal-width bin above 0.40 and every adaptive decile; only the n <= 24 low-confidence bins of Jev and PrismNLI are under-confident): Jev reaches 0.727 accuracy in its top bin at mean confidence 0.988, Laya 0.676 at 0.986, PrismNLI 0.839 at 0.992. PrismNLI's ranking is the sharpest (top adaptive decile 0.970 accuracy); Laya's is the flattest (its 0.73-0.93 equal-width bins sit at 0.36-0.47 accuracy).

### (h) Fewest errors at 50% and 80% coverage
PrismNLI: 105 errors at 50% and 322 at 80%, versus Jev 261 / 561 and Laya 292 / 592. Jev has fewer errors than Laya at 50% (paired bootstrap difference -31 [-58, -4]); at 80% the gap (-31 [-66, +6]) is not distinguishable from zero (Section 4.5).

### (i) Latency and deployment economics
Local compute (M1 Max, MPS, fp32, batch 1): Laya p50 31 ms, PrismNLI p50 58 ms; both fit in about 2 GiB. Remote end-to-end (Jev, Perth client): p50 349 ms concurrent / 362 ms sequential, p95 483-593 ms, dominated by network and service time; 2000 rows completed in 96.6 s wall with 8 streams at $0.0284 total ($0.0142 per 1000 decisions). Local latency is hardware-dependent (the Laya authors' published runs are on a Tesla T4, ours on an M1 Max) and neither is comparable to a remote round trip. Jev trades per-call cost and network latency for zero weight hosting; the local systems trade a one-off load (1 s / 27 s) and hardware for zero marginal cost.

### (j) Which candidate explanation does the evidence support?
- *Jev is a substantial capability improvement over NLI:* not supported; the NLI classifier is 14 points ahead.
- *Jev has similar raw ability but better calibration/efficiency/ergonomics:* not supported on calibration (Jev's ECE 0.281 vs PrismNLI 0.174, Brier 0.667 vs 0.441; 15% of rows get exactly 0 on the gold label) nor on latency; the ergonomic advantage (native choice probabilities, no weights to host) is real but unmeasured here.
- *Laya is an open-weight approximation of Jev's abstraction:* supported for accuracy (statistically indistinguishable, McNemar p = 1.000, paired CI includes 0, 69% prediction agreement) but not for calibration or error profile (Laya is worse calibrated on `plain` though not on `defined`, makes more errors at 50% coverage, and has a distinct fear->sadness failure).
- *The three approaches have different strengths:* partly supported. PrismNLI leads on every quality metric; Laya leads on local latency and cost; Jev leads only on ergonomics. Jev and Laya are statistically indistinguishable in accuracy and both far below PrismNLI, and Jev's calibration is not better than PrismNLI's here.

The bound on this conclusion is the PrismNLI contamination finding: its initialisation checkpoint saw part of dair-ai/emotion train + validation (bounded in Section 5) with a similar declarative emotion-hypothesis template (training: "This example tweet expresses the emotion: {label}"; ours: "The primary emotion expressed in this text is {label}."), so an unknown but non-zero part of its 14-point lead is dataset familiarity rather than zero-shot capability. The `defined` variant (a different template) cost PrismNLI 2.3 points, which is consistent with, but not proof of, template familiarity. The clean test is a re-run on an emotion dataset of different provenance.

---

## 7. Limitations

- Single dataset, single domain (Twitter, 2018), six labels; no claim generalises beyond it.
- Gold labels come from hashtag distant supervision; 389 rows defeat all three systems and the 22-row inspection sample (Section 4.11) shows many are label noise, so absolute accuracies are bounded below 1 by the data.
- Contamination: verified inherited exposure for PrismNLI (train/validation); unverifiable claims for Laya and Jev.
- Jev returns probabilities rounded to 2 decimals; 15.1% of rows have p_gold = 0 exactly, which inflates NLL by construction (eps table) and coarsens ECE bins. Laya rounds to 4 decimals (1.3% exact zeros).
- Local latency and memory are for Apple MPS fp32 at batch size 1, not CUDA; throughput would differ on GPUs and with batching.
- Jev latency is from one client location (Perth) under concurrency 8 (plus one 100-row sequential run); it includes network time and provider queueing.
- No temperature scaling or post-hoc calibration was applied by design; Laya's shipped library temperature is part of its out-of-the-box behaviour and was not removed.
- PrismNLI probabilities are adapter-derived from a binary NLI head; calibration comparisons with native choice models are indirect.
- One prompt per variant, frozen before test inference; no prompt search and no per-model tuning (all three systems received the same instruction, label wording and definitions), so each system may have a better prompt that was not explored.
- Jev's parameter count, architecture and base corpus are undisclosed; Laya's prior checkpoint is undisclosed.

---

## 8. Reproducibility

Commands (from `.`, venv Python 3.11.6, `OPENROUTER_API_KEY` set for Jev only):

```
python benchmark.py --split validation --limit 8 --models jev,prismnli,laya --variants plain,defined   # smoke
python -m models.prismnli --smoke        # writes results/prismnli_pipeline_check.json
python benchmark.py --split test --models jev,prismnli,laya --variants plain      # primary, 2026-09-20T01:08:04Z to 01:13:28Z
python benchmark.py --split test --models jev,prismnli,laya --variants defined    # secondary, 01:13:53Z to 01:19:58Z
pytest -q tests/
```

Versions (`results/env.json`): torch 2.14.0, transformers 4.57.6, laya 0.3.3, datasets 5.0.1, huggingface_hub 0.36.2, safetensors 0.8.0, numpy 2.4.6, pandas 3.0.6, pyarrow 25.0.1, scipy 1.17.1, scikit-learn 1.9.1, statsmodels 0.15.0, matplotlib 3.11.2, httpx 0.28.1; full pins in `requirements.txt`.

Revisions: dataset `dair-ai/emotion` @ `cab853a1dbdf4c42c2b3ef2173804746df8825fe`; `Jaehun/PrismNLI-0.4B` @ `02b9102b34d5dce1bf29c4e6903fea33e1a0ddeb`; `convaiinnovations/laya` @ `c5d78730f3493e4fe16d61507ef4b78eef7318cf`; Jev `typesafe/jev-1.13-20260917` (echoed in every response of both runs).

Seeds: `random`/`numpy`/`torch` = 0; bootstrap `default_rng(0)`, 10 000 resamples; deterministic algorithms requested.

Files:
- `./PROTOCOL.md` - frozen protocol
- `./results/summary.json`, `summary.csv` - all metrics (both variants)
- `./results/raw_predictions.parquet` - per-row predictions and probabilities
- `./results/env.json` - environment of the last (defined) run
- `./results/frozen_primary_plain/{summary.json,summary.csv,raw_predictions.parquet,env.json,SHA256SUMS}` - checksummed snapshot of the primary `plain` run (SHA-256 of `summary.json` starts `7e739a48`, of `raw_predictions.parquet` starts `956efecf`)
- `./results/supplementary.json` - paired plain/defined tests, confusions, agreement, eps sensitivity, confidence slices, errors at coverage, API-confidence comparison, length effect, sequential latency, cost; paired bootstrap CIs for Brier / ECE-15 / errors-at-coverage differences, accuracy on the first 600 / 400 rows, and the Jev confidence = 1.00 group (all recomputed from `frozen_primary_plain/raw_predictions.parquet`)
- `./results/jev_sequential_latency.json`
- `./results/prismnli_pipeline_check.json`
- `./results/contamination_research.json`
- `./results/inspection_sample.md`, `inspection_sample.json`, `error_analysis.md`
- `./results/run_plain.log`, `run_defined.log`
- `./results/cache/jev_plain.jsonl`, `cache/jev_defined.jsonl` (test-split Jev responses, keyed by `dataset_index`); `cache/smoke_validation/`, `cache/seq_latency_validation/` (validation rows only)
- `./models/{common,jev,laya,prismnli}.py`, `metrics.py`, `plots.py`, `benchmark.py`, `tests/`

---

## 9. Figures

Primary (`plain`) figures under `./results/plots/`; `defined` counterparts under `results/plots/defined/` (PNG 200 dpi + PDF).

- `accuracy_macro_f1_ci.png` - accuracy and macro-F1 per model with 95% bootstrap CIs.
- `per_class_f1.png` - per-class F1 for the six emotions, grouped by model.
- `confusion_matrices.png` - 6x6 confusion matrices (rows = gold) for Jev, PrismNLI, Laya.
- `reliability.png` - 15-bin reliability diagrams (accuracy vs max-prob confidence) per model.
- `risk_coverage.png` - selective-classification risk-coverage curves, confidence-sorted.
- `confidence_hist.png` - histograms of max probability per model (shows Jev's 2-decimal mass at 1.0).
- `latency.png` - latency distributions, remote end-to-end (Jev) and local compute (PrismNLI, Laya) plotted separately.
