# model_decisions: zero-shot emotion classification benchmark

Head-to-head, zero-shot evaluation of three "decision" systems on six-way emotion classification
(`dair-ai/emotion`, test split, 2000 rows):

| System | Kind | Where it runs |
|---|---|---|
| **Jev** (`typesafe/jev-1.13-20260917`) | remote, OpenRouter Decisions API | network |
| **Laya** (`convaiinnovations/laya`, repo root checkpoint) | local, `laya` library | Apple MPS |
| **PrismNLI-0.4B** (`Jaehun/PrismNLI-0.4B`) | local, NLI zero-shot via `transformers` | Apple MPS |

Every system gets the same raw text, the same instruction and the same six labels; nothing is
trained or tuned. Each is run twice: `plain` (bare label names, the headline) and `defined`
(each label carries a frozen one-line definition).

## Frozen protocol

**[`PROTOCOL.md`](PROTOCOL.md) is the binding contract.** It fixes the data revision, the exact
request bodies / prompts / hypothesis template, model revisions, retry policy, metric
definitions, latency methodology, output schema and seeds. It was frozen before any test-set
inference and may not change afterwards; any deviation must be recorded in
`results/deviations.md`. `models/common.py` holds the shared constants and dataclasses that all
modules implement against.

## Dataset selection

The benchmark needs one dataset that is fair to all three systems at once: a proprietary
decision model reached only through an API (Jev), a binary NLI classifier that scores each label
as a hypothesis (PrismNLI), and an open non-autoregressive typed-decision model with a fixed
option-token budget (Laya). Candidates were screened before any code was written, using the
models' own documentation as the source for "in training mix" claims.

| Candidate | Why it was not chosen |
|---|---|
| **AG News** (4 topics) | Listed by the Laya authors as *in Laya's training mix*, and used as a zero-shot training/eval set by the `deberta-v3-large-zeroshot` lineage behind PrismNLI. Any lead would be uninterpretable. |
| **banking77** (77 intents) | 77 options exceed Laya's documented ~20-option `choice` budget (options share a 192-token head budget); the Laya authors themselves report a hard ceiling here. Would measure an architectural limit, not decision quality. |
| **MASSIVE intent** (60 intents) | Same option-budget problem; also part of Laya's published evaluation suite with routing tuned on it. |
| **SST-5** (5 ordinal levels) | Ordinal, so the natural Jev/Laya primitive is `score`, not `choice`; PrismNLI has no ordinal primitive, which would force an unfair mapping. |
| **BoolQ**, **XNLI**, **MNLI/ANLI** | Binary / NLI-shaped tasks: PrismNLI's home turf (and XNLI/BoolQ are in Laya's mix). A two-way task also gives weak resolution on calibration. |
| **GoEmotions** (27 labels, multi-label) | Multi-label and 27 options: violates the single-`choice` framing and Laya's option budget. |
| **TweetEval emotion** (4 labels) | Only four classes and a small test split (1,421); less resolution than the six-way task. |
| **typed-decisions** (Laya's own benchmark) | Laya's `typed-decisions` checkpoint is fine-tuned on its training split, and the general checkpoint is near chance on it by the authors' own account; it is also authored by one of the parties being compared. |
| **deepset prompt-injections** (binary, n=116) | Far too small for useful confidence intervals. |

**`dair-ai/emotion` (config `split`, official `test`, 2,000 rows, six mutually exclusive labels)
was selected because it is:**

- small enough to run cheaply (Jev cost for the whole test set was $0.03) yet large enough for
  useful bootstrap confidence intervals (+/- ~2 points on accuracy);
- six-way rather than binary, so calibration, selective classification and confusion structure
  are informative;
- naturally expressible as one typed `choice` question with the identical six label strings for
  every system, and as six NLI hypotheses with one fixed template;
- below Laya's recommended maximum number of `choice` options;
- reported by the Laya authors as *held out* from Laya's training mix, and not named anywhere in
  TypeSafe's (undisclosed) training description;
- not used for any per-model prompt tuning here (the protocol was frozen before test inference).

**What we learned after selection (see `REPORT.md` Section 5).** The contamination research done
for this report found that PrismNLI-0.4B is initialised from `deberta-v3-large-zeroshot-v2.0`
(non-`-c` variant), whose published training list includes the *train* and *validation* splits of
`dair-ai/emotion` (never the test split). That lineage was not visible on the PrismNLI model card
and was only established by tracing the initialisation checkpoint's dataset CSV and notebooks.
The dataset was kept because the protocol was already frozen and because the exposure is bounded
and documented; the report treats PrismNLI's lead as partly non-zero-shot for that reason. A
follow-up on an emotion dataset absent from the `zeroshot-v2.0` list (the list covers 28 public
classification sets, so candidates are scarce) is the clean next step.

### Dataset constraints (follow-up datasets, `PROTOCOL_ADDENDUM_v2.md` §6)

Three follow-up datasets were chosen because they are **absent from every disclosed training list**
of the three systems. The evidence base is the same as Section 5 of `REPORT.md`
(`results/contamination_research.json`): the pinned training CSV of
`MoritzLaurer/deberta-v3-large-zeroshot-v2.0` (the checkpoint PrismNLI-0.4B is initialised from;
the non-`-c` model was trained on every row with `used_in_v1.1 == TRUE`), the PrismNLI synthetic
data seeds (WANLI only, per the dataset card and the paper), Laya's disclosed training mix (model
card table plus the `in_training` flags hard-coded in its public eval harness
`research/scripts/bench_apps.py`), and TypeSafe's statements about Jev's corpus (self-made,
undisclosed). "Absent from disclosed lists" is the strongest statement available; it is not
"never seen".

| dataset (pinned) | evaluated split, n, classes | `deberta-v3-large-zeroshot-v2.0` training CSV (PrismNLI lineage) | PrismNLI synthetic seeds (WANLI) | Laya disclosed training mix / eval harness | Jev (TypeSafe) corpus | verdict |
|---|---|---|---|---|---|---|
| `cardiffnlp/tweet_topic_single` @ `87b7a0d1` | `test_2021`, 1693, 6 | not in the CSV. The sister set `tweet_topic_multi` is listed with `used_in_v1.0/v1.1 = FALSE`, `future_use = excluded` ("multi-label"); the single-label set does not appear at all | WANLI is MNLI-style premise/hypothesis pairs, no tweet-topic data; "tweet"/"topic" absent from the PrismNLI card and paper | not in the model-card table (AG News, BoolQ in mix; DAIR Emotion, SST-5, prompt-injections held out) and not loaded by `bench_apps.py` / `build_benchmark_nb.py`; the dev.to description names intents, NLI, safety, email triage, no tweet topics | training data "made ourselves", no corpus or benchmark named; no public-benchmark policy | absent from all disclosed lists |
| `zeroshot/twitter-financial-news-topic` @ `acbc8af2` | `validation`, 4117, 20 | listed as `twitter_financial_news_topic` with `used_in_v1.0/v1.1 = FALSE`, `future_use = excluded` ("data source and task definition too unclear"); i.e. explicitly **not** trained on (only the sibling `financial_phrasebank` sentiment set is `TRUE`) | not a seed | not in the model-card table; not loaded by the eval harness; no financial-news topic source disclosed | as above | absent from all disclosed lists; explicitly excluded by the zeroshot-v2.0 authors |
| `OpenRL/daily_dialog` @ `1668faf0` (mirror of `li2017dailydialog/daily_dialog`) | `test` flattened to utterances, 7740, 7 | listed as `daily_dialog` with `used_in_v1.0/v1.1 = FALSE`, `future_use = later`; i.e. not in the released model's training set (the emotion sets that **are** `TRUE` are `dair-ai/emotion` and `emo`/EmoContext) | not a seed | not in the model-card table; not loaded by the eval harness. Caveat: Laya's `eval/results.md` shows a trained "emotion and tone" task family (1,825 in-task questions) whose sources are not named, so a dialogue-emotion set cannot be ruled out from the disclosures | as above | absent from all disclosed lists; residual risk through Laya's unnamed emotion family |

Additional per-dataset caveats: `fin_topic` has no test split, so `validation` is used purely as an
evaluation set (never for tuning); its 20 options fall in Laya's `choice:11+` temperature bucket
(0.1006, shipped, applied as-is and recorded in `summary.json` / `env.json`), whereas the 6/7-class
sets use `choice:6-10` (1.00002). `daily_dialog` is 82% `no emotion`, so macro-F1 and per-class
results carry the information there and the majority-class accuracy is reported next to every
accuracy. `tweet_topic`'s loading script is no longer runnable under `datasets` 5, so the raw
`split_temporal/test_2021.single.json` file is read at the pinned revision via `hf_hub_download`.

### Why not an existing Jev benchmark (JevBench, jev-benchmarks, decision-model-benchmark)?

Several community benchmarks for Jev-class models appeared in the days around Jev 1.13's release.
They were reviewed and are useful context, but none fits the question asked here, which is about
*zero-shot decision quality on an independent, externally labelled dataset* where a proprietary
decision model, an open NLI classifier and an open typed-decision model can all be run natively.

| Existing suite | What it is | Why it was not used as the primary benchmark |
|---|---|---|
| [JevBench](https://github.com/fstandhartinger/jevbench) (v1.0 to v1.2.2, 19 Sep 2026) | 534 typed "decisions" across routing, answer-adequacy judging, policy checks, intent, ordinal scoring, enum extraction; composite score = Intelligence, Calibration, Speed, Cost (geometric mean). | (1) **Not independent data**: the hard-tier items were "written by Claude Opus 5 and GPT-5.6 Sol", the rest are hand-written or imported from the author's own router logs; gold labels come from the author or a deterministic grader, not from an external annotation process. (2) **Not fully public**: 109 hard-tier and 24 held-out items are private and 146 imported items are not redistributed, so a third party cannot reproduce the headline number. (3) **Built around Jev's primitives**: tasks are defined as `noul` / `choice` / `score` rubrics in TypeSafe's wire format, so the task distribution is shaped by the API being evaluated; there is no NLI adapter, and mapping `score`/`noul`/extraction items onto entailment hypotheses would require inventing a per-family conversion for PrismNLI, breaking apples-to-apples. (4) **Small and heterogeneous**: 72 public original items; per-family n is too small for the paired tests used here. (5) Its composite folds in a latency adjustment ("x2 + 0.15 s ... an assumption, not a measurement") and a cost scale; this report deliberately keeps local compute latency and remote end-to-end latency apart and reports raw quantities. (6) It was released and re-scored four times on the day before this run; it is a moving target. |
| [AbdelStark/jev-benchmarks](https://github.com/AbdelStark/jev-benchmarks) (BTZSC pilot v1, 17 Sep 2026) | Jev vs GLiNER2.5 on 100 class-balanced rows each of AG News, DAIR Emotion and Banking77, via the `btzsc/btzsc` re-packaged dataset. | Closest in spirit, and its DAIR Emotion Jev number (0.480) is the one the Laya authors cite. Not reused because: 100 rows per dataset gives a +/- 10-point CI; it repackages the data (entailment-pair format, class-balanced subsample) instead of the official 2,000-row test split; its comparator is GLiNER, not an NLI model or Laya; and its prompt (`Which single label best describes the input text?` with `label_000`-style keys and BTZSC label descriptions) differs from a bare six-way `choice`. Our Section 6(c) compares against it as context only. |
| [nibzard/decision-model-benchmark](https://github.com/nibzard/decision-model-benchmark) (18 Sep 2026) | Five suites: 77-way intent, binary spam gate, synthetic cardinality sweep, option-order stability, forced-uncertainty honesty. | Three of five suites are seeded synthetic; the 77-way suite exceeds Laya's option budget; no emotion or six-way classification suite; designed to probe API behaviours (order stability, honesty) rather than accuracy on an external dataset. |

The three suites above are complementary probes of Jev-class behaviour (order stability,
forced uncertainty, schema validity, cost composites). This repository asks a narrower question
with a stricter design: one public, versioned, externally labelled dataset, the full official test
split, one frozen prompt per variant, native probabilities from every system, paired significance
tests, and no composite score.

## Layout

```
PROTOCOL.md                 frozen protocol (read this first)
PROTOCOL_ADDENDUM_v2.md     frozen addendum: three follow-up datasets (tweet_topic, fin_topic, daily_dialog)
REPORT.md                   final report (all numbers, methodology, contamination review, figures)
benchmark.py                end-to-end driver: data -> models -> parquet -> metrics -> plots (--dataset selects the registry entry)
datasets_registry.py        DatasetSpec registry: source/revision, evaluated split, labels, instruction, hypothesis template, smoke rows
metrics.py                  PROTOCOL §5 metrics for any class count (accuracy, majority-class accuracy, macro-F1, NLL, Brier, ECE, selective, bootstrap, McNemar)
plots.py                    PROTOCOL §7 figures (PNG 200 dpi + PDF); scale to 7 and 20 classes
render_plots.py             re-render every figure from results/summary.json + parquet (no model calls)
supplementary_analysis.py   post-hoc analyses from the frozen predictions -> results/supplementary.json (no model calls)
inspect_sample.py           deterministic 22-row qualitative sample -> results/inspection_sample.{md,json} (no model calls)
jev_sequential_latency.py   Jev latency at concurrency 1 on validation rows -> results/jev_sequential_latency.json
models/common.py            LABELS, INSTRUCTION, DEFINITIONS, revisions, Prediction / LoadInfo / Backend (primary benchmark)
models/jev.py               OpenRouter Decisions API adapter (threaded, cached, retrying); takes a DatasetSpec
models/laya.py              Laya local adapter; takes a DatasetSpec, records the temperature bucket applied
models/prismnli.py          PrismNLI-0.4B zero-shot NLI adapter (+ HF pipeline equivalence check); takes a DatasetSpec
tests/                      unit tests (metrics for 6/7/20 classes, registry strings and counts, benchmark merge logic,
                            Jev adapter, plots, and a regression test against results/frozen_primary_plain/)
requirements.txt            pinned package versions
results/                    primary emotion outputs (see below); results/<dataset>/ for the follow-up datasets
```

## Setup

```bash
cd model_decisions
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt        # or: pip install -r requirements.txt
export OPENROUTER_API_KEY=...             # needed for Jev only; never printed or written to disk
```

Hardware used for the reference run: Apple M1 Max, 64 GB, `torch` MPS backend (no CUDA). The
local models run in fp32 at batch size 1; the two checkpoints (about 0.9 GB and 0.6 GB) are
downloaded once into the Hugging Face cache at their pinned revisions.

## Running

### Smoke test (validation split, first 8 rows, all models, both variants)

```bash
python benchmark.py --split validation --limit 8 --models jev,prismnli,laya --variants plain,defined
```

Smoke tests use only the first 8 rows of the **validation** split, per PROTOCOL §2, and write to
`results/smoke_validation/`. The Jev responses for validation rows are cached under
`results/cache/smoke_validation/`, never in the production cache. Per-module smoke tests also exist:

```bash
python -m models.jev --smoke          # also verifies cache resume (second pass makes 0 HTTP calls)
python -m models.prismnli --smoke     # also writes results/prismnli_pipeline_check.json
python -m models.laya --smoke
python plots.py                       # synthetic self-test of every figure
pytest -q tests/                      # unit + regression tests (frozen primary numbers must be reproduced to 1e-9)
```

### Full test run

```bash
# primary variant first (PROTOCOL §4)
python benchmark.py --split test --models jev,prismnli,laya --variants plain
# secondary variant, in a separate invocation once plain results are written and frozen
python benchmark.py --split test --models jev,prismnli,laya --variants defined
```

Runs can be split by model and by variant across invocations: `benchmark.py` merges into the
existing `results/raw_predictions.parquet`, overwriting only the columns of the models and
variants it just ran and recomputing every metric from the merged file. Jev responses are
cached in `results/cache/jev_<variant>.jsonl` keyed by `dataset_index`, so an interrupted run
resumes without re-paying for completed rows. An existing parquet from a *different* split is
moved aside (`raw_predictions.<split>.bak.parquet`) rather than merged.

Flags: `--dataset {emotion,tweet_topic,fin_topic,daily_dialog}` (default `emotion`),
`--split {eval,smoke,validation,test}` (`eval` = the dataset's evaluated split, `smoke` = its
smoke rows; `test`/`validation` are the legacy spellings accepted for `--dataset emotion` only and
rejected for every other dataset, so `validation` can never select `fin_topic`'s evaluated split),
`--smoke`, `--limit N`,
`--models jev,prismnli,laya`, `--variants plain,defined`, `--skip-plots`, `--out-dir DIR`,
`--n-bootstrap N` (protocol value 10000; lower it only for quick checks).

### Follow-up datasets (`PROTOCOL_ADDENDUM_v2.md`)

```bash
# smoke: the addendum's smoke rows only (never the evaluated split), outputs under results/<key>/smoke_<split>/
for d in tweet_topic fin_topic daily_dialog; do
  python benchmark.py --dataset $d --split smoke --limit 8 --models jev,prismnli,laya --variants plain
done
# full evaluated split, outputs under results/<key>/ (Jev cache results/<key>/cache/jev_plain.jsonl)
python benchmark.py --dataset tweet_topic --split eval --models jev,prismnli,laya --variants plain
python benchmark.py --dataset fin_topic   --split eval --models jev,prismnli,laya --variants plain
python benchmark.py --dataset daily_dialog --split eval --models jev,prismnli,laya --variants plain
```

Only `plain` exists for these datasets (no definitions were written; `--variants defined` is
rejected). Each dataset's labels, instruction, hypothesis template, pinned revision and smoke rows
live in `datasets_registry.py`; `env.json` and `summary.json` record the full spec. The `emotion`
entry reproduces the primary run byte-for-byte (same prompts, same `results/` layout). Smoke runs of
any dataset, including `emotion`, are written to a `smoke_<split>/` sub-directory
(`results/smoke_validation/`) so the frozen evaluation files are never overwritten.

### Post-run analyses (no model calls except where noted)

```bash
python supplementary_analysis.py                              # -> results/supplementary.json
python inspect_sample.py                                      # -> results/inspection_sample.{md,json} (from frozen_primary_plain/)
python render_plots.py                                        # re-render results/plots/ and results/plots/defined/
OPENROUTER_API_KEY=... python jev_sequential_latency.py --n 100   # Jev only; validation rows; -> results/jev_sequential_latency.json
shasum -a 256 -c results/frozen_primary_plain/SHA256SUMS      # verify the frozen primary snapshot
```

## Output files (`results/`)

| File | Contents |
|---|---|
| `raw_predictions.parquet` | one row per `(dataset_index, variant)`: text, gold, and per model `pred, p_<label> x6, confidence, latency_ms, error, retries` plus Jev cost/tokens/response id/model/api confidence/raw probs, Laya api confidence/act probability, PrismNLI independent-entailment probs (PROTOCOL §7) |
| `summary.json` | per variant: every §5 metric per model (with reliability tables and risk-coverage curves), pairwise exact McNemar and paired bootstrap for every model pair, latency stats, Jev cost and tokens, error and retry counts |
| `summary.csv` | one headline row per model x variant |
| `env.json` | package versions, model and dataset revisions, hardware, device/dtype, seeds, per-model `LoadInfo`, peak memory, load time, UTC timestamp |
| `plots/` (`plain`) and `plots/defined/` | accuracy/macro-F1 with CIs, per-class F1, confusion matrices, reliability diagrams, risk-coverage curves, confidence histograms, latency box plots (PNG + PDF) |
| `prismnli_pipeline_check.json` | equivalence of the PrismNLI backend with the HF zero-shot pipeline (PROTOCOL §3.3) |
| `frozen_primary_plain/` | checksummed snapshot of the primary `plain` run taken before the `defined` run: `summary.json`, `summary.csv`, `raw_predictions.parquet`, `env.json` plus `SHA256SUMS` (verify with `shasum -a 256 -c results/frozen_primary_plain/SHA256SUMS` from the repo root). All supplementary and qualitative analyses read from this snapshot |
| `supplementary.json` | output of `supplementary_analysis.py`: paired plain-vs-defined tests, top confusions, agreement/oracle, NLL epsilon sensitivity, confidence slices, errors at coverage, API-confidence comparison, length effect, paired bootstrap CIs for Brier/ECE/coverage differences, accuracy on the first 600/400 rows |
| `jev_sequential_latency.json` | output of `jev_sequential_latency.py`: Jev end-to-end latency at concurrency 1 (100 validation rows) |
| `inspection_sample.md` / `inspection_sample.json` | output of `inspect_sample.py`: 22 frozen test rows in nine categories with every model's prediction and probabilities |
| `error_analysis.md` | qualitative error analysis written from the inspection sample and the confusion patterns |
| `contamination_research.json` | verbatim, re-fetched quotes from model cards, training notebooks and press for each system's exposure to `dair-ai/emotion` (REPORT.md Section 5) |
| `run_plain.log` / `run_defined.log` | console logs of the two test-split invocations |
| `cache/` | Jev response caches (test split: `jev_plain.jsonl`, `jev_defined.jsonl`) and `cache/smoke_validation/`, `cache/seq_latency_validation/` (validation rows only) |
| `<dataset>/` (`tweet_topic/`, `fin_topic/`, `daily_dialog/`) | the same file set (`raw_predictions.parquet`, `summary.json`, `summary.csv`, `env.json`, `plots/`, `cache/`) for each follow-up dataset of `PROTOCOL_ADDENDUM_v2.md`; `summary.*` additionally carry `n_classes` and `majority_class_accuracy`, and `env.json`/`summary.json` embed the full `DatasetSpec` (source, revision, split, labels, instruction, template) plus Laya's applied temperature bucket |
| `smoke_validation/`, `<dataset>/smoke_<split>/` | outputs of smoke runs (never the evaluated split) |

Rows on which a model failed (after all retries) carry `{model}_error`, `pred = -1` and NaN
probabilities and are counted in `n_errors`. Per PROTOCOL.md §5 all headline metrics, CIs and pairwise tests of a variant are computed on the rows every model scored (`n_common`); any shortfall is recorded under `deviation` in `summary.json` and, on the test split, appended to `results/deviations.md`.

## Reproducibility notes

- `random`, `numpy` and `torch` are seeded with 0 and `torch.use_deterministic_algorithms(True,
  warn_only=True)` is requested (PROTOCOL §8). No sampling occurs anywhere; all local inference is
  eval-mode fp32 argmax/softmax, so reruns agree to floating-point tolerance on the same device.
- Dataset and both local model revisions are pinned by commit hash; Jev is pinned to a dated model
  snapshot and the `model` string echoed by the API is recorded per response.
- Bootstrap CIs and paired bootstrap tests use `numpy.random.default_rng(0)` with 10000 resamples
  and share one index matrix.
- Remote latency (Jev) is client end-to-end and is **not comparable** to local compute latency
  (Laya, PrismNLI: batch size 1, after 10 validation warm-up calls). The `latency_kind` column and
  all figures label them separately.
- The Jev API rounds probabilities to 2 decimals and Laya to 4, so `p_gold == 0` exactly can occur;
  NLL clips at 1e-6 and `frac_gold_prob_zero` reports how often it happens.
- Full package pins are in `requirements.txt`; the exact versions used for a run are in
  `results/env.json`.

- Absolute home-directory paths in `REPORT.md`, `results/env.json`, `results/frozen_primary_plain/env.json` and the run logs were redacted to `.`/`~` before publication; `results/frozen_primary_plain/SHA256SUMS` was regenerated after that edit (the `summary.json` and `raw_predictions.parquet` digests are unchanged).

## Results

Primary variant (`plain`, bare label names), `dair-ai/emotion` test split, n = 2000, run 2026-09-20.
Full report: [`REPORT.md`](REPORT.md).

| Model | Accuracy | Macro-F1 | Brier ↓ | ECE ↓ | NLL ↓ | p50 latency | p95 latency | Cost |
|---|---|---|---|---|---|---|---|---|
| Jev 1.13 (pinned `typesafe/jev-1.13-20260917`) | 0.587 [0.565, 0.608] | 0.500 [0.471, 0.528] | 0.667 | 0.281 | 2.845 | 349 ms (remote e2e) | 593 ms (remote e2e) | $0.0284 (sum of the per-response `usage.cost` field; equals list price $0.042/M input x 676 323 tokens), 2000 calls |
| PrismNLI-0.4B | 0.725 [0.705, 0.744] | 0.647 [0.620, 0.673] | 0.441 | 0.174 | 1.174 | 58 ms (local, MPS fp32 bs=1) | 83 ms (local, MPS fp32 bs=1) | $0 API (self-hosted); *estimate*: 123 s device wall time for 2000 rows on M1 Max (about 0.034 device-hours) |
| Laya | 0.587 [0.565, 0.609] | 0.493 [0.463, 0.522] | 0.707 | 0.307 | 2.032 | 31 ms (local, MPS fp32 bs=1) | 35 ms (local, MPS fp32 bs=1) | $0 API (self-hosted); *estimate*: 63 s device wall time for 2000 rows on M1 Max (about 0.018 device-hours) |

Brackets are 95% percentile bootstrap CIs (10 000 resamples, seed 0). ECE is the 15-bin equal-width
value on max probability. Jev latency is remote end-to-end under 8 concurrent requests from Perth,
Australia; local latencies are on-device compute at batch size 1 and are not comparable to it.

**Executive summary**

- PrismNLI-0.4B is the best system on every headline quality metric: accuracy 0.725 vs 0.587 for both
  Jev and Laya. The 0.138 gap is significant (exact McNemar p = 1.5e-41; paired bootstrap 95% CI for
  Jev minus PrismNLI [-0.158, -0.118]), and it holds on every per-class F1 point estimate and at every
  selective-classification coverage level (105 vs 261 vs 292 errors at 50% coverage).
- Jev and Laya are statistically indistinguishable in accuracy (218 vs 219 discordant rows out of
  2000; McNemar p = 1.000; paired difference CI [-0.022, +0.020]) and macro-F1 (0.500 vs 0.493). Laya
  therefore reproduces Jev's accuracy with open weights, but the Laya card's claimed +0.115 gap over
  Jev does not survive a matched protocol.
- Jev's "decision model" architecture gives no measurable advantage over the 0.4B NLI classifier on
  this benchmark: it is behind on accuracy, F1, Brier (0.667 vs 0.441), ECE (0.281 vs 0.174) and NLL.
  Its remaining advantages (native choice probabilities, no weights to host, fixed per-call price of
  $0.0142 per 1000 decisions) are operational and were not scored here.
- Calibration order on `plain` is PrismNLI, then Jev, then Laya (ECE 0.174 / 0.281 / 0.307). The
  Jev-over-Laya margin is small (paired bootstrap Brier -0.040 [-0.072, -0.009], ECE -0.026
  [-0.047, -0.004]) and reverses under the `defined` variant (Laya ECE 0.230 vs Jev 0.276). Jev's NLL
  is inflated by its 2-decimal API rounding (15.1% of rows put exactly 0 on the gold label). All three
  systems are over-confident in every bin that carries meaningful mass.
- Both rankings survive the `defined` prompt variant (definitions move Jev +0.013 [+0.003, +0.023],
  PrismNLI -0.023 [-0.036, -0.010], Laya +0.011 [-0.006, +0.027]; only Laya's shift is within noise). Local per-example compute on an M1 Max (Laya p50 31 ms, PrismNLI 58 ms)
  is 6-11x below Jev's remote round trip (p50 349 ms), but these are different quantities.

**Contamination caveat.** PrismNLI-0.4B was initialised from `deberta-v3-large-zeroshot-v2.0`, whose
card and harmonisation notebook show verified training exposure to the *train* and *validation*
splits of `dair-ai/emotion` (up to 500 rows per class, with a declarative emotion-hypothesis template
similar to ours; the test split is stated as held out). An unknown but non-zero part of PrismNLI's
14-point lead is therefore dataset familiarity rather than zero-shot capability, and the 2.3-point
drop under the `defined` template is consistent with (but not proof of) template familiarity. Laya's
authors claim the dataset was held out and Jev's authors claim all training data is self-made, but
neither publishes a training manifest, so neither claim is verifiable. PrismNLI's probabilities are
also adapter-derived (softmax over six entailment logits), so accuracy/F1 comparisons are more direct
than calibration comparisons. Details and verbatim sources: REPORT.md Section 5 and
`results/contamination_research.json`.

**Artifacts**

- [`REPORT.md`](REPORT.md) - full report (headline tables for both variants, methodology, per-class
  and confusion analysis, calibration, selective classification, contamination review, limitations).
- [`results/summary.csv`](results/summary.csv) - one headline row per model x variant
  (`results/summary.json` has every metric, reliability table and pairwise test).
- [`results/raw_predictions.parquet`](results/raw_predictions.parquet) - per-row predictions,
  probabilities, latencies and Jev cost/tokens for both variants.
- [`results/plots/`](results/plots/) - `plain` figures (PNG + PDF); `defined` counterparts in
  [`results/plots/defined/`](results/plots/defined/).
- [`results/frozen_primary_plain/`](results/frozen_primary_plain/) - checksummed snapshot of the
  primary run (`SHA256SUMS`).
