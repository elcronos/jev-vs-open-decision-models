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

## Layout

```
PROTOCOL.md                 frozen protocol (read this first)
REPORT.md                   final report (all numbers, methodology, contamination review, figures)
benchmark.py                end-to-end driver: data -> models -> parquet -> metrics -> plots
metrics.py                  PROTOCOL §5 metrics (accuracy, macro-F1, NLL, Brier, ECE, selective, bootstrap, McNemar)
plots.py                    PROTOCOL §7 figures (PNG 200 dpi + PDF)
render_plots.py             re-render every figure from results/summary.json + parquet (no model calls)
supplementary_analysis.py   post-hoc analyses from the frozen predictions -> results/supplementary.json (no model calls)
inspect_sample.py           deterministic 22-row qualitative sample -> results/inspection_sample.{md,json} (no model calls)
jev_sequential_latency.py   Jev latency at concurrency 1 on validation rows -> results/jev_sequential_latency.json
models/common.py            LABELS, INSTRUCTION, DEFINITIONS, revisions, Prediction / LoadInfo / Backend
models/jev.py               OpenRouter Decisions API adapter (threaded, cached, retrying)
models/laya.py              Laya local adapter
models/prismnli.py          PrismNLI-0.4B zero-shot NLI adapter (+ HF pipeline equivalence check)
tests/                      unit tests (metrics.py, benchmark.py merge logic, Jev adapter)
requirements.txt            pinned package versions
results/                    all outputs (see below)
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

Smoke tests use only the first 8 rows of the **validation** split, per PROTOCOL §2. The Jev
responses for validation rows are cached under `results/cache/smoke_validation/`, never in the
production cache. Per-module smoke tests also exist:

```bash
python -m models.jev --smoke          # also verifies cache resume (second pass makes 0 HTTP calls)
python -m models.prismnli --smoke     # also writes results/prismnli_pipeline_check.json
python -m models.laya --smoke
python plots.py                       # synthetic self-test of every figure
pytest -q tests/                      # metrics unit tests
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

Flags: `--split {validation,test}`, `--limit N`, `--models jev,prismnli,laya`,
`--variants plain,defined`, `--skip-plots`, `--out-dir DIR`, `--n-bootstrap N`
(protocol value 10000; lower it only for quick checks).

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
