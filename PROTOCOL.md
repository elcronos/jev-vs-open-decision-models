# Frozen evaluation protocol (v1, frozen 2026-09-20 before any test-set inference)

This file is the contract every module implements. Nothing here may change after
`benchmark.py --split test` has been run for the first time. Any deviation must be
recorded in `results/deviations.md` with a reason.

## 1. Task

Six-way single-label emotion classification, zero-shot, no demonstrations, no training.

Instruction string (identical for all systems, verbatim):

    Which single primary emotion is expressed in this text?

Label set, canonical order (index = dataset label id):

    0 sadness, 1 joy, 2 love, 3 anger, 4 fear, 5 surprise

## 2. Data

- Hugging Face dataset `dair-ai/emotion`, config `split`, split `test`, all 2000 rows.
- Pin revision `cab853a1dbdf4c42c2b3ef2173804746df8825fe` (`datasets.load_dataset(..., revision=...)`).
- Integration / smoke tests use ONLY the first 8 rows of `validation`. The test split is never
  used for any prompt or code choice.
- `dataset_index` = row position in the test split (0..1999).

## 3. Systems and exactly what each one sees

All systems receive the raw `text` field unmodified, the instruction above, and the six labels.

### 3.1 Jev (remote, OpenRouter Decisions API)

- `POST https://openrouter.ai/api/alpha/decisions`, header `Authorization: Bearer $OPENROUTER_API_KEY`.
- Model id pinned: `typesafe/jev-1.13-20260917` (dated snapshot that the alias `typesafe/jev-1.13`
  currently resolves to). Record the `model` string echoed in every response.
- Request body:
  ```json
  {"model": "typesafe/jev-1.13-20260917",
   "state": "<text>",
   "questions": {"emotion": {"type": "choice",
                              "instructions": "Which single primary emotion is expressed in this text?",
                              "criteria": {"sadness": "", "joy": "", "love": "", "anger": "", "fear": "", "surprise": ""}}}}
  ```
  Empty-string criteria descriptions are accepted by the API and mean "label name only".
- Record per request: `answers.emotion.choice`, `answers.emotion.probabilities` (all six),
  `answers.emotion.confidence`, `usage.input_tokens`, `usage.output_tokens`, `usage.cost`, `id`,
  `provider`, `model`, client wall-clock latency (ms, `time.perf_counter` around the HTTP call,
  successful attempt only), number of retries, final error string if failed.
- The API returns probabilities rounded to 2 decimals; store them exactly as returned in
  `raw_probabilities` and also a renormalised copy (divide by sum; if sum == 0 fall back to one-hot
  on `choice`). Metrics use the renormalised copy. Prediction = argmax of renormalised probs; if it
  disagrees with `choice` record a flag `pred_mismatch`.
- Concurrency 8, timeout 60 s, up to 6 attempts, exponential backoff 1,2,4,8,16,32 s with jitter,
  retry on 429/5xx/timeouts/connection errors. Never print or log the key.
- Cache each successful response to `results/cache/jev_<variant>.jsonl` keyed by dataset_index so
  reruns resume.

### 3.2 Laya (local)

- `huggingface_hub.snapshot_download("convaiinnovations/laya", revision="c5d78730f3493e4fe16d61507ef4b78eef7318cf",
  allow_patterns=[root files only: "*.json", "model.safetensors", "tokenizer/*", "encoder/*", "*.py", "README.md"])`
  then `laya.load(<local_dir>, device=<device>)`. General English checkpoint (repo root), NOT
  `typed-decisions`, NOT `multilingual`.
- Question dict passed to `agent.predict(text, questions)`:
  ```python
  {"emotion": {"type": "choice",
               "instructions": "Which single primary emotion is expressed in this text?",
               "criteria": {"sadness": None, "joy": None, "love": None, "anger": None, "fear": None, "surprise": None}}}
  ```
  `None` renders as the bare label (see `laya.common.render_options`), matching Jev's empty-string
  descriptions.
- Out-of-the-box behaviour: Laya's shipped per-option-count temperature is applied by the library
  (`choice:6-10` = 1.0000158). Do not fit or override it. Record `probabilities`, `choice`,
  `confidence`, `act_probability` (if present).
- One example per `predict` call (batch size 1) for latency. Precision fp32 (library forces fp32 on
  MPS/CPU). Device: `mps` on Apple silicon, else `cuda`, else `cpu`.

### 3.3 PrismNLI-0.4B (local)

- `Jaehun/PrismNLI-0.4B`, revision `02b9102b34d5dce1bf29c4e6903fea33e1a0ddeb`, via
  `AutoModelForSequenceClassification` + `AutoTokenizer`. Label 0 = entailment, 1 = not_entailment.
- Premise = text. Hypothesis template (verbatim, one template for all labels):

      The primary emotion expressed in this text is {label}.

- Six (premise, hypothesis) pairs per example, one forward pass (batch of 6). Truncation
  `only_first`, max_length 512, fp32.
- Primary categorical distribution (HF `zero-shot-classification` `multi_label=False` semantics):
  `softmax_over_labels(entailment_logit[label])`.
- Also store the secondary conversion `p_indep[label] = softmax(binary logits)[entailment]`
  (independent P(entail)) and its L1-normalised version, to quantify the effect of the conversion.
- Verification: on the 8 validation smoke rows, compare primary probs to
  `transformers.pipeline("zero-shot-classification", ..., multi_label=False)` with the same template;
  require `max abs diff < 1e-4`. Save the check to `results/prismnli_pipeline_check.json`.
- No temperature scaling.

## 4. Variants

- `plain` (PRIMARY, headline): bare label names as above.
- `defined` (SECONDARY, run only after `plain` test results are written and frozen): identical
  runs where each label carries the same concise definition for all three systems.
  - Jev: `criteria[label] = DEFINITION[label]`
  - Laya: `criteria[label] = DEFINITION[label]` (renders `label: definition`)
  - PrismNLI: hypothesis `The primary emotion expressed in this text is {label} ({definition}).`

  DEFINITION (frozen now):
  ```
  sadness:  feeling unhappy, down, grief, loss, disappointment or loneliness
  joy:      feeling happy, pleased, cheerful, content or excited
  love:     feeling affection, warmth, tenderness, caring or romantic attachment
  anger:    feeling mad, irritated, annoyed, resentful or hostile
  fear:     feeling afraid, scared, anxious, nervous or worried
  surprise: feeling amazed, shocked, startled or caught off guard by something unexpected
  ```

## 5. Metrics (metrics.py), computed from identical 2000 rows per system

- accuracy; macro-F1; per-class precision/recall/F1/support; 6x6 confusion matrix (rows gold).
- NLL: `-mean(log(clip(p_gold, 1e-6, 1)))`, epsilon documented as 1e-6, no renormalisation after clip.
- Multiclass Brier: `mean(sum_k (p_k - onehot_k)^2)` (range 0..2).
- ECE: confidence = max prob; 15 equal-width bins on [0,1]; `sum_b (n_b/N) |acc_b - conf_b|`.
  Also report adaptive (equal-mass) 10-bin ECE as secondary and the reliability table.
- Mean confidence; accuracy conditional on confidence (per reliability bin).
- Selective classification: sort by confidence descending; risk-coverage curve; accuracy at
  coverage 0.5, 0.8, 0.9, 1.0 (ties broken by dataset_index for determinism).
- Bootstrap 95% CI (percentile) for accuracy and macro-F1: 10 000 resamples, `numpy` RNG seed 0.
- Paired tests for each model pair: exact McNemar (statsmodels `mcnemar(exact=True)`) on
  correct/incorrect; paired bootstrap of accuracy difference (same 10 000 index resamples, seed 0).
- Fraction of examples where p_gold == 0 exactly (Jev rounding artefact check).

## 6. Latency / efficiency

- Remote (Jev): end-to-end client latency per request (successful attempt), p50/p95/mean, plus
  total cost from `usage.cost`, total input/output tokens. Label it "remote API end-to-end".
- Local: model load time (separate); 10 warm-up calls on validation rows; then per-example warm
  latency at batch size 1 for all 2000 rows (p50/p95/mean, examples/s). Report device, dtype,
  batch size, `torch.mps.driver_allocated_memory()` peak (or CUDA peak) and parameter count
  (`sum(p.numel())`) and weight file size on disk. Label it "local compute latency".
- Never compare the two as if equivalent.

## 7. Outputs

- `results/raw_predictions.parquet`: one row per (dataset_index, variant) with columns:
  `dataset_index, variant, text, gold_id, gold_label,` and per model prefix `jev_ / prismnli_ / laya_`:
  `pred, p_sadness, p_joy, p_love, p_anger, p_fear, p_surprise, confidence, latency_ms, error, retries`
  plus `jev_cost_usd, jev_input_tokens, jev_output_tokens, jev_response_id, jev_model,
  jev_api_confidence, jev_raw_probs_json, laya_api_confidence, laya_act_probability,
  prismnli_indep_probs_json`.
- `results/summary.json`, `results/summary.csv` (headline table rows per model x variant).
- `results/env.json`: package versions, model revisions, dataset revision, hardware, dtype, seeds.
- `results/plots/*.png` (+ .pdf).

## 8. Seeds and determinism

`random`, `numpy`, `torch` seeded 0; `torch.use_deterministic_algorithms` where supported; models in
eval mode; no sampling anywhere.
