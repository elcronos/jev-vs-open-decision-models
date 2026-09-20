"""Jev backend: remote zero-shot emotion classification via the OpenRouter Decisions API.

Implements PROTOCOL.md §3.1 exactly:

* ``POST https://openrouter.ai/api/alpha/decisions`` with ``Authorization: Bearer $OPENROUTER_API_KEY``.
* Pinned model id ``common.JEV_MODEL_ID``; the ``model`` string echoed by the API is recorded.
* ``state`` is the raw text; one ``choice`` question (id ``spec.question_id``, ``"emotion"`` for the
  primary benchmark) with the dataset's frozen instruction and a criteria dict whose keys are
  ``spec.labels`` verbatim and whose values are ``""`` (variant ``plain``) or the spec's definitions
  (variant ``defined``; only datasets with definitions support it).
* The dataset is a ``datasets_registry.DatasetSpec`` passed at construction; it defaults to the
  emotion spec, so every pre-existing call produces the exact request body of PROTOCOL.md §3.1.
* Concurrency 8, timeout 60 s, up to 6 attempts, exponential backoff 1,2,4,8,16,32 s with jitter,
  retry on 429 / 5xx / timeouts / connection errors; ``Retry-After`` honoured on 429.
* Probabilities are stored exactly as returned (``extra["raw_probs"]``) and renormalised into
  ``Prediction.probs`` (fallback: one-hot on the API ``choice`` if the sum is 0). ``pred`` is the
  argmax of the renormalised copy; ``extra["pred_mismatch"]`` flags disagreement with ``choice``.
* Every successful response is appended to a JSONL cache keyed by ``dataset_index`` so reruns resume.
  Caching happens inside the worker thread right after the response is parsed, so an interrupted run
  keeps every completed request. A 200 response whose body carries a top-level ``"error"`` envelope
  or lacks ``answers.emotion.probabilities`` is treated as a failed attempt (retried, never cached).

The API key is read from the environment and is never printed, logged, or written to disk.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

from datasets_registry import EMOTION, DatasetSpec
from models.common import (
    JEV_MODEL_ID,
    LABELS,
    VARIANTS,
    LoadInfo,
    Prediction,
    renormalise,
)

# --------------------------------------------------------------------------------------------------
# Protocol constants (§3.1)
# --------------------------------------------------------------------------------------------------

DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
API_KEY_ENV = "OPENROUTER_API_KEY"
CONCURRENCY = 8
TIMEOUT_S = 60.0
MAX_ATTEMPTS = 6
BACKOFF_S: Tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
RETRY_AFTER_CAP_S = 120.0
RETRYABLE_STATUSES = frozenset({408, 425, 429}) | frozenset(range(500, 600))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_DIR = PROJECT_ROOT / "results" / "cache"


class JevAPIError(Exception):
    """Raised internally for one failed HTTP attempt.

    ``retryable`` decides whether the attempt loop continues; ``status_tag`` is appended to the
    per-example ``http_status_history`` (an int HTTP status or a short string such as ``"timeout"``).
    """

    def __init__(self, message: str, *, retryable: bool, status_tag: Any,
                 retry_after_s: Optional[float] = None) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status_tag = status_tag
        self.retry_after_s = retry_after_s


def _parse_retry_after(value: Optional[str]) -> Optional[float]:
    """Parse an HTTP ``Retry-After`` header (delta-seconds or HTTP-date) into seconds, or None."""
    if not value:
        return None
    value = value.strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        dt = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        return None
    return max(0.0, dt.timestamp() - time.time())


def build_request_body(text: str, variant: str = "plain", spec: DatasetSpec = EMOTION) -> Dict[str, Any]:
    """Return the exact request body of PROTOCOL.md §3.1 for ``text``, ``variant`` and dataset ``spec``.

    Criteria keys are ``spec.labels`` verbatim, values ``""`` (plain) or the spec's definitions
    (defined). Raises ``ValueError`` for ``defined`` on a dataset without definitions.
    """
    criteria = spec.criteria(variant, empty="")
    return {
        "model": JEV_MODEL_ID,
        "state": text,
        "questions": {
            spec.question_id: {
                "type": "choice",
                "instructions": spec.instruction,
                "criteria": criteria,
            }
        },
    }


class JevBackend:
    """OpenRouter Decisions API adapter implementing the ``Backend`` protocol (``name == "jev"``).

    Parameters
    ----------
    cache_dir:
        Directory holding ``jev_<variant>.jsonl`` caches. Defaults to ``results/cache``. Smoke
        tests must pass a different directory so validation rows never collide with test-split
        indices in the production cache.
    use_cache:
        If False, the cache is neither read nor written (useful for one-off checks).
    concurrency, timeout_s, max_attempts:
        Protocol values; exposed for tests only, do not change for real runs.
    spec:
        The ``DatasetSpec`` (labels, instruction, question id, definitions). Defaults to the
        emotion spec of the primary benchmark.
    """

    name: str = "jev"

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        *,
        spec: DatasetSpec = EMOTION,
        use_cache: bool = True,
        concurrency: int = CONCURRENCY,
        timeout_s: float = TIMEOUT_S,
        max_attempts: int = MAX_ATTEMPTS,
        endpoint: str = DECISIONS_URL,
        show_progress: bool = False,
    ) -> None:
        self.spec = spec
        self.cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
        self.use_cache = use_cache
        self.concurrency = concurrency
        self.timeout_s = timeout_s
        self.max_attempts = max_attempts
        self.endpoint = endpoint
        self.show_progress = show_progress

        self._api_key: Optional[str] = None
        self._client: Optional[httpx.Client] = None
        self._loaded = False

        # In-memory mirror of the JSONL caches: variant -> {dataset_index: Prediction}.
        self._cache: Dict[str, Dict[int, Prediction]] = {}
        self._cache_lock = threading.Lock()

        # Number of HTTP requests actually sent (all attempts, all variants) since construction.
        self._http_calls = 0
        self._counter_lock = threading.Lock()

    # ------------------------------------------------------------------ Backend interface

    def load(self) -> LoadInfo:
        """Read the API key from the environment and open the shared HTTP client (no weights)."""
        key = os.environ.get(API_KEY_ENV, "").strip()
        if not key:
            raise RuntimeError(f"environment variable {API_KEY_ENV} is not set")
        self._api_key = key
        if self._client is None:
            self._client = httpx.Client(
                timeout=httpx.Timeout(self.timeout_s),
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                limits=httpx.Limits(max_connections=self.concurrency,
                                    max_keepalive_connections=self.concurrency),
            )
        self._loaded = True
        return LoadInfo(
            name=self.name,
            load_time_s=0.0,
            device="remote",
            dtype="n/a",
            revision=JEV_MODEL_ID,
            param_count=None,
            weight_bytes=None,
            extra={
                "endpoint": self.endpoint,
                "concurrency": self.concurrency,
                "timeout_s": self.timeout_s,
                "max_attempts": self.max_attempts,
                "backoff_s": list(BACKOFF_S[: max(0, self.max_attempts - 1)]),
                "dataset": self.spec.key,
                "question_id": self.spec.question_id,
                "n_options": self.spec.n_classes,
            },
        )

    def warmup(self, texts: List[str], variant: str = "plain") -> None:
        """Remote model: nothing to warm. Performs one uncached call to verify authentication.

        Raises ``RuntimeError`` if the call fails (e.g. 401/403), so a bad key is caught before the
        main run starts. The warmup prediction is not cached and not returned.
        """
        self._ensure_loaded()
        if not texts:
            return
        pred = self._predict_uncached(dataset_index=-1, text=texts[0], variant=variant)
        if pred.error is not None:
            raise RuntimeError(f"Jev warmup/auth check failed: {pred.error}")

    def predict_one(self, dataset_index: int, text: str, variant: str = "plain") -> Prediction:
        """Score one example. Serves from the cache when possible and caches a success."""
        self._ensure_loaded()
        cached = self._cache_get(variant, dataset_index)
        if cached is not None:
            return cached
        pred = self._predict_uncached(dataset_index, text, variant)
        if pred.error is None:
            self._cache_put(variant, pred)
        return pred

    def predict_many(self, items: List[tuple], variant: str = "plain") -> List[Prediction]:
        """Score ``items`` = [(dataset_index, text), ...] with a thread pool of ``concurrency``.

        Returns predictions in the same order as ``items``. Indices already in the cache are not
        re-requested; each new success is appended to the cache by the worker thread that produced
        it, so results are persisted even if this method is interrupted. On ``KeyboardInterrupt`` (or
        any other exception escaping the collection loop) queued requests are cancelled before the
        exception propagates, so no further paid calls are issued.
        """
        self._ensure_loaded()
        self._load_cache(variant)
        results: Dict[int, Prediction] = {}
        todo: List[Tuple[int, str]] = []
        for dataset_index, text in items:
            cached = self._cache_get(variant, dataset_index)
            if cached is not None:
                results[dataset_index] = cached
            else:
                todo.append((int(dataset_index), text))

        if todo:
            progress = self._progress_bar(len(todo))
            pool = ThreadPoolExecutor(max_workers=self.concurrency)
            try:
                futures = {
                    pool.submit(self._predict_and_cache, idx, text, variant): idx
                    for idx, text in todo
                }
                for future in as_completed(futures):
                    pred = future.result()
                    results[pred.dataset_index] = pred
                    if progress is not None:
                        progress.update(1)
            except BaseException:
                # Stop issuing new requests; in-flight ones finish and cache themselves.
                pool.shutdown(wait=False, cancel_futures=True)
                raise
            finally:
                pool.shutdown(wait=True)
                if progress is not None:
                    progress.close()

        return [results[int(dataset_index)] for dataset_index, _ in items]

    def _predict_and_cache(self, dataset_index: int, text: str, variant: str) -> Prediction:
        """Worker body for ``predict_many``: score one example and persist it on success.

        Running ``_cache_put`` here (``_cache_lock`` makes the append thread-safe) rather than in the
        consumer loop guarantees that every completed request is on disk regardless of what happens
        to the main thread afterwards.
        """
        pred = self._predict_uncached(dataset_index, text, variant)
        if pred.error is None:
            self._cache_put(variant, pred)
        return pred

    def peak_memory_bytes(self) -> Optional[int]:
        """Remote inference: no local model memory to report."""
        return None

    # ------------------------------------------------------------------ extras

    @property
    def http_calls(self) -> int:
        """Total HTTP requests sent by this instance (every attempt counts)."""
        with self._counter_lock:
            return self._http_calls

    def reset_http_calls(self) -> None:
        with self._counter_lock:
            self._http_calls = 0

    def cache_path(self, variant: str) -> Path:
        return self.cache_dir / f"jev_{variant}.jsonl"

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
        self._loaded = False

    def __enter__(self) -> "JevBackend":
        self.load()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------ HTTP + parsing

    def _ensure_loaded(self) -> None:
        if not self._loaded or self._client is None:
            self.load()

    def _bump_calls(self) -> None:
        with self._counter_lock:
            self._http_calls += 1

    def _scrub(self, message: str) -> str:
        """Defensive: make sure the key can never leak through an error string."""
        if self._api_key and self._api_key in message:
            message = message.replace(self._api_key, "***")
        return message

    def _attempt(self, body: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        """One HTTP attempt. Returns (parsed JSON, latency_ms) or raises ``JevAPIError``."""
        assert self._client is not None
        self._bump_calls()
        t0 = time.perf_counter()
        try:
            response = self._client.post(self.endpoint, json=body)
        except httpx.TimeoutException as exc:
            raise JevAPIError(f"timeout: {exc.__class__.__name__}", retryable=True,
                              status_tag="timeout") from exc
        except httpx.TransportError as exc:
            raise JevAPIError(f"connection error: {exc.__class__.__name__}: {exc}",
                              retryable=True, status_tag="connection_error") from exc
        latency_ms = (time.perf_counter() - t0) * 1000.0

        status = response.status_code
        if status >= 400:
            snippet = response.text[:300].replace("\n", " ")
            retry_after = _parse_retry_after(response.headers.get("Retry-After")) if status == 429 else None
            raise JevAPIError(f"HTTP {status}: {snippet}", retryable=status in RETRYABLE_STATUSES,
                              status_tag=status, retry_after_s=retry_after)
        try:
            data = response.json()
        except ValueError as exc:
            raise JevAPIError(f"HTTP {status}: non-JSON body", retryable=False,
                              status_tag=f"{status}_bad_json") from exc
        if not isinstance(data, dict):
            raise JevAPIError(f"HTTP {status}: JSON body is not an object", retryable=False,
                              status_tag=f"{status}_bad_json")
        if "error" in data:
            # OpenRouter can return an error envelope with a 2xx status (e.g. upstream failure).
            err = data.get("error")
            detail = err.get("message") if isinstance(err, dict) else err
            snippet = str(detail)[:300].replace("\n", " ")
            raise JevAPIError(f"HTTP {status} with error envelope: {snippet}", retryable=True,
                              status_tag=f"{status}_error_body")
        return data, latency_ms

    def _predict_uncached(self, dataset_index: int, text: str, variant: str) -> Prediction:
        """Full retry loop for one example; never raises, errors are recorded on the Prediction.

        A 2xx body that cannot be mapped to a probability vector (see ``_to_prediction``) counts as a
        failed, retryable attempt, so the returned ``Prediction`` always has either finite ``probs``
        and ``error is None`` or ``pred == -1`` and a non-empty ``error`` (never cached).
        """
        body = build_request_body(text, variant, self.spec)
        history: List[Any] = []
        last_error = "no attempt made"
        for attempt in range(1, self.max_attempts + 1):
            try:
                data, latency_ms = self._attempt(body)
                pred = self._to_prediction(dataset_index, data, latency_ms,
                                           retries=attempt - 1, history=history + [200])
                history.append(200)
                return pred
            except JevAPIError as exc:
                history.append(exc.status_tag)
                last_error = self._scrub(str(exc))
                if not exc.retryable or attempt == self.max_attempts:
                    break
                base = BACKOFF_S[min(attempt - 1, len(BACKOFF_S) - 1)]
                sleep_s = base * random.uniform(0.8, 1.2)
                if exc.retry_after_s is not None:
                    sleep_s = max(sleep_s, min(exc.retry_after_s, RETRY_AFTER_CAP_S))
                time.sleep(sleep_s)
        return Prediction(
            dataset_index=int(dataset_index),
            probs=[float("nan")] * self.spec.n_classes,
            pred=-1,
            confidence=float("nan"),
            latency_ms=float("nan"),
            error=last_error,
            retries=len(history) - 1 if history else 0,
            extra={
                "api_choice": None,
                "api_confidence": None,
                "raw_probs": None,
                "cost_usd": None,
                "input_tokens": None,
                "output_tokens": None,
                "response_id": None,
                "model": None,
                "provider": None,
                "pred_mismatch": None,
                "http_status_history": history,
                "variant": variant,
            },
        )

    def _to_prediction(self, dataset_index: int, data: Dict[str, Any], latency_ms: float,
                       *, retries: int, history: List[Any]) -> Prediction:
        """Map a successful response body to a ``Prediction`` (§3.1 recording rules).

        Raises ``JevAPIError`` (retryable, ``status_tag="200_malformed"``) when the body has no
        ``answers.<question_id>``, no ``probabilities`` dict, or a probability vector that sums to 0
        with an unrecognised ``choice`` -- such a body must never become a cached ``error=None``
        prediction.
        """
        labels = self.spec.labels
        label2id = self.spec.label2id
        qid = self.spec.question_id
        answers = data.get("answers")
        answer = answers.get(qid) if isinstance(answers, dict) else None
        if not isinstance(answer, dict) or not answer:
            raise JevAPIError(f"malformed response: missing answers.{qid}", retryable=True,
                              status_tag="200_malformed")
        raw_probs = answer.get("probabilities")
        if not isinstance(raw_probs, dict):
            raise JevAPIError("malformed response: answers.emotion.probabilities is not an object",
                              retryable=True, status_tag="200_malformed")
        try:
            raw_probs = {str(k): float(v) for k, v in raw_probs.items()}
        except (TypeError, ValueError) as exc:
            raise JevAPIError(f"malformed response: non-numeric probability ({exc})",
                              retryable=True, status_tag="200_malformed") from exc
        api_choice = answer.get("choice")
        choice_id = label2id.get(api_choice, -1) if isinstance(api_choice, str) else -1

        ordered = [raw_probs.get(label, 0.0) for label in labels]
        probs = renormalise(ordered, fallback_index=choice_id if choice_id >= 0 else None)
        if sum(probs) <= 0:  # sum was 0 and choice unrecognised: nothing to fall back on
            raise JevAPIError("malformed response: probabilities sum to 0 and choice is unrecognised",
                              retryable=True, status_tag="200_malformed")
        pred = max(range(len(labels)), key=probs.__getitem__)
        confidence = float(probs[pred])

        usage = data.get("usage") or {}
        api_conf = answer.get("confidence")
        return Prediction(
            dataset_index=int(dataset_index),
            probs=probs,
            pred=pred,
            confidence=confidence,
            latency_ms=float(latency_ms),
            error=None,
            retries=retries,
            extra={
                "api_choice": api_choice,
                "api_confidence": float(api_conf) if api_conf is not None else None,
                "raw_probs": raw_probs,
                "cost_usd": usage.get("cost"),
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "response_id": data.get("id"),
                "model": data.get("model"),
                "provider": data.get("provider"),
                "pred_mismatch": bool(pred != choice_id),
                "http_status_history": history,
            },
        )

    # ------------------------------------------------------------------ cache

    def _load_cache(self, variant: str) -> Dict[int, Prediction]:
        """Load ``jev_<variant>.jsonl`` into memory once (corrupt trailing lines are skipped)."""
        with self._cache_lock:
            if variant in self._cache:
                return self._cache[variant]
            entries: Dict[int, Prediction] = {}
            path = self.cache_path(variant)
            if self.use_cache and path.exists():
                with path.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue  # partial line from an interrupted write
                        record.pop("variant", None)
                        pred = Prediction(**record)
                        if pred.error is None:
                            entries[int(pred.dataset_index)] = pred
            self._cache[variant] = entries
            return entries

    def _cache_get(self, variant: str, dataset_index: int) -> Optional[Prediction]:
        if not self.use_cache:
            return None
        entries = self._load_cache(variant)
        with self._cache_lock:
            return entries.get(int(dataset_index))

    def _cache_put(self, variant: str, pred: Prediction) -> None:
        """Append one successful prediction to the JSONL cache and flush immediately."""
        if not self.use_cache or pred.error is not None:
            return
        entries = self._load_cache(variant)
        record = pred.to_dict()
        record["variant"] = variant
        line = json.dumps(record, ensure_ascii=False)
        with self._cache_lock:
            if pred.dataset_index in entries:
                return
            entries[int(pred.dataset_index)] = pred
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with self.cache_path(variant).open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()

    # ------------------------------------------------------------------ misc

    def _progress_bar(self, total: int) -> Any:
        if not self.show_progress:
            return None
        try:
            from tqdm import tqdm
        except ImportError:
            return None
        return tqdm(total=total, desc="jev", unit="req", leave=False)


# --------------------------------------------------------------------------------------------------
# Smoke test CLI (validation rows 0..7 only; never touches the test split or production cache)
# --------------------------------------------------------------------------------------------------

SMOKE_CACHE_DIR = PROJECT_ROOT / "results" / "cache" / "smoke_validation"
SMOKE_N = 8


def _load_smoke_rows() -> List[Tuple[int, str, int]]:
    """First 8 rows of the pinned validation split as (index, text, gold_id)."""
    from datasets import load_dataset  # heavy import; keep local to the CLI
    from models.common import DATASET_CONFIG, DATASET_ID, DATASET_REVISION

    ds = load_dataset(DATASET_ID, DATASET_CONFIG, split="validation", revision=DATASET_REVISION)
    rows = ds.select(range(SMOKE_N))
    return [(i, row["text"], int(row["label"])) for i, row in enumerate(rows)]


def _run_smoke_pass(backend: JevBackend, rows: List[Tuple[int, str, int]], label: str) -> None:
    backend.reset_http_calls()
    total_cost = 0.0
    for variant in VARIANTS:
        print(f"\n=== {label} | variant={variant} ===")
        preds = backend.predict_many([(i, t) for i, t, _ in rows], variant=variant)
        print(f"{'idx':>3} {'gold':>8} {'pred':>8} {'mm':>2} {'ms':>7} {'cost_usd':>10}  probs(sadness,joy,love,anger,fear,surprise)")
        for (i, _, gold), p in zip(rows, preds):
            if p.error is not None:
                print(f"{i:>3} {LABELS[gold]:>8} {'ERROR':>8}  retries={p.retries} history={p.extra['http_status_history']} error={p.error}")
                continue
            cost = p.extra["cost_usd"] or 0.0
            total_cost += float(cost)
            probs = "[" + ", ".join(f"{x:.2f}" for x in p.probs) + "]"
            mm = "*" if p.extra["pred_mismatch"] else ""
            print(f"{i:>3} {LABELS[gold]:>8} {LABELS[p.pred]:>8} {mm:>2} {p.latency_ms:7.0f} {cost:10.2e}  {probs}"
                  f"  api_choice={p.extra['api_choice']} api_conf={p.extra['api_confidence']}"
                  f" tokens={p.extra['input_tokens']}/{p.extra['output_tokens']} model={p.extra['model']}")
        n_ok = sum(p.error is None for p in preds)
        acc = sum(p.pred == gold for (_, _, gold), p in zip(rows, preds) if p.error is None)
        print(f"-- {n_ok}/{len(rows)} scored, {acc}/{n_ok} correct, cache={backend.cache_path(variant)}")
    print(f"\n[{label}] HTTP calls made this pass: {backend.http_calls}   summed cost this pass: {total_cost:.2e} USD")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Jev (OpenRouter Decisions) backend utilities")
    parser.add_argument("--smoke", action="store_true",
                        help="score validation rows 0..7 for both variants, twice (second pass must hit cache)")
    parser.add_argument("--fresh", action="store_true",
                        help="delete the smoke cache before running so the first pass really hits the API")
    args = parser.parse_args(argv)
    if not args.smoke:
        parser.print_help()
        return 0

    if args.fresh and SMOKE_CACHE_DIR.exists():
        for variant in VARIANTS:
            path = SMOKE_CACHE_DIR / f"jev_{variant}.jsonl"
            if path.exists():
                path.unlink()
        print(f"cleared smoke cache in {SMOKE_CACHE_DIR}")

    rows = _load_smoke_rows()
    print(f"loaded {len(rows)} validation rows (smoke); smoke cache dir = {SMOKE_CACHE_DIR}")

    with JevBackend(cache_dir=SMOKE_CACHE_DIR) as backend:
        info = backend.load()
        print("load info:", json.dumps(info.to_dict()))
        backend.warmup([rows[0][1]])
        print(f"warmup/auth check ok (HTTP calls so far: {backend.http_calls})")

        _run_smoke_pass(backend, rows, "pass 1 (fills cache)")
        calls_pass1 = backend.http_calls

    # Brand-new instance: cache must be reloaded from disk, and no HTTP call should be needed.
    with JevBackend(cache_dir=SMOKE_CACHE_DIR) as backend2:
        _run_smoke_pass(backend2, rows, "pass 2 (resume from cache)")
        calls_pass2 = backend2.http_calls

    print(f"\nRESUME CHECK: pass1 HTTP calls={calls_pass1}, pass2 HTTP calls={calls_pass2} "
          f"-> {'OK' if calls_pass2 == 0 else 'FAIL'}")
    return 0 if calls_pass2 == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
