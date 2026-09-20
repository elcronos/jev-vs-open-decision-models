"""Tests for models.jev: response validation and cache persistence. No network; ``_attempt`` is patched."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

import models.jev as J
from models.common import LABELS


def _ok_body(choice: str = "joy") -> Dict[str, Any]:
    probs = {label: 0.0 for label in LABELS}
    probs[choice] = 1.0
    return {
        "id": "gen-1",
        "model": J.JEV_MODEL_ID,
        "provider": "typesafe",
        "answers": {"emotion": {"choice": choice, "probabilities": probs, "confidence": 0.9}},
        "usage": {"input_tokens": 10, "output_tokens": 1, "cost": 1e-6},
    }


@pytest.fixture
def backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> J.JevBackend:
    monkeypatch.setenv(J.API_KEY_ENV, "test-key-not-real")
    monkeypatch.setattr(J.time, "sleep", lambda s: None)  # no backoff waits in tests
    b = J.JevBackend(cache_dir=tmp_path, concurrency=2, max_attempts=3)
    b.load()
    yield b
    b.close()


def _cache_lines(backend: J.JevBackend, variant: str = "plain") -> List[Dict[str, Any]]:
    path = backend.cache_path(variant)
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def test_error_envelope_with_200_is_retried_and_not_cached(backend: J.JevBackend, monkeypatch: pytest.MonkeyPatch) -> None:
    """A 2xx body carrying OpenRouter's ``{"error": ...}`` envelope is a retryable failed attempt."""
    import httpx

    requests: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"error": {"message": "upstream failure"}, "id": "gen-x"})

    backend._client = httpx.Client(transport=httpx.MockTransport(handler),
                                   headers={"Authorization": "Bearer test-key-not-real"})
    pred = backend.predict_many([(0, "hello")], variant="plain")[0]
    assert pred.error is not None and "error envelope" in pred.error and "upstream failure" in pred.error
    assert pred.pred == -1 and pred.retries == backend.max_attempts - 1
    assert len(requests) == backend.max_attempts == backend.http_calls
    assert pred.extra["http_status_history"] == ["200_error_body"] * backend.max_attempts
    assert _cache_lines(backend) == []
    # Fresh instance must re-request (nothing cached).
    b2 = J.JevBackend(cache_dir=backend.cache_dir, concurrency=1, max_attempts=1)
    monkeypatch.setattr(b2, "_attempt", lambda body: (_ok_body(), 5.0))
    with b2:
        assert b2.predict_many([(0, "hello")])[0].pred == LABELS.index("joy")
    assert len(_cache_lines(backend)) == 1


@pytest.mark.parametrize(
    "body",
    [
        {"id": "gen-x"},  # no answers
        {"answers": {"emotion": {"choice": "joy"}}},  # no probabilities
        {"answers": {"emotion": {"choice": "nonsense", "probabilities": {l: 0.0 for l in LABELS}}}},  # zero + bad choice
        {"answers": {"emotion": {"choice": "joy", "probabilities": {"joy": "high"}}}},  # non-numeric
    ],
)
def test_malformed_200_body_yields_error_prediction(backend: J.JevBackend, monkeypatch: pytest.MonkeyPatch, body: Dict[str, Any]) -> None:
    monkeypatch.setattr(backend, "_attempt", lambda b: (body, 1.0))
    pred = backend.predict_one(3, "text")
    assert pred.error is not None and pred.error.startswith("malformed response")
    assert pred.pred == -1
    assert pred.extra["http_status_history"] == ["200_malformed"] * backend.max_attempts
    assert _cache_lines(backend) == []


def test_zero_probs_with_known_choice_falls_back_to_onehot(backend: J.JevBackend, monkeypatch: pytest.MonkeyPatch) -> None:
    body = {"answers": {"emotion": {"choice": "fear", "probabilities": {l: 0.0 for l in LABELS}}}}
    monkeypatch.setattr(backend, "_attempt", lambda b: (body, 1.0))
    pred = backend.predict_one(0, "text")
    assert pred.error is None and pred.pred == LABELS.index("fear") and pred.confidence == 1.0
    assert len(_cache_lines(backend)) == 1


def test_successes_are_cached_from_workers_and_abort_cancels_queue(backend: J.JevBackend, monkeypatch: pytest.MonkeyPatch) -> None:
    executed: List[int] = []
    lock = threading.Lock()

    def fake_attempt(body: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        time.sleep(0.02)
        with lock:
            executed.append(1)
        return _ok_body(), 20.0

    monkeypatch.setattr(backend, "_attempt", fake_attempt)

    real_as_completed = J.as_completed

    def interrupting_as_completed(futures):  # type: ignore[no-untyped-def]
        for i, fut in enumerate(real_as_completed(futures)):
            if i == 2:
                raise KeyboardInterrupt
            yield fut

    monkeypatch.setattr(J, "as_completed", interrupting_as_completed)
    items = [(i, f"text {i}") for i in range(40)]
    t0 = time.perf_counter()
    with pytest.raises(KeyboardInterrupt):
        backend.predict_many(items)
    elapsed = time.perf_counter() - t0

    n_exec = len(executed)
    cached = {rec["dataset_index"] for rec in _cache_lines(backend)}
    assert n_exec < len(items), "queued futures were not cancelled"
    assert elapsed < 0.02 * len(items) / 2 * 0.8, "pool ran the whole queue after the interrupt"
    assert len(cached) == n_exec, "every executed success must be persisted by its worker"
    assert len(cached) >= 3
