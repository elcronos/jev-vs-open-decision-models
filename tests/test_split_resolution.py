"""``--split`` resolution (benchmark.resolve_split / parse_args): the word ``validation`` must never
select an evaluated split. fin_topic's evaluated split is literally named ``validation``, so the
legacy emotion spellings are accepted for ``emotion`` only. No dataset access, no inference."""
from __future__ import annotations

import pytest

import benchmark as B
from datasets_registry import DAILY_DIALOG, EMOTION, FIN_TOPIC, TWEET_TOPIC


def test_generic_spellings_for_every_dataset() -> None:
    assert B.resolve_split(EMOTION, "eval") == (True, "test")
    assert B.resolve_split(EMOTION, "smoke") == (False, "validation")
    assert B.resolve_split(TWEET_TOPIC, "eval") == (True, "test_2021")
    assert B.resolve_split(TWEET_TOPIC, "smoke") == (False, "validation_2021")
    assert B.resolve_split(FIN_TOPIC, "eval") == (True, "validation")
    assert B.resolve_split(FIN_TOPIC, "smoke") == (False, "train")
    assert B.resolve_split(DAILY_DIALOG, "eval") == (True, "test")
    assert B.resolve_split(DAILY_DIALOG, "smoke") == (False, "validation")


def test_legacy_spellings_only_for_emotion() -> None:
    assert B.resolve_split(EMOTION, "test") == (True, "test")
    assert B.resolve_split(EMOTION, "validation") == (False, "validation")
    for spec in (TWEET_TOPIC, FIN_TOPIC, DAILY_DIALOG):
        for legacy in ("test", "validation"):
            with pytest.raises(ValueError, match="accepted for dataset 'emotion' only"):
                B.resolve_split(spec, legacy)


def test_fin_topic_validation_never_resolves_to_eval() -> None:
    with pytest.raises(ValueError):
        B.resolve_split(FIN_TOPIC, "validation")
    # dataset-native names are rejected too
    for spec, native in ((FIN_TOPIC, "train"), (TWEET_TOPIC, "test_2021"), (TWEET_TOPIC, "validation_2021")):
        with pytest.raises(ValueError):
            B.resolve_split(spec, native)


def test_parse_args_rejects_fin_topic_validation() -> None:
    with pytest.raises(SystemExit) as exc:
        B.parse_args(["--dataset", "fin_topic", "--split", "validation", "--limit", "8", "--models", "prismnli"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit):
        B.parse_args(["--dataset", "daily_dialog", "--split", "test", "--models", "prismnli"])
    with pytest.raises(SystemExit):
        B.parse_args(["--dataset", "tweet_topic", "--split", "test_2021", "--models", "prismnli"])


def test_parse_args_keeps_emotion_legacy_commands() -> None:
    a = B.parse_args(["--split", "validation", "--limit", "8", "--models", "prismnli,laya", "--variants", "plain,defined"])
    assert a.split == "validation" and B.resolve_split(EMOTION, a.split) == (False, "validation")
    assert B.default_out_dir(EMOTION, False) == B.RESULTS_DIR / "smoke_validation"
    a = B.parse_args(["--split", "test", "--models", "jev,prismnli,laya", "--variants", "plain"])
    assert B.resolve_split(EMOTION, a.split) == (True, "test")
    assert B.default_out_dir(EMOTION, True) == B.RESULTS_DIR
    assert B.jev_cache_dir(EMOTION, True) == B.RESULTS_DIR / "cache"


def test_smoke_outputs_never_touch_production_dirs() -> None:
    for spec in (TWEET_TOPIC, FIN_TOPIC, DAILY_DIALOG):
        is_eval, _ = B.resolve_split(spec, "smoke")
        assert not is_eval
        assert B.default_out_dir(spec, is_eval) == B.RESULTS_DIR / spec.key / f"smoke_{spec.smoke_split_name}"
        assert B.jev_cache_dir(spec, is_eval) == B.RESULTS_DIR / spec.key / "cache" / f"smoke_{spec.smoke_split_name}"
