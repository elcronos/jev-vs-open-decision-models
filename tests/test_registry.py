"""Tests for datasets_registry (PROTOCOL_ADDENDUM_v2.md §1-§3) and the spec threading through the
model adapters. Row-count tests read the pinned datasets from the Hugging Face cache/network; they
touch the evaluated splits only to count rows and check the label mapping, never to score them."""
from __future__ import annotations

import json

import pytest

import datasets_registry as R
from models import common
from models.jev import build_request_body
from models.laya import build_questions, temperature_bucket
from models.prismnli import hypothesis_for, pipeline_template


EXPECTED = {
    "tweet_topic": {
        "labels": ["arts & culture", "business & entrepreneurs", "pop culture", "daily life",
                   "sports & gaming", "science & technology"],
        "instruction": "Which single topic does this tweet belong to?",
        "template": "The topic of this tweet is {label}.",
        "n": 1693, "eval_split": "test_2021", "smoke_split": "validation_2021",
        "revision": "87b7a0d1c402dbb481db649569c556d9aa27ac05", "source": "cardiffnlp/tweet_topic_single",
        "extras": ("id", "date"),
    },
    "fin_topic": {
        "labels": ["Analyst Update", "Fed | Central Banks", "Company | Product News",
                   "Treasuries | Corporate Debt", "Dividend", "Earnings", "Energy | Oil", "Financials",
                   "Currencies", "General News | Opinion", "Gold | Metals | Materials", "IPO",
                   "Legal | Regulation", "M&A | Investments", "Macro", "Markets", "Politics",
                   "Personnel Change", "Stock Commentary", "Stock Movement"],
        "instruction": "Which single topic does this financial news tweet belong to?",
        "template": "The topic of this tweet is {label}.",
        "n": 4117, "eval_split": "validation", "smoke_split": "train",
        "revision": "acbc8af2a35ccf0916124efcbe9e6cf25f191012", "source": "zeroshot/twitter-financial-news-topic",
        "extras": (),
    },
    "daily_dialog": {
        "labels": ["no emotion", "anger", "disgust", "fear", "happiness", "sadness", "surprise"],
        "instruction": "Which single emotion is expressed in this utterance?",
        "template": "The emotion expressed in this utterance is {label}.",
        "n": 7740, "eval_split": "test", "smoke_split": "validation",
        "revision": "1668faf0c0dc44664f108c489fd0666128db2c48", "source": "OpenRL/daily_dialog",
        "extras": ("dialog_id", "turn_id"),
    },
}


def test_registry_keys() -> None:
    assert list(R.REGISTRY) == ["emotion", "tweet_topic", "fin_topic", "daily_dialog"]
    assert R.DATASET_KEYS == tuple(R.REGISTRY)
    with pytest.raises(KeyError):
        R.get_spec("nope")


def test_emotion_spec_reproduces_protocol_constants() -> None:
    s = R.EMOTION
    assert s.labels == list(common.LABELS) == ["sadness", "joy", "love", "anger", "fear", "surprise"]
    assert s.instruction == common.INSTRUCTION == "Which single primary emotion is expressed in this text?"
    assert s.definitions == common.DEFINITIONS
    assert s.source_id == common.DATASET_ID and s.revision == common.DATASET_REVISION
    assert s.eval_split_name == "test" and s.smoke_split_name == "validation"
    assert s.hypothesis_template == "The primary emotion expressed in this text is {label}."
    assert s.hypothesis("joy", "defined") == (
        "The primary emotion expressed in this text is joy (feeling happy, pleased, cheerful, content or excited).")
    assert s.variants == ("plain", "defined") and s.n_classes == 6 and s.question_id == "emotion"


@pytest.mark.parametrize("key", list(EXPECTED))
def test_followup_spec_strings(key: str) -> None:
    s, e = R.get_spec(key), EXPECTED[key]
    assert s.labels == e["labels"]
    assert s.n_classes == len(e["labels"])
    assert s.instruction == e["instruction"]
    assert s.hypothesis_template == e["template"]
    assert s.eval_split_name == e["eval_split"] and s.smoke_split_name == e["smoke_split"]
    assert s.revision == e["revision"] and s.source_id == e["source"]
    assert s.extra_columns == e["extras"]
    assert s.definitions is None and s.variants == ("plain",)
    assert s.n_eval_rows == e["n"]
    assert [s.hypothesis(l) for l in s.labels] == [e["template"].format(label=l) for l in e["labels"]]
    with pytest.raises(ValueError, match="no label definitions"):
        s.hypothesis(s.labels[0], "defined")
    with pytest.raises(ValueError, match="no label definitions"):
        s.criteria("defined", empty="")
    json.dumps(s.to_dict())


def test_hypothesis_rendering_no_emotion_verbatim() -> None:
    s = R.DAILY_DIALOG
    assert s.hypothesis("no emotion") == "The emotion expressed in this utterance is no emotion."
    assert s.hypotheses()[0] == "The emotion expressed in this utterance is no emotion."
    assert R.TWEET_TOPIC.hypothesis("arts & culture") == "The topic of this tweet is arts & culture."
    assert R.FIN_TOPIC.hypothesis("M&A | Investments") == "The topic of this tweet is M&A | Investments."
    with pytest.raises(KeyError):
        s.hypothesis("joy")


@pytest.mark.parametrize("key", list(EXPECTED))
def test_followup_eval_counts_and_smoke_rows(key: str) -> None:
    s, e = R.get_spec(key), EXPECTED[key]
    df = s.load_eval()
    assert len(df) == e["n"]
    assert list(df.columns) == ["dataset_index", "text", "gold_id", "gold_label", *e["extras"]]
    assert df["dataset_index"].tolist() == list(range(e["n"]))
    assert df["gold_id"].min() >= 0 and df["gold_id"].max() < s.n_classes
    assert sorted(df["gold_label"].unique()) == sorted(set(s.labels))  # every class occurs
    smoke = s.load_smoke()
    assert len(smoke) == R.SMOKE_N == 10
    assert len(s.load_smoke(8)) == 8
    assert len(s.load_eval(3)) == 3
    # smoke rows come from a different split than the evaluated rows
    assert not set(smoke["text"]).issubset(set(df["text"].iloc[:1000]))


def test_emotion_eval_count_and_smoke() -> None:
    df = R.EMOTION.load_eval()
    assert len(df) == 2000 and list(df.columns) == ["dataset_index", "text", "gold_id", "gold_label"]
    assert len(R.EMOTION.load_smoke(8)) == 8


def test_daily_dialog_flattening_and_verbatim_text() -> None:
    """PROTOCOL.md §3: the raw utterance is passed unmodified (the source's trailing space is kept)."""
    from datasets import load_dataset

    df = R.DAILY_DIALOG.load_smoke()
    assert df["dialog_id"].iloc[0] == 0 and df["turn_id"].tolist()[:3] == [0, 1, 2]
    raw = load_dataset(R.DAILY_DIALOG_REPO, split="validation", revision=R.DAILY_DIALOG_REVISION)
    raw_utts = [u for dialog in raw["dialog"] for u in dialog][: len(df)]
    assert df["text"].tolist() == raw_utts  # byte-for-byte, no strip
    assert raw_utts[0] == "Good morning , sir . Is there a bank near here ? "
    assert any(t != t.strip() for t in df["text"])
    ev = R.DAILY_DIALOG.load_eval()
    assert ev["dialog_id"].nunique() == 1000
    assert (ev.groupby("dialog_id")["turn_id"].min() == 0).all()
    # PROTOCOL_ADDENDUM_v2.md §1: 82% no emotion
    assert abs((ev["gold_id"] == 0).mean() - 0.82) < 0.01


def test_tweet_topic_extras() -> None:
    df = R.TWEET_TOPIC.load_eval(5)
    assert df["id"].dtype.kind == "i" and df["date"].str.match(r"\d{4}-\d{2}-\d{2}").all()
    assert "_" not in "".join(R.TWEET_TOPIC.labels)


# ------------------------------------------------------------------ spec threading through adapters


def test_jev_request_body_per_spec() -> None:
    body = build_request_body("t", "plain")  # default = emotion, PROTOCOL.md §3.1 verbatim
    assert body["questions"] == {"emotion": {"type": "choice", "instructions": common.INSTRUCTION,
                                             "criteria": {l: "" for l in common.LABELS}}}
    assert build_request_body("t", "defined")["questions"]["emotion"]["criteria"] == common.DEFINITIONS
    fin = build_request_body("t", "plain", R.FIN_TOPIC)
    q = fin["questions"][R.FIN_TOPIC.question_id]
    assert list(q["criteria"]) == R.FIN_TOPIC.labels and set(q["criteria"].values()) == {""}
    assert q["instructions"] == R.FIN_TOPIC.instruction
    with pytest.raises(ValueError):
        build_request_body("t", "defined", R.FIN_TOPIC)


def test_laya_questions_per_spec() -> None:
    q = build_questions("plain")
    assert q == {"emotion": {"type": "choice", "instructions": common.INSTRUCTION,
                             "criteria": {l: None for l in common.LABELS}}}
    dd = build_questions("plain", R.DAILY_DIALOG)["emotion"]
    assert list(dd["criteria"]) == R.DAILY_DIALOG.labels and set(dd["criteria"].values()) == {None}
    with pytest.raises(ValueError):
        build_questions("defined", R.TWEET_TOPIC)
    assert temperature_bucket(6) == "choice:6-10" and temperature_bucket(7) == "choice:6-10"
    assert temperature_bucket(20) == "choice:11+"


def test_prismnli_hypotheses_per_spec() -> None:
    assert hypothesis_for("joy", "plain") == "The primary emotion expressed in this text is joy."
    assert hypothesis_for("no emotion", "plain", R.DAILY_DIALOG) == "The emotion expressed in this utterance is no emotion."
    assert pipeline_template(R.FIN_TOPIC) == "The topic of this tweet is {}."
    assert pipeline_template().format("joy") == hypothesis_for("joy", "plain")
    with pytest.raises(ValueError):
        hypothesis_for("Macro", "defined", R.FIN_TOPIC)
