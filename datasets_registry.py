"""Dataset registry: one ``DatasetSpec`` per benchmark dataset (PROTOCOL.md §1-§2, PROTOCOL_ADDENDUM_v2.md §1-§2).

A spec pins *everything* a run depends on that is dataset-specific: source repository and revision,
the evaluated split, the label strings in dataset id order, the instruction and the NLI hypothesis
template, the smoke/warm-up rows, and how the evaluated split is materialised as a DataFrame. The
model adapters (``models/jev.py``, ``models/laya.py``, ``models/prismnli.py``) take a spec at
construction and default to :data:`EMOTION`, so every pre-existing call keeps producing the exact
strings of the frozen primary benchmark.

Loaders return a DataFrame with the columns ``dataset_index, text, gold_id, gold_label`` followed by
``spec.extra_columns``. ``dataset_index`` is the row position in the evaluated split file (for
``daily_dialog`` the running utterance index over the flattened split). Texts are passed through
unmodified (PROTOCOL.md §3: "All systems receive the raw ``text`` field unmodified"); this includes
``daily_dialog``, whose source tokenisation leaves a trailing space on every utterance
(``'Good morning , sir . Is there a bank near here ? '``): that space is kept verbatim.

Nothing here is imported by ``models/common.py``, so the primary constants stay the single source of
truth for the emotion benchmark and the emotion spec is built *from* them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from models.common import (
    DATASET_CONFIG,
    DATASET_ID,
    DATASET_REVISION,
    DEFINITIONS,
    INSTRUCTION,
    LABELS,
    VARIANTS,
)

Loader = Callable[[Optional[int]], pd.DataFrame]

#: Number of smoke/warm-up rows fixed by PROTOCOL_ADDENDUM_v2.md §1 (first N rows of the smoke split).
SMOKE_N = 10

BASE_COLUMNS: Tuple[str, ...] = ("dataset_index", "text", "gold_id", "gold_label")


@dataclass(frozen=True)
class DatasetSpec:
    """Frozen description of one dataset as every system sees it."""

    key: str
    display_name: str
    source_id: str
    revision: str
    eval_split_name: str
    labels: List[str]
    instruction: str
    hypothesis_template: str
    smoke_split_name: str
    loader: Loader = field(repr=False, compare=False)
    smoke_loader: Loader = field(repr=False, compare=False)
    extra_columns: Tuple[str, ...] = ()
    #: Per-label definitions for the ``defined`` variant; ``None`` when the dataset has no definitions
    #: (addendum §3: only ``plain`` exists for the follow-up datasets).
    definitions: Optional[Dict[str, str]] = None
    #: Hypothesis template of the ``defined`` variant (``{label}`` and ``{definition}`` placeholders).
    hypothesis_template_defined: Optional[str] = None
    #: Question id used as the key of the single ``choice`` question sent to Jev / Laya.
    question_id: str = "emotion"
    #: Human-readable provenance note stored in env.json.
    source_note: str = ""
    #: Expected number of rows of the evaluated split (checked by the loader).
    n_eval_rows: Optional[int] = None

    # ------------------------------------------------------------------ derived

    @property
    def n_classes(self) -> int:
        return len(self.labels)

    @property
    def label2id(self) -> Dict[str, int]:
        return {label: i for i, label in enumerate(self.labels)}

    @property
    def variants(self) -> Tuple[str, ...]:
        """Variants this dataset supports: ``plain`` always, ``defined`` only with definitions."""
        return VARIANTS if self.definitions is not None else ("plain",)

    def check_variant(self, variant: str) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")
        if variant == "defined" and self.definitions is None:
            raise ValueError(
                f"dataset {self.key!r} has no label definitions; the 'defined' variant is not "
                f"available (PROTOCOL_ADDENDUM_v2.md §3). Use --variants plain."
            )

    def hypothesis(self, label: str, variant: str = "plain") -> str:
        """Render the NLI hypothesis for one label (verbatim template, no special-casing)."""
        self.check_variant(variant)
        if label not in self.label2id:
            raise KeyError(f"{label!r} is not a label of dataset {self.key!r}")
        if variant == "plain":
            return self.hypothesis_template.format(label=label)
        assert self.definitions is not None and self.hypothesis_template_defined is not None
        return self.hypothesis_template_defined.format(label=label, definition=self.definitions[label])

    def hypotheses(self, variant: str = "plain") -> List[str]:
        return [self.hypothesis(label, variant) for label in self.labels]

    def criteria(self, variant: str, empty: Any) -> Dict[str, Any]:
        """Criteria dict for a ``choice`` question: label -> ``empty`` (plain) or definition (defined).

        Jev uses ``empty=""`` (PROTOCOL.md §3.1), Laya ``empty=None`` (§3.2).
        """
        self.check_variant(variant)
        if variant == "plain":
            return {label: empty for label in self.labels}
        assert self.definitions is not None
        return {label: self.definitions[label] for label in self.labels}

    # ------------------------------------------------------------------ data access

    def load_eval(self, limit: Optional[int] = None) -> pd.DataFrame:
        """The evaluated split (first ``limit`` rows if given), validated against the spec."""
        df = self.loader(limit)
        self._validate_frame(df, self.eval_split_name)
        if limit is None and self.n_eval_rows is not None and len(df) != self.n_eval_rows:
            raise RuntimeError(
                f"{self.key}: evaluated split {self.eval_split_name!r} has {len(df)} rows, "
                f"expected {self.n_eval_rows} (PROTOCOL_ADDENDUM_v2.md §1)"
            )
        return df

    def load_smoke(self, limit: Optional[int] = None) -> pd.DataFrame:
        """The smoke/warm-up rows (never part of the evaluated split); at most ``SMOKE_N`` rows unless
        the dataset's smoke split is itself the protocol's smoke split (emotion: whole validation)."""
        df = self.smoke_loader(limit)
        self._validate_frame(df, self.smoke_split_name)
        return df

    def _validate_frame(self, df: pd.DataFrame, split_name: str) -> None:
        expected = list(BASE_COLUMNS) + list(self.extra_columns)
        if list(df.columns) != expected:
            raise RuntimeError(f"{self.key}/{split_name}: columns {list(df.columns)} != {expected}")
        if len(df) == 0:
            raise RuntimeError(f"{self.key}/{split_name}: no rows loaded")
        gold = df["gold_id"].to_numpy()
        if gold.min() < 0 or gold.max() >= self.n_classes:
            raise RuntimeError(f"{self.key}/{split_name}: gold ids outside [0, {self.n_classes - 1}]")
        if not (df["gold_label"] == [self.labels[int(g)] for g in gold]).all():
            raise RuntimeError(f"{self.key}/{split_name}: gold_label does not match gold_id")
        if not (df["dataset_index"].to_numpy() == np.arange(len(df))).all():
            raise RuntimeError(f"{self.key}/{split_name}: dataset_index must be 0..n-1 in file order")

    # ------------------------------------------------------------------ serialisation

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe description recorded in env.json / summary.json."""
        return {
            "key": self.key,
            "display_name": self.display_name,
            "source_id": self.source_id,
            "revision": self.revision,
            "eval_split_name": self.eval_split_name,
            "smoke_split_name": self.smoke_split_name,
            "smoke_rows": SMOKE_N,
            "n_classes": self.n_classes,
            "labels": list(self.labels),
            "instruction": self.instruction,
            "hypothesis_template": self.hypothesis_template,
            "hypothesis_template_defined": self.hypothesis_template_defined,
            "definitions": dict(self.definitions) if self.definitions is not None else None,
            "variants": list(self.variants),
            "question_id": self.question_id,
            "extra_columns": list(self.extra_columns),
            "n_eval_rows": self.n_eval_rows,
            "source_note": self.source_note,
        }


# ----------------------------------------------------------------------------------------------
# Helpers shared by the loaders
# ----------------------------------------------------------------------------------------------


def _frame(texts: Sequence[str], gold: Sequence[int], labels: Sequence[str],
           extras: Optional[Dict[str, Sequence[Any]]] = None, limit: Optional[int] = None) -> pd.DataFrame:
    n = len(texts) if limit is None else min(int(limit), len(texts))
    gold_arr = np.asarray(list(gold)[:n], dtype=np.int64)
    data: Dict[str, Any] = {
        "dataset_index": np.arange(n, dtype=np.int64),
        "text": [str(t) for t in list(texts)[:n]],
        "gold_id": gold_arr,
        "gold_label": [labels[int(i)] for i in gold_arr],
    }
    for name, values in (extras or {}).items():
        data[name] = list(values)[:n]
    return pd.DataFrame(data)


# ----------------------------------------------------------------------------------------------
# emotion (primary benchmark, PROTOCOL.md §1-§2) -- built from models/common.py verbatim
# ----------------------------------------------------------------------------------------------


def _load_emotion(split: str, limit: Optional[int]) -> pd.DataFrame:
    """Identical to the historical ``benchmark.load_split``: ``load_dataset(...).select(range(n))``."""
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID, DATASET_CONFIG, split=split, revision=DATASET_REVISION)
    n = len(ds) if limit is None else min(limit, len(ds))
    rows = ds.select(range(n))
    return _frame(rows["text"], rows["label"], LABELS)


EMOTION = DatasetSpec(
    key="emotion",
    display_name="dair-ai/emotion (test)",
    source_id=DATASET_ID,
    revision=DATASET_REVISION,
    eval_split_name="test",
    smoke_split_name="validation",
    labels=list(LABELS),
    instruction=INSTRUCTION,
    hypothesis_template="The primary emotion expressed in this text is {label}.",
    hypothesis_template_defined="The primary emotion expressed in this text is {label} ({definition}).",
    definitions=dict(DEFINITIONS),
    question_id="emotion",
    loader=lambda limit: _load_emotion("test", limit),
    # PROTOCOL.md §2: smoke tests use the first rows of ``validation`` (``--limit`` picks how many).
    smoke_loader=lambda limit: _load_emotion("validation", limit),
    source_note=f"datasets.load_dataset({DATASET_ID!r}, {DATASET_CONFIG!r}, revision=...)",
    n_eval_rows=2000,
)


# ----------------------------------------------------------------------------------------------
# tweet_topic (cardiffnlp/tweet_topic_single, temporal split, test_2021) -- addendum §1-§2
# ----------------------------------------------------------------------------------------------

TWEET_TOPIC_REPO = "cardiffnlp/tweet_topic_single"
TWEET_TOPIC_REVISION = "87b7a0d1c402dbb481db649569c556d9aa27ac05"
TWEET_TOPIC_EVAL_FILE = "dataset/split_temporal/test_2021.single.json"
TWEET_TOPIC_SMOKE_FILE = "dataset/split_temporal/validation_2021.single.json"
TWEET_TOPIC_RAW_LABELS: Tuple[str, ...] = (
    "arts_&_culture",
    "business_&_entrepreneurs",
    "pop_culture",
    "daily_life",
    "sports_&_gaming",
    "science_&_technology",
)
#: Addendum §2: underscores replaced by spaces, ``&`` kept; order = dataset ids 0..5.
TWEET_TOPIC_LABELS: List[str] = [name.replace("_", " ") for name in TWEET_TOPIC_RAW_LABELS]


def _load_tweet_topic(filename: str, limit: Optional[int]) -> pd.DataFrame:
    """Read the raw JSON-lines file at the pinned revision (the repo's loading script no longer runs
    under ``datasets`` 5). Rows keep their file order; ``id`` and ``date`` are kept as extras."""
    import json

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(TWEET_TOPIC_REPO, filename, repo_type="dataset", revision=TWEET_TOPIC_REVISION)
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    for rec in records:
        expected = TWEET_TOPIC_RAW_LABELS[int(rec["label"])]
        if rec.get("label_name") != expected:
            raise RuntimeError(f"tweet_topic: label id {rec['label']} carries name {rec.get('label_name')!r}, expected {expected!r}")
    return _frame(
        [r["text"] for r in records],
        [int(r["label"]) for r in records],
        TWEET_TOPIC_LABELS,
        extras={"id": [int(r["id"]) for r in records], "date": [str(r["date"]) for r in records]},
        limit=limit,
    )


TWEET_TOPIC = DatasetSpec(
    key="tweet_topic",
    display_name="TweetTopic single-label (test_2021)",
    source_id=TWEET_TOPIC_REPO,
    revision=TWEET_TOPIC_REVISION,
    eval_split_name="test_2021",
    smoke_split_name="validation_2021",
    labels=TWEET_TOPIC_LABELS,
    instruction="Which single topic does this tweet belong to?",
    hypothesis_template="The topic of this tweet is {label}.",
    question_id="topic",
    extra_columns=("id", "date"),
    loader=lambda limit: _load_tweet_topic(TWEET_TOPIC_EVAL_FILE, limit),
    smoke_loader=lambda limit: _load_tweet_topic(TWEET_TOPIC_SMOKE_FILE, SMOKE_N if limit is None else min(limit, SMOKE_N)),
    source_note=f"hf_hub_download raw file {TWEET_TOPIC_EVAL_FILE} (smoke: {TWEET_TOPIC_SMOKE_FILE}, first {SMOKE_N} rows)",
    n_eval_rows=1693,
)


# ----------------------------------------------------------------------------------------------
# fin_topic (zeroshot/twitter-financial-news-topic, validation) -- addendum §1-§2
# ----------------------------------------------------------------------------------------------

FIN_TOPIC_REPO = "zeroshot/twitter-financial-news-topic"
FIN_TOPIC_REVISION = "acbc8af2a35ccf0916124efcbe9e6cf25f191012"
#: Verbatim from the dataset card, ids 0..19.
FIN_TOPIC_LABELS: List[str] = [
    "Analyst Update",
    "Fed | Central Banks",
    "Company | Product News",
    "Treasuries | Corporate Debt",
    "Dividend",
    "Earnings",
    "Energy | Oil",
    "Financials",
    "Currencies",
    "General News | Opinion",
    "Gold | Metals | Materials",
    "IPO",
    "Legal | Regulation",
    "M&A | Investments",
    "Macro",
    "Markets",
    "Politics",
    "Personnel Change",
    "Stock Commentary",
    "Stock Movement",
]


def _load_fin_topic(split: str, limit: Optional[int]) -> pd.DataFrame:
    from datasets import load_dataset

    ds = load_dataset(FIN_TOPIC_REPO, split=split, revision=FIN_TOPIC_REVISION)
    n = len(ds) if limit is None else min(limit, len(ds))
    rows = ds.select(range(n))
    return _frame(rows["text"], rows["label"], FIN_TOPIC_LABELS)


FIN_TOPIC = DatasetSpec(
    key="fin_topic",
    display_name="Twitter financial news topic (validation)",
    source_id=FIN_TOPIC_REPO,
    revision=FIN_TOPIC_REVISION,
    eval_split_name="validation",
    smoke_split_name="train",
    labels=FIN_TOPIC_LABELS,
    instruction="Which single topic does this financial news tweet belong to?",
    hypothesis_template="The topic of this tweet is {label}.",
    question_id="topic",
    loader=lambda limit: _load_fin_topic("validation", limit),
    smoke_loader=lambda limit: _load_fin_topic("train", SMOKE_N if limit is None else min(limit, SMOKE_N)),
    source_note="datasets.load_dataset(revision=...); the dataset has no test split, validation is the "
                "evaluation set and is never used for tuning (smoke: train, first 10 rows)",
    n_eval_rows=4117,
)


# ----------------------------------------------------------------------------------------------
# daily_dialog (OpenRL/daily_dialog parquet mirror, test flattened to utterances) -- addendum §1-§2
# ----------------------------------------------------------------------------------------------

DAILY_DIALOG_REPO = "OpenRL/daily_dialog"
DAILY_DIALOG_REVISION = "1668faf0c0dc44664f108c489fd0666128db2c48"
#: Dataset ClassLabel names, ids 0..6.
DAILY_DIALOG_LABELS: List[str] = ["no emotion", "anger", "disgust", "fear", "happiness", "sadness", "surprise"]


def _load_daily_dialog(split: str, limit: Optional[int]) -> pd.DataFrame:
    """Flatten dialogs to one row per utterance (running index over the split), keeping
    ``dialog_id`` (row in the split) and ``turn_id`` (position in the dialog). Utterances are passed
    verbatim, including the source's trailing space (PROTOCOL.md §3: raw text unmodified)."""
    from datasets import load_dataset

    ds = load_dataset(DAILY_DIALOG_REPO, split=split, revision=DAILY_DIALOG_REVISION)
    names = list(ds.features["emotion"].feature.names)
    if names != DAILY_DIALOG_LABELS:
        raise RuntimeError(f"daily_dialog: ClassLabel names {names} != {DAILY_DIALOG_LABELS}")
    texts: List[str] = []
    gold: List[int] = []
    dialog_ids: List[int] = []
    turn_ids: List[int] = []
    for dialog_id, (utterances, emotions) in enumerate(zip(ds["dialog"], ds["emotion"])):
        if len(utterances) != len(emotions):
            raise RuntimeError(f"daily_dialog: dialog {dialog_id} has {len(utterances)} utterances but {len(emotions)} emotion labels")
        for turn_id, (utt, emo) in enumerate(zip(utterances, emotions)):
            texts.append(str(utt))
            gold.append(int(emo))
            dialog_ids.append(dialog_id)
            turn_ids.append(turn_id)
        if limit is not None and len(texts) >= limit:
            break
    return _frame(texts, gold, DAILY_DIALOG_LABELS,
                  extras={"dialog_id": dialog_ids, "turn_id": turn_ids}, limit=limit)


DAILY_DIALOG = DatasetSpec(
    key="daily_dialog",
    display_name="DailyDialog utterance emotion (test)",
    source_id=DAILY_DIALOG_REPO,
    revision=DAILY_DIALOG_REVISION,
    eval_split_name="test",
    smoke_split_name="validation",
    labels=DAILY_DIALOG_LABELS,
    instruction="Which single emotion is expressed in this utterance?",
    hypothesis_template="The emotion expressed in this utterance is {label}.",
    question_id="emotion",
    extra_columns=("dialog_id", "turn_id"),
    loader=lambda limit: _load_daily_dialog("test", limit),
    smoke_loader=lambda limit: _load_daily_dialog("validation", SMOKE_N if limit is None else min(limit, SMOKE_N)),
    source_note="parquet mirror of li2017dailydialog/daily_dialog; test dialogs flattened to utterances, "
                "each scored without dialogue context; utterance text verbatim, trailing space kept "
                "(smoke: validation, first 10 utterances)",
    n_eval_rows=7740,
)


# ----------------------------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------------------------

REGISTRY: Dict[str, DatasetSpec] = {
    EMOTION.key: EMOTION,
    TWEET_TOPIC.key: TWEET_TOPIC,
    FIN_TOPIC.key: FIN_TOPIC,
    DAILY_DIALOG.key: DAILY_DIALOG,
}

DATASET_KEYS: Tuple[str, ...] = tuple(REGISTRY)


def get_spec(key: str) -> DatasetSpec:
    try:
        return REGISTRY[key]
    except KeyError:
        raise KeyError(f"unknown dataset {key!r}; known: {list(REGISTRY)}") from None


__all__ = [
    "DatasetSpec",
    "REGISTRY",
    "DATASET_KEYS",
    "SMOKE_N",
    "EMOTION",
    "TWEET_TOPIC",
    "FIN_TOPIC",
    "DAILY_DIALOG",
    "get_spec",
]
