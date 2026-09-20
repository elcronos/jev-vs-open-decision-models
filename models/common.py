"""Shared constants and the backend interface every model adapter implements.

See PROTOCOL.md (frozen) for the semantics of every field.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Protocol

LABELS: List[str] = ["sadness", "joy", "love", "anger", "fear", "surprise"]
LABEL2ID: Dict[str, int] = {l: i for i, l in enumerate(LABELS)}

INSTRUCTION = "Which single primary emotion is expressed in this text?"

DEFINITIONS: Dict[str, str] = {
    "sadness": "feeling unhappy, down, grief, loss, disappointment or loneliness",
    "joy": "feeling happy, pleased, cheerful, content or excited",
    "love": "feeling affection, warmth, tenderness, caring or romantic attachment",
    "anger": "feeling mad, irritated, annoyed, resentful or hostile",
    "fear": "feeling afraid, scared, anxious, nervous or worried",
    "surprise": "feeling amazed, shocked, startled or caught off guard by something unexpected",
}

VARIANTS = ("plain", "defined")

DATASET_ID = "dair-ai/emotion"
DATASET_CONFIG = "split"
DATASET_REVISION = "cab853a1dbdf4c42c2b3ef2173804746df8825fe"

JEV_MODEL_ID = "typesafe/jev-1.13-20260917"
PRISMNLI_REPO = "Jaehun/PrismNLI-0.4B"
PRISMNLI_REVISION = "02b9102b34d5dce1bf29c4e6903fea33e1a0ddeb"
LAYA_REPO = "convaiinnovations/laya"
LAYA_REVISION = "c5d78730f3493e4fe16d61507ef4b78eef7318cf"

SEED = 0


@dataclass
class Prediction:
    """One model's output for one example. `probs` is the renormalised categorical distribution
    in canonical LABELS order and is what all metrics consume."""

    dataset_index: int
    probs: List[float]
    pred: int                       # argmax of probs (0..5); -1 if error
    confidence: float               # max(probs); nan if error
    latency_ms: float               # wall-clock for the successful call (remote: end-to-end; local: compute)
    error: Optional[str] = None     # final error string if the example could not be scored
    retries: int = 0                # number of failed attempts before success (remote only)
    extra: Dict[str, Any] = field(default_factory=dict)  # model-specific fields (see PROTOCOL.md §7)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LoadInfo:
    name: str
    load_time_s: float
    device: str
    dtype: str
    revision: Optional[str]
    param_count: Optional[int]
    weight_bytes: Optional[int]
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Backend(Protocol):
    """All adapters expose this surface. `name` is one of 'jev', 'prismnli', 'laya'."""

    name: str

    def load(self) -> LoadInfo: ...

    def warmup(self, texts: List[str], variant: str = "plain") -> None: ...

    def predict_one(self, dataset_index: int, text: str, variant: str = "plain") -> Prediction: ...

    def predict_many(self, items: List[tuple], variant: str = "plain") -> List[Prediction]:
        """items: list of (dataset_index, text). Default sequential; Jev overrides with threads."""
        ...

    def peak_memory_bytes(self) -> Optional[int]: ...


def renormalise(p: List[float], fallback_index: Optional[int] = None) -> List[float]:
    s = float(sum(p))
    if s <= 0:
        out = [0.0] * len(p)
        if fallback_index is not None:
            out[fallback_index] = 1.0
        return out
    return [float(x) / s for x in p]
