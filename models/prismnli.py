"""PrismNLI-0.4B zero-shot NLI backend (PROTOCOL.md §3.3).

Each example is scored with one (premise, hypothesis) pair per label of the dataset, all in a
single forward pass (batch = number of labels: 6 for the primary benchmark, 20 for ``fin_topic``).
The primary categorical distribution follows the HF ``zero-shot-classification`` pipeline with
``multi_label=False``: a softmax over the per-label *entailment* logits. Two secondary conversions
are stored in ``Prediction.extra`` to quantify the effect of that choice.

The dataset is a ``datasets_registry.DatasetSpec`` passed at construction (default: the emotion
spec); hypotheses are ``spec.hypothesis_template.format(label=label)`` verbatim.

Run ``python -m models.prismnli --smoke`` from the project root for the 8-row validation smoke
test plus the pipeline equivalence check required by the protocol.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from datasets_registry import EMOTION, DatasetSpec
from models.common import (
    DATASET_CONFIG,
    DATASET_ID,
    DATASET_REVISION,
    LABELS,
    PRISMNLI_REPO,
    PRISMNLI_REVISION,
    SEED,
    VARIANTS,
    LoadInfo,
    Prediction,
    renormalise,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
PIPELINE_CHECK_PATH = RESULTS_DIR / "prismnli_pipeline_check.json"

# Primary-benchmark templates (PROTOCOL.md §3.3 / §4); other datasets carry their own in the spec.
HYPOTHESIS_TEMPLATE_PLAIN = EMOTION.hypothesis_template
HYPOTHESIS_TEMPLATE_DEFINED = EMOTION.hypothesis_template_defined
# The HF pipeline formats its template positionally (``template.format(label)``).
PIPELINE_TEMPLATE = HYPOTHESIS_TEMPLATE_PLAIN.replace("{label}", "{}")
MAX_LENGTH = 512
PIPELINE_TOLERANCE = 1e-4
SMOKE_ROWS = 8


def pick_device() -> str:
    """Return ``mps`` on Apple silicon, else ``cuda``, else ``cpu`` (PROTOCOL.md §3.2/§6)."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _synchronize(device: str) -> None:
    """Block until all queued kernels on ``device`` have finished (for honest latency timing)."""
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def hypothesis_for(label: str, variant: str, spec: DatasetSpec = EMOTION) -> str:
    """Render the frozen hypothesis for one label under ``variant`` ('plain' or 'defined').

    ``spec.hypothesis_template.format(label=label)`` verbatim; ``defined`` raises ``ValueError`` for
    datasets without definitions.
    """
    return spec.hypothesis(label, variant)


def pipeline_template(spec: DatasetSpec = EMOTION) -> str:
    """The spec's plain template in the HF pipeline's positional form."""
    return spec.hypothesis_template.replace("{label}", "{}")


def pipeline_candidate_label(label: str, variant: str, spec: DatasetSpec = EMOTION) -> str:
    """The string handed to the HF pipeline as a candidate label so that
    ``pipeline_template(spec).format(candidate)`` reproduces :func:`hypothesis_for`."""
    spec.check_variant(variant)
    if variant == "plain":
        return label
    assert spec.definitions is not None
    return f"{label} ({spec.definitions[label]})"


class PrismNLIBackend:
    """Backend adapter for ``Jaehun/PrismNLI-0.4B`` (see :class:`models.common.Backend`)."""

    name: str = "prismnli"

    def __init__(self, device: Optional[str] = None, spec: DatasetSpec = EMOTION) -> None:
        self.spec = spec
        self.device: str = device or pick_device()
        self.dtype: torch.dtype = torch.float32
        self.revision: str = PRISMNLI_REVISION
        self.local_path: Optional[str] = None
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.entailment_id: Optional[int] = None

    # ------------------------------------------------------------------ loading
    def load(self) -> LoadInfo:
        """Download (pinned revision) and load model + tokenizer in fp32 on ``self.device``."""
        t0 = time.perf_counter()
        torch.manual_seed(SEED)
        self.local_path = snapshot_download(PRISMNLI_REPO, revision=self.revision)
        self.tokenizer = AutoTokenizer.from_pretrained(self.local_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.local_path, dtype=self.dtype
        )
        self.model.to(self.device)
        self.model.eval()

        label2id: Dict[str, int] = {k.lower(): int(v) for k, v in self.model.config.label2id.items()}
        entail = [v for k, v in label2id.items() if k.startswith("entail")]
        if len(entail) != 1:
            raise RuntimeError(f"could not identify a unique entailment label in {label2id}")
        self.entailment_id = entail[0]
        _synchronize(self.device)
        load_time_s = time.perf_counter() - t0

        param_count = int(sum(p.numel() for p in self.model.parameters()))
        weights = Path(self.local_path) / "model.safetensors"
        weight_bytes = int(os.path.getsize(weights)) if weights.exists() else None
        return LoadInfo(
            name=self.name,
            load_time_s=load_time_s,
            device=self.device,
            dtype=str(self.dtype).replace("torch.", ""),
            revision=self.revision,
            param_count=param_count,
            weight_bytes=weight_bytes,
            extra={
                "repo": PRISMNLI_REPO,
                "local_path": self.local_path,
                "architecture": list(self.model.config.architectures or []),
                "label2id": dict(self.model.config.label2id),
                "entailment_id": self.entailment_id,
                "checkpoint_torch_dtype": json.loads(
                    (Path(self.local_path) / "config.json").read_text()
                ).get("torch_dtype"),
                "max_length": MAX_LENGTH,
                "batch_size": 1,
                "dataset": self.spec.key,
                "n_hypotheses_per_forward": self.spec.n_classes,
                "hypothesis_template": self.spec.hypothesis_template,
            },
        )

    def _require_loaded(self) -> None:
        if self.model is None or self.tokenizer is None or self.entailment_id is None:
            raise RuntimeError("PrismNLIBackend.load() must be called before inference")

    # ---------------------------------------------------------------- inference
    @torch.inference_mode()
    def _forward(self, text: str, variant: str) -> torch.Tensor:
        """Tokenise one pair per label for ``text`` and return raw logits of shape (n_labels, num_labels)."""
        hypotheses = self.spec.hypotheses(variant)
        enc = self.tokenizer(
            [text] * len(hypotheses),
            hypotheses,
            truncation="only_first",
            max_length=MAX_LENGTH,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        return self.model(**enc).logits.float()

    def warmup(self, texts: List[str], variant: str = "plain") -> None:
        """Run untimed forward passes so the timed loop sees compiled kernels and warm caches."""
        self._require_loaded()
        for text in texts:
            self._forward(text, variant)
        _synchronize(self.device)

    def predict_one(self, dataset_index: int, text: str, variant: str = "plain") -> Prediction:
        """Score one example. ``latency_ms`` covers tokenisation + forward pass only."""
        self._require_loaded()
        try:
            _synchronize(self.device)
            t0 = time.perf_counter()
            logits = self._forward(text, variant)
            _synchronize(self.device)
            latency_ms = (time.perf_counter() - t0) * 1000.0
        except Exception as exc:  # noqa: BLE001 - surfaced as Prediction.error per protocol
            return Prediction(
                dataset_index=dataset_index,
                probs=[float("nan")] * self.spec.n_classes,
                pred=-1,
                confidence=float("nan"),
                latency_ms=float("nan"),
                error=f"{type(exc).__name__}: {exc}",
            )

        logits = logits.detach().cpu()
        entail_logits = logits[:, self.entailment_id]
        probs = torch.softmax(entail_logits, dim=0).tolist()
        indep = torch.softmax(logits, dim=1)[:, self.entailment_id].tolist()
        pred = int(max(range(len(probs)), key=probs.__getitem__))
        return Prediction(
            dataset_index=dataset_index,
            probs=[float(p) for p in probs],
            pred=pred,
            confidence=float(max(probs)),
            latency_ms=latency_ms,
            extra={
                "entail_logits": [float(x) for x in entail_logits.tolist()],
                "indep_probs": [float(x) for x in indep],
                "indep_probs_l1": renormalise([float(x) for x in indep], fallback_index=pred),
                "variant": variant,
            },
        )

    def predict_many(self, items: List[tuple], variant: str = "plain") -> List[Prediction]:
        """Sequential batch-size-1 scoring of ``(dataset_index, text)`` pairs."""
        return [self.predict_one(int(i), str(t), variant) for i, t in items]

    def peak_memory_bytes(self) -> Optional[int]:
        """Driver-allocated bytes on MPS, peak allocated on CUDA, ``None`` on CPU."""
        if self.device == "mps":
            return int(torch.mps.driver_allocated_memory())
        if self.device == "cuda":
            return int(torch.cuda.max_memory_allocated())
        return None

    # ------------------------------------------------------------- verification
    def verify_against_pipeline(
        self,
        rows: Sequence[Tuple[int, str]],
        variants: Sequence[str] = VARIANTS,
        out_path: Path = PIPELINE_CHECK_PATH,
    ) -> Dict[str, Any]:
        """Compare primary probs with ``transformers.pipeline('zero-shot-classification')``.

        The pipeline is given the spec's single template and its candidate labels (for the
        ``defined`` variant the candidate is ``"label (definition)"`` so the rendered
        hypothesis is identical). Pipeline output is ordered by score, so it is aligned back
        to canonical order by label name. Writes the JSON report to ``out_path``.
        """
        from transformers import pipeline as hf_pipeline

        self._require_loaded()
        labels = self.spec.labels
        template = pipeline_template(self.spec)
        device_arg: Any = self.device if self.device != "cpu" else -1
        zs = hf_pipeline(
            "zero-shot-classification",
            model=self.local_path,
            tokenizer=self.local_path,
            device=device_arg,
            dtype=self.dtype,
        )
        per_row: List[Dict[str, Any]] = []
        overall_max = 0.0
        for variant in variants:
            candidates = [pipeline_candidate_label(l, variant, self.spec) for l in labels]
            cand2label = dict(zip(candidates, labels))
            for idx, text in rows:
                ours = self.predict_one(idx, text, variant)
                if ours.error is not None:
                    raise RuntimeError(f"backend failed on row {idx}: {ours.error}")
                out = zs(
                    text,
                    candidate_labels=candidates,
                    hypothesis_template=template,
                    multi_label=False,
                )
                pipe_probs = [0.0] * len(labels)
                for cand, score in zip(out["labels"], out["scores"]):
                    pipe_probs[labels.index(cand2label[cand])] = float(score)
                diffs = [abs(a - b) for a, b in zip(ours.probs, pipe_probs)]
                row_max = max(diffs)
                overall_max = max(overall_max, row_max)
                per_row.append(
                    {
                        "dataset_index": idx,
                        "variant": variant,
                        "ours": ours.probs,
                        "pipeline": pipe_probs,
                        "max_abs_diff": row_max,
                        "argmax_agree": ours.pred == pipe_probs.index(max(pipe_probs)),
                    }
                )
        report = {
            "model": PRISMNLI_REPO,
            "revision": self.revision,
            "device": self.device,
            "dtype": str(self.dtype).replace("torch.", ""),
            "dataset": self.spec.key,
            "hypothesis_template": self.spec.hypothesis_template,
            "pipeline_template": template,
            "variants": list(variants),
            "n_rows": len(rows),
            "tolerance": PIPELINE_TOLERANCE,
            "max_abs_diff": overall_max,
            "passed": bool(overall_max < PIPELINE_TOLERANCE),
            "per_row": per_row,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        return report


# ---------------------------------------------------------------------- smoke CLI
def _load_smoke_rows(n: int = SMOKE_ROWS) -> List[Tuple[int, str, int]]:
    """First ``n`` rows of the *validation* split (never the test split)."""
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID, DATASET_CONFIG, split="validation", revision=DATASET_REVISION)
    return [(i, ds[i]["text"], int(ds[i]["label"])) for i in range(n)]


def _smoke() -> None:
    rows = _load_smoke_rows()
    backend = PrismNLIBackend()
    info = backend.load()
    print("LoadInfo:", json.dumps(info.to_dict(), indent=2))

    texts = [t for _, t, _ in rows]
    backend.warmup(texts)
    print(f"warmup done on {len(texts)} validation rows")

    for variant in VARIANTS:
        print(f"\n=== variant={variant} ===")
        correct = 0
        for idx, text, gold in rows:
            p = backend.predict_one(idx, text, variant)
            correct += int(p.pred == gold)
            print(
                f"[{idx}] gold={LABELS[gold]:8s} pred={LABELS[p.pred]:8s} "
                f"conf={p.confidence:.4f} lat={p.latency_ms:7.1f}ms "
                f"probs={[round(x, 4) for x in p.probs]} "
                f"indep={[round(x, 4) for x in p.extra['indep_probs']]}"
            )
        print(f"smoke accuracy ({variant}): {correct}/{len(rows)}")

    peak = backend.peak_memory_bytes()
    print(f"\npeak_memory_bytes: {peak} ({(peak or 0) / 2**30:.2f} GiB)")

    report = backend.verify_against_pipeline([(i, t) for i, t, _ in rows])
    print(
        f"\npipeline check: max_abs_diff={report['max_abs_diff']:.3e} "
        f"passed={report['passed']} -> {PIPELINE_CHECK_PATH}"
    )
    for r in report["per_row"]:
        print(
            f"  [{r['dataset_index']}] {r['variant']:7s} max_abs_diff={r['max_abs_diff']:.3e} "
            f"argmax_agree={r['argmax_agree']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="PrismNLI backend utilities")
    parser.add_argument("--smoke", action="store_true", help="run the 8-row validation smoke test")
    args = parser.parse_args()
    if args.smoke:
        _smoke()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
