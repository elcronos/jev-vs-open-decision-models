"""Re-render all figures from results/summary.json + results/raw_predictions.parquet without touching any model."""
import json
from pathlib import Path
import pandas as pd
import plots
from benchmark import DISPLAY_NAMES

out_dir = Path("results")
summary = json.load(open(out_dir / "summary.json"))
df = pd.read_parquet(out_dir / "raw_predictions.parquet")
for variant, per_variant in summary["per_variant"].items():
    plot_dir = out_dir / "plots" if variant == "plain" else out_dir / "plots" / variant
    summaries = {DISPLAY_NAMES[n]: e["metrics"] for n, e in per_variant["models"].items() if e["metrics"]}
    prefixes = {DISPLAY_NAMES[n]: n for n in per_variant["models"]}
    written = plots.make_all_plots(df[df["variant"] == variant], summaries, None, plot_dir, model_prefixes=prefixes, variant=variant)
    print(variant, len(written), "files ->", plot_dir)
