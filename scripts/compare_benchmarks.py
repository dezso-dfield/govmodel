"""Vergelijk twee of meer benchmark.json bestanden side-by-side.

Voorbeeld:
    python scripts/compare_benchmarks.py \\
        models/awb-classifier-v0.1/benchmark.json \\
        models/awb-classifier-v0.2/benchmark.json

Geeft per eval-set een tabel: per-label F1 voor elk model + delta.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path


LABEL_ORDER = [
    "aanvraag", "bezwaar", "beroep", "klacht", "melding",
    "zienswijze", "woo_verzoek", "informatieverzoek_3_11", "vraag_overig",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("benchmarks", nargs="+", type=Path,
                        help="Paden naar benchmark.json bestanden om te vergelijken")
    parser.add_argument("--metric", default="argmax_macro_f1",
                        choices=["argmax_macro_f1", "macro_f1", "argmax_micro_f1", "micro_f1",
                                 "prob_gap", "mean_prob_true_class"],
                        help="Welke top-level metric vergelijken (default: argmax_macro_f1)")
    parser.add_argument("--show-per-label", action="store_true",
                        help="Toon ook per-label F1-tabel")
    return parser.parse_args()


def model_label(path: Path) -> str:
    """Korte naam: ouderdir van benchmark.json."""
    return path.parent.name


def main() -> int:
    args = parse_args()
    benchmarks = OrderedDict()
    for p in args.benchmarks:
        data = json.loads(p.read_text(encoding="utf-8"))
        benchmarks[model_label(p)] = data

    # Verzamel alle eval-sets
    all_evals = OrderedDict()
    for name, data in benchmarks.items():
        for eval_path in data.get("evaluations", {}):
            all_evals.setdefault(eval_path, None)

    print()
    print("=" * 90)
    print(f"{'COMPARISON  metric =':>25} {args.metric}")
    print("=" * 90)

    # Top-level vergelijking per eval-set
    header = f"{'EVAL SET':<35} " + " ".join(f"{name:>15}" for name in benchmarks)
    if len(benchmarks) >= 2:
        header += f"{'  Δ (last vs first)':>22}"
    print(header)
    print("-" * len(header))

    rows_data: list[tuple[str, dict[str, float]]] = []
    for eval_path in all_evals:
        eval_name = Path(eval_path).name
        line = f"{eval_name:<35}"
        scores: dict[str, float] = {}
        for name, data in benchmarks.items():
            res = data.get("evaluations", {}).get(eval_path)
            if res is None:
                line += f" {'—':>15}"
                continue
            score = res.get("default_threshold_05", {}).get(args.metric)
            if score is None:
                line += f" {'?':>15}"
                continue
            scores[name] = float(score)
            line += f" {score:>15.4f}"
        if len(scores) >= 2:
            names = list(scores)
            delta = scores[names[-1]] - scores[names[0]]
            arrow = "↑" if delta > 0.001 else ("↓" if delta < -0.001 else "·")
            line += f"   {arrow} {delta:+7.4f}"
        print(line)
        rows_data.append((eval_name, scores))

    print()

    # Per-label F1-tabel (optioneel)
    if args.show_per_label:
        for eval_path in all_evals:
            eval_name = Path(eval_path).name
            print(f"\n--- Per-label argmax F1: {eval_name} ---")
            header = f"{'LABEL':<28} " + " ".join(f"{name:>15}" for name in benchmarks)
            print(header)
            print("-" * len(header))
            for label in LABEL_ORDER:
                line = f"{label:<28}"
                scores: dict[str, float] = {}
                for name, data in benchmarks.items():
                    res = data.get("evaluations", {}).get(eval_path)
                    if res is None:
                        line += f" {'—':>15}"
                        continue
                    val = res.get("default_threshold_05", {}).get(f"f1_{label}_argmax")
                    if val is None:
                        line += f" {'?':>15}"
                        continue
                    scores[name] = float(val)
                    line += f" {val:>15.4f}"
                if len(scores) >= 2:
                    names = list(scores)
                    delta = scores[names[-1]] - scores[names[0]]
                    arrow = "↑" if delta > 0.001 else ("↓" if delta < -0.001 else "·")
                    line += f"   {arrow} {delta:+7.4f}"
                print(line)

    # Overall macro-gemiddelde over alle eval-sets per model
    print()
    print("-" * 90)
    print("MEAN OVER ALL EVAL SETS")
    print("-" * 90)
    line = f"{'avg ' + args.metric:<35}"
    overall_means: dict[str, float] = {}
    for name in benchmarks:
        vals = [scores.get(name) for _, scores in rows_data if name in scores]
        if vals:
            mean = sum(vals) / len(vals)
            overall_means[name] = mean
            line += f" {mean:>15.4f}"
        else:
            line += f" {'—':>15}"
    if len(overall_means) >= 2:
        names = list(overall_means)
        delta = overall_means[names[-1]] - overall_means[names[0]]
        arrow = "↑" if delta > 0.001 else ("↓" if delta < -0.001 else "·")
        line += f"   {arrow} {delta:+7.4f}"
    print(line)
    print("=" * 90)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
