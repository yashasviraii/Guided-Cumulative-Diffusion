"""Compile ``metrics_summary.json`` (produced by ``gcd.evaluation.metrics``)
into a Table-1-style comparison table, optionally split by backbone when
outputs were produced by ``gcd.diffusion.multi_backbone``.

Usage
-----
    # Single-backbone summary (outputs/<method>/...):
    python -m gcd.evaluation.report --summary results/metrics_summary.json

    # Multi-backbone summary, one metrics_summary.json per backbone:
    python -m gcd.evaluation.report \\
        --summary results/sd15_metrics_summary.json=SD-v1.5 \\
        --summary results/dreamshaper8_metrics_summary.json=Dreamshaper-8 \\
        --summary results/sdxl_metrics_summary.json=SDXL
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

METHOD_DISPLAY_NAMES = {
    "attention_modulation": "Attention (Ours)",
    "context_switching": "Context Switching",
    "simple": "Baseline",
    "attend_and_excite": "Attend-and-Excite",
}


def _load_summary(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def format_table(summaries: List[Tuple[str, dict]]) -> str:
    """Render a Markdown table matching the paper's Table 1 layout."""
    lines = ["| Model | Method | CLIP ↑ | LPIPS ↓ | Obj. Acc. ↑ |", "|---|---|---|---|---|"]
    for model_name, summary in summaries:
        # Sort methods by CLIP score (paper convention: best first) if present.
        methods = sorted(
            summary.items(),
            key=lambda kv: kv[1]["summary"].get("avg_clip") or 0.0,
            reverse=True,
        )
        for method_key, data in methods:
            s = data["summary"]
            display_method = METHOD_DISPLAY_NAMES.get(method_key, method_key)
            clip = f"{s['avg_clip']:.4f}" if s.get("avg_clip") is not None else "—"
            lpips = f"{s['avg_lpips']:.4f}" if s.get("avg_lpips") is not None else "—"
            obj_acc = f"{s['overall_object_accuracy']:.4f}" if s.get("overall_object_accuracy") is not None else "—"
            lines.append(f"| {model_name} | {display_method} | {clip} | {lpips} | {obj_acc} |")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compile evaluation summaries into a Table-1-style report")
    parser.add_argument(
        "--summary", action="append", required=True, dest="summaries",
        help="Repeatable. Either a bare path (single-backbone run) or 'path=ModelLabel' for multi-backbone reports.",
    )
    parser.add_argument("--out", default=None, help="Optional path to also write the table as a .md file")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    summaries: List[Tuple[str, dict]] = []
    for entry in args.summaries:
        path, _, label = entry.partition("=")
        summaries.append((label or Path(path).stem, _load_summary(path)))

    table = format_table(summaries)
    print(table)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(table + "\n")
        print(f"\nWrote table to {args.out}")


if __name__ == "__main__":
    main()
