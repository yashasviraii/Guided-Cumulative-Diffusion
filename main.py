#!/usr/bin/env python3
"""Top-level CLI for the Guided Cumulative Diffusion (GCD) repository.

Every pipeline stage is a subcommand delegating to the corresponding
``gcd.*`` module's own argparse parser, so ``python main.py <stage> --help``
always shows the true, up-to-date argument list for that stage.

Data pipeline (Section 3.2):
    python main.py data caption          ...
    python main.py data parse-graph      ...
    python main.py data background       ...
    python main.py data merge            ...
    python main.py data extract-prompts  ...
    python main.py data detect           ...
    python main.py data link-entities    ...
    python main.py data build-graphs     ...
    python main.py data explore          ...

GNN training (Section 3.3):
    python main.py gnn train ...

Generation (Section 3.4-3.5, four experiment runners):
    python main.py generate --method {simple,attend_and_excite,context_switching,attention_modulation} ...
    python main.py generate-multi-backbone ...

Evaluation (Section 4.2):
    python main.py evaluate ...
    python main.py report ...
"""

from __future__ import annotations

import sys

SUBCOMMANDS = {
    ("data", "caption"): "gcd.data.vlm_captioning",
    ("data", "parse-graph"): "gcd.data.graph_parsing",
    ("data", "background"): "gcd.data.background_extraction",
    ("data", "merge"): "gcd.data.merge_context",
    ("data", "extract-prompts"): "gcd.data.prompt_extraction",
    ("data", "detect"): "gcd.data.object_detection",
    ("data", "link-entities"): "gcd.data.entity_linking",
    ("data", "build-graphs"): "gcd.data.graph_builder",
    ("data", "explore"): "gcd.data.dataset_explorer",
    ("gnn", "train"): "gcd.gnn.train",
    ("generate",): "gcd.diffusion.runner",
    ("generate-multi-backbone",): "gcd.diffusion.multi_backbone",
    ("evaluate",): "gcd.evaluation.metrics",
    ("report",): "gcd.evaluation.report",
}


def _print_top_level_help() -> None:
    print(__doc__)
    print("Available subcommands:")
    for keys in SUBCOMMANDS:
        print("  python main.py " + " ".join(keys))


def main() -> None:
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        _print_top_level_help()
        return

    # Match the longest subcommand key first (e.g. ("data", "caption") before ("data",)).
    for length in (2, 1):
        candidate = tuple(args[:length])
        if candidate in SUBCOMMANDS:
            module_name = SUBCOMMANDS[candidate]
            remaining_args = args[length:]
            break
    else:
        print(f"Unknown subcommand: {' '.join(args)}\n")
        _print_top_level_help()
        sys.exit(1)

    import importlib

    module = importlib.import_module(module_name)
    sys.argv = [module_name] + remaining_args
    module.main()


if __name__ == "__main__":
    main()
