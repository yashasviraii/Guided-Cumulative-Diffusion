"""Unified batch-inference CLI, replacing the three near-duplicate
``run_batch_inference*.py`` scripts in the original codebase with one
method-selectable runner.

Two scene-loading modes are supported:

- **Precomputed** (recommended if you already ran the data pipeline):
  pass ``--graphs-dir`` (independent per-image ``gnn_graphs/{file_id}.json``
  files) plus ``--parsed`` (``parsed_images.jsonl``) and either
  ``--background`` (``backgroundContext.jsonl``) or a ``merged.jsonl`` passed
  to ``--parsed`` directly (it already carries ``background_context``). No
  LLM is loaded, objects/relations/background come straight from your
  dataset files, and the GNN scores nodes using the *real* object-relation
  edges saved in the graph JSON — the graph structure it was actually
  trained on.
- **Live** (default if ``--graphs-dir`` is omitted): re-derives objects,
  relations, background, and a hashed (edge-less) feature graph from the raw
  description with a live LLM call, exactly as the original
  ``gnn_inference.py`` / ``gssd_attention_inference.py`` scripts did.

Usage
-----
    # Precomputed mode (reuses your existing dataset files):
    python -m gcd.diffusion.runner \\
        --method attention_modulation \\
        --descriptions data/descriptions.jsonl \\
        --graphs-dir data/gnn_graphs \\
        --parsed data/parsed_images.jsonl \\
        --background data/backgroundContext.jsonl \\
        --test-ids data/splits/test_ids.txt \\
        --gnn-checkpoint checkpoints/gnn.pt \\
        --output-root outputs/attention_modulation

    # Attention-Modulation ramp-size / seed sweep (Section: experiment matrix):
    python -m gcd.diffusion.runner \\
        --method attention_modulation \\
        --descriptions data/descriptions.jsonl \\
        --graphs-dir data/gnn_graphs \\
        --parsed data/parsed_images.jsonl \\
        --background data/backgroundContext.jsonl \\
        --gnn-checkpoint checkpoints/gnn.pt \\
        --output-root outputs/ramp_sweep \\
        --ramp-sizes 0 5 10 15 20 \\
        --seeds 0 1 2

    # Live mode (no precomputed files — parses on the fly with an LLM):
    python -m gcd.diffusion.runner \\
        --method attention_modulation \\
        --descriptions data/descriptions.jsonl \\
        --gnn-checkpoint checkpoints/gnn.pt \\
        --output-root outputs/attention_modulation
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import torch
from gcd.diffusion.backbone import DEFAULT_BASE_MODEL
from gcd.diffusion.methods import METHOD_REGISTRY
from gcd.diffusion.precomputed_scene import load_jsonl_by_file, load_precomputed_scene
from gcd.graph.graph_builder import InferenceGraphBuilder
from gcd.graph.priority_scorer import PriorityScorer
from gcd.graph.sanitize import sanitize_background
from gcd.gnn.models import SimpleGCNInference
from gcd.parsing.description_parser import DEFAULT_LLM_MODEL, DescriptionParser

DEFAULT_SEED = 42


def _load_test_entries(descriptions_path: str, test_ids_path: Optional[str], max_images: int, seed: int) -> List[dict]:
    entries = []
    test_ids = None
    if test_ids_path and Path(test_ids_path).exists():
        test_ids = {line.strip() for line in Path(test_ids_path).read_text().splitlines() if line.strip()}

    with open(descriptions_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            image_id = Path(record.get("file", "")).stem
            if test_ids is None or image_id in test_ids:
                entries.append({
                    "image_id": image_id,
                    "file": record.get("file", ""),
                    "description": record.get("description", ""),
                })

    print(f"Found {len(entries)} candidate descriptions.")
    random.seed(seed)
    return random.sample(entries, min(max_images, len(entries))) if max_images else entries


def _prepare_scene_live(
    parser: DescriptionParser,
    gnn_model: SimpleGCNInference,
    description: str,
) -> Optional[dict]:
    """Fallback path: parse objects/relations/background live with an LLM.

    Features are hashed with no dataset context and no relation edges are
    passed to the GNN (none exist yet at this point in the pipeline) — see
    ``gcd.diffusion.precomputed_scene`` for the preferred, edge-aware path.
    """
    objects = parser.extract_objects(description)
    if not objects:
        return None
    relations = parser.extract_relations(description, objects)
    background = parser.extract_background(description, objects)

    node_features, node_names, edges = InferenceGraphBuilder.build_graph(objects, relations)
    edge_index = None
    if edges:
        import numpy as np

        edge_index = np.array([[s, t] for s, t, _ in edges], dtype=np.int64).T

    raw_scores = PriorityScorer.score_nodes(node_features, gnn_model, edge_index=edge_index)
    priority_scores = PriorityScorer.normalize_scores(raw_scores)

    return {
        "objects": objects,
        "relations": relations,
        "background": sanitize_background(background, node_names, objects),
        "node_names": node_names,
        "priority_scores": priority_scores,
    }


def run_sweep(
    method_name: str,
    entries: List[dict],
    output_root: Path,
    gnn_checkpoint: str,
    llm_model: str,
    device: str,
    num_inference_steps: int,
    ramp_sizes: List[int],
    seeds: List[int],
    graphs_dir: Optional[str] = None,
    parsed_path: Optional[str] = None,
    background_path: Optional[str] = None,
    base_model: str = DEFAULT_BASE_MODEL
) -> None:
    method_cls = METHOD_REGISTRY[method_name]
    precomputed_mode = graphs_dir is not None and parsed_path is not None

    print("\n[Step 1] Loading shared models...")
    gnn_model = SimpleGCNInference(gnn_checkpoint, device=device)
    diffusion = method_cls(base_model=base_model, device=device)
    parser: Optional[DescriptionParser] = None
    parsed_map: Dict[str, dict] = {}
    background_map: Optional[Dict[str, dict]] = None

    if precomputed_mode:
        print(f"Precomputed mode: reading graphs from {graphs_dir}, objects/relations from {parsed_path}")
        parsed_map = load_jsonl_by_file(parsed_path)
        if background_path:
            background_map = load_jsonl_by_file(background_path)
        print("No LLM parser loaded (objects/relations/background come from your dataset files).")
    else:
        print("Live mode: no --graphs-dir/--parsed given, parsing descriptions with an LLM.")
        parser = DescriptionParser(model_name=llm_model, device=device)

    supports_ramp = method_name == "attention_modulation"
    ramp_values = ramp_sizes if supports_ramp else [None]

    for entry in entries:
        img_id, description = entry["image_id"], entry["description"]
        print(f"\n{'=' * 60}\nProcessing image_id={img_id}\nDescription: {description[:150]}...")

        try:
            if precomputed_mode:
                scene = load_precomputed_scene(
                    entry["file"], graphs_dir, parsed_map, background_map, gnn_model
                )
            else:
                scene = _prepare_scene_live(parser, gnn_model, description)
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to prepare scene for {img_id}: {exc}")
            continue
        if scene is None:
            print("No graph/objects found for this image, skipping.")
            continue

        for ramp_len in ramp_values:
            for seed in seeds:
                run_tag = img_id
                extra_kwargs: Dict = {}
                # Only append "_ramp{N}" when actually sweeping more than one ramp size.
                if ramp_len is not None and len(ramp_values) > 1:
                    run_tag += f"_ramp{ramp_len}"
                if ramp_len is not None:
                    extra_kwargs["ramp_len"] = ramp_len
                if len(seeds) > 1:
                    run_tag += f"_seed{seed}"

                output_dir = output_root / run_tag
                if (output_dir / "generated_image.png").exists():
                    print(f"Skipping {run_tag}, output already exists.")
                    continue
                output_dir.mkdir(parents=True, exist_ok=True)

                try:
                    image, final_prompt = diffusion.infer(
                        background=scene["background"],
                        node_names=scene["node_names"],
                        attributes=scene["objects"],
                        relations=scene["relations"],
                        priority_scores=scene["priority_scores"],
                        num_inference_steps=num_inference_steps,
                        seed=seed,
                        output_dir=output_dir,
                        prompt_rewriter=parser,
                        **extra_kwargs,
                    )
                    image.save(output_dir / "generated_image.png")
                    with open(output_dir / "inference_result.json", "w") as f:
                        json.dump({
                            "image_id": img_id,
                            "method": method_name,
                            "seed": seed,
                            "ramp_len": ramp_len,
                            "input_description": description,
                            "extracted_objects": scene["objects"],
                            "extracted_relations": scene["relations"],
                            "cleaned_background": scene["background"],
                            "priority_scores": {
                                n: float(s) for n, s in zip(scene["node_names"], scene["priority_scores"])
                            },
                            "final_prompt": final_prompt,
                        }, f, indent=2)
                except Exception as exc:  # noqa: BLE001
                    print(f"Failed on {run_tag}: {exc}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified batch inference runner for all four methods")
    parser.add_argument("--method", required=True, choices=list(METHOD_REGISTRY.keys()))
    parser.add_argument("--descriptions", required=True)
    parser.add_argument("--test-ids", default=None, help="Optional text file restricting to a test split")
    parser.add_argument(
        "--max-images", type=int, default=50,
        help="Random sample size from --descriptions (after any --test-ids filter). "
             "Pass 0 to run on every matching image with no sampling.",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--gnn-checkpoint", required=True)
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL, help="Only used in live mode")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-inference-steps", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Selection seed and default run seed")
    parser.add_argument(
        "--ramp-sizes", type=int, nargs="+", default=[15],
        help="Attention-Modulation ramp lengths to sweep, e.g. --ramp-sizes 0 5 10 15 20",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="Diffusion seeds to sweep per image (defaults to a single run at --seed)",
    )
    parser.add_argument(
        "--graphs-dir", default=None,
        help="Precomputed mode: directory of independent {file_id}.json graph files",
    )
    parser.add_argument(
        "--parsed", default=None,
        help="Precomputed mode: parsed_images.jsonl (or merged.jsonl, which already has background_context)",
    )
    parser.add_argument(
        "--background", default=None,
        help="Precomputed mode: backgroundContext.jsonl (omit if --parsed is already merged.jsonl)",
    )
    parser.add_argument(
        "--base-model", default=DEFAULT_BASE_MODEL,
        help="Diffusion backbone to use (overrides method default)"
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    entries = _load_test_entries(args.descriptions, args.test_ids, args.max_images, args.seed)
    seeds = args.seeds or [args.seed]

    run_sweep(
        method_name=args.method,
        entries=entries,
        output_root=Path(args.output_root),
        gnn_checkpoint=args.gnn_checkpoint,
        llm_model=args.llm_model,
        device=args.device,
        num_inference_steps=args.num_inference_steps,
        ramp_sizes=args.ramp_sizes,
        seeds=seeds,
        graphs_dir=args.graphs_dir,
        parsed_path=args.parsed,
        background_path=args.background,
        base_model=args.base_model
    )


if __name__ == "__main__":
    main()

