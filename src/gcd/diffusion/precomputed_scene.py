"""Load a test-set scene from your **already-computed** dataset files instead
of re-running the LLM parser live at generation time.

This is the recommended path when you already have, per image:

- an entry in ``parsed_images.jsonl``   ({"file", "objects", "relations"})
- an entry in ``backgroundContext.jsonl`` ({"file", "background_context"})
  — or both folded into a single ``merged.jsonl``
  ({"file", "objects", "relations", "background_context"})
- a graph file ``gnn_graphs/{file_id}.json`` (built by
  ``gcd.data.graph_builder.SceneGraphBuilder`` — the same feature encoding
  and real object-relation edges the GNN was trained on)

Using the precomputed graph's features/edges (rather than re-deriving 7-dim
hashed features with no edges, as the live-LLM path does) means the GNN
checkpoint sees inputs consistent with what it was trained on, and no LLM
needs to be loaded at all unless you explicitly want LLM-based prompt
stacking (see ``prompt_rewriter`` in ``gcd.diffusion.prompt_stacking``).
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from gcd.gnn.models import SimpleGCNInference
from gcd.graph.graph_io import find_graph_path, load_precomputed_graph
from gcd.graph.priority_scorer import PriorityScorer
from gcd.graph.sanitize import sanitize_background



# --- Priority ensemble configuration ---
W_RGCN = 0.6   # ensemble weight for RGCN; 1-W_RGCN goes to class-prior

# --- Debug: track which scenes have had their GNN scores printed ---
_DEBUG_SEEN = set()
DEBUG_GNN = True   # set to False once verified
@lru_cache(maxsize=1)
def _load_class_prior() -> dict:
    for p in [Path("checkpoints/class_prior_clean.json"),
              Path("checkpoints/class_prior.json")]:
        if p.exists():
            return json.load(open(p))
    return {}


def _canonical(name: str) -> str:
    if name.startswith("##"):
        return ""
    name = re.sub(r"\s*\d+\s*$", "", name)
    return re.sub(r"(?<!^)(?=[A-Z])", " ", name).lower().strip()


def _minmax(a: np.ndarray, floor: float = 0.3) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    stretched = (a - lo) / max(hi - lo, 1e-6)
    return floor + (1.0 - floor) * stretched   # range [0.3, 1.0], not [0, 1]


def _ensemble_priority(rgcn_scores: np.ndarray, node_names: List[str]) -> np.ndarray:
    """0.6 * normalized(RGCN) + 0.4 * normalized(class-prior)."""
    prior = _load_class_prior()
    cm = np.array([prior.get(_canonical(n), 0.15) for n in node_names], dtype=np.float32)
    return W_RGCN * _minmax(rgcn_scores) + (1.0 - W_RGCN) * _minmax(cm)



def load_jsonl_by_file(path: str) -> Dict[str, dict]:
    """Load a ``{"file": ..., ...}``-per-line JSONL, keyed by the full ``file`` path."""
    records: Dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            file_path = rec.get("file")
            if file_path:
                records[file_path] = rec
    return records


def load_precomputed_scene(
    file_path: str,
    graphs_dir: str,
    parsed_map: Dict[str, dict],
    background_map: Optional[Dict[str, dict]],
    gnn_model: SimpleGCNInference,
) -> Optional[dict]:
    """Assemble everything a diffusion method's ``infer()`` needs from cached
    dataset files, scoring priority with the GNN using the *real* graph.

    ``background_map`` may be ``None`` if ``parsed_map`` entries already
    carry a ``"background_context"`` field (i.e. you passed ``merged.jsonl``
    for both).
    """
    file_id = Path(file_path).stem

    graph_path = find_graph_path(graphs_dir, file_id)
    if graph_path is None:
        return None
    node_features, node_names, edge_index, edge_type = load_precomputed_graph(graph_path)
    parsed_rec = parsed_map.get(file_path)
    if parsed_rec is None:
        return None
    objects = parsed_rec.get("objects", {})
    relations = parsed_rec.get("relations", {})

    if background_map is not None:
        bg_rec = background_map.get(file_path, {})
        background = bg_rec.get("background_context", "")
    else:
        background = parsed_rec.get("background_context", "")

    rgcn_raw = PriorityScorer.score_nodes(
        node_features,
        gnn_model,
        edge_index=edge_index,
        edge_type=edge_type,
    )
    ensemble_raw = _ensemble_priority(rgcn_raw, node_names)
    priority_scores = PriorityScorer.normalize_scores(ensemble_raw)

    cleaned_background = sanitize_background(background, node_names, objects)

    if DEBUG_GNN and file_id not in _DEBUG_SEEN:
        _DEBUG_SEEN.add(file_id)
        print(f"[GNN] {file_id} | nodes={len(node_names)} | "
            f"names={node_names} | "
            f"ensemble={[round(float(x), 3) for x in ensemble_raw[:6]]} | "
            f"normalized={[round(float(x), 3) for x in priority_scores[:6]]}")
    return {
        "objects": objects,
        "relations": relations,
        "background": cleaned_background,
        "node_names": node_names,
        "priority_scores": priority_scores,
    }
