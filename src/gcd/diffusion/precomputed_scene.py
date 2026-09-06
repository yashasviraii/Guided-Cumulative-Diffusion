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
from pathlib import Path
from typing import Dict, Optional

from gcd.gnn.models import SimpleGCNInference
from gcd.graph.graph_io import find_graph_path, load_precomputed_graph
from gcd.graph.priority_scorer import PriorityScorer
from gcd.graph.sanitize import sanitize_background


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
    node_features, node_names, edge_index = load_precomputed_graph(graph_path)

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

    raw_scores = PriorityScorer.score_nodes(node_features, gnn_model, edge_index=edge_index)
    priority_scores = PriorityScorer.normalize_scores(raw_scores)

    cleaned_background = sanitize_background(background, node_names, objects)

    return {
        "objects": objects,
        "relations": relations,
        "background": cleaned_background,
        "node_names": node_names,
        "priority_scores": priority_scores,
    }
