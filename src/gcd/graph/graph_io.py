"""Load a single precomputed graph JSON produced by
``gcd.data.graph_builder.SceneGraphBuilder`` for reuse at inference time.

In the original codebase (and in your exported dataset), ``gnn_graphs/`` is a
flat directory of independent per-image files —
``gnn_graphs/{file_id}.json`` — not nested under ``train/``/``test/``
subfolders. ``SceneGraphBuilder.save_graphs`` (used for GNN *training*) adds
the split subfolder; this loader instead checks, in order:

1. ``{graphs_dir}/{file_id}.json``            (flat — matches your test set)
2. ``{graphs_dir}/test/{file_id}.json``        (nested — matches a training run's split)
3. ``{graphs_dir}/train/{file_id}.json``

so the same directory works whichever way it was produced.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


def find_graph_path(graphs_dir: str, file_id: str) -> Optional[Path]:
    base = Path(graphs_dir)
    for candidate in (base / f"{file_id}.json", base / "test" / f"{file_id}.json", base / "train" / f"{file_id}.json"):
        if candidate.exists():
            return candidate
    return None


def load_precomputed_graph(graph_path: Path) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Return (node_features [N, 7], node_names [N], edge_index [2, E])."""
    with open(graph_path) as f:
        graph_data = json.load(f)

    node_features = np.array([n["features"] for n in graph_data["nodes"]], dtype=np.float32)
    node_names = graph_data["node_names"]

    edges = graph_data.get("edges", [])
    if edges:
        edge_index = np.array([[e["source"], e["target"]] for e in edges], dtype=np.int64).T
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)

    return node_features, node_names, edge_index
