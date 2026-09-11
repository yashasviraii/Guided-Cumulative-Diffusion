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
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def find_graph_path(graphs_dir: str, file_id: str) -> Optional[Path]:
    base = Path(graphs_dir)
    for candidate in (base / f"{file_id}.json",
                      base / "test" / f"{file_id}.json",
                      base / "train" / f"{file_id}.json"):
        if candidate.exists():
            return candidate
    return None


def _load_name_vocab() -> Dict[str, int]:
    for p in [Path("checkpoints/gnn_name_vocab.json"),
              Path("src/gcd/gnn/gnn_name_vocab.json")]:
        if p.exists():
            return json.load(open(p))
    return {}


def _load_relation_vocab() -> Dict[str, int]:
    for p in [Path("checkpoints/gnn_relation_vocab.json"),
              Path("src/gcd/gnn/gnn_relation_vocab.json")]:
        if p.exists():
            return json.load(open(p))
    return {}


def _load_class_prior() -> Dict[str, float]:
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


def load_precomputed_graph(graph_path: Path) -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray]:
    """Return (node_features [N, 509], node_names [N], edge_index [2,E], edge_type [E])."""
    with open(graph_path) as f:
        graph_data = json.load(f)

    edges = graph_data.get("edges", [])
    n_nodes = len(graph_data["nodes"])
    degree = [0] * n_nodes
    for e in edges:
        degree[e["source"]] += 1
        degree[e["target"]] += 1

    name_to_idx  = _load_name_vocab()
    rel_to_idx   = _load_relation_vocab()
    class_prior  = _load_class_prior()
    n_classes    = len(name_to_idx)

    def _encode(node, deg):
        feats = [0.0] * n_classes
        name = node.get("name", "")
        if name in name_to_idx:
            feats[name_to_idx[name]] = 1.0
        feats.extend(list(node.get("features", [0.0] * 7)))
        feats.append(min(deg, 10) / 10.0)
        feats.append(class_prior.get(_canonical(name), 0.15))
        return feats

    node_features = np.array(
        [_encode(n, degree[i]) for i, n in enumerate(graph_data["nodes"])],
        dtype=np.float32,
    )
    node_names = graph_data["node_names"]

    if edges:
        edge_index = np.array([[e["source"], e["target"]] for e in edges], dtype=np.int64).T
        edge_type  = np.array([rel_to_idx.get(e["relation"], 0) for e in edges], dtype=np.int64)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_type  = np.zeros((0,), dtype=np.int64)

    return node_features, node_names, edge_index, edge_type