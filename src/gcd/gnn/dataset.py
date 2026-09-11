"""PyTorch Geometric ``Dataset`` wrapping the per-image graph JSON files
produced by ``gcd.data.graph_builder.SceneGraphBuilder``.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import torch
from torch_geometric.data import Data, Dataset


# Load clean class prior if available; else empty
_PRIOR_PATH = Path("checkpoints/class_prior_clean.json")
if _PRIOR_PATH.exists():
    CLASS_PRIOR = json.load(open(_PRIOR_PATH))
    print(f"Loaded class prior with {len(CLASS_PRIOR)} classes")
else:
    CLASS_PRIOR = {}
    print("WARNING: no class_prior_clean.json, using 0.15 for all classes")

def _canonical(name: str) -> str:
    import re
    if name.startswith("##"):
        return ""
    name = re.sub(r"\s*\d+\s*$", "", name)
    return re.sub(r"(?<!^)(?=[A-Z])", " ", name).lower().strip()

class GraphDataset(Dataset):
    """Loads {graph_dir}/{split}/*.json graphs into PyG Data objects.

    Node features (dims):
        [n_classes one-hot] + [7 attribute slots] + [degree] = n_classes + 8
    """

    MAX_CLASSES = 500

    def __init__(self, graph_dir: str, split: str = "train", transform=None) -> None:
        super().__init__()
        self.graph_dir = Path(graph_dir) / split
        self.root_dir = Path(graph_dir)
        self.split = split
        self.graph_files = sorted(self.graph_dir.glob("*.json"))
        self.transform = transform
        print(f"Found {len(self.graph_files)} graphs in {split} set")
        self._build_name_vocab()
        self._build_relation_vocab()   # <-- MAKE SURE THIS LINE EXISTS

    def _build_name_vocab(self) -> None:
        """Scan BOTH splits so train/test use identical vocab."""
        counter = Counter()
        for split in ["train", "test"]:
            split_dir = self.root_dir / split
            if not split_dir.exists():
                continue
            for gf in split_dir.glob("*.json"):
                try:
                    g = json.load(open(gf))
                    for n in g["nodes"]:
                        counter[n["name"]] += 1
                except Exception:
                    continue
        self.name_to_idx = {name: i for i, (name, _) in enumerate(counter.most_common(self.MAX_CLASSES))}
        self.n_classes = len(self.name_to_idx)
        print(f"Shared name vocab: {self.n_classes} classes")

    def _encode_node(self, node, degree):
        """[class one-hot (500) | 7 attrs | degree | class_prior]."""
        feats = [0.0] * self.n_classes
        name = node.get("name", "")
        if name in self.name_to_idx:
            feats[self.name_to_idx[name]] = 1.0
        feats.extend(list(node.get("features", [0.0] * 7)))
        feats.append(min(degree, 10) / 10.0)
        # NEW: class prior lookup
        canonical = _canonical(name)
        feats.append(CLASS_PRIOR.get(canonical, 0.15))
        return feats

    def len(self) -> int:
        return len(self.graph_files)

    def get(self, idx: int) -> Data:
        with open(self.graph_files[idx]) as f:
            graph_data = json.load(f)

        edges = graph_data.get("edges", [])
        n_nodes = len(graph_data["nodes"])
        degree = [0] * n_nodes
        for e in edges:
            degree[e["source"]] += 1
            degree[e["target"]] += 1

        x = torch.tensor(
            [self._encode_node(n, degree[i]) for i, n in enumerate(graph_data["nodes"])],
            dtype=torch.float32,
        )
        # Old
        # if edges:
        #     edge_index = torch.tensor(
        #         [[e["source"], e["target"]] for e in edges], dtype=torch.long
        #     ).T.contiguous()
        #     relation_types = sorted({e["relation"] for e in edges})
        #     relation_to_id = {r: i for i, r in enumerate(relation_types)}
        #     edge_attr = torch.tensor(
        #         [[relation_to_id[e["relation"]]] for e in edges], dtype=torch.float32
        #     )
        # else:
        #     edge_index = torch.zeros((2, 0), dtype=torch.long)
        #     edge_attr = torch.zeros((0, 1), dtype=torch.float32)

        if edges:
            edge_index = torch.tensor(
                [[e["source"], e["target"]] for e in edges], dtype=torch.long
            ).T.contiguous()
            # Use GLOBAL relation vocabulary (not per-graph)
            edge_attr = torch.tensor(
                [[self.relation_to_idx[e["relation"]]] for e in edges], dtype=torch.long
            )
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 1), dtype=torch.long)

        y = torch.tensor(graph_data["targets"], dtype=torch.float32)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            num_nodes=n_nodes,
            file_id=graph_data["file_id"],
            node_names=graph_data["node_names"],
        )
        return self.transform(data) if self.transform else data

    def _build_relation_vocab(self) -> None:
        """Global relation vocabulary across all train + test graphs."""
        rels = set()
        for split in ["train", "test"]:
            split_dir = self.root_dir / split
            if not split_dir.exists():
                continue
            for gf in split_dir.glob("*.json"):
                try:
                    g = json.load(open(gf))
                    for e in g.get("edges", []):
                        rels.add(e["relation"])
                except Exception:
                    continue
        self.relation_to_idx = {r: i for i, r in enumerate(sorted(rels))}
        self.num_relations = len(self.relation_to_idx)
        print(f"Global relation vocab: {self.num_relations} relations -> {sorted(rels)}")