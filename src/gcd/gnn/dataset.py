"""PyTorch Geometric ``Dataset`` wrapping the per-image graph JSON files
produced by ``gcd.data.graph_builder.SceneGraphBuilder``.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch_geometric.data import Data, Dataset


class GraphDataset(Dataset):
    """Loads ``{graph_dir}/{split}/*.json`` graphs into PyG ``Data`` objects."""

    def __init__(self, graph_dir: str, split: str = "train", transform=None) -> None:
        super().__init__()
        self.graph_dir = Path(graph_dir) / split
        self.split = split
        self.graph_files = sorted(self.graph_dir.glob("*.json"))
        self.transform = transform
        print(f"Found {len(self.graph_files)} graphs in {split} set")

    def len(self) -> int:
        return len(self.graph_files)

    def get(self, idx: int) -> Data:
        with open(self.graph_files[idx]) as f:
            graph_data = json.load(f)

        x = torch.tensor([n["features"] for n in graph_data["nodes"]], dtype=torch.float32)

        edges = graph_data["edges"]
        if edges:
            edge_index = torch.tensor(
                [[e["source"], e["target"]] for e in edges], dtype=torch.long
            ).T.contiguous()
            relation_types = sorted({e["relation"] for e in edges})
            relation_to_id = {r: i for i, r in enumerate(relation_types)}
            edge_attr = torch.tensor(
                [[relation_to_id[e["relation"]]] for e in edges], dtype=torch.float32
            )
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 1), dtype=torch.float32)

        y = torch.tensor(graph_data["targets"], dtype=torch.float32)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            num_nodes=len(graph_data["nodes"]),
            file_id=graph_data["file_id"],
            node_names=graph_data["node_names"],
        )
        return self.transform(data) if self.transform else data
