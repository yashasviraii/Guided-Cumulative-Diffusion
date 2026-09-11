"""Model definitions for the entity-priority predictor (Section 3.3).

Two architectures are supported:

- ``NodeLevelGNN``: the 3-layer Graph Convolutional Network described in the
  paper (Eq. 1), with residual-friendly BatchNorm/ReLU/Dropout blocks and a
  final linear projection to a scalar priority score per node.
- ``SimpleMLP``: a node-wise MLP baseline that ignores graph structure
  entirely (ablation / fast-training option), sharing a Sequential layout
  compatible with the checkpoint format loaded by ``SimpleGCNInference``.

``SimpleGCNInference`` is the deployment-side loader used by every
inference-time diffusion method: it accepts a checkpoint in either format,
auto-detects which architecture it belongs to, and falls back to a randomly
initialized MLP (with a loud warning) rather than crashing, so a missing or
partially-trained checkpoint never blocks a smoke-test run.
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Dropout
from torch_geometric.nn import GCNConv


from torch_geometric.nn import RGCNConv


class NodeLevelGNN(torch.nn.Module):
    """Relation-aware 3-layer GCN (RGCN) predicting per-node priority."""

    def __init__(self, in_channels, hidden_channels=64, num_layers=3, num_relations=10):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_relations = num_relations

        self.convs = torch.nn.ModuleList([RGCNConv(in_channels, hidden_channels, num_relations)])
        self.bns = torch.nn.ModuleList([BatchNorm1d(hidden_channels)])
        for _ in range(num_layers - 2):
            self.convs.append(RGCNConv(hidden_channels, hidden_channels, num_relations))
            self.bns.append(BatchNorm1d(hidden_channels))
        self.convs.append(RGCNConv(hidden_channels, 1, num_relations))

        self.dropouts = torch.nn.ModuleList([Dropout(0.3) for _ in range(num_layers - 1)])

    def forward(self, x, edge_index, edge_type=None, batch=None):
        for i in range(self.num_layers - 1):
            h = x
            x = self.convs[i](x, edge_index, edge_type)
            x = self.bns[i](x)
            x = F.relu(x)
            x = self.dropouts[i](x)
            if h.shape == x.shape:
                x = x + h
        x = self.convs[-1](x, edge_index, edge_type)
        return x.squeeze(-1)


class SimpleMLP(torch.nn.Module):
    """Node-wise MLP baseline with the same Sequential layout as inference's fc_layers."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(in_channels, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor, edge_index=None, batch=None) -> torch.Tensor:
        return self.model(x).squeeze(-1)


class SimpleGCNInference:
    """Deployment-side priority model: loads a checkpoint from either
    ``NodeLevelGNN`` or the plain-MLP format and scores a batch of node
    features with no PyTorch Geometric ``Data`` object required.
    """

    IN_CHANNELS = 7  # must match gcd.graph.graph_builder.NODE_FEATURE_DIM

    def __init__(self, model_path: str, device: str = "cpu") -> None:
        self.device = device
        self.gnn_model: Optional[NodeLevelGNN] = None
        self.fc_layers: Optional[torch.nn.Sequential] = torch.nn.Sequential(
            torch.nn.Linear(self.IN_CHANNELS, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )
        self._load_checkpoint(model_path)

    def _load_checkpoint(self, model_path: str) -> None:
        try:
            ckpt = torch.load(model_path, map_location=self.device)
        except Exception as exc:  # noqa: BLE001 - deliberately broad, see fallback note above
            print(f"Warning: could not read checkpoint at {model_path}: {exc!r}")
            print("Using a randomly initialized priority model.")
            self.fc_layers.to(self.device).eval()
            return

        state_dict = ckpt
        if isinstance(ckpt, dict):
            state_dict = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))

        has_gcn_layers = any(str(k).startswith("convs.") for k in state_dict.keys())
        if has_gcn_layers and self._try_load_gcn(state_dict):
            return
        self._try_load_mlp(state_dict, model_path)

    def _try_load_gcn(self, state_dict: dict) -> bool:
        conv_idxs = {int(m.group(1)) for k in state_dict if (m := re.match(r"convs\.(\d+)\.", k))}
        num_layers = max(conv_idxs) + 1 if conv_idxs else 0

        hidden_channels = 64
        in_channels = self.IN_CHANNELS
        num_relations = 10

        for k, v in state_dict.items():
            # RGCNConv params are like 'convs.0.weight' of shape [num_relations, in, out]
            if re.match(r"convs\.0\.weight$", k) and v.ndim == 3:
                num_relations, in_channels, hidden_channels = v.shape
                break
            # GCNConv params are 'convs.0.lin.weight' of shape [out, in]
            if re.match(r"convs\.0\.(lin\.)?weight$", k) and v.ndim == 2:
                hidden_channels, in_channels = v.shape
                break

        try:
            gnn = NodeLevelGNN(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                num_layers=num_layers,
                num_relations=num_relations,
            )
            gnn.load_state_dict(state_dict, strict=False)
            self.gnn_model = gnn.to(self.device).eval()
            self.fc_layers = None
            print(f"Loaded RGCN priority model (in={in_channels}, hidden={hidden_channels}, "
                f"layers={num_layers}, relations={num_relations})")
            return True
        except Exception as e:
            print(f"RGCN load failed ({e!r}), trying MLP fallback")
            return False

    def _try_load_mlp(self, state_dict: dict, model_path: str) -> None:
        try:
            cleaned = {}
            for k, v in state_dict.items():
                new_k = k
                for prefix in ("fc_layers.", "model.", "module."):
                    if new_k.startswith(prefix):
                        new_k = new_k[len(prefix):]
                cleaned[new_k] = v
            self.fc_layers.load_state_dict(cleaned)
            print(f"Loaded MLP priority model from {model_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: could not match checkpoint format ({exc!r}). Using random init.")
        self.fc_layers.to(self.device).eval()

    def score_nodes(
        self,
        node_features: np.ndarray,
        edge_index: Optional[np.ndarray] = None,
        edge_type: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        x = torch.tensor(node_features, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            if self.gnn_model is not None:
                # --- edge_index ---
                if edge_index is not None and edge_index.size:
                    edges_t = torch.tensor(edge_index, dtype=torch.long, device=self.device)
                else:
                    edges_t = torch.empty((2, 0), dtype=torch.long, device=self.device)

                # --- edge_type (relation IDs) ---
                if edge_type is not None and len(edge_type):
                    et_t = torch.tensor(edge_type, dtype=torch.long, device=self.device)
                else:
                    # Always produce a tensor (RGCN asserts edge_type is not None)
                    n_edges = edges_t.shape[1] if edges_t.numel() else 0
                    et_t = torch.zeros(n_edges, dtype=torch.long, device=self.device)

                scores = self.gnn_model(x, edges_t, edge_type=et_t).cpu().numpy()
            else:
                scores = self.fc_layers(x).cpu().numpy()
        return scores.flatten()
