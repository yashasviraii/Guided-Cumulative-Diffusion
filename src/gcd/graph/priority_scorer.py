"""Normalize raw GCN outputs into a proper [0, 1] priority distribution that
sums to one, so that the diffusion step budget (Section 3.3/3.4 of the
paper) can be partitioned proportionally.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class PriorityScorer:
    """Score entities with a trained GCN and normalize into a step-budget distribution."""

    @staticmethod
    def score_nodes(
        node_features: np.ndarray,
        gnn_model,
        edge_index: Optional[np.ndarray] = None,
        edge_type: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        scores = gnn_model.score_nodes(
            node_features, edge_index=edge_index, edge_type=edge_type
        )
        return np.clip(scores, 0, 1)
    @staticmethod
    def normalize_scores(scores: np.ndarray, min_score: float = 0.01) -> np.ndarray:
        """Enforce a minimum per-object priority, then renormalize to sum to 1."""
        scores = np.maximum(scores, min_score)
        return scores / scores.sum()
