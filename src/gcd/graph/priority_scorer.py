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
    def score_nodes(node_features: np.ndarray, gnn_model, edge_index: Optional[np.ndarray] = None) -> np.ndarray:
        """Run the GCN/MLP priority model and clip raw scores to [0, 1].

        Pass ``edge_index`` (shape ``[2, E]``, from a precomputed graph JSON
        or an inference-time relation graph) whenever real object-relation
        structure is available — a GCN checkpoint uses it; an MLP checkpoint
        ignores it.
        """
        scores = gnn_model.score_nodes(node_features, edge_index=edge_index)
        return np.clip(scores, 0, 1)

    @staticmethod
    def normalize_scores(scores: np.ndarray, min_score: float = 0.01) -> np.ndarray:
        """Enforce a minimum per-object priority, then renormalize to sum to 1."""
        scores = np.maximum(scores, min_score)
        return scores / scores.sum()
