"""Turn (objects, relations) extracted by ``DescriptionParser`` into the
7-dimensional node-feature graph consumed by the GCN priority predictor at
inference time.

This is deliberately a *different, lighter* builder than
``gcd.data.graph_builder.SceneGraphBuilder``, which builds richer,
vocabulary-fitted training graphs from the full dataset-construction
pipeline (Section 3.2 of the paper). At inference time we only have a single
image's worth of objects, so features are hashed rather than vocabulary
indexed — this mirrors the ``gnn_inference.py`` / ``gssd_attention_inference.py``
behaviour exactly so that scores match a model trained with
``gcd.gnn.models.SimpleMLP``.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

ATTRIBUTE_KEYS = ["type", "color", "material", "size", "style", "shape", "texture"]
NODE_FEATURE_DIM = len(ATTRIBUTE_KEYS)  # last slot is overwritten below with a count feature


class InferenceGraphBuilder:
    """Build a single-image entity graph from parsed objects and relations."""

    @staticmethod
    def encode_attributes(attributes: Dict[str, str]) -> np.ndarray:
        """Hash-encode an object's attribute dict into a fixed-size feature vector."""
        features = np.zeros(NODE_FEATURE_DIM, dtype=np.float32)
        for i, key in enumerate(ATTRIBUTE_KEYS):
            if key in attributes and attributes[key]:
                features[i] = hash(str(attributes[key])) % 100 / 100.0
        features[-1] = len([v for v in attributes.values() if v]) / 10.0
        return features

    @staticmethod
    def build_graph(
        objects: Dict[str, Dict[str, str]],
        relations: Dict[str, Dict[str, str]],
    ) -> Tuple[np.ndarray, List[str], List[Tuple[int, int, str]]]:
        """Return (node_features [N, 7], node_names [N], edges [(src, tgt, relation)])."""
        node_names = list(objects.keys())
        node_id_map = {name: i for i, name in enumerate(node_names)}

        node_features = np.array(
            [
                InferenceGraphBuilder.encode_attributes(
                    objects[name] if isinstance(objects[name], dict) else {}
                )
                for name in node_names
            ]
        )

        edges: List[Tuple[int, int, str]] = []
        for src_obj, rel_dict in relations.items():
            if src_obj not in node_id_map or not isinstance(rel_dict, dict):
                continue
            src_id = node_id_map[src_obj]
            for tgt_obj, rel_type in rel_dict.items():
                if tgt_obj in node_id_map:
                    edges.append((src_id, node_id_map[tgt_obj], str(rel_type)))

        if not edges:
            # Fall back to a fully-connected co-occurrence graph so the GCN
            # always has message-passing structure to work with.
            edges = [
                (i, j, "co_occurs")
                for i in range(len(node_names))
                for j in range(i + 1, len(node_names))
            ]

        return node_features, node_names, edges
