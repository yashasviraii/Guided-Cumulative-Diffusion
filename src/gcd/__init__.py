"""Guided Cumulative Diffusion (GCD) reference implementation.

This package contains the full research pipeline described in
"Guided Cumulative Diffusion: Graph-Guided Entity Prioritization and
Log-Bias Attention Modulation for Compositional Text-to-Image Synthesis":

- ``gcd.data``       Scene-graph dataset construction (VLM captioning, LLM
                      graph parsing, detection, entity linking, GCN graph
                      building).
- ``gcd.parsing``     Shared inference-time LLM description parser.
- ``gcd.graph``       Lightweight inference-time graph construction and
                      priority-score normalization.
- ``gcd.gnn``         GCN training pipeline and model definitions.
- ``gcd.diffusion``   Inference-only generation methods: Simple Diffusion,
                      Attend-and-Excite, Context Switching, and Attention
                      Modulation (GCD, ours).
- ``gcd.evaluation``  CLIP / LPIPS / Object-Accuracy evaluation suite.

No diffusion-model *training* code is included by design: every generation
method in ``gcd.diffusion`` operates at inference time on a frozen,
pretrained backbone.
"""

__version__ = "0.1.0"
