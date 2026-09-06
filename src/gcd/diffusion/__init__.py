"""Inference-only generation methods.

Every class in ``gcd.diffusion.methods`` operates on a frozen, pretrained
diffusion backbone (no training). Available methods:

- ``simple_diffusion.SimpleDiffusion``               Unmodified one-shot baseline.
- ``attend_and_excite.AttendAndExciteDiffusion``      Chefer et al. 2023 baseline.
- ``context_switching.ContextSwitchingDiffusion``     Hard prompt-switching ablation
                                                       (triggers the Crystallization
                                                       Problem, Section 3.5).
- ``attention_modulation.AttentionModulationDiffusion`` Guided Cumulative Diffusion
                                                       (GCD, ours): log-bias cross-
                                                       attention modulation.

All four share the same public ``infer(...)`` signature so
``gcd.diffusion.runner`` can drive any of them interchangeably.
"""
