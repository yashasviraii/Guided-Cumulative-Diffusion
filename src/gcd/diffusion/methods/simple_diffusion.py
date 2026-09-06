"""Method 1: Simple Diffusion — the unmodified one-shot baseline.

Generates the entire image in a single unguided pass, no phase structure and
no attention modulation, but conditioned on the *same* cumulative prompt the
other three methods use, so the comparison isolates each method's inference
mechanism rather than differences in prompt content.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from gcd.diffusion.backbone import DEFAULT_BASE_MODEL, load_pipeline, swap_long_horizon_scheduler
from gcd.diffusion.prompt_stacking import build_cumulative_prompt
from gcd.parsing.description_parser import DescriptionParser


class SimpleDiffusion:
    """Unmodified diffusion inference: single forward denoising pass, no phases."""

    def __init__(self, base_model: str = DEFAULT_BASE_MODEL, device: str = "cuda") -> None:
        self.device = device
        print(f"[SimpleDiffusion] Loading base model: {base_model}")
        self.pipeline = load_pipeline(base_model, device)
        swap_long_horizon_scheduler(self.pipeline)

    def infer(
        self,
        background: str,
        node_names: List[str],
        attributes: Dict[str, Dict[str, str]],
        relations: Dict[str, Dict[str, str]],
        priority_scores: np.ndarray,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 1200,
        guidance_scale: float = 7.5,
        seed: int = 42,
        output_dir: Optional[str] = None,
        prompt_rewriter: Optional[DescriptionParser] = None,
    ) -> Tuple[Image.Image, str]:
        final_prompt, _ = build_cumulative_prompt(
            background, node_names, attributes, relations, priority_scores, prompt_rewriter
        )
        print(f"[SimpleDiffusion] Prompt: {final_prompt[:120]}...")

        num_steps = self._safe_step_count(num_inference_steps)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        with torch.no_grad():
            image = self.pipeline(
                prompt=final_prompt,
                height=height,
                width=width,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images[0]

        if output_dir is not None:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        return image, final_prompt

    def _safe_step_count(self, requested_steps: int) -> int:
        max_ts = getattr(self.pipeline.scheduler.config, "num_train_timesteps", 1000)
        return requested_steps if max_ts >= requested_steps else max_ts - 1
