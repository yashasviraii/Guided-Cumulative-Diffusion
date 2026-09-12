"""Method 2: Attend-and-Excite (Chefer et al., 2023) baseline.

Wraps diffusers' official ``StableDiffusionAttendAndExcitePipeline``, which
performs latent-space gradient updates during generation to maximize the
peak cross-attention activation of each subject token, mitigating
"catastrophic neglect" of objects. This is the most directly comparable
prior attention-control method referenced in Section 2.1 of the paper, and
is included here as the fourth experiment-runner column requested for the
ablation table.

Unlike Context Switching and Attention Modulation, Attend-and-Excite excites
attention toward each object *simultaneously* rather than in a priority
schedule — there is no phase structure and no GCN priority score involved
beyond picking which object nouns to excite.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from gcd.diffusion.backbone import DEFAULT_BASE_MODEL
from gcd.diffusion.prompt_stacking import build_cumulative_prompt
from gcd.parsing.description_parser import DescriptionParser


class AttendAndExciteDiffusion:
    """Attend-and-Excite baseline via ``diffusers.StableDiffusionAttendAndExcitePipeline``."""

    def __init__(self, base_model: str = DEFAULT_BASE_MODEL, device: str = "cuda") -> None:
        from diffusers import StableDiffusionAttendAndExcitePipeline, DDIMScheduler

        self.device = device
        print(f"[AttendAndExcite] Loading base model: {base_model}")
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.pipeline = StableDiffusionAttendAndExcitePipeline.from_pretrained(
            base_model, torch_dtype=dtype, safety_checker=None
        ).to(device)

        # Force DDIM to match the scheduler used by Simple / Context Switching / GCD.
        try:
            config = dict(self.pipeline.scheduler.config)
            config["num_train_timesteps"] = max(
                1000, config.get("num_train_timesteps", 1000)
            )
            self.pipeline.scheduler = DDIMScheduler.from_config(config)
            print(
                f"[AttendAndExcite] Scheduler: DDIM, "
                f"num_train_timesteps={config['num_train_timesteps']}"
            )
        except Exception as exc:
            print(f"[AttendAndExcite] Warning: DDIM swap failed ({exc!r}); keeping default.")
    def _subject_token_indices(self, prompt: str, node_names: List[str]) -> List[int]:
        """Find the token index of each object's first word in ``prompt``."""
        tokenizer = self.pipeline.tokenizer
        token_ids = tokenizer(prompt).input_ids
        tokens = [tokenizer.decode([tid]).strip().lower() for tid in token_ids]

        indices: List[int] = []
        for name in node_names:
            first_word = name.split()[0].lower() if name else ""
            for i, tok in enumerate(tokens):
                if tok and (tok in first_word or first_word.startswith(tok)):
                    indices.append(i)
                    break
        return sorted(set(indices)) or [1]  # never pass an empty index list

    def infer(
        self,
        background: str,
        node_names: List[str],
        attributes: Dict[str, Dict[str, str]],
        relations: Dict[str, Dict[str, str]],
        priority_scores: np.ndarray,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: int = 42,
        output_dir: Optional[str] = None,
        prompt_rewriter: Optional[DescriptionParser] = None,
        max_iter_to_alter: int = 25,
        raw_description: Optional[str] = None,
    ) -> Tuple[Image.Image, str]:
        # Attend-and-Excite operates on a normal short prompt, not a 1200-step
        # priority schedule; we still use the same cumulative-prompt builder
        # so all four methods describe the same scene content.
        if raw_description is not None:
            final_prompt = raw_description
        else:
            final_prompt, _ = build_cumulative_prompt(
                background, node_names, attributes, relations, priority_scores, prompt_rewriter
            )
        print(f"[AttendAndExcite] Prompt: {final_prompt[:120]}...")

        token_indices = self._subject_token_indices(final_prompt, node_names)
        print(f"[AttendAndExcite] Exciting token indices: {token_indices}")

        generator = torch.Generator(device=self.device).manual_seed(seed)
        image = self.pipeline(
            prompt=final_prompt,
            token_indices=token_indices,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            max_iter_to_alter=max_iter_to_alter,
        ).images[0]

        if output_dir is not None:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        return image, final_prompt
