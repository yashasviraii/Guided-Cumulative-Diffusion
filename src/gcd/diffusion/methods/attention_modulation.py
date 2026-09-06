"""Method 4: Attention Modulation — Guided Cumulative Diffusion (GCD, ours).

The full scene prompt is encoded once, from step 0 (avoiding the
Crystallization Problem of Context Switching), while a per-token log-bias
injected into every cross-attention layer (``gcd.diffusion.attention_processor``)
suppresses not-yet-introduced objects and ramps each one in over
``ramp_len`` steps in GCN-determined priority order (Section 3.4-3.5).

``ramp_len`` and ``obj_suppress`` are exposed as constructor / ``infer()``
arguments so the ``ramp_sizes = [0, 5, 10, 15, 20]`` sweep requested for the
experiment matrix can be run without touching this file.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from gcd.diffusion.attention_processor import (
    OBJ_SUPPRESS,
    RAMP_LEN,
    AttentionState,
    GCDAttentionProcessor,
    build_weight_schedule,
    find_token_spans,
)
from gcd.diffusion.backbone import DEFAULT_BASE_MODEL, decode_latents, load_pipeline, swap_long_horizon_scheduler
from gcd.diffusion.prompt_stacking import build_cumulative_prompt, compute_step_allocations
from gcd.graph.sanitize import sanitize_background
from gcd.parsing.description_parser import DescriptionParser

BG_STEPS = 100
INTRO_PHASE_END = 1000


class AttentionModulationDiffusion:
    """Guided Cumulative Diffusion (ours): log-bias cross-attention scheduling."""

    def __init__(self, base_model: str = DEFAULT_BASE_MODEL, device: str = "cuda") -> None:
        self.device = device
        print(f"[AttentionModulation/GCD] Loading base model: {base_model}")
        self.pipeline = load_pipeline(base_model, device)
        swap_long_horizon_scheduler(self.pipeline)

        self._attn_state = AttentionState()
        self._register_processors()

    def _register_processors(self) -> None:
        processor = GCDAttentionProcessor(self._attn_state)
        procs = {key: processor for key in self.pipeline.unet.attn_processors}
        self.pipeline.unet.set_attn_processor(procs)
        print(f"GCD attention processors registered on {len(procs)} layers.")

    @staticmethod
    def sanitize_background(background: str, node_names: List[str], attributes: Dict[str, Dict]) -> str:
        return sanitize_background(background, node_names, attributes)

    def _encode_text(self, text: str) -> torch.Tensor:
        tokenizer = self.pipeline.tokenizer
        ids = tokenizer(
            text, padding="max_length", truncation=True, max_length=tokenizer.model_max_length, return_tensors="pt"
        ).input_ids.to(self.device)
        with torch.no_grad():
            return self.pipeline.text_encoder(ids)[0]

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
        ramp_len: int = RAMP_LEN,
        obj_suppress: float = OBJ_SUPPRESS,
    ) -> Tuple[Image.Image, str]:
        torch.manual_seed(seed)
        np.random.seed(seed)

        step_allocations = compute_step_allocations(node_names, priority_scores, BG_STEPS, INTRO_PHASE_END)

        full_prompt, concept_searches = build_cumulative_prompt(
            background, node_names, attributes, relations, priority_scores, prompt_rewriter
        )
        print(f"[AttentionModulation/GCD] Full prompt: {full_prompt[:120]}...")

        token_spans = find_token_spans(full_prompt, concept_searches, self.pipeline.tokenizer)
        n_tok = self.pipeline.tokenizer.model_max_length
        weight_schedule = build_weight_schedule(
            total_steps=num_inference_steps, bg_steps=BG_STEPS, step_allocations=step_allocations,
            token_spans=token_spans, n_tokens=n_tok, obj_suppress=obj_suppress, ramp_len=ramp_len,
        )

        cond_emb = self._encode_text(full_prompt)
        uncond_emb = self._encode_text("")

        interm_dir = None
        if output_dir is not None:
            interm_dir = Path(output_dir) / "intermediate"
            interm_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_steps = sorted({BG_STEPS, INTRO_PHASE_END, num_inference_steps} | {e for _, e in step_allocations.values()})

        try:
            image = self._denoise(
                height, width, num_inference_steps, guidance_scale, seed,
                cond_emb, uncond_emb, weight_schedule, step_allocations, checkpoint_steps, interm_dir, full_prompt,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Attention denoising failed ({exc!r}); falling back to pipeline() call.")
            image = self.pipeline(
                prompt=full_prompt, height=height, width=width,
                num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
                generator=torch.Generator(device=self.device).manual_seed(seed),
            ).images[0]

        return image, full_prompt

    def _denoise(
        self, height, width, num_inference_steps, guidance_scale, seed,
        cond_emb, uncond_emb, weight_schedule, step_allocations, checkpoint_steps, interm_dir, full_prompt,
    ):
        unet, scheduler = self.pipeline.unet, self.pipeline.scheduler
        scheduler.set_timesteps(num_inference_steps)
        timesteps = scheduler.timesteps
        max_ts = getattr(scheduler.config, "num_train_timesteps", None)
        if max_ts is not None:
            timesteps = torch.clamp(timesteps, max=int(max_ts) - 1)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        latents = torch.randn(
            (1, unet.config.in_channels, height // 8, width // 8),
            generator=generator, device=self.device, dtype=getattr(unet, "dtype", torch.float16),
        )
        init_sigma = getattr(scheduler, "init_noise_sigma", None)
        if init_sigma is not None:
            latents = latents * init_sigma

        for step_index, timestep in enumerate(timesteps):
            current_step = step_index + 1
            self._attn_state.weight_vector = weight_schedule[step_index].to(self.device)

            with torch.no_grad():
                self._attn_state.enabled = True
                noise_cond = unet(latents, timestep, encoder_hidden_states=cond_emb).sample
                self._attn_state.enabled = False
                noise_uncond = unet(latents, timestep, encoder_hidden_states=uncond_emb).sample

            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            step_out = scheduler.step(noise_pred, timestep, latents, return_dict=True)
            latents = step_out.prev_sample

            if interm_dir is not None and current_step in checkpoint_steps:
                self._save_snapshot(
                    interm_dir, current_step, step_index,
                    getattr(step_out, "pred_original_sample", latents), step_allocations, full_prompt,
                )

        return decode_latents(self.pipeline, latents)

    def _save_snapshot(self, interm_dir, step_count, step_index, latents_tensor, step_allocations, full_prompt):
        active = [n for n, (s, e) in step_allocations.items() if s <= step_index < e]
        if step_index < BG_STEPS or not active:
            label = "background"
        elif step_index >= INTRO_PHASE_END:
            label = "harmonise"
        else:
            label = "priority_" + "_".join(active[:3])
        safe_label = re.sub(r"[^A-Za-z0-9_\-]+", "_", label).strip("_") or "step"
        decode_latents(self.pipeline, latents_tensor).save(interm_dir / f"step_{step_count:04d}_{safe_label}.png")
        (interm_dir / f"step_{step_count:04d}_{safe_label}_prompt.txt").write_text(full_prompt)
