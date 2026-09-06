"""Method 3: Context Switching — hard prompt-switching ablation (Section 3.5).

Instead of a single, complete conditioning prompt from step 0, the text
prompt is *swapped* at every phase boundary: background-only for the first
``bg_steps``, then progressively grown as each object is introduced. Section
3.5 argues (and Section 4.3 empirically confirms) that this reliably
triggers the "Crystallization Problem": the first ~20% of diffusion steps
determine global layout, so background-only conditioning at that stage
causes the model to commit the canvas to a background-only interpretation
that later object prompts cannot reclaim.

This class is retained deliberately as an ablation, not a recommended
inference method.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from gcd.diffusion.backbone import DEFAULT_BASE_MODEL, decode_latents, load_pipeline, swap_long_horizon_scheduler
from gcd.diffusion.prompt_stacking import compute_step_allocations
from gcd.parsing.description_parser import DescriptionParser

BG_STEPS = 100
INTRO_PHASE_END = 1000


class ContextSwitchingDiffusion:
    """Diffusion inference that swaps the conditioning prompt at each phase boundary."""

    def __init__(self, base_model: str = DEFAULT_BASE_MODEL, device: str = "cuda") -> None:
        self.device = device
        print(f"[ContextSwitching] Loading base model: {base_model}")
        self.pipeline = load_pipeline(base_model, device)
        swap_long_horizon_scheduler(self.pipeline)

    def _build_prompt_schedule(
        self,
        background: str,
        node_names: List[str],
        attributes: Dict[str, Dict[str, str]],
        relations: Dict[str, Dict[str, str]],
        priority_scores: np.ndarray,
        step_allocations: Dict[str, Tuple[int, int]],
        num_inference_steps: int,
        prompt_rewriter: Optional[DescriptionParser],
    ) -> Dict[int, str]:
        base_prompt = background if background else "a scene"
        schedule: Dict[int, str] = {step: base_prompt for step in range(min(BG_STEPS, num_inference_steps))}

        if not step_allocations:
            schedule.update({step: base_prompt for step in range(BG_STEPS, num_inference_steps)})
            return schedule

        ordered = sorted(
            (
                (idx, node_names[idx], *step_allocations[node_names[idx]])
                for idx in np.argsort(priority_scores)[::-1]
                if node_names[idx] in step_allocations
            ),
            key=lambda item: (item[2], -priority_scores[item[0]]),
        )

        prompt, mentioned = base_prompt, []
        for idx, obj_name, start_step, end_step in ordered:
            attrs = attributes.get(obj_name, {}) if isinstance(attributes.get(obj_name, {}), dict) else {}
            rels_to_prev = {
                prev: relations[obj_name][prev]
                for prev in mentioned
                if obj_name in relations and prev in relations.get(obj_name, {})
            }

            if prompt_rewriter is not None:
                try:
                    prompt = prompt_rewriter.rewrite_prompt_stack(
                        previous_prompt=prompt, object_name=obj_name, attributes=attrs,
                        previous_objects=mentioned, relations_to_previous=rels_to_prev, background=base_prompt,
                    )
                except Exception:  # noqa: BLE001
                    prompt = self._deterministic_stack(prompt, obj_name, attrs, rels_to_prev)
            else:
                prompt = self._deterministic_stack(prompt, obj_name, attrs, rels_to_prev)

            for step in range(start_step, min(end_step, num_inference_steps)):
                schedule[step] = prompt
            mentioned.append(obj_name)

        last_prompt = prompt
        for step in range(BG_STEPS, num_inference_steps):
            schedule.setdefault(step, last_prompt)
        return schedule

    @staticmethod
    def _deterministic_stack(previous_prompt: str, obj_name: str, attrs: dict, rels_to_prev: dict) -> str:
        attr_parts = [f"{v} {k}" for k, v in attrs.items() if v and k != "type"]
        obj_desc = f"{obj_name} with {', '.join(attr_parts)}" if attr_parts else obj_name
        prompt = f"{obj_desc}. {previous_prompt}"
        if rels_to_prev:
            rel_text = "; ".join(f"{obj_name} is {rel} {prev}" for prev, rel in rels_to_prev.items())
            prompt += f" Relations: {rel_text}"
        return prompt

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
        torch.manual_seed(seed)
        np.random.seed(seed)

        step_allocations = compute_step_allocations(node_names, priority_scores, BG_STEPS, INTRO_PHASE_END)
        prompt_schedule = self._build_prompt_schedule(
            background, node_names, attributes, relations, priority_scores,
            step_allocations, num_inference_steps, prompt_rewriter,
        )
        final_prompt = prompt_schedule.get(num_inference_steps - 1, background or "a scene")

        interm_dir = None
        if output_dir is not None:
            interm_dir = Path(output_dir) / "intermediate"
            interm_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_steps = sorted({BG_STEPS, num_inference_steps} | {end for _, end in step_allocations.values()})

        try:
            image, final_prompt = self._denoise(
                node_names, height, width, num_inference_steps, guidance_scale, seed,
                prompt_schedule, final_prompt, step_allocations, checkpoint_steps, interm_dir,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Manual denoising failed ({exc!r}); falling back to pipeline() call.")
            image = self.pipeline(
                prompt=final_prompt, height=height, width=width,
                num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
                generator=torch.Generator(device=self.device).manual_seed(seed),
            ).images[0]

        return image, final_prompt

    def _denoise(
        self, node_names, height, width, num_inference_steps, guidance_scale, seed,
        prompt_schedule, final_prompt, step_allocations, checkpoint_steps, interm_dir,
    ):
        unet, scheduler = self.pipeline.unet, self.pipeline.scheduler
        tokenizer, text_encoder = self.pipeline.tokenizer, self.pipeline.text_encoder

        scheduler.set_timesteps(num_inference_steps)
        timesteps = scheduler.timesteps
        max_train_ts = getattr(scheduler.config, "num_train_timesteps", None)
        if max_train_ts is not None:
            timesteps = torch.clamp(timesteps, max=int(max_train_ts) - 1)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        latents = torch.randn(
            (1, unet.config.in_channels, height // 8, width // 8),
            generator=generator, device=self.device, dtype=getattr(unet, "dtype", torch.float32),
        )
        init_sigma = getattr(scheduler, "init_noise_sigma", None)
        if init_sigma is not None:
            latents = latents * init_sigma

        def encode(text: str) -> torch.Tensor:
            ids = tokenizer(
                text, padding="max_length", truncation=True, max_length=tokenizer.model_max_length, return_tensors="pt"
            ).input_ids.to(self.device)
            with torch.no_grad():
                return text_encoder(ids)[0]

        uncond_emb = encode("") if guidance_scale > 1.0 else None
        last_prompt, cond_emb = None, None

        for step_index, timestep in enumerate(timesteps):
            current_step = step_index + 1
            current_prompt = prompt_schedule.get(step_index, final_prompt)
            if current_prompt != last_prompt or cond_emb is None:
                last_prompt, cond_emb = current_prompt, encode(current_prompt)

            with torch.no_grad():
                noise_pred_text = unet(latents, timestep, encoder_hidden_states=cond_emb).sample
                if guidance_scale > 1.0:
                    noise_pred_uncond = unet(latents, timestep, encoder_hidden_states=uncond_emb).sample
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                else:
                    noise_pred = noise_pred_text

            step_out = scheduler.step(noise_pred, timestep, latents, return_dict=True)
            latents = step_out.prev_sample

            if interm_dir is not None and current_step in checkpoint_steps:
                self._save_snapshot(
                    interm_dir, current_step, step_index, current_prompt,
                    getattr(step_out, "pred_original_sample", latents), step_allocations,
                )

        return decode_latents(self.pipeline, latents), final_prompt

    def _save_snapshot(self, interm_dir, step_count, step_index, prompt_text, latents_tensor, step_allocations):
        active = [n for n, (s, e) in step_allocations.items() if s <= step_index < e]
        label = "background" if (step_index < BG_STEPS or not active) else "priority_" + "_".join(active[:3])
        safe_label = re.sub(r"[^A-Za-z0-9_\-]+", "_", label).strip("_") or "step"
        decode_latents(self.pipeline, latents_tensor).save(interm_dir / f"step_{step_count:04d}_{safe_label}.png")
        (interm_dir / f"step_{step_count:04d}_{safe_label}_prompt.txt").write_text(prompt_text)
