"""Shared, inference-only backbone loading utilities.

Centralizes the "load a Stable Diffusion pipeline and optionally swap in a
long-horizon scheduler" logic that was previously duplicated across every
`*_inference.py` / `run_batch_inference_*.py` script in the original
codebase.
"""

from __future__ import annotations

import torch
from diffusers import DDIMScheduler, DPMSolverMultistepScheduler, StableDiffusionPipeline

DEFAULT_BASE_MODEL = "Lykon/dreamshaper-8"


def load_pipeline(base_model: str = DEFAULT_BASE_MODEL, device: str = "cuda") -> StableDiffusionPipeline:
    """Load a Stable-Diffusion-family pipeline with the safety checker disabled."""
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    pipeline = StableDiffusionPipeline.from_pretrained(base_model, torch_dtype=dtype, safety_checker=None)
    pipeline = pipeline.to(device)
    try:
        pipeline.enable_xformers_memory_efficient_attention()
        print("xformers enabled for faster inference.")
    except Exception:
        print("xformers not available, skipping.")
    pipeline.enable_attention_slicing()
    return pipeline


def swap_long_horizon_scheduler(pipeline, min_train_timesteps: int | None = None) -> None:
    from diffusers import DDIMScheduler
    try:
        config = dict(pipeline.scheduler.config)
        if min_train_timesteps is not None:
            config["num_train_timesteps"] = max(
                min_train_timesteps, config.get("num_train_timesteps", 1000)
            )
        pipeline.scheduler = DDIMScheduler.from_config(config)
        print(f"Scheduler: DDIM, num_train_timesteps={config['num_train_timesteps']}")
    except Exception as exc:
        print(f"Warning: DDIM swap failed ({exc!r}); keeping default.")

def decode_latents(pipeline: StableDiffusionPipeline, latents: torch.Tensor):
    """Decode VAE latents to a PIL image."""
    import numpy as np
    from PIL import Image

    latents = latents / 0.18215
    with torch.no_grad():
        decoded = pipeline.vae.decode(latents.to(dtype=pipeline.vae.dtype)).sample
    decoded = (decoded / 2 + 0.5).clamp(0, 1)
    decoded = decoded.detach().cpu().permute(0, 2, 3, 1).numpy()[0]
    return Image.fromarray((decoded * 255).round().astype(np.uint8))
