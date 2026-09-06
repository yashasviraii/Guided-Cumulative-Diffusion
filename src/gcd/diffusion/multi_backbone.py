"""Multi-backbone experiment driver: run Simple Diffusion, Context Switching,
and Attention Modulation across *several diffusion backbones* (e.g. SD v1.5,
Dreamshaper-8, SDXL) with a single shared LLM parser + GCN priority model,
reproducing the per-backbone comparison in Table 1 of the paper.

``gcd.diffusion.runner`` (single-backbone) is enough for most experiments;
this module exists specifically because SDXL's two-stage UNet needs a
different text-encoding and ``added_cond_kwargs`` call signature than SD
1.x, so the attention-modulation and context-switching denoising loops must
branch on backbone type. ``ModelRunner`` centralizes that branching so the
three per-method ``run_*`` functions stay backbone-agnostic.

Note: Attend-and-Excite is intentionally excluded from this multi-backbone
driver — diffusers' ``StableDiffusionAttendAndExcitePipeline`` only supports
the SD 1.x UNet/text-encoder architecture, not SDXL's dual text encoders.
Run it per-backbone via ``gcd.diffusion.methods.attend_and_excite`` instead.

Usage
-----
    python -m gcd.diffusion.multi_backbone \\
        --descriptions data/descriptions.jsonl \\
        --test-ids data/splits/test_ids.txt \\
        --gnn-checkpoint checkpoints/gnn_model.pt \\
        --output-root outputs/multi_backbone \\
        --backbone sd15=runwayml/stable-diffusion-v1-5 \\
        --backbone dreamshaper8=Lykon/dreamshaper-8 \\
        --backbone sdxl=stabilityai/stable-diffusion-xl-base-1.0
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline, StableDiffusionXLPipeline
from PIL import Image

from gcd.diffusion.attention_processor import (
    AttentionState,
    GCDAttentionProcessor,
    build_weight_schedule,
    find_token_spans,
)
from gcd.diffusion.precomputed_scene import load_jsonl_by_file, load_precomputed_scene
from gcd.graph.graph_builder import InferenceGraphBuilder
from gcd.graph.priority_scorer import PriorityScorer
from gcd.graph.sanitize import sanitize_background
from gcd.gnn.models import SimpleGCNInference
from gcd.parsing.description_parser import DEFAULT_LLM_MODEL, DescriptionParser

GUIDANCE_SCALE = 7.5
# Same 0-100-1000-1200 phase proportions used everywhere else, rescaled to
# whatever step budget a given backbone is run at (SDXL needs far fewer
# steps for comparable quality than SD 1.x).
_BG_FRAC = 100 / 1200
_INTRO_FRAC = 1000 / 1200


def phase_boundaries(num_steps: int) -> Tuple[int, int]:
    bg = max(1, round(num_steps * _BG_FRAC))
    intro = min(num_steps, round(num_steps * _INTRO_FRAC))
    return bg, intro


class ModelRunner:
    """Wraps an SD 1.x or SDXL pipeline behind one text-encode / UNet-step / decode API."""

    def __init__(self, model_name: str, model_id: str, device: str, num_steps: int) -> None:
        self.model_name = model_name
        self.model_id = model_id
        self.device = device
        self.is_xl = "xl" in model_name.lower()

        print(f"\nLoading {model_name}: {model_id} (XL={self.is_xl})")
        dtype = torch.float16 if device == "cuda" else torch.float32
        if self.is_xl:
            self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                model_id, torch_dtype=dtype, use_safetensors=True, variant="fp16"
            ).to(device)
        else:
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                model_id, torch_dtype=dtype, safety_checker=None
            ).to(device)

        try:
            config = dict(self.pipeline.scheduler.config)
            self.pipeline.scheduler = DDIMScheduler.from_config(config)
            print(f"  Scheduler: DDIM ({config.get('num_train_timesteps', 1000)} train steps)")
        except Exception as exc:  # noqa: BLE001
            print(f"  Warning: could not set DDIM ({exc})")

        self.pipeline.enable_attention_slicing()

        max_ts = getattr(self.pipeline.scheduler.config, "num_train_timesteps", 1000)
        self.num_steps = min(num_steps, max_ts)
        self.bg_steps, self.intro_end = phase_boundaries(self.num_steps)
        print(
            f"  Steps: {self.num_steps} | bg: 0-{self.bg_steps} | "
            f"objects: {self.bg_steps}-{self.intro_end} | harmonise: {self.intro_end}-{self.num_steps}"
        )

    def encode_text(self, text: str) -> Dict:
        if self.is_xl:
            emb, neg_emb, pooled, neg_pooled = self.pipeline.encode_prompt(
                prompt=text, prompt_2=text, device=self.device, num_images_per_prompt=1,
                do_classifier_free_guidance=True, negative_prompt="", negative_prompt_2="",
            )
            return {"embeds": emb, "pooled": pooled, "neg_embeds": neg_emb, "neg_pooled": neg_pooled}

        tokenizer, text_encoder = self.pipeline.tokenizer, self.pipeline.text_encoder

        def _encode(t: str) -> torch.Tensor:
            ids = tokenizer(
                t, padding="max_length", truncation=True, max_length=tokenizer.model_max_length, return_tensors="pt"
            ).input_ids.to(self.device)
            with torch.no_grad():
                return text_encoder(ids)[0]

        return {"embeds": _encode(text), "pooled": None, "neg_embeds": _encode(""), "neg_pooled": None}

    def _sdxl_time_ids(self, height: int, width: int) -> torch.Tensor:
        return torch.tensor([[height, width, 0, 0, height, width]], dtype=torch.float32, device=self.device)

    def unet_forward(self, latents: torch.Tensor, timestep: torch.Tensor, enc: Dict, height: int, width: int) -> torch.Tensor:
        """One conditional UNet forward pass (no CFG combination here)."""
        unet = self.pipeline.unet
        with torch.no_grad():
            if self.is_xl:
                added_cond = {"text_embeds": enc["pooled"], "time_ids": self._sdxl_time_ids(height, width)}
                return unet(latents, timestep, encoder_hidden_states=enc["embeds"], added_cond_kwargs=added_cond).sample
            return unet(latents, timestep, encoder_hidden_states=enc["embeds"]).sample

    def make_latents(self, seed: int, height: int = 512, width: int = 512) -> torch.Tensor:
        unet = self.pipeline.unet
        generator = torch.Generator(device=self.device).manual_seed(seed)
        latents = torch.randn(
            (1, unet.config.in_channels, height // 8, width // 8),
            generator=generator, device=self.device, dtype=getattr(unet, "dtype", torch.float16),
        )
        sigma = getattr(self.pipeline.scheduler, "init_noise_sigma", None)
        return latents * sigma if sigma is not None else latents

    def decode_latents(self, latents: torch.Tensor) -> Image.Image:
        latents = latents / 0.18215
        with torch.no_grad():
            img = self.pipeline.vae.decode(latents.to(dtype=self.pipeline.vae.dtype)).sample
        img = (img / 2 + 0.5).clamp(0, 1).detach().cpu().permute(0, 2, 3, 1).numpy()[0]
        return Image.fromarray((img * 255).round().astype(np.uint8))

    def compute_step_allocations(self, node_names: List[str], priority_scores: np.ndarray) -> Dict[str, Tuple[int, int]]:
        obj_budget = self.intro_end - self.bg_steps
        allocations: Dict[str, Tuple[int, int]] = {}
        cursor = self.bg_steps
        for idx in np.argsort(priority_scores)[::-1]:
            name = node_names[int(idx)]
            n_steps = round(float(priority_scores[int(idx)]) * obj_budget)
            if n_steps <= 0:
                continue
            end = min(self.intro_end, cursor + n_steps)
            allocations[name] = (cursor, end)
            cursor = end
        if allocations:
            last = next(reversed(allocations))
            allocations[last] = (allocations[last][0], self.intro_end)
        return allocations

    def save_snapshot(self, interm_dir: Path, step_count: int, step_index: int, latents: torch.Tensor, allocations: Dict, prompt_text: str) -> None:
        active = [n for n, (s, e) in allocations.items() if s <= step_index < e]
        if step_index < self.bg_steps:
            label = "background"
        elif step_index >= self.intro_end:
            label = "harmonise"
        else:
            label = "priority_" + "_".join(active[:2]) if active else "transition"
        safe = re.sub(r"[^A-Za-z0-9_\-]+", "_", label).strip("_") or "step"
        self.decode_latents(latents).save(interm_dir / f"step_{step_count:04d}_{safe}.png")
        (interm_dir / f"step_{step_count:04d}_{safe}_prompt.txt").write_text(prompt_text)

    # ── Shared scene preparation ────────────────────────────────────────────

    def _prepare_scene(
        self,
        description: str,
        parser: Optional[DescriptionParser],
        gnn_model: SimpleGCNInference,
        file_path: Optional[str] = None,
        precomputed: Optional[Dict] = None,
    ):
        """Build (objects, relations, background, node_names, priority_scores).

        If ``precomputed`` is given (``{"graphs_dir", "parsed_map", "background_map"}``),
        reuse your existing dataset files and the GNN scores the *real*
        object-relation edges from the graph JSON. Otherwise falls back to a
        live LLM parse with hashed, edge-less features (requires ``parser``).
        """
        if precomputed is not None and file_path:
            scene = load_precomputed_scene(
                file_path, precomputed["graphs_dir"], precomputed["parsed_map"],
                precomputed.get("background_map"), gnn_model,
            )
            if scene is None:
                return None
            return scene["objects"], scene["relations"], scene["background"], scene["node_names"], scene["priority_scores"]

        if parser is None:
            raise ValueError("Either `precomputed` (with file_path) or a `parser` must be provided.")

        objects = parser.extract_objects(description)
        if not objects:
            return None
        relations = parser.extract_relations(description, objects)
        background = parser.extract_background(description, objects)
        node_features, node_names, edges = InferenceGraphBuilder.build_graph(objects, relations)
        edge_index = np.array([[s, t] for s, t, _ in edges], dtype=np.int64).T if edges else None
        priority_scores = PriorityScorer.normalize_scores(
            PriorityScorer.score_nodes(node_features, gnn_model, edge_index=edge_index)
        )
        cleaned_background = sanitize_background(background, node_names, objects)
        return objects, relations, cleaned_background, node_names, priority_scores

    def _stack_prompt(self, parser: Optional[DescriptionParser], base_prompt: str, order: List[str], objects: dict, relations: dict) -> str:
        prompt, mentioned = base_prompt, []
        for name in order:
            attrs = objects.get(name, {}) if isinstance(objects.get(name, {}), dict) else {}
            rels = {m: relations[name][m] for m in mentioned if name in relations and m in relations.get(name, {})}
            if parser is not None:
                try:
                    prompt = parser.rewrite_prompt_stack(
                        previous_prompt=prompt, object_name=name, attributes=attrs,
                        previous_objects=mentioned, relations_to_previous=rels, background=base_prompt,
                    )
                except Exception:  # noqa: BLE001
                    prompt = f"{name}. {prompt}"
            else:
                prompt = f"{name}. {prompt}"
            mentioned.append(name)
        return prompt

    # ── Method: Simple Diffusion (one-shot baseline) ────────────────────────

    def run_simple(self, description, output_dir, parser, gnn_model, seed, height=512, width=512, file_path=None, precomputed=None):
        scene = self._prepare_scene(description, parser, gnn_model, file_path, precomputed)
        if scene is None:
            return None
        objects, relations, background, node_names, scores = scene
        order = [node_names[i] for i in np.argsort(scores)[::-1]]
        final_prompt = self._stack_prompt(parser, background or "a scene", order, objects, relations)

        max_ts = getattr(self.pipeline.scheduler.config, "num_train_timesteps", 1000)
        num_steps = self.num_steps if self.num_steps < max_ts else max_ts - 1
        generator = torch.Generator(device=self.device).manual_seed(seed)
        with torch.no_grad():
            image = self.pipeline(
                prompt=final_prompt, height=height, width=width,
                num_inference_steps=num_steps, guidance_scale=GUIDANCE_SCALE, generator=generator,
            ).images[0]

        output_dir.mkdir(parents=True, exist_ok=True)
        image.save(output_dir / "generated_image.png")
        self._dump_result(output_dir, "simple", description, final_prompt, objects, node_names, scores)
        return final_prompt

    # ── Method: Context Switching (hard prompt-switching ablation) ─────────

    def run_context_switching(self, description, output_dir, parser, gnn_model, seed, height=512, width=512, file_path=None, precomputed=None):
        scene = self._prepare_scene(description, parser, gnn_model, file_path, precomputed)
        if scene is None:
            return None
        objects, relations, background, node_names, scores = scene
        allocations = self.compute_step_allocations(node_names, scores)
        base_prompt = background or "a scene"

        schedule: Dict[int, str] = {s: base_prompt for s in range(self.bg_steps)}
        prompt, mentioned = base_prompt, []
        for name, (start, end) in sorted(allocations.items(), key=lambda item: item[1][0]):
            attrs = objects.get(name, {}) if isinstance(objects.get(name, {}), dict) else {}
            rels = {m: relations[name][m] for m in mentioned if name in relations and m in relations.get(name, {})}
            if parser is not None:
                try:
                    prompt = parser.rewrite_prompt_stack(
                        previous_prompt=prompt, object_name=name, attributes=attrs,
                        previous_objects=mentioned, relations_to_previous=rels, background=base_prompt,
                    )
                except Exception:  # noqa: BLE001
                    prompt = f"{name}. {prompt}"
            else:
                prompt = f"{name}. {prompt}"
            for s in range(start, end):
                schedule[s] = prompt
            mentioned.append(name)
        final_prompt = prompt
        for s in range(self.bg_steps, self.num_steps):
            schedule.setdefault(s, final_prompt)

        self.pipeline.scheduler.set_timesteps(self.num_steps)
        timesteps = self.pipeline.scheduler.timesteps
        max_ts = getattr(self.pipeline.scheduler.config, "num_train_timesteps", None)
        if max_ts:
            timesteps = torch.clamp(timesteps, max=int(max_ts) - 1)

        latents = self.make_latents(seed, height, width)
        interm = output_dir / "intermediate"
        interm.mkdir(parents=True, exist_ok=True)
        checkpoints = sorted({self.bg_steps, self.intro_end, self.num_steps} | {e for _, e in allocations.values()})

        last_prompt, enc = None, None
        for step_idx, timestep in enumerate(timesteps):
            current_prompt = schedule.get(step_idx, final_prompt)
            if current_prompt != last_prompt:
                enc, last_prompt = self.encode_text(current_prompt), current_prompt

            noise_cond = self.unet_forward(latents, timestep, enc, height, width)
            uncond_enc = {**enc, "embeds": enc["neg_embeds"], "pooled": enc["neg_pooled"]}
            noise_uncond = self.unet_forward(latents, timestep, uncond_enc, height, width)
            noise = noise_uncond + GUIDANCE_SCALE * (noise_cond - noise_uncond)

            step_out = self.pipeline.scheduler.step(noise, timestep, latents, return_dict=True)
            latents = step_out.prev_sample

            current_step = step_idx + 1
            if current_step in checkpoints:
                self.save_snapshot(interm, current_step, step_idx, getattr(step_out, "pred_original_sample", latents), allocations, current_prompt)

        self.decode_latents(latents).save(output_dir / "generated_image.png")
        self._dump_result(output_dir, "context_switching", description, final_prompt, objects, node_names, scores, allocations)
        return final_prompt

    # ── Method: Attention Modulation (GCD, ours) ────────────────────────────

    def run_attention_modulation(self, description, output_dir, parser, gnn_model, seed, height=512, width=512, file_path=None, precomputed=None):
        scene = self._prepare_scene(description, parser, gnn_model, file_path, precomputed)
        if scene is None:
            return None
        objects, relations, background, node_names, scores = scene
        allocations = self.compute_step_allocations(node_names, scores)
        base_prompt = background or "a scene"

        order = [name for name, _ in sorted(allocations.items(), key=lambda item: item[1][0])]
        full_prompt = self._stack_prompt(parser, base_prompt, order, objects, relations)
        concept_searches = {"background": re.split(r"[.,;]", base_prompt)[0].strip()[:40], **{n: n for n in order}}

        tokenizer = self.pipeline.tokenizer
        token_spans = find_token_spans(full_prompt, concept_searches, tokenizer)
        weight_schedule = build_weight_schedule(
            total_steps=self.num_steps, bg_steps=self.bg_steps, step_allocations=allocations,
            token_spans=token_spans, n_tokens=tokenizer.model_max_length,
        )

        attn_state = AttentionState()
        processor = GCDAttentionProcessor(attn_state)
        saved_processors = dict(self.pipeline.unet.attn_processors)
        self.pipeline.unet.set_attn_processor({k: processor for k in saved_processors})

        enc_cond = self.encode_text(full_prompt)
        enc_uncond = self.encode_text("")

        self.pipeline.scheduler.set_timesteps(self.num_steps)
        timesteps = self.pipeline.scheduler.timesteps
        max_ts = getattr(self.pipeline.scheduler.config, "num_train_timesteps", None)
        if max_ts:
            timesteps = torch.clamp(timesteps, max=int(max_ts) - 1)

        latents = self.make_latents(seed, height, width)
        interm = output_dir / "intermediate"
        interm.mkdir(parents=True, exist_ok=True)
        checkpoints = sorted({self.bg_steps, self.intro_end, self.num_steps} | {e for _, e in allocations.values()})

        for step_idx, timestep in enumerate(timesteps):
            attn_state.weight_vector = weight_schedule[step_idx].to(self.device)

            attn_state.enabled = True
            noise_cond = self.unet_forward(latents, timestep, enc_cond, height, width)
            attn_state.enabled = False
            noise_uncond = self.unet_forward(latents, timestep, enc_uncond, height, width)

            noise = noise_uncond + GUIDANCE_SCALE * (noise_cond - noise_uncond)
            step_out = self.pipeline.scheduler.step(noise, timestep, latents, return_dict=True)
            latents = step_out.prev_sample

            current_step = step_idx + 1
            if current_step in checkpoints:
                self.save_snapshot(interm, current_step, step_idx, getattr(step_out, "pred_original_sample", latents), allocations, full_prompt)

        self.pipeline.unet.set_attn_processor(saved_processors)
        attn_state.enabled = False

        self.decode_latents(latents).save(output_dir / "generated_image.png")
        self._dump_result(output_dir, "attention_modulation", description, full_prompt, objects, node_names, scores, allocations, token_spans)
        return full_prompt

    @staticmethod
    def _dump_result(output_dir, method, description, final_prompt, objects, node_names, scores, allocations=None, token_spans=None) -> None:
        payload = {
            "method": method, "description": description, "final_prompt": final_prompt,
            "objects": objects, "priority_scores": dict(zip(node_names, scores.tolist())),
        }
        if allocations is not None:
            payload["step_allocations"] = {n: list(v) for n, v in allocations.items()}
        if token_spans is not None:
            payload["token_spans"] = token_spans
        with open(output_dir / "inference_result.json", "w") as f:
            json.dump(payload, f, indent=2)


METHOD_FUNCS = {
    "simple": ModelRunner.run_simple,
    "context_switching": ModelRunner.run_context_switching,
    "attention_modulation": ModelRunner.run_attention_modulation,
}


def _load_test_entries(descriptions_path: str, test_ids_path: Optional[str], n_images: int, seed: int) -> List[dict]:
    test_ids = None
    if test_ids_path and Path(test_ids_path).exists():
        test_ids = {line.strip() for line in Path(test_ids_path).read_text().splitlines() if line.strip()}

    entries = []
    with open(descriptions_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            image_id = Path(record.get("file", "")).stem
            if test_ids is None or image_id in test_ids:
                entries.append({
                    "image_id": image_id,
                    "file": record.get("file", ""),
                    "description": record.get("description", ""),
                })

    random.seed(seed)
    if not n_images:
        print(f"Selected all {len(entries)} test images (--n-images 0 means 'no limit').")
        return entries
    selected = random.sample(entries, min(n_images, len(entries)))
    print(f"Selected {len(selected)} test images.")
    return selected


def run_all(
    backbones: Dict[str, str],
    descriptions_path: str,
    test_ids_path: Optional[str],
    output_root: Path,
    gnn_checkpoint: str,
    llm_model: str,
    steps_per_backbone: Dict[str, int],
    n_images: int,
    seed: int,
    graphs_dir: Optional[str] = None,
    parsed_path: Optional[str] = None,
    background_path: Optional[str] = None,
) -> None:
    entries = _load_test_entries(descriptions_path, test_ids_path, n_images, seed)
    if not entries:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n[Setup] Loading GNN (shared across all backbones)...")
    gnn_model = SimpleGCNInference(gnn_checkpoint, device=device)

    precomputed_mode = graphs_dir is not None and parsed_path is not None
    parser: Optional[DescriptionParser] = None
    precomputed: Optional[Dict] = None
    if precomputed_mode:
        print(f"Precomputed mode: reading graphs from {graphs_dir}, objects/relations from {parsed_path}")
        precomputed = {
            "graphs_dir": graphs_dir,
            "parsed_map": load_jsonl_by_file(parsed_path),
            "background_map": load_jsonl_by_file(background_path) if background_path else None,
        }
        print("No LLM parser loaded (objects/relations/background come from your dataset files).")
    else:
        print("Live mode: loading LLM parser (no --graphs-dir/--parsed given).")
        parser = DescriptionParser(model_name=llm_model, device=device)

    errors: List[tuple] = []
    for backbone_name, model_id in backbones.items():
        num_steps = steps_per_backbone.get(backbone_name, 1200)
        runner = ModelRunner(backbone_name, model_id, device, num_steps)

        for method_name, method_fn in METHOD_FUNCS.items():
            print(f"\n{'=' * 70}\n  Backbone: {backbone_name} | Method: {method_name}\n{'=' * 70}")
            for i, entry in enumerate(entries):
                img_id, description = entry["image_id"], entry["description"]
                out_dir = output_root / backbone_name / method_name / img_id
                if (out_dir / "generated_image.png").exists():
                    print(f"  [{i + 1}/{len(entries)}] {img_id} — already exists, skipping.")
                    continue
                out_dir.mkdir(parents=True, exist_ok=True)
                print(f"  [{i + 1}/{len(entries)}] {img_id}")
                try:
                    result = method_fn(
                        runner, description, out_dir, parser, gnn_model, seed,
                        file_path=entry["file"], precomputed=precomputed,
                    )
                    print("  No objects found — skipped." if result is None else f"  Saved -> {out_dir / 'generated_image.png'}")
                except Exception as exc:  # noqa: BLE001
                    print(f"  Failed: {exc}")
                    errors.append((backbone_name, method_name, str(exc)))

        del runner.pipeline
        del runner
        torch.cuda.empty_cache()
        print(f"\nFreed {backbone_name} pipeline from VRAM.")

    print("\n\nAll done.")
    if errors:
        print(f"\n{len(errors)} error(s):")
        for backbone_name, method_name, err in errors:
            print(f"  [{backbone_name} / {method_name}] {err}")
    print(f"Results in: {output_root}")


def _parse_backbone_arg(values: List[str]) -> Dict[str, str]:
    backbones = {}
    for value in values:
        name, _, model_id = value.partition("=")
        if not model_id:
            raise argparse.ArgumentTypeError(f"--backbone must be 'name=model_id', got: {value}")
        backbones[name] = model_id
    return backbones


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multi-backbone experiment driver (Table 1 reproduction)")
    parser.add_argument("--descriptions", required=True)
    parser.add_argument("--test-ids", default=None)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--gnn-checkpoint", required=True)
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument(
        "--n-images", type=int, default=50,
        help="Random sample size from --descriptions (after any --test-ids filter). "
             "Pass 0 to run on every matching image with no sampling.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--backbone", action="append", required=True, dest="backbones",
        help="Repeatable 'name=huggingface_model_id', e.g. --backbone sd15=runwayml/stable-diffusion-v1-5",
    )
    parser.add_argument(
        "--steps", action="append", default=[],
        help="Repeatable 'name=num_steps' override, e.g. --steps sdxl=500 (default: 1200)",
    )
    parser.add_argument(
        "--graphs-dir", default=None,
        help="Precomputed mode: directory of independent {file_id}.json graph files",
    )
    parser.add_argument(
        "--parsed", default=None,
        help="Precomputed mode: parsed_images.jsonl (or merged.jsonl, which already has background_context)",
    )
    parser.add_argument(
        "--background", default=None,
        help="Precomputed mode: backgroundContext.jsonl (omit if --parsed is already merged.jsonl)",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    backbones = _parse_backbone_arg(args.backbones)
    steps_per_backbone = {"": 1200}
    for entry in args.steps:
        name, _, steps = entry.partition("=")
        steps_per_backbone[name] = int(steps)

    run_all(
        backbones=backbones,
        descriptions_path=args.descriptions,
        test_ids_path=args.test_ids,
        output_root=Path(args.output_root),
        gnn_checkpoint=args.gnn_checkpoint,
        llm_model=args.llm_model,
        steps_per_backbone=steps_per_backbone,
        n_images=args.n_images,
        seed=args.seed,
        graphs_dir=args.graphs_dir,
        parsed_path=args.parsed,
        background_path=args.background,
    )


if __name__ == "__main__":
    main()
