"""Build the single, cumulative, priority-ordered prompt shared by all four
generation methods, so that the ablation study isolates the *inference
mechanism* (one-shot vs. prompt switching vs. attention modulation vs.
Attend-and-Excite) rather than prompt content.

Highest-priority object is folded in last, so it appears first in the final
prompt string (LLMs and diffusion text encoders both weight earlier tokens
more heavily).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from gcd.parsing.description_parser import DescriptionParser


def build_cumulative_prompt(
    background: str,
    node_names: List[str],
    attributes: Dict[str, Dict[str, str]],
    relations: Dict[str, Dict[str, str]],
    priority_scores: np.ndarray,
    prompt_rewriter: Optional[DescriptionParser] = None,
) -> Tuple[str, Dict[str, str]]:
    """Return (full_prompt, concept_search_phrases).

    ``concept_search_phrases`` maps each object name (plus ``"background"``)
    to the short phrase used for cross-attention token-span matching in the
    Attention-Modulation method; Simple Diffusion and Context Switching
    ignore the second return value.
    """
    base_prompt = background if background else "a scene"
    if len(base_prompt.split()) > 25:
        base_prompt = " ".join(base_prompt.split()[:25]).rstrip(",.;") + "."
    sorted_idx = np.argsort(priority_scores)

    concept_searches: Dict[str, str] = {"background": base_prompt.split(".")[0].strip()[:40]}
    prompt = base_prompt
    mentioned: List[str] = []

    for idx in sorted_idx:
        obj_name = node_names[int(idx)]
        attrs = attributes.get(obj_name, {})
        if not isinstance(attrs, dict):
            attrs = {}

        rels_to_prev = {
            prev: relations[obj_name][prev]
            for prev in mentioned
            if obj_name in relations and prev in relations.get(obj_name, {})
        }

        if prompt_rewriter is not None:
            try:
                prompt = prompt_rewriter.rewrite_prompt_stack(
                    previous_prompt=prompt,
                    object_name=obj_name,
                    attributes=attrs,
                    previous_objects=mentioned,
                    relations_to_previous=rels_to_prev,
                    background=base_prompt,
                )
            except Exception:  # noqa: BLE001 - fall back to a deterministic template
                prompt = _deterministic_stack(prompt, obj_name, attrs)
        else:
            prompt = _deterministic_stack(prompt, obj_name, attrs)

        concept_searches[obj_name] = obj_name
        mentioned.append(obj_name)

    return prompt, concept_searches


def _deterministic_stack(previous_prompt: str, obj_name: str, attrs: Dict[str, str]) -> str:
    attr_parts = [f"{v} {k}" for k, v in attrs.items() if v and k != "type"]
    obj_desc = f"{obj_name} with {', '.join(attr_parts)}" if attr_parts else obj_name
    return f"{obj_desc}. {previous_prompt}"


def compute_step_allocations(
    node_names: List[str],
    priority_scores: np.ndarray,
    bg_steps: int,
    intro_phase_end: int,
    ramp_len: int = 3,
    max_share: float = 0.5,
) -> Dict[str, Tuple[int, int]]:
    """..."""
    n = len(node_names)
    obj_steps = intro_phase_end - bg_steps
    if n == 0 or obj_steps <= 0:
        return {}
    name_to_score: Dict[str, float] = {}
    for name, score in zip(node_names, priority_scores):
        name_to_score[name] = name_to_score.get(name, 0.0) + float(score)
    unique_names = list(name_to_score.keys())
    unique_scores = np.array([name_to_score[n] for n in unique_names], dtype=np.float64)
    unique_scores = unique_scores / unique_scores.sum()
    n_unique = len(unique_names)
    order = np.argsort(unique_scores)[::-1]
    sorted_names = [unique_names[int(i)] for i in order]
    sorted_scores = np.array([unique_scores[int(i)] for i in order])

    min_steps = max(ramp_len + 2, obj_steps // (n_unique * 4))
    alloc_steps = np.full(n_unique, float(min_steps))
    remaining = obj_steps - min_steps * n_unique

    if remaining > 0:
        s = sorted_scores / max(sorted_scores.sum(), 1e-9)
        capped = np.minimum(s, max_share)
        capped = capped / capped.sum()
        alloc_steps += capped * remaining

    alloc_steps = np.round(alloc_steps).astype(int)
    alloc_steps[0] += obj_steps - alloc_steps.sum()

    allocations: Dict[str, Tuple[int, int]] = {}
    cursor = bg_steps
    for i in range(n_unique):
        end = min(intro_phase_end, cursor + int(alloc_steps[i]))
        allocations[sorted_names[i]] = (cursor, end)
        cursor = end

    if allocations:
        last = next(reversed(allocations))
        allocations[last] = (allocations[last][0], intro_phase_end)
    return allocations