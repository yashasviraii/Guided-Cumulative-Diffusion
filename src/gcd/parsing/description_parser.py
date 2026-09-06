"""Inference-time natural-language scene parsing.

``DescriptionParser`` wraps a causal LLM (default: Qwen2.5-3B-Instruct) and
exposes the three extraction calls needed by every generation method, plus
the "prompt-stacking" rewrite used by the Context-Switching and Attention
Modulation methods to build a single cumulative prompt in priority order:

1. ``extract_objects``   -> {object_name: {attribute: value, ...}, ...}
2. ``extract_relations`` -> {object_name: {other_object_name: relation}, ...}
3. ``extract_background``-> free-text background/environment description
4. ``rewrite_prompt_stack`` -> fold one new object into a running prompt

This module is intentionally free of diffusion-model imports so it can be
reused by the data-construction pipeline (``gcd.data``) as well as every
inference-time diffusion method in ``gcd.diffusion``.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"

SYS_PROMPT_OBJECTS = (
    "You are a strict data extraction assistant.\n"
    "Extract every visible physical object and its visual attributes from the "
    "description into a compact JSON dictionary.\n\n"
    "Format (compact, no extra spaces):\n"
    '  {"Object Name":{"color":"red","size":"large"},"Object2":{"type":"round"}}\n\n'
    "Rules:\n"
    "1. Object names: SHORT GENERIC NOUNS only (1-3 words max).\n"
    "   CORRECT: 'Bear', 'Snow', 'Mountain'   WRONG: 'PolarBear', 'SnowyCovered'\n"
    "2. All properties (color, size, material, type, texture) go in attributes dict.\n"
    "3. For multiple same objects: 'Bear 1', 'Bear 2'.\n"
    "4. Include clothing/accessories as objects.\n"
    "5. Normalise values: 'reddish'->'red', 'dark blue'->'dark_blue'.\n"
    "6. Output ONLY valid compact JSON — no markdown, no explanation."
)

SYS_PROMPT_RELATIONS = (
    "You are a relationship extraction assistant.\n"
    "Given objects and a description, extract spatial and semantic relationships.\n\n"
    "Format (compact JSON):\n"
    '  {"Object1": {"Object2": "relation_type", ...}, ...}\n\n'
    "Valid relations: on, under, inside, in_front_of, behind, next_to, above, below, "
    "holding, wearing, attached_to, part_of, contains.\n\n"
    "Rules:\n"
    "1. Only include relationships explicitly mentioned or strongly implied.\n"
    "2. For each relation, pick the MOST SPECIFIC type.\n"
    "3. Output ONLY compact JSON — no explanation."
)

SYS_PROMPT_BACKGROUND = (
    "You are a background extraction expert.\n"
    "Given a description and foreground objects to skip, extract ONLY background context.\n\n"
    "Background includes: location/setting, environment (floor, walls, sky), "
    "lighting, atmosphere, weather, distant scenery.\n\n"
    "Rules:\n"
    "1. Do NOT describe the foreground objects listed. STRICT: do not mention object "
    "names, colors, sizes, clothing, accessories, or any attribute tied to foreground "
    "objects.\n"
    "2. Output plain text in 1-3 sentences.\n"
    "3. If minimal background, output an empty string or a single short sentence "
    "focusing only on environment (no objects).\n"
    "4. No bullet points, no labels, plain descriptive text only."
)

SYS_PROMPT_PROMPT_STACKING = (
    "You are a diffusion prompt editor.\n"
    "Rewrite the prompt using the previous prompt and one new object.\n\n"
    "Rules:\n"
    "1. The new object must appear first and be the main focus.\n"
    "2. Keep the previous prompt's background and already mentioned objects.\n"
    "3. Use only the provided object name, attributes, and relations. Do not invent "
    "details.\n"
    "4. If a relation to a previous object is given, express it naturally.\n"
    "5. Keep it concise and readable; one sentence preferred.\n"
    "6. Output only the rewritten prompt."
)


def _clean_name(text: str) -> str:
    """Convert a CamelCase / snake_case object name into a spaced phrase."""
    text = text.replace("_", " ")
    return re.sub(r"(?<!^)(?=[A-Z])", " ", text)


class DescriptionParser:
    """LLM-backed extractor of objects, relations, and background context."""

    def __init__(self, model_name: str = DEFAULT_LLM_MODEL, device: str = "cuda") -> None:
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device,
        )
        self.model.eval()

    def _call_llm(self, system_prompt: str, user_message: str, max_tokens: int = 256) -> str:
        """Run one deterministic (greedy) chat completion and return the text."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=None,
                top_p=None,
                do_sample=False,
            )

        generated = outputs[0][input_len:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    @staticmethod
    def _extract_json_object(text: str) -> Dict:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return {}

    def extract_objects(self, description: str) -> Dict[str, Dict[str, str]]:
        """Extract {object_name: {attribute: value}} from a caption."""
        response = self._call_llm(SYS_PROMPT_OBJECTS, description, max_tokens=512)
        return self._extract_json_object(response)

    def extract_relations(
        self, description: str, objects: Dict[str, Dict[str, str]]
    ) -> Dict[str, Dict[str, str]]:
        """Extract a directed adjacency list of spatial/semantic relations."""
        if not objects:
            return {}
        obj_list = ", ".join(objects.keys())
        user_msg = f"Description: {description}\n\nObjects: {obj_list}\n\nRelations:"
        response = self._call_llm(SYS_PROMPT_RELATIONS, user_msg, max_tokens=384)
        return self._extract_json_object(response)

    def extract_background(self, description: str, objects: Dict[str, Dict[str, str]]) -> str:
        """Extract the background/environment description, excluding foreground objects."""
        if objects:
            obj_list = ", ".join(objects.keys())
            user_msg = (
                f"Description: {description}\n\nSkip (foreground): {obj_list}\n\n"
                "Background: (Return only background environment; do NOT mention or "
                "describe any of the skipped objects.)"
            )
        else:
            user_msg = (
                f"Description: {description}\n\n"
                "Background: (Return only background environment; do NOT mention or "
                "describe any foreground objects.)"
            )
        return self._call_llm(SYS_PROMPT_BACKGROUND, user_msg, max_tokens=128).strip()

    def rewrite_prompt_stack(
        self,
        previous_prompt: str,
        object_name: str,
        attributes: Dict[str, str],
        previous_objects: List[str],
        relations_to_previous: Dict[str, str],
        background: str,
    ) -> str:
        """Fold one new object into ``previous_prompt``, placing it first.

        Used by the Context-Switching and Attention-Modulation methods to build
        the cumulative, priority-ordered prompt one object at a time.
        """
        clean_object_name = _clean_name(object_name)
        clean_previous_objects = [_clean_name(n) for n in previous_objects] if previous_objects else []
        clean_relations = (
            {_clean_name(k): v for k, v in relations_to_previous.items()}
            if relations_to_previous
            else {}
        )

        user_message = (
            f"Previous prompt:\n{previous_prompt}\n\n"
            f"Background:\n{background if background else 'a scene'}\n\n"
            f"New object:\n{clean_object_name}\n\n"
            f"New object attributes (JSON):\n{json.dumps(attributes, ensure_ascii=False)}\n\n"
            f"Previously mentioned objects:\n"
            f"{', '.join(clean_previous_objects) if clean_previous_objects else 'None'}\n\n"
            f"Relations to previous objects (JSON):\n{json.dumps(clean_relations, ensure_ascii=False)}\n\n"
            "Rewrite the prompt so the new object is first, previous context is "
            "preserved, and the output is a single concise prompt."
        )
        return self._call_llm(SYS_PROMPT_PROMPT_STACKING, user_message, max_tokens=128)
