"""Cross-attention log-bias mechanism (paper Section 3.5, Eq. 2-3).

``GCDAttentionProcessor`` replaces every cross-attention op in the UNet with
one that adds a per-token log-bias to the pre-softmax attention logits:

    Attn_mod(q, K, V) = softmax(qK^T / sqrt(d) + log(w)) V

Since ``softmax(x + log w) ∝ w * softmax(x)``, this is exactly equivalent to
multiplying each token's post-softmax attention weight by ``w`` — a
differentiable-in-spirit, purely inference-time soft gate. Self-attention and
the unconditional (classifier-free-guidance) branch are passed through
unmodified.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

BG_BOOST = 1.3
OBJ_SUPPRESS = 0.15
RAMP_LEN = 3


class AttentionState:
    """Mutable per-step state shared by every attention-processor instance."""

    def __init__(self) -> None:
        self.weight_vector: Optional[torch.Tensor] = None  # [n_text_tokens]
        self.enabled: bool = False  # True only during the conditional forward pass


class GCDAttentionProcessor:
    """Cross-attention processor applying ``AttentionState.weight_vector`` as a log-bias."""

    def __init__(self, state: AttentionState) -> None:
        self.state = state

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        is_cross = encoder_hidden_states is not None
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            B, C, H, W = hidden_states.shape
            hidden_states = hidden_states.view(B, C, H * W).transpose(1, 2)

        B = hidden_states.shape[0]
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        kv = encoder_hidden_states if is_cross else hidden_states
        if is_cross and attn.norm_cross:
            kv = attn.norm_encoder_hidden_states(kv)
        key, value = attn.to_k(kv), attn.to_v(kv)

        heads = attn.heads
        head_dim = key.shape[-1] // heads
        query = query.view(B, -1, heads, head_dim).transpose(1, 2)
        key = key.view(B, -1, heads, head_dim).transpose(1, 2)
        value = value.view(B, -1, heads, head_dim).transpose(1, 2)

        if getattr(attn, "norm_q", None) is not None:
            query = attn.norm_q(query)
        if getattr(attn, "norm_k", None) is not None:
            key = attn.norm_k(key)

        scores = torch.matmul(query, key.transpose(-2, -1)) * (head_dim ** -0.5)

        if is_cross and self.state.enabled and self.state.weight_vector is not None:
            n_k = scores.shape[-1]
            wv = self.state.weight_vector.to(device=scores.device, dtype=scores.dtype)
            if wv.shape[0] < n_k:
                wv = torch.cat([wv, torch.ones(n_k - wv.shape[0], device=wv.device, dtype=wv.dtype)])
            else:
                wv = wv[:n_k]
            log_bias = torch.log(wv.clamp(min=1e-6)).view(1, 1, 1, -1)
            if B >= 2:
                half = B // 2
                scores = torch.cat([scores[:half], scores[half:] + log_bias], dim=0)
            else:
                scores = scores + log_bias

        if attention_mask is not None:
            if attention_mask.ndim == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            scores = scores + attention_mask

        attn_w = torch.softmax(scores.float(), dim=-1).to(query.dtype)
        out = torch.matmul(attn_w, value)
        out = out.transpose(1, 2).reshape(B, -1, heads * head_dim).to(query.dtype)
        out = attn.to_out[1](attn.to_out[0](out))

        if input_ndim == 4:
            out = out.transpose(-1, -2).reshape(B, C, H, W)
        if attn.residual_connection:
            out = out + residual
        return out / attn.rescale_output_factor


def find_token_spans(full_prompt: str, concept_phrases: Dict[str, str], tokenizer) -> Dict[str, List[int]]:
    """Sliding-window exact token match, returning {name: [token_indices]}."""
    full_ids: List[int] = tokenizer(
        full_prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True
    )["input_ids"]

    spans: Dict[str, List[int]] = {}
    for name, phrase in concept_phrases.items():
        phrase_ids = tokenizer(phrase.strip(), add_special_tokens=False)["input_ids"]
        found: List[int] = []
        if not phrase_ids:
            spans[name] = found
            continue
        for start in range(1, len(full_ids) - len(phrase_ids) + 1):
            if full_ids[start : start + len(phrase_ids)] == phrase_ids:
                found.extend(range(start, start + len(phrase_ids)))
                # No break — keep scanning for further occurrences
        spans[name] = sorted(set(found))
    return spans

def build_weight_schedule(
    total_steps: int,
    bg_steps: int,
    step_allocations: Dict[str, Tuple[int, int]],
    token_spans: Dict[str, List[int]],
    n_tokens: int = 77,
    bg_boost: float = BG_BOOST,
    obj_suppress: float = OBJ_SUPPRESS,
    ramp_len: int = RAMP_LEN,
) -> List[torch.Tensor]:
    """Return one ``[n_tokens]`` weight tensor per denoising step (Eq. 3)."""
    bg_idx = set(token_spans.get("background", []))
    obj_idx = {name: set(token_spans.get(name, [])) for name in step_allocations}

    schedule: List[torch.Tensor] = []
    for step in range(total_steps):
        w = np.ones(n_tokens, dtype=np.float32)

        for idx in bg_idx:
            if 0 <= idx < n_tokens:
                w[idx] = bg_boost if step < bg_steps else 1.0

        for name, (start, end) in step_allocations.items():
            for idx in obj_idx.get(name, set()):
                if not (0 <= idx < n_tokens):
                    continue
                if step < start:
                    w[idx] = obj_suppress
                elif step < start + ramp_len:
                    frac = (step - start) / max(1, ramp_len - 1)
                    w[idx] = obj_suppress + frac * (1.0 - obj_suppress)
                # else: already fully active (1.0)

        schedule.append(torch.tensor(w, dtype=torch.float32))
    return schedule
