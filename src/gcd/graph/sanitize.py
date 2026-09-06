"""Remove foreground object mentions from an LLM-extracted background string.

The background-extraction prompt (``SYS_PROMPT_BACKGROUND``) already instructs
the LLM not to mention foreground objects, but this regex pass is a defensive
second line to guarantee Phase 1 ("background only") never leaks object
names or attribute values into the conditioning prompt.
"""

from __future__ import annotations

import re
from typing import Dict, List


def _split_camel_case(text: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", " ", text)


def sanitize_background(
    background: str,
    node_names: List[str],
    attributes: Dict[str, Dict[str, str]],
) -> str:
    """Strip object names and attribute values from a background description."""
    if not background:
        return ""

    text = background

    for name in node_names:
        if not name:
            continue
        spaced_name = _split_camel_case(name)
        text = re.sub(r"\b" + re.escape(name) + r"\b", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\b" + re.escape(spaced_name) + r"\b", "", text, flags=re.IGNORECASE)
        parts = spaced_name.split()
        if len(parts) > 1:
            text = re.sub(r"\b" + re.escape(parts[-1]) + r"\b", "", text, flags=re.IGNORECASE)

    if isinstance(attributes, dict):
        for attrs in attributes.values():
            if not isinstance(attrs, dict):
                continue
            for value in attrs.values():
                if not value:
                    continue
                clean_value = str(value).replace("_", " ")
                text = re.sub(r"\b" + re.escape(clean_value) + r"\b", "", text, flags=re.IGNORECASE)

    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.])", r"\1", text)
    return text
