#!/usr/bin/env python3
"""Helpers for OpenAI-compatible chat responses under parser-separated reasoning."""

from __future__ import annotations

import re
from typing import Any

_VISIBLE_REASONING_PREFIX_RE = re.compile(
    r"(?im)^\s*(?:\*{0,2}\s*)?"
    r"(?:(?:thinking process|reasoning|analysis|chain of thought|"
    r"an[áa]lisis|comentario|commentary|nota|note|"
    r"cambios realizados|changes made)\s*:\s*(?:\*{0,2}\s*)?"
    r".*?\n\s*\n|\d+\.\s+\*\*?analyze\*\*?)\n?"
)


def strip_visible_reasoning_prefix(content: str) -> str:
    """Remove leading visible reasoning blocks while keeping final passage text."""
    text = content.strip()
    while True:
        match = _VISIBLE_REASONING_PREFIX_RE.match(text)
        if match is None:
            return text
        text = text[match.end() :].strip()


def extract_final_message_content(message: Any, *, context: str) -> str:
    """Return final assistant content or raise if only reasoning was emitted."""
    content = strip_visible_reasoning_prefix(getattr(message, "content", None) or "")
    reasoning = getattr(message, "reasoning_content", None) or getattr(
        message,
        "reasoning",
        None,
    )
    if not content.strip() and isinstance(reasoning, str) and reasoning.strip():
        raise RuntimeError(
            f"model returned reasoning without final content during {context}; "
            "increase the generation token budget or tighten the output contract"
        )
    return content
