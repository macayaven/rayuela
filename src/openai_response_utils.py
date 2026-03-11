#!/usr/bin/env python3
"""Helpers for OpenAI-compatible chat responses under parser-separated reasoning."""

from __future__ import annotations

from typing import Any


def extract_final_message_content(message: Any, *, context: str) -> str:
    """Return final assistant content or raise if only reasoning was emitted."""
    content = getattr(message, "content", None) or ""
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
