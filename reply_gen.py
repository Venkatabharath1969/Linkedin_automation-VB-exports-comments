"""
reply_gen.py
────────────
Generates contextual LinkedIn comment replies using Gemini AI.
Generic — driven entirely by the profile's reply_persona config.
"""

from __future__ import annotations

import logging
import os

from google import genai
from google.genai import types as genai_types

log = logging.getLogger(__name__)

_MODEL_PRIMARY  = "gemini-2.5-flash"
_MODEL_FALLBACK = "gemini-2.5-flash-lite"


def _client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")
    return genai.Client(api_key=api_key)


def generate_reply(
    comment_text: str,
    post_context: str,
    persona: str,
    max_chars: int = 250,
    signature: str = "",
) -> str:
    """
    Generates a professional, contextual reply to a LinkedIn comment.

    Args:
        comment_text:  The text of the incoming comment.
        post_context:  Brief description of what the post was about.
        persona:       The reply_persona from the profile config.
        max_chars:     Maximum character length of the reply.
        signature:     Optional signature appended to the reply.

    Returns:
        A reply string ready to post, or empty string on failure.
    """
    prompt = (
        f"Someone commented on a LinkedIn post:\n\n"
        f"Post topic: {post_context}\n\n"
        f"Their comment: \"{comment_text}\"\n\n"
        f"Write a short, genuine reply from this perspective:\n{persona}\n\n"
        f"Rules:\n"
        f"- Maximum {max_chars - len(signature) - 2} characters (leave room for signature)\n"
        f"- Do NOT use emojis excessively — maximum 1\n"
        f"- Do NOT start with 'I' or 'We'\n"
        f"- Sound human, warm, professional\n"
        f"- If the comment is a question, answer it briefly\n"
        f"- If it's appreciation, thank them and add one insight\n"
        f"- If it's spam or irrelevant, reply with a polite generic acknowledgement\n"
        f"- Output ONLY the reply text, no quotes, no prefix\n"
    )

    for model in [_MODEL_PRIMARY, _MODEL_FALLBACK]:
        try:
            config = genai_types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=512,
                top_p=0.95,
            )
            response = _client().models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            text = response.text.strip()
            # Enforce length
            if len(text) + len(signature) + 2 > max_chars:
                text = text[: max_chars - len(signature) - 5].rsplit(" ", 1)[0] + "…"
            if signature:
                text = text + "\n" + signature
            log.info("  Reply generated (%d chars, model=%s)", len(text), model)
            return text
        except Exception as e:
            log.warning("  generate_reply failed on %s: %s", model, e)

    log.error("  All models failed — skipping reply")
    return ""
