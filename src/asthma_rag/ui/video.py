"""Gradio-safe YouTube video embed helpers.

Produces raw iframe HTML for ``gr.HTML`` — not ``gr.Video``, which does not
support YouTube embeds. The embed URL is always the ``/embed/<id>`` form;
``watch?v=`` URLs are never used as the iframe source.
"""

from __future__ import annotations

import os
import re

# American Lung Association inhaler how-to video.
INHALER_VIDEO_ID: str = os.getenv("INHALER_VIDEO_ID", "TuzCfpeieFA")

_YOUTUBE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")

# Explicitly safe characters for attribute values inside double quotes.
_ALLOWED_TITLE_CHARS = re.compile(r'[^A-Za-z0-9 _.,:;\-()/&\'"]')


def is_youtube_id(video_id: str) -> bool:
    """Return True if ``video_id`` is a valid 11-char YouTube video ID."""
    return bool(_YOUTUBE_ID_RE.fullmatch(video_id))


def build_youtube_embed(
    video_id: str,
    title: str = "Inhaler how-to video",
    width: int = 560,
    height: int = 315,
) -> str:
    """Build a Gradio-safe iframe embed for a YouTube video.

    Raises:
        ValueError: if ``video_id`` is not a valid 11-char YouTube ID, or if
            ``width``/``height`` are not positive ints.
    """
    if not is_youtube_id(video_id):
        raise ValueError(
            f"Invalid YouTube video ID {video_id!r}: expected 11 chars of "
            "[A-Za-z0-9_-]"
        )
    if width <= 0 or height <= 0:
        raise ValueError(f"width and height must be positive, got {width}x{height}")
    if not isinstance(width, int) or not isinstance(height, int):
        raise ValueError("width and height must be ints")

    # HTML-escape attribute values (defends against quote/XSS injection).
    escaped_title = (
        title.replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

    return (
        '<iframe width="{w}" height="{h}" src="https://www.youtube.com/embed/'
        '{vid}" title="{title}" frameborder="0" '
        'allow="accelerometer; autoplay; clipboard-write; encrypted-media; '
        'gyroscope; picture-in-picture; web-share" '
        'referrerpolicy="strict-origin-when-cross-origin" '
        'allowfullscreen></iframe>'
    ).format(w=width, h=height, vid=video_id, title=escaped_title)
