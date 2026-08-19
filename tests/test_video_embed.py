"""Tests for the Gradio-safe YouTube embed helper."""

import re

import pytest

from asthma_rag.ui.video import (
    INHALER_VIDEO_ID,
    build_youtube_embed,
    is_youtube_id,
)


def test_default_inhaler_video_id():
    assert INHALER_VIDEO_ID == "TuzCfpeieFA"


def test_embed_contains_expected_youtube_src():
    html = build_youtube_embed("TuzCfpeieFA")
    assert "youtube.com/embed/TuzCfpeieFA" in html


def test_embed_contains_iframe_and_allowfullscreen():
    html = build_youtube_embed("TuzCfpeieFA")
    assert "<iframe" in html
    assert "allowfullscreen" in html
    assert "youtube.com/embed/" in html
    assert "watch?v=" not in html


def test_is_youtube_id_accepts_valid_ids():
    assert is_youtube_id("TuzCfpeieFA")
    assert is_youtube_id("dQw4w9WgXcQ")
    assert is_youtube_id("aB1_x-yZ-0q")


def test_is_youtube_id_rejects_malformed():
    assert not is_youtube_id("TuzCfpeie")  # too short
    assert not is_youtube_id("TuzCfpeieFAA")  # too long
    assert not is_youtube_id("Tuz CfpeieF")  # space
    assert not is_youtube_id("watch?v=TuzCfpeieFA")  # URL, not an ID
    assert not is_youtube_id("")  # empty


@pytest.mark.parametrize(
    "bad_id",
    [
        "watch?v=TuzCfpeieFA",
        "https://www.youtube.com/watch?v=TuzCfpeieFA",
        "short",
        "TuzCfpeieFA-extra",
        "bad id here!!",
        "",
    ],
)
def test_build_embed_rejects_malformed_ids(bad_id):
    with pytest.raises(ValueError):
        build_youtube_embed(bad_id)


def test_build_embed_rejects_bad_dimensions():
    with pytest.raises(ValueError):
        build_youtube_embed("TuzCfpeieFA", width=0)
    with pytest.raises(ValueError):
        build_youtube_embed("TuzCfpeieFA", height=-5)
    with pytest.raises(ValueError):
        build_youtube_embed("TuzCfpeieFA", width=560.5)


def test_embed_escapes_unsafe_title():
    html = build_youtube_embed("TuzCfpeieFA", title='"><script>alert(1)</script>')
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_embed_title_has_no_unescaped_angle_brackets():
    html = build_youtube_embed("TuzCfpeieFA", title="How <to> use & inhaler \"now\"")
    match = re.search(r'title="([^"]*)"', html)
    assert match is not None
    title_attr = match.group(1)
    assert "<" not in title_attr and ">" not in title_attr
    assert "&lt;to&gt;" in title_attr
    assert "&amp;" in title_attr
    assert "&quot;now&quot;" in title_attr
