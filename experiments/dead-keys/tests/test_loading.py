"""Tests for deadkeys.common.loading — model tag registrations and invariants."""

from deadkeys.common.loading import DROPPED_ROPE_TAGS, uses_dropped_rope


def test_rope_recal_not_in_dropped_rope_tags():
    """G-ctrl.1 depends on this exclusion — `qwen3-rope-recal` is a RoPE-active control,
    not a dropped-RoPE variant, so it MUST NOT be in DROPPED_ROPE_TAGS."""
    assert "qwen3-rope-recal" not in DROPPED_ROPE_TAGS
    assert uses_dropped_rope("qwen3-rope-recal") is False
