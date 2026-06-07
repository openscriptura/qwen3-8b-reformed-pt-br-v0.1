"""Tests for canonical content hashing (VALIDATION_REPORT.md M5)."""

from utils.hash import content_hash


def _record(**overrides):
    base = {
        "messages": [{"role": "user", "content": "Quem é Deus?"}],
        "tradition": "reformed",
        "lang": "pt-BR",
    }
    base.update(overrides)
    return base


def test_hash_is_deterministic():
    assert content_hash(_record()) == content_hash(_record())


def test_hash_ignores_non_content_metadata():
    # Extra fields (timestamps, tier, etc.) must not change the hash.
    a = _record()
    b = _record(tier="C", created_at="2026-06-07T00:00:00Z")
    assert content_hash(a) == content_hash(b)


def test_hash_changes_with_content():
    base = content_hash(_record())
    assert content_hash(_record(tradition="lutheran")) != base
    assert content_hash(_record(lang="en")) != base
