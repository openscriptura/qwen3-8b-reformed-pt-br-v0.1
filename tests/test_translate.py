"""Tests for translate_benchmark helpers — quote cleaning only strips true pairs."""

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "translate_benchmark",
    Path(__file__).resolve().parent.parent / "scripts" / "translate_benchmark.py",
)
tb = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tb)


def test_clean_strips_matching_pairs():
    assert tb._clean('"hello"') == "hello"        # straight double pair
    assert tb._clean("'olá'") == "olá"            # straight single pair
    assert tb._clean("“olá”") == "olá"            # curly pair


def test_clean_does_not_strip_mismatched_quotes():
    # old code stripped any leading+trailing quote char even when mismatched —
    # that corrupts a genuine quotation. Now only a true matching pair is stripped.
    assert tb._clean('“olá"') == '“olá"'          # curly-open / straight-close: kept
    assert tb._clean('"olá”') == '"olá”'          # straight-open / curly-close: kept


def test_clean_plain_text_untouched():
    assert tb._clean("  Os seres humanos têm responsabilidades?  ") == "Os seres humanos têm responsabilidades?"
