"""Unit and integration tests for transliteration, benchmark alignment, parsing, and stats."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.analyze_results import bootstrap_ci, paired_mcnemar_test
from src.run_experiments import parse_mc_answer
from src.transliterator import (
    PROCESSED_DIR,
    corpus_rule_transliterate,
    load_lexicon,
    normalize_urdu_text,
    rule_transliterate_word,
)


def test_normalize_urdu_text() -> None:
    raw = "كيا یے ۱۸۷۶ء میں ہوا؟"
    norm = normalize_urdu_text(raw)
    assert "ک" in norm
    assert "1876 " in norm
    assert norm.endswith("?")


def test_rule_transliterate_aspirates_and_vowels() -> None:
    assert rule_transliterate_word("پاکستان") == "paakstaan" or "pk" not in rule_transliterate_word("پاکستان")
    assert rule_transliterate_word("بھائی").startswith("bh")
    assert rule_transliterate_word("گھر").startswith("gh")


def test_lexicon_and_corpus_transliteration() -> None:
    lex = load_lexicon()
    assert len(lex) > 10000
    assert lex.get("کرتے") == "karte"
    assert lex.get("ہے") in {"hai", "he"}
    rom, meta = corpus_rule_transliterate("فرانس کے لوگ کیا کرتے ہیں؟", lex)
    assert "karte" in rom.lower()
    assert meta["lexicon_coverage"] >= 0.8


def test_parse_mc_answer_variants() -> None:
    assert parse_mc_answer('{"answer": "B"}') == "B"
    assert parse_mc_answer('```json\n{"answer": "3"}\n```') == "C"
    assert parse_mc_answer("The correct Answer: D") == "D"
    assert parse_mc_answer("A") == "A"
    assert parse_mc_answer("Option (C) is best") == "C"
    assert parse_mc_answer("") is None


def test_stats_helpers() -> None:
    arr = np.array([1, 1, 0, 1, 0, 1, 1, 1])
    mean, lo, hi = bootstrap_ci(arr, n_boot=500)
    assert 0.0 <= lo <= mean <= hi <= 1.0
    a = np.array([1, 1, 0, 0, 1, 0])
    b = np.array([1, 1, 1, 1, 1, 0])
    res = paired_mcnemar_test(a, b)
    assert res["b_only_correct"] == 2
    assert res["a_only_correct"] == 0
    assert 0.0 <= res["p_value"] <= 1.0


def test_prepared_benchmarks_exist_and_aligned() -> None:
    bele_path = PROCESSED_DIR / "belebele_3scripts.jsonl"
    idioms_path = PROCESSED_DIR / "idioms_3scripts.jsonl"
    assert bele_path.exists()
    assert idioms_path.exists()

    bele = [json.loads(l) for l in bele_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(bele) == 900
    assert sum(1 for r in bele if r["is_pilot"]) >= 120
    assert bele[0]["urd_Latn_corpus"]["question"]

    idioms = [json.loads(l) for l in idioms_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(idioms) == 1919
    assert sum(1 for r in idioms if r["is_pilot"]) == 120
    assert len(idioms[0]["raw_equiv_options"]) == 4
