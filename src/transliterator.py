"""Step 1: Corpus-backed (Dakshina + Roman-Urdu-Parl) & hybrid Urdu -> Roman Urdu transliterator."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from rapidfuzz.distance import Levenshtein

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

URDU_CHAR_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]")
URDU_WORD_OR_OTHER_RE = re.compile(
    r"([\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]+|[0-9]+|[A-Za-z]+|[^\s])"
)

BIDI_AND_TATWEEL_RE = re.compile(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069\u0640]")
DIACRITICS_RE = re.compile(r"[\u064B-\u065F\u0670]")

CHAR_NORMALIZE_MAP = str.maketrans(
    {
        "ك": "ک",
        "ي": "ی",
        "ى": "ی",
        "ئ": "ئ",
        "أ": "ا",
        "إ": "ا",
        "ٱ": "ا",
        "ة": "ہ",
        "ه": "ہ",
        "ۀ": "ہ",
        "ؤ": "ؤ",
        "۔": ".",
        "،": ",",
        "؟": "?",
        "؛": ";",
        "٪": "%",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "۰": "0",
        "۱": "1",
        "۲": "2",
        "۳": "3",
        "۴": "4",
        "۵": "5",
        "۶": "6",
        "۷": "7",
        "۸": "8",
        "۹": "9",
        "٠": "0",
        "١": "1",
        "٢": "2",
        "٣": "3",
        "٤": "4",
        "٥": "5",
        "٦": "6",
        "٧": "7",
        "٨": "8",
        "٩": "9",
        "\u200c": " ",  # ZWNJ as word separator for cleaner Roman tokens
    }
)

# Urdu aspirated consonant digraphs (consonant + do-chashmi he 'ھ')
ASPIRATED_DIGRAPHS: dict[str, str] = {
    "بھ": "bh",
    "پھ": "ph",
    "تھ": "th",
    "ٹھ": "th",
    "جھ": "jh",
    "چھ": "chh",
    "دھ": "dh",
    "ڈھ": "dh",
    "ڑھ": "rh",
    "کھ": "kh",
    "گھ": "gh",
    "لھ": "lh",
    "مھ": "mh",
    "نھ": "nh",
    "وھ": "wh",
    "یھ": "yh",
}

CONSONANT_MAP: dict[str, str] = {
    "ب": "b",
    "پ": "p",
    "ت": "t",
    "ٹ": "t",
    "ث": "s",
    "ج": "j",
    "چ": "ch",
    "ح": "h",
    "خ": "kh",
    "د": "d",
    "ڈ": "d",
    "ذ": "z",
    "ر": "r",
    "ڑ": "r",
    "ز": "z",
    "ژ": "zh",
    "س": "s",
    "ش": "sh",
    "ص": "s",
    "ض": "z",
    "ط": "t",
    "ظ": "z",
    "غ": "gh",
    "ف": "f",
    "ق": "q",
    "ک": "k",
    "گ": "g",
    "ل": "l",
    "م": "m",
    "ن": "n",
    "ں": "n",
    "ھ": "h",
    "ء": "",
}

DIACRITIC_VOWELS: dict[str, str] = {
    "\u064e": "a",  # zabar
    "\u0650": "i",  # zer
    "\u064f": "u",  # pesh
    "\u064b": "an",
    "\u064d": "in",
    "\u064c": "un",
    "\u0670": "a",
}


def normalize_urdu_text(text: str) -> str:
    """Normalize Arabic/Urdu variant codepoints, digits, and punctuation."""
    text = BIDI_AND_TATWEEL_RE.sub("", text)
    text = text.translate(CHAR_NORMALIZE_MAP)
    # Remove Urdu honorific / era suffix 'ء' immediately following digits (e.g. 1876ء -> 1876)
    text = re.sub(r"(\d)ء", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_urdu_word(word: str) -> str:
    """Strip diacritics and normalize a single Urdu word for lexicon lookup."""
    w = normalize_urdu_text(word)
    w = DIACRITICS_RE.sub("", w)
    return w


@lru_cache(maxsize=65536)
def rule_transliterate_word(word: str) -> str:
    """Deterministic contextual Perso-Arabic -> Roman Urdu fallback for OOV words."""
    clean = normalize_urdu_text(word)
    if not clean:
        return ""
    if not URDU_CHAR_RE.search(clean):
        return clean

    out: list[str] = []
    n = len(clean)
    i = 0
    while i < n:
        ch = clean[i]
        nxt = clean[i + 1] if i + 1 < n else ""
        prev = clean[i - 1] if i > 0 else ""

        if ch in DIACRITIC_VOWELS:
            out.append(DIACRITIC_VOWELS[ch])
            i += 1
            continue
        if ch == "\u0651":  # shadda: geminate previous consonant
            if out and out[-1] and out[-1][-1].isalpha():
                out.append(out[-1][-1])
            i += 1
            continue
        if ch in ("\u0652", "\u0654", "\u0655"):
            i += 1
            continue

        # Digraph aspiration
        if nxt == "ھ" and (ch + nxt) in ASPIRATED_DIGRAPHS:
            out.append(ASPIRATED_DIGRAPHS[ch + nxt])
            i += 2
            continue

        if ch == "آ":
            out.append("aa")
        elif ch == "ا":
            if i == 0:
                if nxt == "و":
                    out.append("o")
                    i += 2
                    continue
                if nxt == "ی":
                    out.append("ai")
                    i += 2
                    continue
                out.append("a")
            elif i == n - 1:
                out.append("a")
            else:
                out.append("aa" if prev in CONSONANT_MAP else "a")
        elif ch == "ع":
            if i == 0 or i == n - 1:
                out.append("a")
            else:
                out.append("a")
        elif ch == "و":
            if i == 0:
                out.append("w")
            elif i == n - 1:
                out.append("o")
            else:
                out.append("oo" if prev in CONSONANT_MAP else "o")
        elif ch == "ؤ":
            out.append("o")
        elif ch in ("ی", "ئ"):
            if i == 0:
                out.append("y")
            elif i == n - 1:
                out.append("i")
            else:
                out.append("ee" if nxt in CONSONANT_MAP else "i")
        elif ch == "ے":
            out.append("ay" if i == n - 1 else "e")
        elif ch == "ہ":
            if i == 0:
                out.append("h")
            elif i == n - 1 and prev in CONSONANT_MAP:
                out.append("a")
            else:
                out.append("h")
        elif ch in CONSONANT_MAP:
            out.append(CONSONANT_MAP[ch])
        else:
            out.append(ch)
        i += 1

    res = "".join(out)
    # Collapse 3+ repeated vowels/consonants
    res = re.sub(r"([a-z])\1{2,}", r"\1\1", res)
    return res


def _is_valid_roman_token(roman_tok: str) -> bool:
    """Reject noisy numeric or non-Latin alignments from parallel corpora."""
    if not roman_tok:
        return False
    if URDU_CHAR_RE.search(roman_tok):
        return False
    # Reject pure digits mapped to non-digit Urdu words (e.g., 'ایک' -> '1' in parl)
    if roman_tok.isdigit():
        return False
    return bool(re.search(r"[A-Za-z]", roman_tok))


@lru_cache(maxsize=65536)
def _consonant_skeleton(roman: str) -> str:
    """Normalize a Roman string into a coarse phonetic consonant skeleton for alignment filtering."""
    s = roman.lower()
    s = s.replace("ph", "f").replace("gh", "g").replace("kh", "k").replace("sh", "s").replace("ch", "c")
    s = s.replace("th", "t").replace("dh", "d").replace("zh", "z").replace("bh", "b")
    s = s.replace("c", "k").replace("q", "k").replace("v", "w").replace("x", "ks")
    # Strip vowels and collapse consecutive duplicates
    s = re.sub(r"[aeiouyh']", "", s)
    s = re.sub(r"(.)\1+", r"\1", s)
    return s


def _is_phonetically_compatible(u_word: str, r_tok: str) -> bool:
    """Check whether a Roman token plausibly matches the Urdu word's phonetic form."""
    if not _is_valid_roman_token(r_tok):
        return False
    rule_rom = rule_transliterate_word(u_word)
    u_skel = _consonant_skeleton(rule_rom)
    r_skel = _consonant_skeleton(r_tok)
    if not u_skel or not r_skel:
        # Short vowel-heavy words like 'او', 'یہ', 'ہے', 'کی', 'کو'
        return abs(len(rule_rom) - len(r_tok)) <= 3
    dist = Levenshtein.distance(u_skel, r_skel)
    skel_sim = 1.0 - dist / max(len(u_skel), len(r_skel), 1)
    return skel_sim >= 0.45


def build_lexicon(
    dakshina_train_rows: int = 8000,
    min_parl_count: int = 2,
    save_path: Path | None = None,
) -> dict[str, str]:
    """
    Build a high-coverage Urdu -> Roman Urdu word lexicon from:
      1. Dakshina Urdu token alignments (rows 0..dakshina_train_rows) — highest priority
      2. Deduplicated Roman-Urdu-Parl train_sample + small_validation_set filtered by
         phonetic consonant-skeleton compatibility to eliminate alignment-shift noise.
    Held-out Dakshina rows (8000+) and Roman-Urdu-Parl test sets are never touched.
    """
    dak_counts: dict[str, Counter[str]] = defaultdict(Counter)
    parl_counts: dict[str, Counter[str]] = defaultdict(Counter)

    # 1. Dakshina Urdu train slice (human-curated token alignments)
    dakshina_path = RAW_DIR / "dakshina" / "ur.parquet"
    if dakshina_path.exists():
        df = pd.read_parquet(dakshina_path).iloc[:dakshina_train_rows]
        for alignments in df["token_alignment"]:
            if alignments is None:
                continue
            for pair in alignments:
                u_tok = normalize_urdu_word(str(pair.get("native_token", "")))
                r_tok = str(pair.get("romanized_token", "")).strip()
                if u_tok and URDU_CHAR_RE.search(u_tok) and _is_valid_roman_token(r_tok):
                    dak_counts[u_tok][r_tok.lower()] += 1

    # 2. Roman-Urdu-Parl train_sample + small_validation_set (deduplicated + phonetically filtered)
    parl_dir = RAW_DIR / "roman_urdu_parl"
    parl_files = list(parl_dir.glob("train_sample_*.csv")) + [parl_dir / "small_validation_set.csv"]
    seen_sentences: set[str] = set()
    for pfile in parl_files:
        if not pfile.exists():
            continue
        with pfile.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) < 2:
                    continue
                u_line = normalize_urdu_text(row[0])
                if not u_line or u_line in seen_sentences:
                    continue
                seen_sentences.add(u_line)
                u_tokens = u_line.split()
                r_tokens = row[1].strip().split()
                if not u_tokens or len(u_tokens) != len(r_tokens):
                    continue
                for u_raw, r_raw in zip(u_tokens, r_tokens):
                    u_clean = normalize_urdu_word(re.sub(r"[^\w\u0600-\u06FF]", "", u_raw))
                    r_clean = re.sub(r"^[^A-Za-z0-9']+|[^A-Za-z0-9']+$", "", r_raw)
                    if (
                        u_clean
                        and URDU_CHAR_RE.search(u_clean)
                        and _is_phonetically_compatible(u_clean, r_clean)
                    ):
                        parl_counts[u_clean][r_clean.lower()] += 1

    lexicon: dict[str, str] = {}
    for u_word, ctr in parl_counts.items():
        best_rom, best_cnt = ctr.most_common(1)[0]
        if best_cnt >= min_parl_count:
            lexicon[u_word] = best_rom
    # Dakshina gold alignments always take priority
    for u_word, ctr in dak_counts.items():
        lexicon[u_word] = ctr.most_common(1)[0][0]

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(lexicon, ensure_ascii=False, indent=2), encoding="utf-8")

    return lexicon


def load_lexicon(path: Path | None = None) -> dict[str, str]:
    """Load saved lexicon JSON or build it if absent."""
    target = path or (PROCESSED_DIR / "roman_lexicon.json")
    if target.exists():
        return json.loads(target.read_text(encoding="utf-8"))
    return build_lexicon(save_path=target)


def corpus_rule_transliterate(
    text: str, lexicon: dict[str, str]
) -> tuple[str, dict[str, Any]]:
    """
    Transliterate Urdu text to Roman Urdu using the corpus lexicon + rule fallback.
    Returns (roman_text, metadata_with_oov_stats).
    """
    norm = normalize_urdu_text(text)
    if not norm:
        return "", {"total_urdu_tokens": 0, "lexicon_hits": 0, "oov_tokens": []}

    parts: list[str] = []
    total_urdu = 0
    hits = 0
    oov_tokens: list[str] = []

    for tok in URDU_WORD_OR_OTHER_RE.findall(norm):
        if URDU_CHAR_RE.search(tok):
            total_urdu += 1
            key = normalize_urdu_word(tok)
            if key in lexicon:
                hits += 1
                parts.append(lexicon[key])
            else:
                oov_tokens.append(key)
                parts.append(rule_transliterate_word(tok))
        else:
            parts.append(tok)

    # Detokenize punctuation cleanly
    joined = " ".join(parts)
    joined = re.sub(r"\s+([.,?;:!%)\]}])", r"\1", joined)
    joined = re.sub(r"([(\[{])\s+", r"\1", joined)
    joined = re.sub(r"\s+'s\b", "'s", joined)
    joined = re.sub(r"\s+", " ", joined).strip()

    # Capitalize sentence starts
    joined = re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), joined)

    meta = {
        "total_urdu_tokens": total_urdu,
        "lexicon_hits": hits,
        "lexicon_coverage": round(hits / total_urdu, 4) if total_urdu else 1.0,
        "oov_tokens": oov_tokens,
    }
    return joined, meta


def evaluate_transliterator(lexicon: dict[str, str]) -> dict[str, dict[str, float]]:
    """
    Evaluate corpus_rule_transliterate on strictly held-out Dakshina (rows 8000+)
    and Roman-Urdu-Parl small_test_set.csv using CER and normalized similarity.
    """
    results: dict[str, dict[str, float]] = {}

    def score_pairs(pairs: list[tuple[str, str]]) -> dict[str, float]:
        total_chars = 0
        total_edit = 0
        sims: list[float] = []
        tok_hits = 0
        tok_total = 0
        for native, gold_rom in pairs:
            pred_rom, meta = corpus_rule_transliterate(native, lexicon)
            p_low = re.sub(r"\s+", " ", pred_rom.lower().strip())
            g_low = re.sub(r"\s+", " ", gold_rom.lower().strip())
            dist = Levenshtein.distance(p_low, g_low)
            total_edit += dist
            total_chars += max(len(g_low), 1)
            sims.append(1.0 - dist / max(len(p_low), len(g_low), 1))
            tok_hits += meta["lexicon_hits"]
            tok_total += meta["total_urdu_tokens"]
        return {
            "n_sentences": float(len(pairs)),
            "cer": round(total_edit / max(total_chars, 1), 4),
            "char_similarity": round(sum(sims) / max(len(sims), 1), 4),
            "lexicon_token_coverage": round(tok_hits / max(tok_total, 1), 4),
        }

    dakshina_path = RAW_DIR / "dakshina" / "ur.parquet"
    if dakshina_path.exists():
        df = pd.read_parquet(dakshina_path).iloc[8000:]
        pairs = list(zip(df["native"].astype(str), df["romanized"].astype(str)))
        results["dakshina_heldout_test"] = score_pairs(pairs)

    parl_test = RAW_DIR / "roman_urdu_parl" / "small_test_set.csv"
    if parl_test.exists():
        with parl_test.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            pairs = [(r[0], r[1]) for r in reader if len(r) >= 2 and r[0].strip() and r[1].strip()]
        results["roman_urdu_parl_small_test"] = score_pairs(pairs)

    return results


HYBRID_SYSTEM_PROMPT = (
    "You are an expert Urdu-to-Roman-Urdu transliterator following Pakistani Roman Urdu "
    "orthography (as in Dakshina and Roman-Urdu-Parl). Given an Urdu (Nastaliq) text and a "
    "corpus-lexicon draft transliteration, output ONLY a clean, natural, faithful Roman Urdu "
    "transliteration in Latin script. Rules:\n"
    "1. Do NOT translate into English — keep Urdu words in Roman Urdu.\n"
    "2. Fix vowelization and OOV words from the draft using the original Urdu.\n"
    "3. Preserve numbers, punctuation, and standard English proper nouns.\n"
    "4. Return JSON with keys matching the input JSON."
)


def hybrid_normalize_fields(
    urdu_fields: dict[str, str],
    lexicon: dict[str, str],
    gemini_client: Any,
    max_output_tokens: int = 1024,
) -> dict[str, Any]:
    """
    Produce both corpus_rule Roman Urdu and Gemini-verified/normalized Roman Urdu
    for an arbitrary dict of Urdu fields (e.g., passage alone or question+choices).
    """
    corpus_draft: dict[str, str] = {}
    cov_stats: list[float] = []
    for f, text in urdu_fields.items():
        rom, meta = corpus_rule_transliterate(text, lexicon)
        corpus_draft[f] = rom
        cov_stats.append(meta["lexicon_coverage"])

    keys_str = ", ".join(urdu_fields.keys())
    prompt = (
        "Examples of standard Dakshina / Roman-Urdu-Parl transliteration:\n"
        "- فرانس کے ووٹنگ کے عمل -> France ke voting ke amal\n"
        "- اٹھارویں صدی کے وسط میں ایک انسائیکلوپیڈیا -> atharwin sadi ke wast mein aik encyclopedia\n"
        "- یہ تنخواہ تو میرے اخراجات کے مقابلے میں -> yeh tankhwa to mere ikhrajat ke muqable mein\n\n"
        "Convert the following Urdu fields into natural Pakistani Roman Urdu, using the corpus draft "
        "as an anchor and fixing OOV/vowel ambiguities. Return ONLY valid JSON with the exact keys: "
        f"{keys_str}.\n\n"
        f"URDU:\n{json.dumps(urdu_fields, ensure_ascii=False)}\n\n"
        f"CORPUS_DRAFT:\n{json.dumps(corpus_draft, ensure_ascii=False)}"
    )

    resp = gemini_client.generate(
        prompt=prompt,
        system_instruction=HYBRID_SYSTEM_PROMPT,
        json_mode=True,
        temperature=0.0,
        max_output_tokens=max_output_tokens,
    )

    hybrid_fields = dict(corpus_draft)
    hybrid_valid = False
    try:
        parsed = json.loads(resp.text)
        if isinstance(parsed, dict) and all(
            isinstance(parsed.get(f), str) and parsed[f].strip() and not URDU_CHAR_RE.search(parsed[f])
            for f in urdu_fields
        ):
            hybrid_fields = {f: parsed[f].strip() for f in urdu_fields}
            hybrid_valid = True
    except json.JSONDecodeError:
        pass

    return {
        "corpus_draft": corpus_draft,
        "hybrid": hybrid_fields,
        "hybrid_valid": hybrid_valid,
        "mean_lexicon_coverage": round(sum(cov_stats) / max(len(cov_stats), 1), 4),
    }


def hybrid_normalize_belebele_item(
    item: dict[str, Any],
    lexicon: dict[str, str],
    gemini_client: Any,
) -> dict[str, Any]:
    """Convenience wrapper normalizing all 6 Belebele fields."""
    fields = ["flores_passage", "question", "mc_answer1", "mc_answer2", "mc_answer3", "mc_answer4"]
    return hybrid_normalize_fields({f: item[f] for f in fields}, lexicon, gemini_client, max_output_tokens=1500)


if __name__ == "__main__":
    lex = build_lexicon(save_path=PROCESSED_DIR / "roman_lexicon.json")
    print(f"Built lexicon with {len(lex):,} Urdu -> Roman Urdu entries.")
    metrics = evaluate_transliterator(lex)
    print("Held-out transliteration metrics:")
    print(json.dumps(metrics, indent=2))
