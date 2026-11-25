"""Build aligned 3-script (English, Urdu Nastaliq, Roman Urdu) Belebele & Urdu Idioms benchmarks."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import openpyxl
from tqdm import tqdm

from src.gemini_client import CachedGeminiClient, DEFAULT_MODEL
from src.transliterator import (
    PROCESSED_DIR,
    RAW_DIR,
    corpus_rule_transliterate,
    hybrid_normalize_fields,
    load_lexicon,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "data" / "results"

STOPWORDS = {
    "a", "an", "the", "in", "on", "at", "of", "to", "for", "and", "or", "is",
    "are", "be", "not", "no", "with", "by", "from", "as", "it", "one", "own",
    "your", "you", "his", "her", "their", "its", "than", "that", "this", "who",
    "what", "when", "where", "do", "does", "did", "make", "out", "into", "up",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _select_pilot_belebele_links(rows: list[dict[str, Any]], target_count: int = 120) -> set[str]:
    """Select full passage links deterministically so pilot items cover diverse passages."""
    by_link: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_link[r["link"]].append(r)
    links = sorted(by_link.keys())
    rng = random.Random(42)
    rng.shuffle(links)
    selected: set[str] = set()
    total = 0
    for lk in links:
        selected.add(lk)
        total += len(by_link[lk])
        if total >= target_count:
            break
    return selected


def prepare_belebele(
    hybrid: bool = False,
    hybrid_scope: str = "pilot",
    model: str = DEFAULT_MODEL,
    workers: int = 2,
) -> list[dict[str, Any]]:
    """Align all 900 Belebele English & Urdu items and convert Urdu -> Roman Urdu."""
    ur_rows = _read_jsonl(RAW_DIR / "belebele" / "urd_Arab.jsonl")
    en_rows = _read_jsonl(RAW_DIR / "belebele" / "eng_Latn.jsonl")

    en_by_key = {(r["link"], int(r["question_number"])): r for r in en_rows}
    ur_rows_sorted = sorted(ur_rows, key=lambda r: (r["link"], int(r["question_number"])))
    pilot_links = _select_pilot_belebele_links(ur_rows_sorted, target_count=120)

    lexicon = load_lexicon()
    fields = ["flores_passage", "question", "mc_answer1", "mc_answer2", "mc_answer3", "mc_answer4"]
    q_fields = ["question", "mc_answer1", "mc_answer2", "mc_answer3", "mc_answer4"]

    out_path = PROCESSED_DIR / "belebele_3scripts.jsonl"
    existing_by_id: dict[str, dict[str, Any]] = {}
    if out_path.exists():
        for rec in _read_jsonl(out_path):
            existing_by_id[rec["item_id"]] = rec

    client = (
        CachedGeminiClient(
            cache_path=RESULTS_DIR / f"cache_translit_{model}.jsonl",
            model=model,
        )
        if hybrid
        else None
    )

    # First normalize unique passages once (cuts passage tokens ~2x and avoids duplicate calls)
    passage_by_link: dict[str, str] = {}
    if hybrid and client is not None:
        target_links: dict[str, str] = {}
        for idx, ur in enumerate(ur_rows_sorted):
            item_id = f"belebele_{idx:04d}"
            is_pilot = ur["link"] in pilot_links
            prev = existing_by_id.get(item_id)
            if prev and prev.get("transliteration_source") == "hybrid_gemini":
                passage_by_link[ur["link"]] = prev["urd_Latn"]["flores_passage"]
            elif (hybrid_scope == "all" or (hybrid_scope == "pilot" and is_pilot)) and ur["link"] not in target_links:
                target_links[ur["link"]] = ur["flores_passage"]

        missing_links = [(lk, txt) for lk, txt in target_links.items() if lk not in passage_by_link]
        if missing_links:
            def norm_passage(pair: tuple[str, str]) -> tuple[str, str]:
                lk, txt = pair
                res = hybrid_normalize_fields({"flores_passage": txt}, lexicon, client, max_output_tokens=1024)
                return lk, res["hybrid"]["flores_passage"] if res["hybrid_valid"] else ""

            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                for lk, rom_p in tqdm(
                    pool.map(norm_passage, missing_links),
                    total=len(missing_links),
                    desc="Belebele unique passages",
                ):
                    if rom_p:
                        passage_by_link[lk] = rom_p

    def build_record(idx_and_ur: tuple[int, dict[str, Any]]) -> dict[str, Any]:
        idx, ur = idx_and_ur
        key = (ur["link"], int(ur["question_number"]))
        en = en_by_key[key]
        assert str(ur["correct_answer_num"]) == str(en["correct_answer_num"])

        item_id = f"belebele_{idx:04d}"
        is_pilot = ur["link"] in pilot_links

        corpus_fields: dict[str, str] = {}
        covs: list[float] = []
        for f in fields:
            rom, meta = corpus_rule_transliterate(ur[f], lexicon)
            corpus_fields[f] = rom
            covs.append(meta["lexicon_coverage"])
        mean_cov = round(sum(covs) / len(covs), 4)

        prev = existing_by_id.get(item_id)
        if prev and prev.get("transliteration_source") == "hybrid_gemini":
            urd_latn = prev["urd_Latn"]
            trans_source = "hybrid_gemini"
        else:
            urd_latn = dict(corpus_fields)
            trans_source = "corpus_rule"

        if hybrid and client is not None and (hybrid_scope == "all" or (hybrid_scope == "pilot" and is_pilot)):
            if trans_source != "hybrid_gemini" and ur["link"] in passage_by_link:
                q_res = hybrid_normalize_fields(
                    {f: ur[f] for f in q_fields}, lexicon, client, max_output_tokens=512
                )
                if q_res["hybrid_valid"]:
                    urd_latn = {"flores_passage": passage_by_link[ur["link"]], **q_res["hybrid"]}
                    trans_source = "hybrid_gemini"

        return {
            "item_id": item_id,
            "link": ur["link"],
            "question_number": int(ur["question_number"]),
            "correct_answer_num": int(ur["correct_answer_num"]),
            "is_pilot": is_pilot,
            "transliteration_source": trans_source,
            "lexicon_coverage": mean_cov,
            "eng_Latn": {f: en[f] for f in fields},
            "urd_Arab": {f: ur[f] for f in fields},
            "urd_Latn_corpus": corpus_fields,
            "urd_Latn": urd_latn,
        }

    indexed = list(enumerate(ur_rows_sorted))
    if hybrid and client is not None:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            records = list(
                tqdm(pool.map(build_record, indexed), total=len(indexed), desc="Belebele Q+options")
            )
    else:
        records = [build_record(pair) for pair in indexed]

    _write_jsonl(out_path, records)
    return records


def _load_valid_idiom_rows() -> list[dict[str, str]]:
    """Load non-empty aligned triples from urdu_idiom_sentences.xlsx & roman_idiom_sentences.xlsx."""
    wb_ur = openpyxl.load_workbook(RAW_DIR / "urdu_idioms" / "urdu_idiom_sentences.xlsx", read_only=True)
    wb_rm = openpyxl.load_workbook(RAW_DIR / "urdu_idioms" / "roman_idiom_sentences.xlsx", read_only=True)
    ws_ur = wb_ur.active
    ws_rm = wb_rm.active

    ur_iter = ws_ur.iter_rows(values_only=True)
    rm_iter = ws_rm.iter_rows(values_only=True)
    next(ur_iter, None)
    next(rm_iter, None)

    seen_eng_idioms: set[str] = set()
    rows: list[dict[str, str]] = []
    for u, r in zip(ur_iter, rm_iter):
        if not u or not r or len(u) < 4 or len(r) < 4:
            continue
        vals_u = [str(x).strip() if x is not None else "" for x in u[:4]]
        vals_r = [str(x).strip() if x is not None else "" for x in r[:4]]
        if any(not v for v in vals_u) or any(not v for v in vals_r):
            continue
        if vals_u[2] != vals_r[2] or vals_u[3] != vals_r[3]:
            continue
        eng_key = vals_u[2].lower()
        if eng_key in seen_eng_idioms:
            continue
        seen_eng_idioms.add(eng_key)
        rows.append(
            {
                "urdu_idiom": vals_u[0],
                "urdu_sentence": vals_u[1],
                "roman_idiom": vals_r[0],
                "roman_sentence": vals_r[1],
                "english_idiom": vals_u[2],
                "english_sentence": vals_u[3],
            }
        )
    return rows


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w not in STOPWORDS and len(w) > 2}


def prepare_idioms(
    build_plain_meanings: bool = False,
    scope: str = "pilot",
    eval_limit: int = 2000,
    pilot_size: int = 120,
    model: str = DEFAULT_MODEL,
    workers: int = 4,
) -> list[dict[str, Any]]:
    """
    Build aligned Urdu Idioms MC-QA records across English, Urdu Nastaliq, and Roman Urdu.
    Each record supports:
      - Task 2B (raw_equiv_choices): 4-choice English idiom match (zero synthetic labels)
      - Task 2A (plain_meaning + meaning_choices): leak-free figurative meaning across all 3 scripts
    """
    raw_rows = _load_valid_idiom_rows()
    rng = random.Random(42)
    indices = list(range(len(raw_rows)))
    rng.shuffle(indices)
    selected_rows = [raw_rows[i] for i in indices[:eval_limit]]

    out_path = PROCESSED_DIR / "idioms_3scripts.jsonl"
    existing_by_id: dict[str, dict[str, Any]] = {}
    if out_path.exists():
        for rec in _read_jsonl(out_path):
            existing_by_id[rec["item_id"]] = rec

    client = (
        CachedGeminiClient(
            cache_path=RESULTS_DIR / f"cache_idiom_meanings_{model}.jsonl",
            model=model,
        )
        if build_plain_meanings
        else None
    )

    # Precompute deterministic distractor indices cleanly before parallel meaning generation
    precomputed: list[dict[str, Any]] = []
    for idx, row in enumerate(selected_rows):
        item_id = f"idiom_{idx:04d}"
        is_pilot = idx < pilot_size
        pool_end = pilot_size if is_pilot else len(selected_rows)
        distractor_pool = [j for j in range(pool_end) if j != idx]
        dist_idxs = rng.sample(distractor_pool, 3)
        raw_opts = [row["english_idiom"]] + [selected_rows[j]["english_idiom"] for j in dist_idxs]
        opt_order = list(range(4))
        rng.shuffle(opt_order)
        raw_shuffled = [raw_opts[k] for k in opt_order]
        raw_correct_num = opt_order.index(0) + 1
        prev = existing_by_id.get(item_id)
        precomputed.append(
            {
                "idx": idx,
                "item_id": item_id,
                "is_pilot": is_pilot,
                "row": row,
                "dist_idxs": dist_idxs,
                "raw_shuffled": raw_shuffled,
                "raw_correct_num": raw_correct_num,
                "prev_plain": prev.get("plain_meaning", "") if prev else "",
            }
        )

    def build_idiom_rec(meta: dict[str, Any]) -> dict[str, Any]:
        plain_meaning = meta["prev_plain"]
        row = meta["row"]
        if (
            build_plain_meanings
            and client is not None
            and not plain_meaning
            and (scope == "all" or (scope == "pilot" and meta["is_pilot"]))
        ):
            forbidden = sorted(_content_words(row["english_idiom"]))
            prompt = (
                "Write a concise (8-16 word) plain-English figurative definition of the idiom "
                f"'{row['english_idiom']}' (Urdu: '{row['urdu_idiom']}'). "
                f"Do NOT use any of these words: {', '.join(forbidden) if forbidden else 'none'}. "
                "Output ONLY the plain definition sentence."
            )
            resp = client.generate(prompt=prompt, temperature=0.0, max_output_tokens=128)
            candidate = resp.text.strip().strip('"').strip(".")
            if candidate:
                plain_meaning = candidate

        return {
            "item_id": meta["item_id"],
            "is_pilot": meta["is_pilot"],
            **row,
            "raw_equiv_options": meta["raw_shuffled"],
            "raw_equiv_correct_num": meta["raw_correct_num"],
            "plain_meaning": plain_meaning,
            "meaning_distractor_indices": meta["dist_idxs"],
        }

    if build_plain_meanings and client is not None:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            records = list(
                tqdm(pool.map(build_idiom_rec, precomputed), total=len(precomputed), desc="Idiom plain meanings")
            )
    else:
        records = [build_idiom_rec(m) for m in precomputed]

    # Populate 4-choice plain-meaning MC options wherever all 4 items have plain_meaning
    for rec in records:
        dist_idxs = rec["meaning_distractor_indices"]
        if rec["plain_meaning"] and all(records[j]["plain_meaning"] for j in dist_idxs):
            opts = [rec["plain_meaning"]] + [records[j]["plain_meaning"] for j in dist_idxs]
            order = list(range(4))
            random.Random(rec["item_id"]).shuffle(order)
            rec["meaning_options"] = [opts[k] for k in order]
            rec["meaning_correct_num"] = order.index(0) + 1
        else:
            rec["meaning_options"] = []
            rec["meaning_correct_num"] = None

    _write_jsonl(out_path, records)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare 3-script Belebele & Urdu Idioms benchmarks.")
    parser.add_argument("--hybrid", action="store_true", help="Run Gemini hybrid normalization on Belebele.")
    parser.add_argument("--build-plain-meanings", action="store_true", help="Generate leak-free plain meanings for Idioms.")
    parser.add_argument("--scope", choices=["pilot", "all"], default="pilot")
    parser.add_argument("--idiom-limit", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    bele = prepare_belebele(
        hybrid=args.hybrid,
        hybrid_scope=args.scope,
        model=args.model,
        workers=args.workers,
    )
    n_pilot_b = sum(1 for r in bele if r["is_pilot"])
    n_hyb = sum(1 for r in bele if r["transliteration_source"] == "hybrid_gemini")
    print(f"Belebele prepared: {len(bele)} total, {n_pilot_b} pilot, {n_hyb} hybrid-normalized.")

    idioms = prepare_idioms(
        build_plain_meanings=args.build_plain_meanings,
        scope=args.scope,
        eval_limit=args.idiom_limit,
        model=args.model,
        workers=args.workers,
    )
    n_pilot_i = sum(1 for r in idioms if r["is_pilot"])
    n_mean = sum(1 for r in idioms if r["meaning_options"])
    print(f"Idioms prepared: {len(idioms)} unique idioms, {n_pilot_i} pilot, {n_mean} with 3-script plain-meaning QA.")


if __name__ == "__main__":
    main()
