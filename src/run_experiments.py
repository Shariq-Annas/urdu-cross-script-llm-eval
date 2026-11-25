"""Steps 2-4: Evaluate Gemini on English, Urdu, Roman Urdu, and Transliterate-First-Then-Answer."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.gemini_client import CachedGeminiClient, DEFAULT_MODEL
from src.transliterator import PROCESSED_DIR

RESULTS_DIR = Path(__file__).resolve().parents[1] / "data" / "results"
LETTERS = ["A", "B", "C", "D"]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def parse_mc_answer(text: str) -> str | None:
    """Extract A/B/C/D answer choice from JSON or free-form model output."""
    if not text:
        return None
    cleaned = text.strip()
    # Strip markdown code fences if present
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()

    # 1. Try JSON object parsing
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            ans = str(obj.get("answer", "")).strip().upper()
            if ans in LETTERS:
                return ans
            if ans in {"1", "2", "3", "4"}:
                return LETTERS[int(ans) - 1]
    except json.JSONDecodeError:
        pass

    # 2. Try explicit patterns like "answer": "B" or Answer: B
    m = re.search(r'(?i)"answer"\s*:\s*"([A-D1-4])"', cleaned)
    if m:
        ch = m.group(1).upper()
        return LETTERS[int(ch) - 1] if ch.isdigit() else ch

    m2 = re.search(r"(?i)\b(?:answer|option|choice)\s*[:\-]?\s*\(?([A-D])\)?\b", cleaned)
    if m2:
        return m2.group(1).upper()

    # 3. Single letter on first or last line
    if cleaned.upper() in LETTERS:
        return cleaned.upper()
    m3 = re.match(r"^\(?([A-D])\)?(?:[\.\):]|\s|$)", cleaned.upper())
    if m3:
        return m3.group(1)
    return None


def format_belebele_prompt(fields: dict[str, str]) -> str:
    return (
        "Read the following passage carefully and choose the single best answer (A, B, C, or D).\n"
        'Respond ONLY with valid JSON: {"answer": "A"|"B"|"C"|"D"}.\n\n'
        f"Passage:\n{fields['flores_passage']}\n\n"
        f"Question:\n{fields['question']}\n\n"
        f"A) {fields['mc_answer1']}\n"
        f"B) {fields['mc_answer2']}\n"
        f"C) {fields['mc_answer3']}\n"
        f"D) {fields['mc_answer4']}"
    )


def format_belebele_tfta_single_prompt(fields: dict[str, str]) -> str:
    return (
        "The following reading comprehension item is written in Roman Urdu (Urdu written in Latin script).\n"
        "Follow these two steps strictly:\n"
        "1. First transliterate the Roman Urdu passage, question, and all four options into native Urdu script (Perso-Arabic / Nastaliq).\n"
        "2. Then read your Urdu transliteration and choose the single best answer (A, B, C, or D).\n\n"
        'Respond ONLY with valid JSON: {"urdu_transliteration": "<passage and question in Urdu script>", "answer": "A"|"B"|"C"|"D"}.\n\n'
        f"Passage:\n{fields['flores_passage']}\n\n"
        f"Question:\n{fields['question']}\n\n"
        f"A) {fields['mc_answer1']}\n"
        f"B) {fields['mc_answer2']}\n"
        f"C) {fields['mc_answer3']}\n"
        f"D) {fields['mc_answer4']}"
    )


def run_belebele_condition(
    items: list[dict[str, Any]],
    condition: str,
    client: CachedGeminiClient,
    workers: int = 2,
    cached_only: bool = False,
) -> list[dict[str, Any]]:
    """
    Evaluate Belebele under one condition:
      eng_Latn | urd_Arab | urd_Latn_corpus | urd_Latn_hybrid |
      urd_Latn_tfta_single | urd_Latn_tfta_twostage
    """
    # Pre-transliterate unique passages once for two-stage TFTA to avoid duplicate passage calls
    tfta_passage_cache: dict[str, tuple[str, int, int, float]] = {}
    if condition == "urd_Latn_tfta_twostage":
        unique_passages = {it["link"]: it["urd_Latn"]["flores_passage"] for it in items}

        def translit_p(pair: tuple[str, str]) -> tuple[str, str, int, int, float]:
            lk, rom_p = pair
            p_prompt = (
                "Transliterate the following Roman Urdu passage into standard Urdu (Perso-Arabic / Nastaliq script). "
                "Do NOT translate into English. Return ONLY valid JSON: {\"flores_passage\": \"...\"}.\n\n"
                + rom_p
            )
            r = client.generate(prompt=p_prompt, json_mode=True, temperature=0.0, max_output_tokens=1200)
            ur_p = rom_p
            try:
                parsed = json.loads(r.text)
                if isinstance(parsed, dict) and isinstance(parsed.get("flores_passage"), str) and parsed["flores_passage"].strip():
                    ur_p = parsed["flores_passage"].strip()
            except Exception:
                pass
            return lk, ur_p, r.input_tokens, r.output_tokens, r.latency_s

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for lk, ur_p, itok, otok, l_s in tqdm(
                pool.map(translit_p, list(unique_passages.items())),
                total=len(unique_passages),
                desc="Belebele-TFTA[passages]",
            ):
                tfta_passage_cache[lk] = (ur_p, itok, otok, l_s)

    def eval_one(item: dict[str, Any]) -> dict[str, Any] | None:
        gold_letter = LETTERS[int(item["correct_answer_num"]) - 1]
        in_tok = 0
        out_tok = 0
        lat = 0.0
        reconstructed_urdu = ""

        def call_or_skip(prompt: str, max_out: int) -> Any:
            if cached_only:
                k = CachedGeminiClient.make_key(client.model, "", prompt, True, True)
                if k not in client._cache or not client._cache[k].get("text"):
                    return None
            return client.generate(prompt=prompt, json_mode=True, temperature=0.0, max_output_tokens=max_out)

        if condition in ("eng_Latn", "urd_Arab", "urd_Latn_corpus"):
            prompt = format_belebele_prompt(item[condition])
            resp = call_or_skip(prompt, 256)
            if resp is None:
                return None
            in_tok, out_tok, lat = resp.input_tokens, resp.output_tokens, resp.latency_s
            raw_text = resp.text
        elif condition == "urd_Latn_hybrid":
            prompt = format_belebele_prompt(item["urd_Latn"])
            resp = call_or_skip(prompt, 256)
            if resp is None:
                return None
            in_tok, out_tok, lat = resp.input_tokens, resp.output_tokens, resp.latency_s
            raw_text = resp.text
        elif condition == "urd_Latn_tfta_single":
            src_fields = item["urd_Latn"]
            prompt = format_belebele_tfta_single_prompt(src_fields)
            resp = call_or_skip(prompt, 2048)
            if resp is None:
                return None
            in_tok, out_tok, lat = resp.input_tokens, resp.output_tokens, resp.latency_s
            raw_text = resp.text
            try:
                reconstructed_urdu = str(json.loads(resp.text).get("urdu_transliteration", ""))
            except Exception:
                reconstructed_urdu = ""
        elif condition == "urd_Latn_tfta_twostage":
            src_fields = item["urd_Latn"]
            ur_p, p_in, p_out, p_lat = tfta_passage_cache[item["link"]]
            q_only = {k: src_fields[k] for k in ["question", "mc_answer1", "mc_answer2", "mc_answer3", "mc_answer4"]}
            t_prompt = (
                "Transliterate the following Roman Urdu question and four answer choices "
                "into standard Urdu (Perso-Arabic / Nastaliq script). Do NOT translate into English. "
                "Return ONLY valid JSON with the exact keys: question, mc_answer1, mc_answer2, mc_answer3, mc_answer4.\n\n"
                + json.dumps(q_only, ensure_ascii=False)
            )
            t_resp = client.generate(prompt=t_prompt, json_mode=True, temperature=0.0, max_output_tokens=600)
            in_tok += p_in + t_resp.input_tokens
            out_tok += p_out + t_resp.output_tokens
            lat += p_lat + t_resp.latency_s

            ur_fields = dict(src_fields)
            ur_fields["flores_passage"] = ur_p
            try:
                parsed_ur = json.loads(t_resp.text)
                req_keys = ["question", "mc_answer1", "mc_answer2", "mc_answer3", "mc_answer4"]
                if isinstance(parsed_ur, dict) and all(isinstance(parsed_ur.get(k), str) and parsed_ur[k].strip() for k in req_keys):
                    for k in req_keys:
                        ur_fields[k] = parsed_ur[k].strip()
                    reconstructed_urdu = ur_fields["flores_passage"] + " " + ur_fields["question"]
            except Exception:
                pass

            q_prompt = format_belebele_prompt(ur_fields)
            q_resp = client.generate(prompt=q_prompt, json_mode=True, temperature=0.0, max_output_tokens=256)
            in_tok += q_resp.input_tokens
            out_tok += q_resp.output_tokens
            lat += q_resp.latency_s
            raw_text = q_resp.text
        else:
            raise ValueError(f"Unknown condition: {condition}")

        pred = parse_mc_answer(raw_text)
        return {
            "benchmark": "belebele",
            "item_id": item["item_id"],
            "is_pilot": item["is_pilot"],
            "condition": condition,
            "gold": gold_letter,
            "pred": pred,
            "correct": int(pred == gold_letter),
            "parse_ok": pred is not None,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "latency_s": round(lat, 4),
            "reconstructed_urdu": reconstructed_urdu,
            "raw_response": raw_text[:500],
        }

    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for res in tqdm(
            pool.map(eval_one, items),
            total=len(items),
            desc=f"Belebele[{condition}]",
        ):
            if res is not None:
                results.append(res)
    return results


def _format_idiom_prompt(sentence: str, options: list[str], task: str, tfta: bool = False) -> str:
    opt_block = "\n".join(f"{LETTERS[i]}) {opt}" for i, opt in enumerate(options))
    if task == "meaning":
        q_text = "Which option best describes the figurative meaning of the idiomatic expression used in the sentence?"
    else:
        q_text = "Which English idiom or expression best matches the idiomatic meaning of the Urdu sentence?"

    if tfta:
        return (
            "The sentence below is written in Roman Urdu (Urdu written in Latin script) and contains an Urdu idiom.\n"
            "Step 1: First transliterate the Roman Urdu sentence into native Urdu script (Perso-Arabic / Nastaliq) to recognize the idiom accurately.\n"
            f"Step 2: {q_text}\n"
            'Respond ONLY with valid JSON: {"urdu_transliteration": "<sentence in Urdu script>", "answer": "A"|"B"|"C"|"D"}.\n\n'
            f"Sentence: {sentence}\n\n"
            f"{opt_block}"
        )
    return (
        f"{q_text}\n"
        'Respond ONLY with valid JSON: {"answer": "A"|"B"|"C"|"D"}.\n\n'
        f"Sentence: {sentence}\n\n"
        f"{opt_block}"
    )


def run_idioms_condition(
    items: list[dict[str, Any]],
    task: str,
    condition: str,
    client: CachedGeminiClient,
    workers: int = 6,
) -> list[dict[str, Any]]:
    """
    Evaluate Idioms MC-QA:
      task='meaning' -> 3-script parallel plain-meaning QA (eng_Latn, urd_Arab, urd_Latn, urd_Latn_tfta)
      task='equiv'   -> Urdu vs Roman Urdu -> English Idiom QA (urd_Arab, urd_Latn, urd_Latn_tfta)
    """
    filtered = [
        it for it in items
        if (task == "meaning" and it.get("meaning_options")) or (task == "equiv" and it.get("raw_equiv_options"))
    ]

    def eval_one(item: dict[str, Any]) -> dict[str, Any]:
        if task == "meaning":
            options = item["meaning_options"]
            gold_letter = LETTERS[int(item["meaning_correct_num"]) - 1]
        else:
            options = item["raw_equiv_options"]
            gold_letter = LETTERS[int(item["raw_equiv_correct_num"]) - 1]

        if condition == "eng_Latn":
            sent = item["english_sentence"]
            tfta = False
        elif condition == "urd_Arab":
            sent = item["urdu_sentence"]
            tfta = False
        elif condition == "urd_Latn":
            sent = item["roman_sentence"]
            tfta = False
        elif condition == "urd_Latn_tfta":
            sent = item["roman_sentence"]
            tfta = True
        else:
            raise ValueError(f"Unknown condition: {condition}")

        prompt = _format_idiom_prompt(sent, options, task=task, tfta=tfta)
        resp = client.generate(
            prompt=prompt,
            json_mode=True,
            temperature=0.0,
            max_output_tokens=512 if tfta else 128,
        )
        pred = parse_mc_answer(resp.text)
        reconstructed_urdu = ""
        if tfta:
            try:
                reconstructed_urdu = str(json.loads(resp.text).get("urdu_transliteration", ""))
            except Exception:
                pass
        return {
            "benchmark": f"idioms_{task}",
            "item_id": item["item_id"],
            "is_pilot": item["is_pilot"],
            "condition": condition,
            "gold": gold_letter,
            "pred": pred,
            "correct": int(pred == gold_letter),
            "parse_ok": pred is not None,
            "input_tokens": resp.input_tokens,
            "output_tokens": resp.output_tokens,
            "latency_s": round(resp.latency_s, 4),
            "reconstructed_urdu": reconstructed_urdu,
            "raw_response": resp.text[:400],
        }

    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for res in tqdm(
            pool.map(eval_one, filtered),
            total=len(filtered),
            desc=f"Idioms-{task}[{condition}]",
        ):
            results.append(res)
    return results


def _write_results(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 3-script & TFTA evaluation experiments.")
    parser.add_argument("--scope", choices=["pilot", "all"], default="pilot")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--min-interval", type=float, default=2.2)
    parser.add_argument("--limit", type=int, default=0, help="Optional cap per benchmark (0=no cap).")
    parser.add_argument("--cached-only", action="store_true", help="Export only cached responses without API calls.")
    parser.add_argument(
        "--belebele-conditions",
        nargs="+",
        default=None,
        help="Optional explicit list of Belebele conditions to run.",
    )
    parser.add_argument("--idioms-limit", type=int, default=300)
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["belebele", "idioms_meaning", "idioms_equiv"],
        choices=["belebele", "idioms_meaning", "idioms_equiv"],
    )
    args = parser.parse_args()

    client = CachedGeminiClient(
        cache_path=RESULTS_DIR / f"cache_eval_{args.model}.jsonl",
        model=args.model,
        min_interval_s=args.min_interval,
    )

    if "belebele" in args.benchmarks:
        bele_items = _read_jsonl(PROCESSED_DIR / "belebele_3scripts.jsonl")
        if args.belebele_conditions:
            conds = args.belebele_conditions
        elif args.scope == "pilot":
            conds = [
                "eng_Latn",
                "urd_Arab",
                "urd_Latn_corpus",
                "urd_Latn_hybrid",
                "urd_Latn_tfta_single",
                "urd_Latn_tfta_twostage",
            ]
        else:
            conds = [
                "eng_Latn",
                "urd_Arab",
                "urd_Latn_corpus",
                "urd_Latn_hybrid",
                "urd_Latn_tfta_twostage",
            ]
        if args.scope == "pilot":
            bele_items = [x for x in bele_items if x["is_pilot"]]
        if args.limit > 0:
            bele_items = bele_items[: args.limit]
        all_bele: list[dict[str, Any]] = []
        for c in conds:
            res = run_belebele_condition(
                bele_items, c, client, workers=args.workers, cached_only=args.cached_only
            )
            if res:
                acc = sum(r["correct"] for r in res) / len(res)
                print(f"[Belebele | {c}] n={len(res)} acc={acc:.4f}")
                all_bele.extend(res)
        _write_results(RESULTS_DIR / f"belebele_{args.scope}_{args.model}.jsonl", all_bele)

    if "idioms_meaning" in args.benchmarks or "idioms_equiv" in args.benchmarks:
        idiom_items = _read_jsonl(PROCESSED_DIR / "idioms_3scripts.jsonl")
        if args.scope == "pilot":
            idiom_items = [x for x in idiom_items if x["is_pilot"]]
        else:
            idiom_items = idiom_items[: args.idioms_limit]
        if args.limit > 0:
            idiom_items = idiom_items[: args.limit]

        if "idioms_meaning" in args.benchmarks:
            all_m: list[dict[str, Any]] = []
            for c in ["eng_Latn", "urd_Arab", "urd_Latn", "urd_Latn_tfta"]:
                res = run_idioms_condition(idiom_items, "meaning", c, client, workers=args.workers)
                if res:
                    acc = sum(r["correct"] for r in res) / len(res)
                    print(f"[Idioms-Meaning | {c}] n={len(res)} acc={acc:.4f}")
                    all_m.extend(res)
            _write_results(RESULTS_DIR / f"idioms_meaning_{args.scope}_{args.model}.jsonl", all_m)

        if "idioms_equiv" in args.benchmarks:
            all_e: list[dict[str, Any]] = []
            for c in ["urd_Arab", "urd_Latn", "urd_Latn_tfta"]:
                res = run_idioms_condition(idiom_items, "equiv", c, client, workers=args.workers)
                if res:
                    acc = sum(r["correct"] for r in res) / len(res)
                    print(f"[Idioms-Equiv | {c}] n={len(res)} acc={acc:.4f}")
                    all_e.extend(res)
            _write_results(RESULTS_DIR / f"idioms_equiv_{args.scope}_{args.model}.jsonl", all_e)


if __name__ == "__main__":
    main()
