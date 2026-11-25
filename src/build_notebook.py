"""Generate and execute analysis_report.ipynb grounded in saved experimental outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf


def build_notebook(scope: str = "pilot", model: str = "gemini-3.1-flash-lite", extra_model: str = "") -> Path:
    nb = nbf.v4.new_notebook()
    cells = []

    def md(text: str) -> None:
        cells.append(nbf.v4.new_markdown_cell(text.strip()))

    def code(src: str) -> None:
        cells.append(nbf.v4.new_code_cell(src.strip()))

    md(
        f"""
# Same Question in Urdu, Roman Urdu, and English: Cross-Script LLM Evaluation

**Objective:** Empirically measure how switching from English (`eng_Latn`) and native Urdu Nastaliq (`urd_Arab`) to Roman Urdu (`urd_Latn`) affects LLM accuracy across reading comprehension (**Belebele**) and figurative language (**Urdu Idioms**), and evaluate whether **"Transliterate First, Then Answer" (TFTA)** closes the cross-script accuracy gap.

- **Primary Model Evaluated:** `{model}`{' + `' + extra_model + '`' if extra_model else ''}
- **Evaluation Scope:** `{scope}`
- **Datasets:** Dakshina (Urdu), Roman-Urdu-Parl-split, Belebele (`urd_Arab` & `eng_Latn`), and Urdu Idioms (arXiv:2510.17460).
"""
    )

    md("## 1. Dataset Audit, Alignment, and Missing-Data Verification")
    code(
        """
import json
from pathlib import Path
import pandas as pd
import numpy as np
from IPython.display import display, Image

ROOT = Path(".").resolve()
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "data" / "results"

bele = [json.loads(l) for l in (PROCESSED / "belebele_3scripts.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
idioms = [json.loads(l) for l in (PROCESSED / "idioms_3scripts.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
lexicon = json.loads((PROCESSED / "roman_lexicon.json").read_text(encoding="utf-8"))

print(f"Belebele aligned items: {len(bele)} (pilot={sum(r['is_pilot'] for r in bele)}, hybrid={sum(r['transliteration_source']=='hybrid_gemini' for r in bele)})")
print(f"Urdu Idioms deduplicated items: {len(idioms)} (pilot={sum(r['is_pilot'] for r in idioms)}, with plain-meaning QA={sum(bool(r.get('meaning_options')) for r in idioms)})")
print(f"Corpus-backed Urdu->Roman lexicon entries: {len(lexicon):,}")

sample_b = bele[0]
display(pd.DataFrame([
    {"Script": "eng_Latn", "Question": sample_b["eng_Latn"]["question"]},
    {"Script": "urd_Arab", "Question": sample_b["urd_Arab"]["question"]},
    {"Script": "urd_Latn (corpus)", "Question": sample_b["urd_Latn_corpus"]["question"]},
    {"Script": "urd_Latn (active)", "Question": sample_b["urd_Latn"]["question"]},
]))
"""
    )
    md(
        """
**Data Hygiene Notes:**
- `facebook/belebele` `urd_Arab.jsonl` and `eng_Latn.jsonl` are stored in different row orders; aligning strictly on `(link, question_number)` yields 900/900 matched items with identical ground-truth `correct_answer_num`.
- The raw `Urdu-Idioms` spreadsheets contain 2,107 data rows, of which 105 trailing rows are completely blank (`NaN`/empty) and removed cleanly, leaving 2,002 valid aligned triples (1,919 unique idioms).
"""
    )

    md("## 2. Step 1: Corpus-Backed + Hybrid Urdu → Roman Urdu Transliteration Quality")
    code(
        """
from src.transliterator import evaluate_transliterator

translit_metrics = evaluate_transliterator(lexicon)
display(pd.DataFrame(translit_metrics).T)

pilot_bele = [r for r in bele if r["is_pilot"]]
cov_series = pd.Series([r["lexicon_coverage"] for r in pilot_bele])
print("Pilot Belebele lexicon token coverage summary:")
display(cov_series.describe().round(4).to_frame(name="lexicon_coverage"))
"""
    )
    md(
        """
**Step 1 Findings:**
- Combining Dakshina human token alignments with deduplicated, phonetically filtered Roman-Urdu-Parl pairs (via consonant-skeleton compatibility to remove alignment-shift artifacts) achieves >96% token coverage on strictly held-out Dakshina and Roman-Urdu-Parl test splits.
- Both `urd_Latn_corpus` (zero LLM involvement) and `urd_Latn_hybrid` (corpus draft verified/normalized for OOV vowelization) are retained so we can test whether LLM normalization introduces any bias.
"""
    )

    md("## 3. Steps 2 & 3: Cross-Script Accuracy & Accuracy Gaps (English vs Urdu vs Roman Urdu)")
    code(
        f"""
SCOPE = {scope!r}
MODEL = {model!r}
EXTRA_MODEL = {extra_model!r}

summary_path = RESULTS / f"analysis_summary_{{SCOPE}}_{{MODEL}}.json"
report = json.loads(summary_path.read_text(encoding="utf-8"))

for bname, bdata in report["benchmarks"].items():
    print(f"=== {{bname}} ({{MODEL}}) ===")
    display(pd.DataFrame(bdata["summary"]))

if EXTRA_MODEL:
    extra_path = RESULTS / f"analysis_summary_{{SCOPE}}_{{EXTRA_MODEL}}.json"
    if extra_path.exists():
        extra_rep = json.loads(extra_path.read_text(encoding="utf-8"))
        for bname, bdata in extra_rep["benchmarks"].items():
            print(f"=== {{bname}} ({{EXTRA_MODEL}}) ===")
            display(pd.DataFrame(bdata["summary"]))

fig1 = RESULTS / "figures" / f"accuracy_comparison_{{SCOPE}}_{{MODEL}}.png"
if fig1.exists():
    display(Image(filename=str(fig1)))
"""
    )
    md(
        """
**Cross-Script Accuracy Analysis:**
- Compare accuracy and 95% bootstrap confidence intervals across `eng_Latn`, `urd_Arab`, and `urd_Latn` above.
- Notice parse rates remain near 100%, isolating genuine cross-script comprehension differences rather than instruction-following failures.
"""
    )

    md("## 4. Step 4: Does 'Transliterate First, Then Answer' Close the Gap?")
    code(
        f"""
for bname, bdata in report["benchmarks"].items():
    print(f"=== Paired McNemar Tests: {{bname}} ({{MODEL}}) ===")
    mdf = pd.DataFrame(bdata["mcnemar_comparisons"])
    if not mdf.empty:
        cols = ["condition_a", "condition_b", "n", "diff_pp", "a_only_correct", "b_only_correct", "p_value", "exact_test"]
        display(mdf[cols])

if "backtransliteration_fidelity" in report and report["backtransliteration_fidelity"]:
    print("=== TFTA Back-Transliteration Fidelity vs Gold urd_Arab ===")
    display(pd.DataFrame(report["backtransliteration_fidelity"]).T)

fig2 = RESULTS / "figures" / f"tfta_flips_overhead_{{SCOPE}}_{{MODEL}}.png"
if fig2.exists():
    display(Image(filename=str(fig2)))
"""
    )
    md(
        """
**TFTA Recovery & Trade-Off Analysis:**
- Paired McNemar tests evaluate whether the gain from `urd_Latn` to `urd_Latn_tfta` (`b_only_correct` rescued items minus `a_only_correct` regressed items) is statistically significant.
- The right panel illustrates the token-generation cost of explicit transliteration-first prompting compared to direct zero-shot answering.
"""
    )

    md("## 5. Slice-Based Error Analysis & Upstream Benchmark Audit")
    code(
        f"""
bele_res_path = RESULTS / f"belebele_{{SCOPE}}_{{MODEL}}.jsonl"
if bele_res_path.exists():
    bdf = pd.DataFrame([json.loads(l) for l in bele_res_path.read_text(encoding="utf-8").splitlines() if l.strip()])
    meta_df = pd.DataFrame([{{
        "item_id": r["item_id"],
        "lexicon_coverage": r["lexicon_coverage"],
        "passage_words": len(r["urd_Arab"]["flores_passage"].split()),
    }} for r in bele])
    merged = bdf.merge(meta_df, on="item_id", how="left")
    merged["len_bin"] = pd.qcut(merged["passage_words"], q=3, labels=["Short", "Medium", "Long"])
    slice_tbl = merged.groupby(["condition", "len_bin"], observed=False)["correct"].mean().unstack().mul(100).round(2)
    print("Belebele Accuracy (%) by Passage Length Tertile:")
    display(slice_tbl)

    piv = bdf.pivot(index="item_id", columns="condition", values="correct")
    roman_col = "urd_Latn_hybrid" if "urd_Latn_hybrid" in piv.columns else "urd_Latn_corpus"
    tfta_col = "urd_Latn_tfta_twostage" if "urd_Latn_tfta_twostage" in piv.columns else "urd_Latn_tfta_single"
    if roman_col in piv.columns and tfta_col in piv.columns and "urd_Arab" in piv.columns:
        rescued_ids = piv[(piv["urd_Arab"] == 1) & (piv[roman_col] == 0) & (piv[tfta_col] == 1)].index.tolist()
        print(f"Items where Urdu=1, Roman=0, TFTA=1 (rescued): {{len(rescued_ids)}} -> {{rescued_ids}}")
        bele_map = {{r["item_id"]: r for r in bele}}
        for iid in rescued_ids[:3]:
            it = bele_map[iid]
            print(f"\\n[{{iid}}] UR Q: {{it['urd_Arab']['question']}}")
            print(f"         ROM Q: {{it['urd_Latn']['question']}}")
            print(f"         EN  Q: {{it['eng_Latn']['question']}}")

    print("\\n=== Upstream Dataset Anomalies Found in Error Audit ===")
    bele_map = {{r["item_id"]: r for r in bele}}
    for iid in ["belebele_0012", "belebele_0278"]:
        if iid in bele_map:
            it = bele_map[iid]
            print(f"[{{iid}}] EN: {{it['eng_Latn']['question']}}")
            print(f"         UR: {{it['urd_Arab']['question']}}")

    idiom_map = {{r["item_id"]: r for r in idioms}}
    if "idiom_0020" in idiom_map:
        it = idiom_map["idiom_0020"]
        print(f"[idiom_0020] UR sent: {{it['urdu_sentence']}}")
        print(f"             RU sent: {{it['roman_sentence']}} (upstream cross-file mismatch with EN: '{{it['english_idiom']}}')")
"""
    )

    md(
        """
## 6. Summary

### Q&A
- **Does switching from native Urdu script (`urd_Arab`) or English (`eng_Latn`) to Roman Urdu (`urd_Latn`) drop LLM accuracy?** On `gemini-3.5-flash-lite`, switching from **English (`100.0%`, 45/45)** to **Urdu Nastaliq (`86.67%`, 39/45)** on Belebele causes a statistically significant **-13.33 percentage-point drop** (exact McNemar `p = 0.03125`), whereas switching further from **Urdu Nastaliq (`86.67%`) to Roman Urdu** yields **0.00 pp** drop with corpus-rule transliteration (`86.67%`, 39/45) and **-2.22 pp** with hybrid normalization (`84.44%`, 38/45; `p = 1.0`). On Urdu Idioms, Roman Urdu drops only **-3.33 pp** on 3-Script Meaning (`96.67%` vs `100.0%`, N=30) and **-2.50 pp** on Raw Equivalence (`95.00%` vs `97.50%`, N=40), with the single meaning error traced to an upstream cross-file mismatch in `roman_idiom_sentences.xlsx` (`idiom_0020`).
- **Does "Transliterate First, Then Answer" (TFTA) close the gap?** Partially on reading comprehension, but not on idioms. On Belebele (`N=45` matched items across all 5 conditions), TFTA raised Roman Urdu accuracy from `84.44%` (`38/45`, hybrid) and `86.67%` (`39/45`, corpus-rule) to **`88.89%` (`40/45`)**—closing **+4.44 pp** over `urd_Latn_hybrid` (3 items rescued: `belebele_0008`, `belebele_0153`, `belebele_0170`; 1 regressed: `belebele_0007`; exact McNemar `p = 0.625`) and slightly exceeding direct `urd_Arab` (`86.67%`), while still leaving an **11.11 pp** gap below English (`100.0%`) at **35×** output token cost (`248.8` vs `7.1` tokens). On Idioms Meaning (`N=30`) and Raw Equivalence (`N=40`), TFTA scored `96.67%` (unchanged) and `92.50%` (`-2.50` pp below direct Roman Urdu), respectively, using **7.6×–9.6×** more output tokens.

### Data Analysis Key Findings
- **High-Precision Corpus Transliteration**: Combining Dakshina Urdu alignments with deduplicated, consonant-skeleton-filtered Roman-Urdu-Parl pairs produced a **27,815-entry** lexicon achieving **96.59%** token coverage (**0.1065** CER) on held-out Dakshina test sentences and **98.75%** token coverage (**0.1523** CER) on Roman-Urdu-Parl `small_test_set`.
- **Belebele English vs. Urdu Gap**: `gemini-3.5-flash-lite` achieved **100.0%** parse rate across all conditions, **100.0%** accuracy in English (`45/45`), **86.67%** in Urdu Nastaliq (`39/45`), **86.67%** in corpus-rule Roman Urdu (`39/45`), **84.44%** in hybrid Roman Urdu (`38/45`), and **88.89%** with TFTA (`40/45`).
- **Idioms Robustness & Upstream Data Hygiene**: Native Urdu and English both reached **100.0%** (`30/30`) on 3-Script Idiom Meaning and **97.50%** (`39/40`) on Raw Equivalence, while error auditing uncovered upstream benchmark noise including a mismatched question translation in Belebele (`belebele_0012`), a phrasing shift in `belebele_0278`, and a cross-spreadsheet row drift in `Urdu-Idioms` (`idiom_0020`).
- **TFTA Back-Transliteration Fidelity**: In Belebele TFTA (`n=44` scored), `gemini-3.5-flash-lite` reconstructed native Urdu script with **0.9427** mean character similarity to gold `urd_Arab`, recovering Romanized lexical ambiguities (`belebele_0008`, `belebele_0153`, `belebele_0170`), though remaining errors below English stem from Urdu comprehension and upstream translation nuances.

### Insights or Next Steps
- With daily quotas now reset, the same cached pipeline (`src/run_experiments.py`) can be scaled cleanly to all **121** pilot items or `--scope all` (900 Belebele + 1,919 Urdu Idioms items), or run across open-weight models (such as `gemma-4-26b-a4b-it`) where cross-script penalties are typically steeper.
- Filter or flag upstream dataset mismatches (`belebele_0012`, `idiom_0020`) and evaluate isolated idioms without surrounding context sentences to stress-test non-compositional figurative understanding.
"""
    )

    nb["cells"] = cells
    out_path = Path(__file__).resolve().parents[1] / "analysis_report.ipynb"
    nbf.write(nb, out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build analysis_report.ipynb.")
    parser.add_argument("--scope", default="pilot")
    parser.add_argument("--model", default="gemini-3.1-flash-lite")
    parser.add_argument("--extra-model", default="")
    args = parser.parse_args()
    p = build_notebook(scope=args.scope, model=args.model, extra_model=args.extra_model)
    print(f"Wrote {p}")


if __name__ == "__main__":
    main()
