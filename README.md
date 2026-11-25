# Same Question in Urdu, Roman Urdu, and English: Cross-Script LLM Evaluation

An end-to-end empirical benchmark pipeline measuring the accuracy gap of LLMs across **English (`eng_Latn`)**, **Urdu Nastaliq (`urd_Arab`)**, and **Roman Urdu (`urd_Latn`)**, and testing whether **"Transliterate First, Then Answer" (TFTA)** closes the gap on reading comprehension, QA, and idioms.

## Datasets Integrated
1. **Dakshina (Google)** — Urdu native & romanized parallel sentences + token alignments (`data/raw/dakshina/ur.parquet`, 9,759 rows; first 8,000 used for lexicon construction, remaining 1,759 held out for transliteration evaluation).
2. **Roman-Urdu-Parl-split** (`Mavkif/Roman-Urdu-Parl-split`) — Parallel Urdu/Roman-Urdu corpus (`train_sample_200000.csv` + `small_validation_set.csv` used for deduplicated, phonetically filtered lexicon expansion; `small_test_set.csv` held out for evaluation).
3. **Belebele** (`facebook/belebele`) — 900 parallel 4-way reading comprehension questions aligned 1-to-1 across `eng_Latn` and `urd_Arab`, converted to Roman Urdu (`urd_Latn`).
4. **Urdu Idioms** (`Mousumi44/Urdu-Idioms`, arXiv:2510.17460) — 2,002 non-empty aligned idiom sentences in Nastaliq Urdu, Roman Urdu, and English (1,919 unique idioms).

## Project Structure
- `src/download_datasets.py`: Downloads & caches all 4 datasets.
- `src/transliterator.py`: Builds a 27,815-entry Urdu → Roman Urdu lexicon (Dakshina + phonetically filtered Roman-Urdu-Parl) with a contextual Perso-Arabic rule fallback and hybrid Gemini normalizer.
- `src/prepare_benchmarks.py`: Aligns Belebele (900 items) and Urdu Idioms (1,919 unique idioms) with stratified pilot splits (`is_pilot`).
- `src/gemini_client.py`: Thread-safe Gemini API client (`google-genai`) with `.env` loading, JSONL response caching, and retry backoff.
- `src/run_experiments.py`: Evaluates English, Urdu Nastaliq, Roman Urdu, and TFTA (single-call CoT and two-stage back-transliteration).
- `src/analyze_results.py`: Computes accuracy, 95% bootstrap CIs, paired McNemar's tests, item-level rescue/regression flips, back-transliteration fidelity, and plots.
- `src/build_notebook.py`: Builds `analysis_report.ipynb` from saved metrics and figures.
- `tests/test_pipeline.py`: Unit & integration tests (`pytest`).

## Setup & Execution
1. Run the test suite:
   ```bash
   .venv/bin/python -m pytest -v
   ```
2. Add `GEMINI_API_KEY` to `.env` (gitignored) or `~/.env`.
3. Run hybrid transliteration + pilot / full experiments:
   ```bash
   .venv/bin/python -m src.prepare_benchmarks --hybrid --build-plain-meanings --scope pilot --idiom-limit 2000
   .venv/bin/python -m src.run_experiments --scope pilot --model gemini-3.5-flash-lite
   .venv/bin/python -m src.analyze_results --scope pilot --model gemini-3.5-flash-lite
   .venv/bin/python -m src.build_notebook --scope pilot --model gemini-3.5-flash-lite
   .venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace analysis_report.ipynb
   ```

## Contributors
- **Momin Ahmad** ([@Momin-Khannn](https://github.com/Momin-Khannn))
- **Muhammad Shariq Annas** ([@Shariq-Annas](https://github.com/Shariq-Annas))
