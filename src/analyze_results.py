"""Statistical analysis, paired McNemar tests, bootstrap CIs, slice analysis, and plotting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein
from statsmodels.stats.contingency_tables import mcnemar

from src.transliterator import PROCESSED_DIR, normalize_urdu_text

RESULTS_DIR = Path(__file__).resolve().parents[1] / "data" / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

CONDITION_LABELS = {
    "eng_Latn": "English\n(eng_Latn)",
    "urd_Arab": "Urdu Nastaliq\n(urd_Arab)",
    "urd_Latn_corpus": "Roman Urdu\n(Corpus-Rule)",
    "urd_Latn_hybrid": "Roman Urdu\n(Hybrid)",
    "urd_Latn": "Roman Urdu\n(urd_Latn)",
    "urd_Latn_tfta_single": "Roman Urdu\n+ TFTA (1-Call)",
    "urd_Latn_tfta_twostage": "Roman Urdu\n+ TFTA (2-Stage)",
    "urd_Latn_tfta": "Roman Urdu\n+ TFTA",
}

PALETTE = {
    "eng_Latn": "#2E6F95",
    "urd_Arab": "#1B9E77",
    "urd_Latn_corpus": "#D95F02",
    "urd_Latn_hybrid": "#E67E22",
    "urd_Latn": "#D95F02",
    "urd_Latn_tfta_single": "#7570B3",
    "urd_Latn_tfta_twostage": "#5E3C99",
    "urd_Latn_tfta": "#5E3C99",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def bootstrap_ci(values: np.ndarray, n_boot: int = 5000, seed: int = 42) -> tuple[float, float, float]:
    """Return (mean, ci_low, ci_high) at 95% confidence via bootstrap resampling."""
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    if len(arr) <= 1:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    boots = arr[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return mean, float(lo), float(hi)


def paired_mcnemar_test(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    """Run paired McNemar test on binary correctness arrays a and b."""
    n11 = int(np.sum((a == 1) & (b == 1)))
    n10 = int(np.sum((a == 1) & (b == 0)))
    n01 = int(np.sum((a == 0) & (b == 1)))
    n00 = int(np.sum((a == 0) & (b == 0)))
    table = [[n11, n10], [n01, n00]]
    use_exact = (n10 + n01) < 25
    res = mcnemar(table, exact=use_exact, correction=not use_exact)
    return {
        "n": len(a),
        "both_correct": n11,
        "a_only_correct": n10,
        "b_only_correct": n01,
        "both_wrong": n00,
        "diff_pp": round(float(b.mean() - a.mean()) * 100.0, 2),
        "p_value": float(res.pvalue),
        "exact_test": use_exact,
    }


def summarize_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    """Compute accuracy, 95% bootstrap CI, parse rate, tokens, and latency per condition."""
    rows: list[dict[str, Any]] = []
    for cond, grp in df.groupby("condition", sort=False):
        mean, lo, hi = bootstrap_ci(grp["correct"].to_numpy())
        rows.append(
            {
                "condition": cond,
                "n": len(grp),
                "accuracy": round(mean * 100.0, 2),
                "ci_low": round(lo * 100.0, 2),
                "ci_high": round(hi * 100.0, 2),
                "parse_rate": round(float(grp["parse_ok"].mean()) * 100.0, 2),
                "avg_input_tokens": round(float(grp["input_tokens"].mean()), 1),
                "avg_output_tokens": round(float(grp["output_tokens"].mean()), 1),
                "avg_latency_s": round(float(grp["latency_s"].mean()), 3),
            }
        )
    return pd.DataFrame(rows)


def compute_paired_comparisons(df: pd.DataFrame, pairs: list[tuple[str, str]]) -> list[dict[str, Any]]:
    piv = df.pivot(index="item_id", columns="condition", values="correct")
    out: list[dict[str, Any]] = []
    for c_a, c_b in pairs:
        if c_a not in piv.columns or c_b not in piv.columns:
            continue
        sub = piv[[c_a, c_b]].dropna()
        res = paired_mcnemar_test(sub[c_a].to_numpy(dtype=int), sub[c_b].to_numpy(dtype=int))
        res["condition_a"] = c_a
        res["condition_b"] = c_b
        out.append(res)
    return out


def analyze_backtransliteration(df: pd.DataFrame, belebele_path: Path) -> dict[str, Any]:
    """Compare LLM reconstructed Urdu in TFTA against gold urd_Arab."""
    if not belebele_path.exists():
        return {}
    gold_map = {
        r["item_id"]: normalize_urdu_text(
            r["urd_Arab"]["flores_passage"] + " " + r["urd_Arab"]["question"]
        )
        for r in read_jsonl(belebele_path)
    }
    sub = df[df["condition"].str.contains("tfta", na=False) & (df["reconstructed_urdu"].str.len() > 0)]
    if sub.empty:
        return {}
    out: dict[str, Any] = {}
    for cond, grp in sub.groupby("condition"):
        sims_correct, sims_wrong = [], []
        for _, row in grp.iterrows():
            gold = gold_map.get(row["item_id"], "")
            pred = normalize_urdu_text(str(row["reconstructed_urdu"]))
            if not gold or not pred:
                continue
            dist = Levenshtein.distance(pred[: len(gold)], gold[: len(pred)])
            sim = 1.0 - dist / max(len(pred), len(gold), 1)
            if row["correct"] == 1:
                sims_correct.append(sim)
            else:
                sims_wrong.append(sim)
        out[str(cond)] = {
            "n_scored": len(sims_correct) + len(sims_wrong),
            "mean_char_sim_all": round(float(np.mean(sims_correct + sims_wrong)), 4) if (sims_correct + sims_wrong) else None,
            "mean_char_sim_correct": round(float(np.mean(sims_correct)), 4) if sims_correct else None,
            "mean_char_sim_wrong": round(float(np.mean(sims_wrong)), 4) if sims_wrong else None,
        }
    return out


def plot_accuracy_bars(summaries: dict[str, pd.DataFrame], out_path: Path) -> None:
    """Plot cross-script accuracy & TFTA recovery with 95% bootstrap CIs."""
    n_panels = len(summaries)
    fig, axes = plt.subplots(1, n_panels, figsize=(6.2 * n_panels, 5.0), squeeze=False)
    for ax, (title, sdf) in zip(axes[0], summaries.items()):
        x = np.arange(len(sdf))
        vals = sdf["accuracy"].to_numpy()
        yerr = np.vstack([vals - sdf["ci_low"].to_numpy(), sdf["ci_high"].to_numpy() - vals])
        colors = [PALETTE.get(c, "#666666") for c in sdf["condition"]]
        labels = [CONDITION_LABELS.get(c, c) for c in sdf["condition"]]
        bars = ax.bar(x, vals, yerr=yerr, capsize=4, color=colors, edgecolor="#222222", linewidth=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, 114)
        ax.axhline(25.0, color="#888888", linestyle="--", linewidth=0.9, label="Random (25%)")
        ax.set_ylabel("Accuracy (%)", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
        ax.grid(axis="y", alpha=0.25)
        for bar, v, hi in zip(bars, vals, sdf["ci_high"].to_numpy()):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                max(v, hi) + 2.0,
                f"{v:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_flip_and_overhead(
    bele_df: pd.DataFrame,
    summaries: dict[str, pd.DataFrame],
    out_path: Path,
) -> None:
    """Plot TFTA item-level rescued vs broken flips and token/latency cost."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))

    # Left: Item-level flips (Roman -> TFTA) across benchmarks
    piv = bele_df.pivot(index="item_id", columns="condition", values="correct")
    flip_rows = []
    roman_col = "urd_Latn_hybrid" if "urd_Latn_hybrid" in piv.columns else "urd_Latn_corpus"
    for tfta_col in ["urd_Latn_tfta_single", "urd_Latn_tfta_twostage"]:
        if roman_col in piv.columns and tfta_col in piv.columns:
            sub = piv[[roman_col, tfta_col]].dropna()
            rescued = int(((sub[roman_col] == 0) & (sub[tfta_col] == 1)).sum())
            broken = int(((sub[roman_col] == 1) & (sub[tfta_col] == 0)).sum())
            same_ok = int(((sub[roman_col] == 1) & (sub[tfta_col] == 1)).sum())
            same_bad = int(((sub[roman_col] == 0) & (sub[tfta_col] == 0)).sum())
            flip_rows.append(
                {
                    "label": CONDITION_LABELS.get(tfta_col, tfta_col).replace("\n", " "),
                    "Rescued (0->1)": rescued,
                    "Regressed (1->0)": broken,
                    "Both Correct": same_ok,
                    "Both Wrong": same_bad,
                }
            )
    if flip_rows:
        fdf = pd.DataFrame(flip_rows)
        x = np.arange(len(fdf))
        w = 0.32
        max_y = max(int(fdf["Rescued (0->1)"].max()), int(fdf["Regressed (1->0)"].max()), 1)
        ax1.bar(x - w / 2, fdf["Rescued (0->1)"], width=w, color="#1B9E77", label="Rescued (0 → 1)")
        ax1.bar(x + w / 2, fdf["Regressed (1->0)"], width=w, color="#E7298A", label="Regressed (1 → 0)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(fdf["label"], fontsize=9)
        ax1.set_ylim(0, max_y * 1.38 + 0.2)
        ax1.set_ylabel("Number of Items", fontsize=10)
        ax1.set_title("Belebele: Items Rescued vs Regressed by TFTA", fontsize=11, fontweight="bold")
        ax1.legend(loc="upper right")
        ax1.grid(axis="y", alpha=0.25)
        for i, row in fdf.iterrows():
            ax1.text(i - w / 2, row["Rescued (0->1)"] + max_y * 0.03, str(row["Rescued (0->1)"]), ha="center", va="bottom", fontweight="bold")
            ax1.text(i + w / 2, row["Regressed (1->0)"] + max_y * 0.03, str(row["Regressed (1->0)"]), ha="center", va="bottom", fontweight="bold")

    # Right: Accuracy vs Avg Output Tokens on Belebele
    bsum = next(iter(summaries.values()))
    offsets = {
        "eng_Latn": (10, -3),
        "urd_Arab": (10, 6),
        "urd_Latn_corpus": (10, -10),
        "urd_Latn_hybrid": (10, -4),
        "urd_Latn_tfta_single": (-12, 10),
        "urd_Latn_tfta_twostage": (-12, -12),
    }
    for _, row in bsum.iterrows():
        c = row["condition"]
        ax2.scatter(
            row["avg_output_tokens"],
            row["accuracy"],
            s=130,
            color=PALETTE.get(c, "#444444"),
            edgecolors="#111111",
            zorder=3,
        )
        dx, dy = offsets.get(c, (8, 4))
        ha = "right" if dx < 0 else "left"
        ax2.annotate(
            CONDITION_LABELS.get(c, c).replace("\n", " "),
            (row["avg_output_tokens"], row["accuracy"]),
            textcoords="offset points",
            xytext=(dx, dy),
            ha=ha,
            fontsize=8.5,
        )
    max_tok = max(float(bsum["avg_output_tokens"].max()), 50.0)
    min_acc = float(bsum["accuracy"].min())
    ax2.set_xlim(-15, max_tok * 1.12)
    ax2.set_ylim(max(0.0, min_acc - 3.0), 102.5)
    ax2.set_xlabel("Average Output Tokens per Question", fontsize=10)
    ax2.set_ylabel("Accuracy (%)", fontsize=10)
    ax2.set_title("Accuracy vs Token Overhead", fontsize=11, fontweight="bold")
    ax2.grid(alpha=0.25)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_analysis(scope: str, model: str) -> dict[str, Any]:
    files = {
        "Belebele Reading Comprehension": RESULTS_DIR / f"belebele_{scope}_{model}.jsonl",
        "Urdu Idioms (3-Script Meaning)": RESULTS_DIR / f"idioms_meaning_{scope}_{model}.jsonl",
        "Urdu Idioms (Raw Equivalence)": RESULTS_DIR / f"idioms_equiv_{scope}_{model}.jsonl",
    }
    summaries: dict[str, pd.DataFrame] = {}
    report: dict[str, Any] = {"scope": scope, "model": model, "benchmarks": {}}

    bele_df = pd.DataFrame()
    for title, path in files.items():
        if not path.exists():
            continue
        df = pd.DataFrame(read_jsonl(path))
        if df.empty:
            continue
        if "Belebele" in title:
            bele_df = df
        sdf = summarize_benchmark(df)
        summaries[title] = sdf

        if "Belebele" in title:
            pairs = [
                ("eng_Latn", "urd_Arab"),
                ("urd_Arab", "urd_Latn_corpus"),
                ("urd_Arab", "urd_Latn_hybrid"),
                ("urd_Latn_corpus", "urd_Latn_hybrid"),
                ("urd_Latn_corpus", "urd_Latn_tfta_single"),
                ("urd_Latn_hybrid", "urd_Latn_tfta_single"),
                ("urd_Latn_hybrid", "urd_Latn_tfta_twostage"),
                ("urd_Arab", "urd_Latn_tfta_twostage"),
            ]
        elif "Meaning" in title:
            pairs = [
                ("eng_Latn", "urd_Arab"),
                ("urd_Arab", "urd_Latn"),
                ("urd_Latn", "urd_Latn_tfta"),
                ("urd_Arab", "urd_Latn_tfta"),
            ]
        else:
            pairs = [
                ("urd_Arab", "urd_Latn"),
                ("urd_Latn", "urd_Latn_tfta"),
                ("urd_Arab", "urd_Latn_tfta"),
            ]

        comparisons = compute_paired_comparisons(df, pairs)
        report["benchmarks"][title] = {
            "summary": sdf.to_dict(orient="records"),
            "mcnemar_comparisons": comparisons,
        }

    if not bele_df.empty:
        report["backtransliteration_fidelity"] = analyze_backtransliteration(
            bele_df, PROCESSED_DIR / "belebele_3scripts.jsonl"
        )

    if summaries:
        plot_accuracy_bars(summaries, FIGURES_DIR / f"accuracy_comparison_{scope}_{model}.png")
    if not bele_df.empty and summaries:
        plot_flip_and_overhead(bele_df, summaries, FIGURES_DIR / f"tfta_flips_overhead_{scope}_{model}.png")

    out_json = RESULTS_DIR / f"analysis_summary_{scope}_{model}.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze cross-script evaluation results.")
    parser.add_argument("--scope", choices=["pilot", "all"], default="pilot")
    parser.add_argument("--model", default="gemini-3.1-flash-lite")
    args = parser.parse_args()
    rep = run_analysis(args.scope, args.model)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
