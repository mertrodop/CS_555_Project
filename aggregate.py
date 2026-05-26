"""
Aggregate LLM-AGR reproduction logs into CSVs and markdown tables.

Usage (after all 80 runs complete):
    python aggregate.py

Outputs (all in results/):
    amazon_book.csv, yelp.csv   -- long-format per-run data
    table3_repro.md             -- mean ± std, significance markers
    paper_vs_ours.md            -- gap to reported paper numbers
    SUMMARY.md                  -- missing baselines, patches, anomalies
"""

import os
import re
import ast
import csv
import math
import glob
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

# ── Paper-reported numbers from Table 3 ───────────────────────────────────────
# Source: Zha et al., KBS 2025, Table 3.
# Fill in the values from the paper for each (base, augmentation, dataset) cell.
# Keys: (base_model, augmentation, dataset)
# Values: dict mapping metric_name -> reported float
# Set to None for cells not reported or not applicable.
PAPER_RESULTS = {
    # LightGCN rows
    ("lightgcn", "base",    "amazon"): {"recall@5": 0.0570, "recall@10": 0.0915, "recall@20": 0.1411, "ndcg@5": 0.0574, "ndcg@10": 0.0694, "ndcg@20": 0.0856},
    ("lightgcn", "llm_agr", "amazon"): {"recall@5": 0.0658, "recall@10": 0.1040, "recall@20": 0.1567, "ndcg@5": 0.0651, "ndcg@10": 0.0781, "ndcg@20": 0.0952},
    ("lightgcn", "base",    "yelp"):   {"recall@5": 0.0421, "recall@10": 0.0706, "recall@20": 0.1157, "ndcg@5": 0.0491, "ndcg@10": 0.0580, "ndcg@20": 0.0733},
    ("lightgcn", "llm_agr", "yelp"):   {"recall@5": 0.0475, "recall@10": 0.0790, "recall@20": 0.1289, "ndcg@5": 0.0563, "ndcg@10": 0.0655, "ndcg@20": 0.0821},
    # SGL rows
    ("sgl",      "base",    "amazon"): {"recall@5": 0.0637, "recall@10": 0.0994, "recall@20": 0.1473, "ndcg@5": 0.0632, "ndcg@10": 0.0756, "ndcg@20": 0.0913},
    ("sgl",      "llm_agr", "amazon"): {"recall@5": 0.0672, "recall@10": 0.1033, "recall@20": 0.1543, "ndcg@5": 0.0667, "ndcg@10": 0.0791, "ndcg@20": 0.0957},
    ("sgl",      "base",    "yelp"):   {"recall@5": 0.0432, "recall@10": 0.0722, "recall@20": 0.1197, "ndcg@5": 0.0501, "ndcg@10": 0.0592, "ndcg@20": 0.0753},
    ("sgl",      "llm_agr", "yelp"):   {"recall@5": 0.0481, "recall@10": 0.0779, "recall@20": 0.1274, "ndcg@5": 0.0558, "ndcg@10": 0.0648, "ndcg@20": 0.0813},
    # SimGCL rows
    ("simgcl",   "base",    "amazon"): {"recall@5": 0.0618, "recall@10": 0.0992, "recall@20": 0.1512, "ndcg@5": 0.0619, "ndcg@10": 0.0749, "ndcg@20": 0.0919},
    ("simgcl",   "llm_agr", "amazon"): {"recall@5": 0.0647, "recall@10": 0.1023, "recall@20": 0.1565, "ndcg@5": 0.0638, "ndcg@10": 0.0770, "ndcg@20": 0.0945},
    ("simgcl",   "base",    "yelp"):   {"recall@5": 0.0467, "recall@10": 0.0772, "recall@20": 0.1254, "ndcg@5": 0.0546, "ndcg@10": 0.0638, "ndcg@20": 0.0801},
    ("simgcl",   "llm_agr", "yelp"):   {"recall@5": 0.0476, "recall@10": 0.0801, "recall@20": 0.1317, "ndcg@5": 0.0565, "ndcg@10": 0.0665, "ndcg@20": 0.0838},
    # BIGCF rows
    ("bigcf",    "base",    "amazon"): {"recall@5": 0.0662, "recall@10": 0.1028, "recall@20": 0.1552, "ndcg@5": 0.0658, "ndcg@10": 0.0784, "ndcg@20": 0.0955},
    ("bigcf",    "llm_agr", "amazon"): {"recall@5": 0.0697, "recall@10": 0.1076, "recall@20": 0.1638, "ndcg@5": 0.0699, "ndcg@10": 0.0828, "ndcg@20": 0.1010},
    ("bigcf",    "base",    "yelp"):   {"recall@5": 0.0458, "recall@10": 0.0758, "recall@20": 0.1237, "ndcg@5": 0.0536, "ndcg@10": 0.0627, "ndcg@20": 0.0789},
    ("bigcf",    "llm_agr", "yelp"):   {"recall@5": 0.0507, "recall@10": 0.0842, "recall@20": 0.1356, "ndcg@5": 0.0583, "ndcg@10": 0.0685, "ndcg@20": 0.0858},
}

METRICS = ["recall@5", "recall@10", "recall@20", "ndcg@5", "ndcg@10", "ndcg@20"]
BASES   = ["lightgcn", "sgl", "simgcl", "bigcf"]
DATASETS = {"amazon": "amazon_book", "yelp": "yelp"}

MISSING_BASELINES = ["KAR", "RLMRec-Con", "RLMRec-Gen", "Semantic Only", "AlphaRec"]
PATCHES_APPLIED = [
    "main.py:11 — set_seed(2025) → set_seed(configs['train']['seed'])",
    "config/configurator.py:36 — if args.seed: → if args.seed is not None:",
    "trainer/trainer.py — added wall-clock timing and GPU memory logging ([REPRO] line)",
    "venv — installed torch_sparse and torch_scatter from PyG wheel index",
]


# ── Log parsing ────────────────────────────────────────────────────────────────

def model_to_base_aug(model_name):
    if model_name.endswith("_agr"):
        return model_name[:-4], "llm_agr"
    return model_name, "base"


def parse_log(log_path):
    with open(log_path) as f:
        content = f.read()

    result = {}

    # Extract test metrics from "Final test result: {...}."
    m = re.search(r"Final test result:\s*(\{.+?\})\.", content, re.DOTALL)
    if m:
        raw = m.group(1)
        # Replace numpy array(...) notation with plain lists
        raw_clean = re.sub(r"array\(\[([\d\s\.,e\-]+)\]\)", r"[\1]", raw)
        try:
            d = ast.literal_eval(raw_clean)
            ks = [5, 10, 20]
            for metric_name, vals in d.items():
                for k, v in zip(ks, vals):
                    result[f"{metric_name}@{k}"] = float(v)
        except Exception as e:
            print(f"  Warning: could not parse metrics from {log_path}: {e}")

    # Extract [REPRO] line
    m2 = re.search(r"\[REPRO\] wall_clock_s=([\d.]+) peak_mem_mb=([\d.]+)", content)
    if m2:
        result["wall_clock_s"] = float(m2.group(1))
        result["peak_mem_mb"] = float(m2.group(2))
    else:
        result["wall_clock_s"] = float("nan")
        result["peak_mem_mb"] = float("nan")

    return result


def collect_all_runs(logs_root="logs"):
    rows = []
    for dataset_dir in ["amazon", "yelp"]:
        path = os.path.join(logs_root, dataset_dir)
        if not os.path.isdir(path):
            continue
        for fname in sorted(os.listdir(path)):
            if not fname.endswith(".log"):
                continue
            # Expected: <model>_seed<n>.log
            m = re.match(r"(.+)_seed(\d+)\.log$", fname)
            if not m:
                print(f"  Skipping unexpected filename: {fname}")
                continue
            model_name, seed = m.group(1), int(m.group(2))
            base, aug = model_to_base_aug(model_name)
            log_path = os.path.join(path, fname)
            parsed = parse_log(log_path)
            for metric in METRICS:
                rows.append({
                    "dataset":      dataset_dir,
                    "base":         base,
                    "augmentation": aug,
                    "seed":         seed,
                    "metric":       metric,
                    "value":        parsed.get(metric, float("nan")),
                    "wall_clock_s": parsed.get("wall_clock_s", float("nan")),
                    "peak_mem_mb":  parsed.get("peak_mem_mb", float("nan")),
                })
    return pd.DataFrame(rows)


# ── Table 3 reproduction ───────────────────────────────────────────────────────

def compute_stats(df, dataset):
    sub = df[df["dataset"] == dataset]
    records = []
    for base in BASES:
        for aug in ["base", "llm_agr"]:
            group = sub[(sub["base"] == base) & (sub["augmentation"] == aug)]
            for metric in METRICS:
                vals = group[group["metric"] == metric]["value"].dropna().values
                mean = np.mean(vals) if len(vals) else float("nan")
                std  = np.std(vals, ddof=1) if len(vals) > 1 else float("nan")
                records.append({"base": base, "aug": aug, "metric": metric,
                                "mean": mean, "std": std, "vals": vals})
    return records


def significance_mark(agr_vals, best_base_vals):
    if len(agr_vals) < 2 or len(best_base_vals) < 2:
        return ""
    if len(agr_vals) != len(best_base_vals):
        return ""
    try:
        _, p = ttest_rel(agr_vals, best_base_vals)
        return "*" if p < 0.05 else ""
    except Exception:
        return ""


def build_table3_md(df, dataset):
    stats = {(r["base"], r["aug"], r["metric"]): r for r in compute_stats(df, dataset)}
    lines = [f"## {dataset.replace('_', '-').title()} Dataset", ""]
    header = "| Base | Aug | " + " | ".join(METRICS) + " |"
    sep    = "|------|-----|" + "|".join(["------"] * len(METRICS)) + "|"
    lines += [header, sep]

    for base in BASES:
        for aug in ["base", "llm_agr"]:
            cells = [base, aug]
            for metric in METRICS:
                r = stats.get((base, aug, metric))
                if r is None or math.isnan(r["mean"]):
                    cells.append("—")
                    continue
                mark = ""
                if aug == "llm_agr":
                    # Find best baseline value among base runs
                    best_base_vals = stats.get((base, "base", metric), {}).get("vals", [])
                    mark = significance_mark(r["vals"], best_base_vals)
                cells.append(f"{r['mean']:.4f}±{r['std']:.4f}{mark}")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    lines.append("\\* p < 0.05 paired t-test (LLM-AGR vs. best base for same backbone)")
    return "\n".join(lines)


# ── Paper comparison ───────────────────────────────────────────────────────────

def build_paper_vs_ours_md(df):
    lines = ["# Paper vs. Our Reproduction", "",
             "Cells where |relative diff| > 5% are flagged with ⚠️", "",
             "| Dataset | Base | Aug | Metric | Paper | Ours (mean) | Abs diff | Rel diff |",
             "|---------|------|-----|--------|-------|-------------|----------|----------|"]

    for dataset in ["amazon", "yelp"]:
        stats = {(r["base"], r["aug"], r["metric"]): r for r in compute_stats(df, dataset)}
        for base in BASES:
            for aug in ["base", "llm_agr"]:
                paper_row = PAPER_RESULTS.get((base, aug, dataset), {})
                for metric in METRICS:
                    paper_val = paper_row.get(metric)
                    ours_r    = stats.get((base, aug, metric))
                    ours_mean = ours_r["mean"] if ours_r and not math.isnan(ours_r["mean"]) else None

                    paper_str = f"{paper_val:.4f}" if paper_val is not None else "N/A"
                    ours_str  = f"{ours_mean:.4f}" if ours_mean is not None else "N/A"

                    if paper_val is not None and ours_mean is not None:
                        abs_diff = ours_mean - paper_val
                        rel_diff = abs_diff / paper_val * 100 if paper_val != 0 else float("nan")
                        flag = " ⚠️" if abs(rel_diff) > 5 else ""
                        abs_str = f"{abs_diff:+.4f}"
                        rel_str = f"{rel_diff:+.1f}%{flag}"
                    else:
                        abs_str = rel_str = "—"

                    lines.append(f"| {dataset} | {base} | {aug} | {metric} | "
                                 f"{paper_str} | {ours_str} | {abs_str} | {rel_str} |")

    return "\n".join(lines)


# ── SUMMARY ───────────────────────────────────────────────────────────────────

def build_summary_md(df):
    total_expected = 80
    total_done = df["seed"].nunique() * 0  # recount properly
    # Count unique (base, aug, dataset, seed) tuples
    done_runs = df.drop_duplicates(subset=["dataset", "base", "augmentation", "seed"])
    n_done = len(done_runs)

    lines = [
        "# Reproduction Summary",
        "",
        f"## Completion: {n_done}/{total_expected} runs",
        "",
        "## Missing Baselines (not implemented in repo)",
        "",
    ]
    for b in MISSING_BASELINES:
        lines.append(f"- {b}")

    lines += [
        "",
        "## Patches Applied",
        "",
    ]
    for p in PATCHES_APPLIED:
        lines.append(f"- {p}")

    lines += [
        "",
        "## Anomalies / Notes",
        "",
        "- BIGCF and BIGCF-AGR require `torch_sparse` and `torch_scatter` which were not in the "
        "original venv. They were installed from the PyG wheel index.",
        "- `aggregate.py` paper comparison values are placeholders (None). Fill them in from "
        "Table 3 of Zha et al., KBS 2025.",
        "",
        "## Runs with missing or malformed logs",
        "",
    ]

    # Check for missing runs
    missing = []
    for dataset in ["amazon", "yelp"]:
        for base in BASES:
            for aug in ["base", "llm_agr"]:
                model = base if aug == "base" else f"{base}_agr"
                for seed in range(5):
                    mask = ((df["dataset"] == dataset) & (df["base"] == base) &
                            (df["augmentation"] == aug) & (df["seed"] == seed))
                    if not mask.any():
                        missing.append(f"  {dataset}/{model}_seed{seed}.log")
    if missing:
        lines += [f"- {m}" for m in missing]
    else:
        lines.append("- None")

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("results", exist_ok=True)

    print("Collecting runs from logs/...")
    df = collect_all_runs("logs")

    if df.empty:
        print("No log files found. Run the grid first.")
        return

    print(f"Found {len(df) // len(METRICS)} run(s) across {len(df)} metric rows.")

    # ── Long-format CSVs
    for dataset, csv_name in DATASETS.items():
        sub = df[df["dataset"] == dataset]
        out = f"results/{csv_name}.csv"
        sub.to_csv(out, index=False)
        print(f"Wrote {out}  ({len(sub)} rows)")

    # ── Table 3
    table_parts = []
    for dataset in ["amazon", "yelp"]:
        table_parts.append(build_table3_md(df, dataset))
    table3_md = "# Table 3 Reproduction (LLM-AGR)\n\n" + "\n\n".join(table_parts)
    with open("results/table3_repro.md", "w") as f:
        f.write(table3_md)
    print("Wrote results/table3_repro.md")

    # ── Paper comparison
    pvo_md = build_paper_vs_ours_md(df)
    with open("results/paper_vs_ours.md", "w") as f:
        f.write(pvo_md)
    print("Wrote results/paper_vs_ours.md")

    # ── Summary
    summary_md = build_summary_md(df)
    with open("results/SUMMARY.md", "w") as f:
        f.write(summary_md)
    print("Wrote results/SUMMARY.md")

    print("\nDone. Fill in PAPER_RESULTS at the top of aggregate.py to complete paper_vs_ours.md.")


if __name__ == "__main__":
    main()
