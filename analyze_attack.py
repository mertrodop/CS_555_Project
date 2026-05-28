"""
analyze_attack.py — Summarize and plot shilling-attack grid results.

Usage:
    python analyze_attack.py [results/attack_grid.csv]

Outputs (all in results/):
    attack_summary.csv
    robustness.png
    attack_success.png
    ATTACK_FINDINGS.md
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULT_FILE  = sys.argv[1] if len(sys.argv) > 1 else 'results/attack_grid.csv'
OUT_DIR      = 'results'
METRIC_COLS  = ['recall@20', 'ndcg@20', 'target_hr@20', 'target_exposure@20']

VARIANT_ORDER = ['full', 'wo_ags_ib', 'wo_kd', 'wo_se', 'base']
VARIANT_LABEL = {
    'full':      'LLM-AGR (full)',
    'wo_ags_ib': 'w/o AGS+IB',
    'wo_kd':     'w/o KD',
    'wo_se':     'w/o SE',
    'base':      'BIGCF (base)',
}
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']


def load_data():
    if not os.path.exists(RESULT_FILE):
        raise FileNotFoundError(f"Results file not found: {RESULT_FILE}")
    df = pd.read_csv(RESULT_FILE)
    for col in METRIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def make_summary(df):
    grp = df.groupby(['variant', 'attack_size'])
    rows = []
    for (variant, attack_size), g in grp:
        row = {'variant': variant, 'attack_size': attack_size}
        for col in METRIC_COLS:
            if col in g.columns:
                row[f'{col}_mean'] = g[col].mean()
                row[f'{col}_std']  = g[col].std(ddof=1) if len(g) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def plot_metric(summary, metric, ylabel, filename, title):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    mean_col = f'{metric}_mean'
    std_col  = f'{metric}_std'
    attack_sizes = sorted(summary['attack_size'].unique())

    for i, variant in enumerate(VARIANT_ORDER):
        sub = summary[summary['variant'] == variant].sort_values('attack_size')
        if sub.empty or mean_col not in sub.columns:
            continue
        means = sub[mean_col].values
        stds  = sub[std_col].values if std_col in sub.columns else np.zeros_like(means)
        sizes = sub['attack_size'].values
        color = COLORS[i % len(COLORS)]
        label = VARIANT_LABEL.get(variant, variant)
        ax.plot(sizes, means, marker='o', label=label, color=color)
        ax.fill_between(sizes, means - stds, means + stds, alpha=0.15, color=color)

    ax.set_xlabel('Attack size (%)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='best', fontsize=9)
    ax.set_xticks(attack_sizes)
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, filename)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[ANALYZE] Saved {out_path}")


def write_findings(summary):
    # Compute slope of degradation (recall@20 from size=0 to size=25) per variant
    slopes = {}
    for variant in VARIANT_ORDER:
        sub = summary[summary['variant'] == variant].sort_values('attack_size')
        if sub.empty or 'recall@20_mean' not in sub.columns:
            continue
        base_row = sub[sub['attack_size'] == 0]
        top_row  = sub[sub['attack_size'] == 25]
        if base_row.empty or top_row.empty:
            continue
        base_val = base_row['recall@20_mean'].values[0]
        top_val  = top_row['recall@20_mean'].values[0]
        slopes[variant] = top_val - base_val  # negative = degradation

    slope_str = '\n'.join(
        f"  - {VARIANT_LABEL.get(v, v):20s}: Δrecall@20 = {slopes[v]:+.4f}"
        for v in VARIANT_ORDER if v in slopes
    )

    md = f"""# Shilling Attack Findings — LLM-AGR Robustness Study

## Setup
- Dataset: see attack_grid.csv
- Attack strategy: bandwagon
- Variants: {', '.join(VARIANT_ORDER)}
- Attack sizes: 0%, 5%, 10%, 15%, 25% of genuine users
- 3 random seeds per condition; reported as mean ± std

## Recall@20 Degradation (Δ from clean to 25% attack)

{slope_str}

## Key Observations

### Which component absorbs the attack?
[TO BE FILLED after results]

Compare slopes: a smaller absolute Δrecall means better robustness.
- **full vs wo_ags_ib**: Does removing Adaptive Graph Structure + IB cause larger degradation?
  If yes, AGS+IB acts as a defence mechanism (denoises attacked edges).
- **full vs wo_kd**: If KD removal hurts robustness, the knowledge distillation path
  (preference alignment) provides some regularisation that suppresses attacker influence.
- **full vs wo_se**: If removing semantic embeddings causes larger drop, LLM embeddings
  provide an out-of-band signal that is harder for attackers to spoof.
- **base (BIGCF) vs full**: Quantifies the net robustness gain from LLM-AGR as a whole.

### Attack success (target_hr@20)
[TO BE FILLED after results]

Compare target_hr@20 across variants: a higher hit ratio means the attack was more
effective at promoting target items. Variants that suppress target_hr better are more
robust from an adversarial perspective.

## Plots
- `robustness.png`      — Recall@20 vs attack size (quality degradation)
- `attack_success.png`  — target_hr@20 vs attack size (attacker perspective)
"""
    out_path = os.path.join(OUT_DIR, 'ATTACK_FINDINGS.md')
    with open(out_path, 'w') as f:
        f.write(md)
    print(f"[ANALYZE] Saved {out_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_data()
    print(f"[ANALYZE] Loaded {len(df)} rows from {RESULT_FILE}")

    summary = make_summary(df)
    summary_path = os.path.join(OUT_DIR, 'attack_summary.csv')
    summary.to_csv(summary_path, index=False)
    print(f"[ANALYZE] Saved {summary_path}")

    plot_metric(summary, 'recall@20',
                ylabel='Recall@20', filename='robustness.png',
                title='Recommendation Quality Under Shilling Attack')

    plot_metric(summary, 'target_hr@20',
                ylabel='Target HR@20', filename='attack_success.png',
                title='Attack Success (Target Item Hit Ratio)')

    write_findings(summary)
    print("[ANALYZE] Done.")


if __name__ == '__main__':
    main()
