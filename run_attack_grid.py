"""
run_attack_grid.py — Resumable 75-run shilling-attack grid.

Usage:
    python run_attack_grid.py [dataset] [strategy]

Defaults: dataset=amazon, strategy=bandwagon
Second pass (random strategy): python run_attack_grid.py amazon random
Yelp pass: python run_attack_grid.py yelp bandwagon
"""

import csv
import glob
import os
import re
import subprocess
import sys
import tempfile
import yaml

# ---------------------------------------------------------------------------
# Grid configuration
# ---------------------------------------------------------------------------
VARIANTS     = ['full', 'wo_ags_ib', 'wo_kd', 'wo_se', 'base']
ATTACK_SIZES = [0, 5, 10, 15, 25]
SEEDS        = [0, 1, 2]
DATASET      = sys.argv[1] if len(sys.argv) > 1 else 'amazon'
STRATEGY     = sys.argv[2] if len(sys.argv) > 2 else 'bandwagon'
TARGET_SEED  = 42
NUM_TARGETS  = 10
RESULT_FILE  = 'results/attack_grid.csv'
CSV_COLS     = [
    'dataset', 'base_model', 'variant', 'strategy', 'attack_size', 'seed',
    'recall@20', 'ndcg@20', 'target_hr@20', 'target_exposure@20',
    'wall_clock_s', 'peak_mem_mb',
]

# Mapping variant → (model_name, ablation_flag)
VARIANT_MODEL = {
    'full':      ('bigcf_agr', 'none'),
    'wo_ags_ib': ('bigcf_agr', 'wo_ags_ib'),
    'wo_kd':     ('bigcf_agr', 'wo_kd'),
    'wo_se':     ('bigcf_agr', 'wo_se'),
    'base':      ('bigcf',     'none'),
}


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------
def load_done():
    done = set()
    if not os.path.exists(RESULT_FILE):
        return done
    with open(RESULT_FILE, newline='') as f:
        for row in csv.DictReader(f):
            done.add((row['dataset'], row['variant'], row['strategy'],
                      int(row['attack_size']), int(row['seed'])))
    return done


def append_row(row_dict):
    os.makedirs('results', exist_ok=True)
    write_header = not os.path.exists(RESULT_FILE)
    with open(RESULT_FILE, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        if write_header:
            w.writeheader()
        w.writerow(row_dict)


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------
def find_latest_log(model_name):
    pattern1 = os.path.join('log', model_name, '*.log')
    pattern2 = os.path.join('logs', model_name, '*.log')
    files = glob.glob(pattern1) + glob.glob(pattern2)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def parse_repro_line(log_path):
    """Extract key=value pairs from the [REPRO] line."""
    if log_path is None or not os.path.exists(log_path):
        return {}
    with open(log_path) as f:
        for line in f:
            if '[REPRO]' in line:
                pairs = re.findall(r'(\w[\w@]+)=([\d.]+)', line)
                return {k: float(v) for k, v in pairs}
    return {}


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------
def run_single(variant, attack_size, seed):
    model_name, ablation = VARIANT_MODEL[variant]

    # Write temp attack config YAML
    attack_cfg = {
        'attack': {
            'enabled':     attack_size > 0,
            'attack_size': attack_size,
            'num_targets': NUM_TARGETS,
            'strategy':    STRATEGY,
            'target_seed': TARGET_SEED,
            'emb_mode':    'clone',
        }
    }
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml',
                                     delete=False, prefix='atk_') as tf:
        yaml.dump(attack_cfg, tf)
        atk_path = tf.name

    cmd = [
        'python', 'main.py',
        '--model',         model_name,
        '--dataset',       DATASET,
        '--seed',          str(seed),
        '--ablation',      ablation,
        '--attack_config', atk_path,
    ]

    label = f"{variant} size={attack_size} seed={seed}"
    print(f"\n[GRID] Starting: {label}", flush=True)
    print(f"[GRID] CMD: {' '.join(cmd)}", flush=True)

    result = subprocess.run(cmd, capture_output=False)

    os.unlink(atk_path)

    if result.returncode != 0:
        print(f"[GRID] FAILED: {label} (returncode={result.returncode})", flush=True)
        return

    log_path = find_latest_log(model_name)
    metrics  = parse_repro_line(log_path)

    if not metrics:
        print(f"[GRID] WARNING: no [REPRO] line found in {log_path}", flush=True)
        return

    row = {
        'dataset':          DATASET,
        'base_model':       model_name,
        'variant':          variant,
        'strategy':         STRATEGY if attack_size > 0 else 'none',
        'attack_size':      attack_size,
        'seed':             seed,
        'recall@20':        metrics.get('recall@20',        ''),
        'ndcg@20':          metrics.get('ndcg@20',          ''),
        'target_hr@20':     metrics.get('target_hr@20',     ''),
        'target_exposure@20': metrics.get('target_exposure@20', ''),
        'wall_clock_s':     metrics.get('wall_clock_s',     ''),
        'peak_mem_mb':      metrics.get('peak_mem_mb',      ''),
    }
    append_row(row)
    print(f"[GRID] Done: {label} → recall@20={row['recall@20']}  "
          f"target_hr@20={row['target_hr@20']}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    done = load_done()
    total = len(VARIANTS) * len(ATTACK_SIZES) * len(SEEDS)
    completed = 0

    for variant in VARIANTS:
        for attack_size in ATTACK_SIZES:
            for seed in SEEDS:
                key = (DATASET, variant, STRATEGY if attack_size > 0 else 'none',
                       attack_size, seed)
                if key in done:
                    print(f"[GRID] Skip (done): variant={variant} "
                          f"size={attack_size} seed={seed}")
                    completed += 1
                    continue
                run_single(variant, attack_size, seed)
                completed += 1
                print(f"[GRID] Progress: {completed}/{total}", flush=True)

    print(f"\n[GRID] Finished. Results in {RESULT_FILE}")
