# LLM-AGR: Large Language Model Augmented Graph Representation Learning for Recommendation

LLM-AGR enhances graph-based collaborative filtering models with LLM-derived user/item embeddings. It wraps four base GNN recommenders (LightGCN, SGL, SimGCL, BiGCF) with semantic embedding injection, preference knowledge distillation, adaptive graph structure learning, and an information bottleneck regularizer. Experiments run on Amazon-book and Yelp.

---

## Setup

**Prerequisites:** Python 3.9+, CUDA GPU recommended.

```bash
# 1. Install base dependencies
pip install -r requirements.txt

# 2. Install PyTorch (CUDA 12.4)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 3. Install PyG sparse ops (match your torch version)
pip install torch_sparse torch_scatter -f https://data.pyg.org/whl/torch-2.6.0+cu124.html

# 4. Download datasets (auto-downloads to data/)
python -c "from load_data.download_data import ensure_datasets; ensure_datasets()"
```

---

## Running Normal Training

**Entry point:** `main.py`

```bash
python main.py [--model MODEL] [--dataset DATASET] [--seed SEED] [--device DEVICE] [--cuda CUDA_ID] [--ablation VARIANT]
```

### CLI Flags

| Flag | Default | Options |
|------|---------|---------|
| `--model` | `lightgcn_agr` | `lightgcn`, `lightgcn_agr`, `sgl`, `sgl_agr`, `simgcl`, `simgcl_agr`, `bigcf`, `bigcf_agr` |
| `--dataset` | `amazon` | `amazon`, `yelp` |
| `--seed` | `2025` | any integer |
| `--device` | `cuda` | `cuda`, `cpu` |
| `--cuda` | `0` | GPU device index |
| `--ablation` | `none` | `none`, `wo_ags_ib`, `wo_kd`, `wo_se` |

**Ablation variants:**
- `none` — full LLM-AGR (all components enabled)
- `wo_ags_ib` — disable Adaptive Graph Structure + Information Bottleneck
- `wo_kd` — disable preference Knowledge Distillation
- `wo_se` — disable Semantic Embedding losses

### Examples

```bash
# LightGCN-AGR on Amazon, seed 42
python main.py --model lightgcn_agr --dataset amazon --seed 42

# Base LightGCN (no LLM) on Yelp
python main.py --model lightgcn --dataset yelp --seed 0

# BiGCF-AGR ablation: without knowledge distillation
python main.py --model bigcf_agr --dataset amazon --ablation wo_kd --seed 1

# Run on CPU
python main.py --model simgcl_agr --dataset yelp --device cpu --seed 0

# Use GPU 1
python main.py --model sgl_agr --dataset amazon --cuda 1 --seed 2
```

### Hyperparameter Configuration

Edit the YAML files in `config/models_config/` to tune hyperparameters. Each file has dataset-specific sections (`amazon:`, `yelp:`) that override the base values. Key LLM-AGR parameters:

| Parameter | Meaning |
|-----------|---------|
| `alpha` | LLM semantic embedding weight |
| `beta` | HSIC / information bottleneck regularization strength |
| `prf_weight` | User/item profile alignment weight |
| `kd_weight` | Knowledge distillation weight |
| `str_weight` | Structural knowledge weight |

Logs are saved to `logs/{dataset}/`, checkpoints to `checkpoint/`.

---

## Running Shilling Attacks

### Single Attack Run

Pass a YAML attack config to `main.py` via `--attack_config`:

```bash
# 1. Write an attack config file
cat > /tmp/attack.yml << 'EOF'
attack:
  enabled: true
  attack_size: 10       # % of genuine users to inject as fake users
  num_targets: 10       # number of target items to promote
  strategy: bandwagon   # bandwagon (popular filler) or random
  target_seed: 42       # RNG seed for target item selection
  emb_mode: clone       # fake embeddings: clone (copy genuine) or mean
EOF

# 2. Run training with the attack
python main.py --model bigcf_agr --dataset amazon --seed 0 --attack_config /tmp/attack.yml
```

**Attack config fields:**

| Field | Description |
|-------|-------------|
| `attack_size` | Injected fake users as % of genuine users (e.g. 10 → 10%) |
| `strategy` | `bandwagon`: fill profiles with popular items; `random`: random items |
| `emb_mode` | `clone`: copy random genuine user embeddings; `mean`: use mean embedding |
| `target_seed` | Fixes which items are selected as promotion targets |

### Attack Grid (75 runs per dataset)

`run_attack_grid.py` runs a resumable sweep over all variants, attack sizes, and seeds:

```bash
python run_attack_grid.py [dataset] [strategy]
```

**Grid dimensions (per dataset/strategy):**
- **Variants:** `full`, `wo_ags_ib`, `wo_kd`, `wo_se`, `base` (5)
- **Attack sizes:** 0%, 5%, 10%, 15%, 25% (5)
- **Seeds:** 0, 1, 2 (3)
- **Total:** 75 runs

All variants use `bigcf_agr` as the underlying model (`base` uses plain `bigcf`).

```bash
# Amazon, bandwagon strategy
python run_attack_grid.py amazon bandwagon

# Amazon, random strategy
python run_attack_grid.py amazon random

# Yelp, bandwagon strategy
python run_attack_grid.py yelp bandwagon
```

Results are appended to `results/attack_grid.csv`. Runs already present in the CSV are skipped automatically, so the script is safe to interrupt and resume.

**Output columns:** `dataset`, `base_model`, `variant`, `strategy`, `attack_size`, `seed`, `recall@20`, `ndcg@20`, `target_hr@20`, `target_exposure@20`, `wall_clock_s`, `peak_mem_mb`

### Analyzing Attack Results

```bash
python analyze_attack.py [results/attack_grid.csv]
```

Produces in `results/`:
- `attack_summary.csv` — mean ± std per (variant, attack_size)
- `robustness.png` — recall/NDCG degradation curves by variant
- `attack_success.png` — target hit-rate curves by variant
- `ATTACK_FINDINGS.md` — written summary of robustness findings

---

## Running the Reproduction Grid (Colab)

`llm_agr_repro.ipynb` runs the full 80-run reproduction sweep on Google Colab:

**Grid:** 8 models × 2 datasets × 5 seeds = **80 runs**

**Steps:**
1. Open `llm_agr_repro.ipynb` in Google Colab
2. Mount Google Drive when prompted
3. Run all cells top-to-bottom (cells copy data from Drive to local SSD, install deps, generate and execute the run script)
4. The notebook syncs logs to Drive after each run — safe to interrupt and re-run

After all 80 logs exist, aggregate results locally:

```bash
python aggregate.py
```

**Outputs in `results/`:**
| File | Contents |
|------|---------|
| `table3_repro.md` | Mean ± std table matching paper Table 3; `*` marks significance (p < 0.05) |
| `paper_vs_ours.md` | Cell-by-cell gap to reported paper numbers; ⚠️ flags > 5% relative difference |
| `SUMMARY.md` | Run count, missing logs, patches applied, anomalies |
| `amazon_book.csv` / `yelp.csv` | Long-format per-seed metric data |

---

## Project Structure

```
LLM-AGR/
├── main.py                      # Single training run entry point
├── run_attack_grid.py           # 75-run shilling attack grid
├── analyze_attack.py            # Attack result summarizer & plotter
├── aggregate.py                 # Aggregate 80 repro logs into tables
├── llm_agr_repro.ipynb          # Colab reproduction notebook
├── requirements.txt
├── config/
│   ├── configurator.py          # CLI arg + YAML config parser
│   └── models_config/           # Per-model YAML configs
│       ├── lightgcn.yml / lightgcn_agr.yml
│       ├── sgl.yml / sgl_agr.yml
│       ├── simgcl.yml / simgcl_agr.yml
│       ├── bigcf.yml / bigcf_agr.yml
│       └── default.yml
├── models/
│   ├── general_cf/              # Model implementations
│   ├── base_model.py
│   └── aug_utils.py / loss_utils.py / model_utils.py
├── trainer/
│   ├── trainer.py               # Training loop
│   └── metrics.py / logger.py / utils.py
├── load_data/
│   ├── data_handler_general_cf.py
│   ├── datasets_general_cf.py
│   └── download_data.py         # Auto-download from Google Drive
├── attack/
│   └── shilling.py              # Shilling attack injector
├── data/
│   ├── amazon/                  # trn_mat, val_mat, tst_mat, usr/itm embeddings
│   └── yelp/
├── logs/                        # Training logs (dataset/model_seedN.log)
├── checkpoint/                  # Saved model weights
└── results/                     # CSVs, markdown tables, plots
```

---

## Supported Models

| Model | Type | Description |
|-------|------|-------------|
| `lightgcn` | Base | LightGCN collaborative filtering |
| `lightgcn_agr` | LLM-enhanced | LightGCN + AGR augmentation |
| `sgl` | Base | Self-supervised Graph Learning |
| `sgl_agr` | LLM-enhanced | SGL + AGR augmentation |
| `simgcl` | Base | Simple Graph Contrastive Learning |
| `simgcl_agr` | LLM-enhanced | SimGCL + AGR augmentation |
| `bigcf` | Base | BiGCF collaborative filtering |
| `bigcf_agr` | LLM-enhanced | BiGCF + AGR augmentation |

## Supported Datasets

| Dataset | Key | Users | Notes |
|---------|-----|-------|-------|
| Amazon-book | `amazon` | — | Default dataset |
| Yelp | `yelp` | — | Restaurant reviews |
