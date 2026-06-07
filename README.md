# LLM-AGR: Large Language Model Augmented Graph Representation Learning for Recommendation

This repository contains the implementation, reproduction pipelines, adversarial robustness evaluations, and generative explainability extensions for the **LLM-AGR** (Large Language Model Augmented Graph Representation Learning) framework. 

LLM-AGR enhances standard graph-based collaborative filtering (GNN) recommenders by integrating LLM-derived semantic embeddings, optimizing user/item profile alignments, executing preference knowledge distillation, and employing adaptive graph structure learning with an information bottleneck regularizer to filter out noise.

---

## Key Features & Extensions

1. **Base GNN Models & AGR Wrappers:** Implements four base collaborative filtering backbones (`LightGCN`, `SGL`, `SimGCL`, and `BiGCF`) and their respective augmented `_agr` variants.
2. **Adversarial Robustness (Augmentation 1):** Robustness sweeps and shilling attack simulations (injecting fake bot profiles with Random or Bandwagon filler strategies to promote specific target items) to evaluate adaptive structural denoising.
3. **Generative Explainability Module (Augmentation 2):** Maps collaborative graph representations back to the LLM semantic space using trained MLP projectors. It retrieves semantic nearest-neighbor items and leverages a local **Qwen2.5-7B-Instruct** model to generate fluent, personalized recommendations explanations.

---

## System Requirements & Hardware Context
- **OS:** Linux (tested on Ubuntu)
- **Python:** Python 3.9+
- **GPU:** CUDA compatible GPU with high VRAM (tested on RTX 5090).
- **Local Language Models:**
  - **Generator:** `Qwen/Qwen2.5-7B-Instruct` (loaded with `device_map='auto'` and `torch_dtype='auto'` for optimal GPU/VRAM utilization).
  - **Semantic Evaluator:** SentenceTransformer `all-MiniLM-L6-v2` for computing cosine similarities between generated explanations and ground-truth reasoning texts.

---

## Setup & Installation

### 1. Install Base & PyTorch Dependencies
Install the required packages:

```bash
# Clone the repository and navigate inside
cd CS_555_Project

# Install base dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA 12.4 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install PyG sparse operations matching PyTorch 2.6.0
pip install torch_sparse torch_scatter -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
```

### 2. Install Explainability Dependencies
The explainability module relies on standard text-processing and transformer packages:

```bash
pip install transformers accelerate sentence-transformers scikit-learn matplotlib pandas
```

### 3. Download Datasets
Initialize and download the required pre-processed datasets (`Amazon-book` and `Yelp` containing user/item graphs and LLM profile embeddings):

```bash
python -c "from load_data.download_data import ensure_datasets; ensure_datasets()"
```

---

## Workflow 1: Paper Replication (80-Run Grid)

To replicate the paper's main recommendation performance results (Table 3), we evaluate 8 models across 2 datasets with 5 random seeds (totaling 80 runs).

### Option A: Local Grid Sweep (Recommended)
You can run the full sequential grid sweep locally using the provided shell script. It is designed to automatically skip already completed runs (resumable):

```bash
chmod +x run_grid.sh
./run_grid.sh
```

### Option B: Google Colab Sweep
Use the Jupyter Notebook `llm_agr_repro.ipynb` to mount Google Drive, install dependencies, and execute the reproduction grid in a cloud GPU environment. Logs are synchronized dynamically to Google Drive to allow seamless resume states.

### Aggregating & Visualizing Replication Results
Once the log files are generated under `log/{dataset}/`, parse and format them into comparative tables matching the paper's formats:

```bash
python aggregate.py
```

This generates three files under `results/`:
- `table3_repro.md`: Mean $\pm$ std dev metrics table (Recall@K and NDCG@K) matching paper Table 3, with significance markers (`*` for $p < 0.05$).
- `paper_vs_ours.md`: A cell-by-cell comparison outlining the exact metric differences against the paper's reported values.
- `SUMMARY.md`: High-level execution summary, anomalies, and logs inventory.

---

## Workflow 2: Shilling Attack Robustness (75-Run Grid)

To test the robustness of the Adaptive Graph Structure (AGS) learning and Information Bottleneck (IB) regularizers, we perform malicious bot injection sweeps.

### 1. Running the Attack Sweep
Run the grid sweep over variants (full, ablations, base models), attack sizes (0%, 5%, 10%, 15%, 25%), and seeds:

```bash
# Run attack sweep for a specific dataset and strategy
python run_attack_grid.py amazon bandwagon
python run_attack_grid.py amazon random
```

Results are saved to `results/attack_grid.csv`. The script skips existing lines in the CSV file, allowing you to stop and resume at any time.

### 2. Plotting and Analysing Results
Once the grid runs complete, summarize the attack data and generate trend curves:

```bash
python analyze_attack.py results/attack_grid.csv
```

This outputs the following artifacts in the `results/` directory:
- `attack_summary.csv`: Aggregated mean and standard deviations of metric degradations.
- `robustness.png`: Line plots showing Recall/NDCG degradation under increasing noise.
- `attack_success.png`: Line plots tracking the target item promotion success rate (Target HR@20 and Exposure@20).
- `ATTACK_FINDINGS.md`: A summary markdown file analyzing model vulnerabilities under each attack strategy.

### 3. Custom Single Attack Config
For ad-hoc testing, write a custom configuration YAML file and pass it using `--attack_config`:

```bash
# example_attack.yml
attack:
  enabled: true
  attack_size: 10       # % of fake users to inject relative to genuine users
  num_targets: 10       # number of long-tail items to push
  strategy: bandwagon   # filler profile selection strategy: bandwagon or random
  target_seed: 42
  emb_mode: clone       # fake bot embeddings: clone (copy from user) or mean
```

Execute training under attack:
```bash
python main.py --model bigcf_agr --dataset amazon --seed 42 --attack_config example_attack.yml
```

---

## Workflow 3: Generative Explainability & Semantic Projector

The Generative Explainability Module bridges graph collaborative representations and natural language preferences using a multi-step pipeline:
1. Extract user and item collaborative embeddings from the GNN backbone.
2. Project collaborative embeddings to the LLM semantic space using the trained projector MLP (`gen_mlp`).
3. Retrieve semantic nearest-neighbor item descriptions from the dataset.
4. Prompt a local `Qwen2.5-7B-Instruct` generator to synthesize explanations matching user interests to item traits.

### Running the Module
Generate personalized recommendations explanations and calculate semantic alignments for a sample of users:

```bash
python explain.py --model lightgcn_agr --dataset amazon --checkpoint ./checkpoint/lightgcn_agr/lightgcn_agr-amazon-42.pth --num_users 5 --output results/explainability_report.md
```

### Key Architectural Insight
* **MLP Projector Behavior:** During testing, we noticed that in `LightGCN_AGR`, the projected embeddings had low cosine similarity (around random) with the ground-truth LLM embeddings. Code analysis revealed that `LightGCN_AGR.cal_loss` does not compute or optimize the reconstruction loss (`recon_loss`), leaving its `self.gen_mlp` mapping unoptimized. In contrast, `BiGCF_AGR` optimizes `recon_loss` alongside standard collaborative filtering objectives, leading to high-quality, aligned semantic embeddings.

---

## Project Structure

```
CS_555_Project/
├── README.md                    # This instructions file
├── requirements.txt             # Primary environment dependencies
├── main.py                      # Training & model execution script
├── run_grid.sh                  # Sequential 80-run replication sweep
├── aggregate.py                 # Replication log aggregator & compiler
├── run_attack_grid.py           # Resumable 75-run shilling grid sweep
├── analyze_attack.py            # Shilling attack data analyzer & plotter
├── explain.py                   # Generative recommendation explainability module
├── llm_agr_repro.ipynb          # Google Colab replication notebook
├── config/                      # Configurations and parameters
│   ├── configurator.py          # Argument parser and configs merger
│   └── models_config/           # Per-model and per-dataset YAML parameters
├── models/                      # Model architecture implementations
│   ├── general_cf/              # GNN backbones (LightGCN, SGL, SimGCL, BiGCF)
│   ├── base_model.py            # Shared GNN model logic
│   └── aug_utils.py             # AGR augmentations (AGS, IB, KD, projectors)
├── trainer/                     # Training loops, loss metrics, and loggers
├── load_data/                   # Graph construction and profile loaders
├── attack/                      # Shilling attack injector logic
├── data/                        # Graph datasets, raw profiles & pre-saved LLM embeddings
├── log/                         # Output training logs
├── checkpoint/                  # Saved .pth weights
└── results/                     # CSV logs, markdown reports, and trend curves
```
