# Reproduction Summary

## Completion: 80/80 runs

## Missing Baselines (not implemented in repo)

- KAR
- RLMRec-Con
- RLMRec-Gen
- Semantic Only
- AlphaRec

## Patches Applied

- main.py:11 — set_seed(2025) → set_seed(configs['train']['seed'])
- config/configurator.py:36 — if args.seed: → if args.seed is not None:
- trainer/trainer.py — added wall-clock timing and GPU memory logging ([REPRO] line)
- venv — installed torch_sparse and torch_scatter from PyG wheel index

## Anomalies / Notes

- BIGCF and BIGCF-AGR require `torch_sparse` and `torch_scatter` which were not in the original venv. They were installed from the PyG wheel index.
- `aggregate.py` paper comparison values are placeholders (None). Fill them in from Table 3 of Zha et al., KBS 2025.

## Runs with missing or malformed logs

- None