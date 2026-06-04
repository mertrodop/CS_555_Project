#!/bin/bash
# run_grid.sh — Run the full 80-run paper reproduction grid locally.
# Grid size: 8 models × 2 datasets × 5 seeds = 80 runs.
set -e

# Make sure virtual environment python is used
PYTHON_BIN=~/.env/bin/python3

for model in lightgcn lightgcn_agr sgl sgl_agr simgcl simgcl_agr bigcf bigcf_agr; do
  for dataset in amazon yelp; do
    for seed in 0 1 2 3 4; do
      LOG="log/${dataset}/${model}_seed${seed}.log"
      mkdir -p "log/${dataset}"
      
      if [ -f "$LOG" ] && grep -q "Final test result" "$LOG"; then
        echo "[SKIP] Run for model=${model}, dataset=${dataset}, seed=${seed} already completed."
        continue
      fi
      
      echo "=== Running model=${model} | dataset=${dataset} | seed=${seed} ==="
      $PYTHON_BIN main.py --model $model --dataset $dataset --seed $seed 2>&1 | tee "$LOG"
    done
  done
done

echo "Grid complete. Aggregating results..."
$PYTHON_BIN aggregate.py
