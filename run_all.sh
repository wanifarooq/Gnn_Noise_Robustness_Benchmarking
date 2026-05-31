#!/usr/bin/env bash
# ============================================================================
# run_all.sh — run ALL 8 benchmark configs across N GPUs (one process per GPU).
#
#   - Queue: keeps one job per GPU; when a GPU frees, it starts the next config.
#   - Resumable: main.py skips already-completed runs (run_N/training_log.json),
#     so you can Ctrl-C / reboot and just run this again — it continues.
#   - Per-config logs in logs/<config>.log ; overall log in logs/run_all.log.
#
# Usage (from the repo root):
#   chmod +x run_all.sh
#   PY=python NGPU=3 nohup ./run_all.sh > logs/run_all.log 2>&1 &
#
#   # then disconnect; check progress any time with:
#   tail -f logs/run_all.log
#   grep -h SAVED logs/*.log | wc -l        # how many (config,method,noise,run) done
#   grep -h SKIP  logs/*.log                # any skipped combos (e.g. NRGNN OOM on dense)
# ============================================================================
set -u
cd "$(dirname "$0")"                 # repo root
mkdir -p logs

PY=${PY:-python}                     # set PY=/path/to/conda/python if 'python' isn't your env
NGPU=${NGPU:-3}                      # number of GPUs to use

# Heaviest first (long gcn_modified runs) so the queue balances well.
CONFIGS=(
  configs/roman-empire_gcn_modified.yaml      # L9 h512, 2500 epochs (longest)
  configs/amazon-computers_gcn_modified.yaml  # h512 dense, 1000 epochs
  configs/cora_gcn_modified.yaml              # h512, 1000 epochs
  configs/dblp_gcn_modified.yaml              # h256, 1000 epochs
  configs/roman-empire.yaml                   # gcn+gat baseline, 500 epochs
  configs/amazon-computers.yaml               # gcn+gat baseline, dense
  configs/dblp.yaml                           # gcn+gat baseline
  configs/cora.yaml                           # gcn+gat baseline
)

echo "[$(date '+%F %T')] START  ${#CONFIGS[@]} configs on $NGPU GPUs  (PY=$PY)"

declare -A GPU_PID                   # gpu -> pid of its current job
launch() {                           # $1=gpu  $2=config
  local gpu=$1 cfg=$2 name
  name=$(basename "$cfg" .yaml)
  echo "[$(date '+%F %T')] GPU$gpu  START  $name"
  CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" main.py -c "$cfg" > "logs/${name}.log" 2>&1 &
  GPU_PID[$gpu]=$!
}

i=0
# seed the first NGPU GPUs
for ((g=0; g<NGPU && i<${#CONFIGS[@]}; g++, i++)); do launch "$g" "${CONFIGS[$i]}"; done

# as each GPU's job ends, hand it the next config
while ((i < ${#CONFIGS[@]})) || ((${#GPU_PID[@]} > 0)); do
  for g in "${!GPU_PID[@]}"; do
    if ! kill -0 "${GPU_PID[$g]}" 2>/dev/null; then
      wait "${GPU_PID[$g]}" 2>/dev/null
      echo "[$(date '+%F %T')] GPU$g  FINISHED a config"
      unset 'GPU_PID[$g]'
      if ((i < ${#CONFIGS[@]})); then launch "$g" "${CONFIGS[$i]}"; ((i++)); fi
    fi
  done
  sleep 15
done

echo "[$(date '+%F %T')] ALL CONFIGS COMPLETE."
echo "Results in results/ ; make plots with:  $PY analysis/compare_methods.py --results-dir results --out-dir analysis/plots"
