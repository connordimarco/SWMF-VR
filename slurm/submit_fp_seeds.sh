#!/bin/bash
#SBATCH --job-name=swmfvr_fpseeds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem-per-cpu=4g
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --get-user-env
# account/partition from SBATCH_* (set by .env). Submit:
#   source slurm/env.sh
#   sbatch --chdir="$SWMF_VR_ROOT" --output="$SWMF_VR_ROOT/run/logs/fpseeds_%j.out" \
#          slurm/submit_fp_seeds.sh

# Foot-point boundary seeds for every frame (system python3, sharded). Each
# worker streams the whole flux NPZ, so 8 workers is a good balance between
# wall time and turbo I/O. Resumable: frames whose 3 CSVs exist are skipped.
set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
source "$SCRIPT_DIR/slurm/env.sh"
OUT_ROOT="${OUT_ROOT:-$SWMF_VR_ROOT/run}"
LOG_DIR="$OUT_ROOT/logs"
mkdir -p "$LOG_DIR"

NUM_TASKS="${SLURM_NTASKS_PER_NODE:-8}"
echo "fp seeds: ${NUM_TASKS} workers -> $OUT_ROOT"

pids=()
for ((i=0; i<NUM_TASKS; i++)); do
    python3 "$SWMF_VR_ROOT/scripts/footpoint_seeds.py" "$i" "$NUM_TASKS" \
        --out-root "$OUT_ROOT" > "$LOG_DIR/fpseeds_worker_${i}.log" 2>&1 &
    pids+=("$!")
done

failures=0
for pid in "${pids[@]}"; do
    wait "$pid" || failures=$((failures + 1))
done
[[ "$failures" -eq 0 ]] || { echo "fp seeds failed: ${failures} worker(s)"; exit 1; }
echo "fp seeds done."
