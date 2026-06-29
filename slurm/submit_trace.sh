#!/bin/bash
#SBATCH --job-name=swmfvr_trace
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem-per-cpu=4g
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=run/logs/trace_%x_%j.out
#SBATCH --error=run/logs/trace_%x_%j.err
#SBATCH --get-user-env
# account/partition from SBATCH_* (set by .env); chdir/mail passed by submit_all.sh.

# trace: trace field-line OBJ shells + colored screenshots (pvbatch, sharded).
# Three surfaces per timestep -> longer walltime than a single-surface render.
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
echo "trace: ${NUM_TASKS} workers -> $OUT_ROOT  (PV_BATCH=$PV_BATCH)"

pids=()
for ((i=0; i<NUM_TASKS; i++)); do
    "$PV_BATCH" "$SWMF_VR_ROOT/scripts/trace.py" "$i" "$NUM_TASKS" \
        --out-root "$OUT_ROOT" > "$LOG_DIR/render_worker_${i}.log" 2>&1 &
    pids+=("$!")
done

failures=0
for pid in "${pids[@]}"; do
    wait "$pid" || failures=$((failures + 1))
done
[[ "$failures" -eq 0 ]] || { echo "trace failed: ${failures} worker(s)"; exit 1; }
echo "trace done."
