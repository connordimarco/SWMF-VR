#!/bin/bash
#SBATCH --account=tuija98
#SBATCH --job-name=swmfvr_seeds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem-per-cpu=4g
#SBATCH --time=12:00:00
#SBATCH --partition=standard
#SBATCH --mail-user=cdimarco@umich.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --chdir=/nfs/turbo/coe-tuija/shared/connor_austin/CRA/SWMF-VR
#SBATCH --output=/nfs/turbo/coe-tuija/shared/connor_austin/CRA/SWMF-VR/run/logs/seeds_%x_%j.out
#SBATCH --error=/nfs/turbo/coe-tuija/shared/connor_austin/CRA/SWMF-VR/run/logs/seeds_%x_%j.err
#SBATCH --get-user-env

# Stage A: 2D flux-slice plots + per-surface seed CSVs (system python3, sharded).
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
echo "Stage A (seeds): ${NUM_TASKS} workers -> $OUT_ROOT"

pids=()
for ((i=0; i<NUM_TASKS; i++)); do
    python3 "$SWMF_VR_ROOT/scripts/stage_a_flux_seeds.py" "$i" "$NUM_TASKS" \
        --out-root "$OUT_ROOT" > "$LOG_DIR/seeds_worker_${i}.log" 2>&1 &
    pids+=("$!")
done

failures=0
for pid in "${pids[@]}"; do
    wait "$pid" || failures=$((failures + 1))
done
[[ "$failures" -eq 0 ]] || { echo "Stage A failed: ${failures} worker(s)"; exit 1; }
echo "Completed Stage A."
