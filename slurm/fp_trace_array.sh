#!/bin/bash
#SBATCH --job-name=swmfvr_fptrace
#SBATCH --array=0-31
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --mail-type=NONE
#SBATCH --get-user-env

# Foot-point hybrid field-line OBJs for every frame (pvbatch, 32-shard array).
# Each task runs one long-lived pvbatch over its ~53 frames -- the same shape
# that submit_trace.sh used for the old pipeline. Resumable via
# --skip-existing (frames whose OBJs all exist are skipped). Submit after the
# seeds job, with a dependency:
#   source slurm/env.sh
#   sbatch --chdir="$SWMF_VR_ROOT" \
#          --output="$SWMF_VR_ROOT/run/logs/fptrace_%A_%a.out" \
#          --dependency=afterok:<SEEDS_JOB_ID> slurm/fp_trace_array.sh
set -uo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$SCRIPT_DIR/slurm/env.sh"
OUT_ROOT="${OUT_ROOT:-$SWMF_VR_ROOT/run}"
RANK="${SLURM_ARRAY_TASK_ID:-0}"
NSHARDS="${SLURM_ARRAY_TASK_COUNT:-32}"
mkdir -p "$OUT_ROOT/logs"

echo "fp trace shard ${RANK}/${NSHARDS} -> $OUT_ROOT  (PV_BATCH=$PV_BATCH)"
"$PV_BATCH" "$SWMF_VR_ROOT/scripts/footpoint_trace.py" "$RANK" "$NSHARDS" \
    --skip-existing --out-root "$OUT_ROOT"
echo "fp trace shard ${RANK} done."
