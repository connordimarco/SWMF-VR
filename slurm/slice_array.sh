#!/bin/bash
#SBATCH --job-name=swmfvr_slice
#SBATCH --array=0-31
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=13G
#SBATCH --time=02:00:00
#SBATCH --mail-type=NONE
#SBATCH --get-user-env

# One array task = one shard of frames (t % NSHARDS == TASK_ID), one ~9 GB worker.
# Small mem => backfills into busy nodes instantly (like the MIDL jobs), and the
# whole array runs in parallel. Resumable: frames whose png+npz exist are skipped.
# No machine specifics here -- account/partition come from SBATCH_* (set by .env),
# chdir/output are passed on the sbatch line by submit_slice_all.sh.
set -uo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$SCRIPT_DIR/slurm/env.sh"
OUT_ROOT="${OUT_ROOT:-$SWMF_VR_ROOT/run}"
N="${N:-121}"
QUANTITY="${QUANTITY:-omni}"
RANK="${SLURM_ARRAY_TASK_ID:-0}"
NSHARDS="${SLURM_ARRAY_TASK_COUNT:-32}"

echo "shard ${RANK}/${NSHARDS}  n=${N}  quantity=${QUANTITY}"
"$PV_BATCH" "$SWMF_VR_ROOT/scripts/slice3d.py" "$RANK" "$NSHARDS" \
    --n "$N" --quantity "$QUANTITY" --out-root "$OUT_ROOT"
echo "shard ${RANK} done."
