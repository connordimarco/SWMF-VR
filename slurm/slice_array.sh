#!/bin/bash
#SBATCH --job-name=swmfvr_slice
#SBATCH --array=0-31
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --mail-type=NONE
#SBATCH --get-user-env

# One array task = one shard of frames (t % NSHARDS == TASK_ID). Each frame is its
# OWN pvbatch process -- ParaView accumulates memory across pipeline rebuilds, so a
# single long-lived process OOMs after a few frames; per-frame processes reclaim it.
# 24 GB easily holds one frame and still backfills into busy nodes. Resumable: frames
# whose png+npz already exist are skipped. No machine specifics (account/partition
# from SBATCH_*; chdir/output on the sbatch line via submit_slice_all.sh).
set -uo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$SCRIPT_DIR/slurm/env.sh"
OUT_ROOT="${OUT_ROOT:-$SWMF_VR_ROOT/run}"
N="${N:-121}"
QUANTITY="${QUANTITY:-omni}"
RANK="${SLURM_ARRAY_TASK_ID:-0}"
NSHARDS="${SLURM_ARRAY_TASK_COUNT:-32}"

NFRAMES=1683                                              # frames in the CIMI npz
BASE_EPOCH=$(date -u -d "2024-05-10 13:00:00 UTC" +%s)   # frame 0 (== config base_iso, 1-min cadence)
PNG="$OUT_ROOT/slice_renders/png"
NPZ="$OUT_ROOT/slice_renders/npz"

echo "shard ${RANK}/${NSHARDS}  n=${N}  quantity=${QUANTITY}  -- one process per frame"
n=0
for ((t=RANK; t<NFRAMES; t+=NSHARDS)); do
    TS=$(date -u -d "@$((BASE_EPOCH + t * 60))" +%Y%m%d_%H%M%S)
    [[ -f "$PNG/slice_y0_${TS}_bands_${QUANTITY}.png" && -f "$NPZ/slice_y0_${TS}_bands.npz" ]] && continue
    "$PV_BATCH" "$SWMF_VR_ROOT/scripts/slice3d.py" --ts "$TS" \
        --n "$N" --quantity "$QUANTITY" --out-root "$OUT_ROOT" \
        || echo "  frame $TS failed (continuing)"
    n=$((n + 1))
done
echo "shard ${RANK}: processed $n frames"
