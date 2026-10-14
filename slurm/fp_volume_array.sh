#!/bin/bash
#SBATCH --job-name=swmfvr_fpvol
#SBATCH --array=0-31
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --mail-type=NONE
#SBATCH --get-user-env

# The 3D flux volume for every frame, foot-point method (footpoint_volume.py:
# no stream tracing -- BATSRUS theta_1/phi_1 + RBE grid interpolation +
# Liouville). Same shape as volume_array.sh: one shard of frames per task, one
# pvbatch process per frame. ~30-60 s/frame (dominated by streaming the flux
# NPZ to the frame), vs ~6 min/frame for the traced version. Resumable:
# frames whose csv+npz exist are skipped. Outputs to volume_renders_fp/
# (the old traced product in volume_renders/ is untouched). GEI rotation
# afterwards, offline, same as before:
#   python3 scripts/volume_gei.py 'run/volume_renders_fp/npz/volume_*_bands.npz'
# Submit:
#   source slurm/env.sh
#   sbatch --chdir="$SWMF_VR_ROOT" --output="$SWMF_VR_ROOT/run/logs/fpvol_%A_%a.out" \
#          slurm/fp_volume_array.sh
set -uo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$SCRIPT_DIR/slurm/env.sh"
OUT_ROOT="${OUT_ROOT:-$SWMF_VR_ROOT/run}"
STEP="${STEP:-0.5}"
EXTENT="${EXTENT:-10}"
QUANTITY="${QUANTITY:-omni}"
RANK="${SLURM_ARRAY_TASK_ID:-0}"
NSHARDS="${SLURM_ARRAY_TASK_COUNT:-32}"

NFRAMES=1683                                              # frames in the RBE npz
BASE_EPOCH=$(date -u -d "2024-05-10 13:00:00 UTC" +%s)   # frame 0 (== config base_iso, 1-min cadence)
CSV="$OUT_ROOT/volume_renders_fp/csv"
NPZ="$OUT_ROOT/volume_renders_fp/npz"
mkdir -p "$OUT_ROOT/logs" "$CSV" "$NPZ"

echo "fp volume shard ${RANK}/${NSHARDS}  step=${STEP}  extent=${EXTENT}  quantity=${QUANTITY}"
n=0
for ((t=RANK; t<NFRAMES; t+=NSHARDS)); do
    TS=$(date -u -d "@$((BASE_EPOCH + t * 60))" +%Y%m%d_%H%M%S)
    [[ -f "$CSV/volume_${TS}_bands.csv" && -f "$NPZ/volume_${TS}_bands.npz" ]] && continue
    "$PV_BATCH" "$SWMF_VR_ROOT/scripts/footpoint_volume.py" --ts "$TS" \
        --step "$STEP" --extent "$EXTENT" --quantity "$QUANTITY" --out-root "$OUT_ROOT" \
        || echo "frame $TS failed"
    n=$((n + 1))
done
echo "fp volume shard ${RANK} done: $n frames processed."
