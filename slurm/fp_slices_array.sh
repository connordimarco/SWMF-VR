#!/bin/bash
#SBATCH --job-name=swmfvr_fpslices
#SBATCH --array=0-31
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --mail-type=NONE
#SBATCH --get-user-env

# Meridional flux slices (every 360/N deg about the dipole axis, no NaNs) for
# every frame -- scripts/fp_slices.py. Same shape as fp_volume_array.sh: one
# shard of frames per task, one pvbatch process per frame (ParaView leaks
# across pipeline rebuilds). Resumable: frames whose npz exists are skipped.
# Outputs to run/fp_slices/{csv,npz}/. Submit:
#   source slurm/env.sh
#   sbatch --chdir="$SWMF_VR_ROOT" --output="$SWMF_VR_ROOT/run/logs/fpslices_%A_%a.out" \
#          slurm/fp_slices_array.sh
# Knobs (env): STEP (Re, 0.1), EXTENT (Re, 10), NSLICES (24), FRAME (gei|sm|gsm).
set -uo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$SCRIPT_DIR/slurm/env.sh"
OUT_ROOT="${OUT_ROOT:-$SWMF_VR_ROOT/run}"
STEP="${STEP:-0.1}"
EXTENT="${EXTENT:-10}"
NSLICES="${NSLICES:-24}"
FRAME="${FRAME:-gei}"
RANK="${SLURM_ARRAY_TASK_ID:-0}"
NSHARDS="${SLURM_ARRAY_TASK_COUNT:-32}"

NFRAMES=1683                                              # frames in the RBE npz
BASE_EPOCH=$(date -u -d "2024-05-10 13:00:00 UTC" +%s)   # frame 0 (== config base_iso, 1-min cadence)
NPZ="$OUT_ROOT/fp_slices/npz"
mkdir -p "$OUT_ROOT/logs" "$NPZ"

echo "fp slices shard ${RANK}/${NSHARDS}  step=${STEP}  extent=${EXTENT}  nslices=${NSLICES}  frame=${FRAME}"
n=0
for ((t=RANK; t<NFRAMES; t+=NSHARDS)); do
    TS=$(date -u -d "@$((BASE_EPOCH + t * 60))" +%Y%m%d_%H%M%S)
    [[ -f "$NPZ/slices_${TS}.npz" ]] && continue
    "$PV_BATCH" "$SWMF_VR_ROOT/scripts/fp_slices.py" --ts "$TS" --no-plot \
        --step "$STEP" --extent "$EXTENT" --n-slices "$NSLICES" --frame "$FRAME" \
        --out-root "$OUT_ROOT" 2>&1 | grep -vE 'IGRF|^Warning|vtk|Generic Warning' \
        || echo "frame $TS failed"
    n=$((n + 1))
done
echo "fp slices shard ${RANK} done: $n frames processed."
