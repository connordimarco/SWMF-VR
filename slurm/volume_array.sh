#!/bin/bash
#SBATCH --job-name=swmfvr_vol
#SBATCH --array=0-31
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --mail-type=NONE
#SBATCH --get-user-env

# The 3D flux volume for every frame -- same shape as slice_array.sh (one shard of
# frames per task, one pvbatch process per frame so ParaView memory is reclaimed),
# just the volume script and its bigger memory box (~2.3x the slice's trace points).
# Resumable: frames whose csv+npz already exist are skipped. After it finishes,
# rotate everything to GEI offline in one pass:
#   python3 scripts/volume_gei.py 'run/volume_renders/npz/volume_*_bands.npz'
# Submit:
#   source slurm/env.sh
#   sbatch --chdir="$SWMF_VR_ROOT" --output="$SWMF_VR_ROOT/run/logs/vol_arr_%A_%a.out" \
#          slurm/volume_array.sh
set -uo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$SCRIPT_DIR/slurm/env.sh"
OUT_ROOT="${OUT_ROOT:-$SWMF_VR_ROOT/run}"
STEP="${STEP:-0.5}"
EXTENT="${EXTENT:-10}"
QUANTITY="${QUANTITY:-omni}"
RANK="${SLURM_ARRAY_TASK_ID:-0}"
NSHARDS="${SLURM_ARRAY_TASK_COUNT:-32}"

NFRAMES=1683                                              # frames in the CIMI npz
BASE_EPOCH=$(date -u -d "2024-05-10 13:00:00 UTC" +%s)   # frame 0 (== config base_iso, 1-min cadence)
CSV="$OUT_ROOT/volume_renders/csv"
NPZ="$OUT_ROOT/volume_renders/npz"
mkdir -p "$OUT_ROOT/logs" "$CSV" "$NPZ" "$OUT_ROOT/volume_renders/seeds"

echo "shard ${RANK}/${NSHARDS}  step=${STEP}  extent=${EXTENT}  quantity=${QUANTITY}  -- one process per frame"
n=0
for ((t=RANK; t<NFRAMES; t+=NSHARDS)); do
    TS=$(date -u -d "@$((BASE_EPOCH + t * 60))" +%Y%m%d_%H%M%S)
    [[ -f "$CSV/volume_${TS}_bands.csv" && -f "$NPZ/volume_${TS}_bands.npz" ]] && continue
    "$PV_BATCH" "$SWMF_VR_ROOT/scripts/volume3d.py" --ts "$TS" \
        --step "$STEP" --extent "$EXTENT" --quantity "$QUANTITY" --out-root "$OUT_ROOT" \
        || echo "  frame $TS failed (continuing)"
    n=$((n + 1))
done
echo "shard ${RANK}: processed $n frames"
