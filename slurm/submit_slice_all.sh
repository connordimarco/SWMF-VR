#!/bin/bash
# Launch the whole y=0 slice run as a JOB ARRAY (32 small ~13 GB tasks that
# backfill into busy nodes instantly), then a dependent job to stitch the movie.
# Run on the login node -- it just submits:
#
#   ./slurm/submit_slice_all.sh
#
# Reads everything from .env (via env.sh): account/partition come through SBATCH_*,
# and chdir/output/mail are passed on the sbatch line. Resumable -- re-running skips
# frames whose png+npz already exist. Override via env: N, QUANTITY, FPS.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SCRIPT_DIR/slurm/env.sh"
OUT_ROOT="${OUT_ROOT:-$SWMF_VR_ROOT/run}"
SLICE="$OUT_ROOT/slice_renders"
mkdir -p "$OUT_ROOT/logs" "$SLICE/png" "$SLICE/npz" "$SLICE/seeds"

# context strips (Bz/AL/Dst) -- build once if missing (light, runs here)
[[ -f "$SLICE/indices.npz" ]] || python3 "$SWMF_VR_ROOT/scripts/parse_indices.py" --out-root "$OUT_ROOT"

# cluster-specific sbatch flags built from .env (job scripts carry none)
common=(--chdir="$SWMF_VR_ROOT")
[[ -n "${SWMFVR_MAIL:-}" ]] && common+=(--mail-user="$SWMFVR_MAIL" --mail-type=END,FAIL)

ARR=$(sbatch --parsable "${common[@]}" \
      --output="$OUT_ROOT/logs/slice_arr_%A_%a.out" "$SCRIPT_DIR/slurm/slice_array.sh")
echo "slice array: job $ARR  (32 tasks x 24 GB, one process per frame)"
MOV=$(sbatch --parsable "${common[@]}" --dependency=afterany:"$ARR" \
      --output="$OUT_ROOT/logs/slice_movie_%j.out" "$SCRIPT_DIR/slurm/slice_movie.sh")
echo "movie:       job $MOV  (after $ARR)  -> $SLICE/movie.mp4"
echo
echo "monitor:  squeue -u \$USER | grep swmfvr"
echo "          ls $SLICE/png | wc -l   # frames done / 1683"
