#!/bin/bash
#SBATCH --job-name=swmfvr_cutaway
#SBATCH --array=0-47
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=05:00:00
#SBATCH --mail-type=NONE
#SBATCH --get-user-env

# One cutaway render (fp_cutaway.py) per frame -> run/fp_cutaway_frames/.
# Same shape as fp_volume_array.sh: a shard of frames per task, one pvbatch
# process per frame, frames whose PNG exists are skipped. Needs the foot-point
# OBJs (obj_fp_outer / obj_fp_inner) for the frame; frames without them
# (the empty-plt gap) are skipped. Assemble afterwards:
#   module load ffmpeg
#   ffmpeg -framerate 30 -pattern_type glob -i 'run/fp_cutaway_frames/cutaway_*.png' \
#          -c:v libx264 -pix_fmt yuv420p run/fp_cutaway.mp4
# Submit (FRAME_DIR=$SWMF_VR_ROOT/run/fp_cutaway_frames_annotated for the 2026-09-30 annotated re-render):
#   source slurm/env.sh
#   sbatch --chdir="$SWMF_VR_ROOT" --output="$SWMF_VR_ROOT/run/logs/cutaway_%A_%a.out" \
#          slurm/fp_cutaway_array.sh
set -uo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$SCRIPT_DIR/slurm/env.sh"
OUT_ROOT="${OUT_ROOT:-$SWMF_VR_ROOT/run}"
RANK="${SLURM_ARRAY_TASK_ID:-0}"
NSHARDS="${SLURM_ARRAY_TASK_COUNT:-48}"
FRAME_DIR="${FRAME_DIR:-$OUT_ROOT/fp_cutaway_frames}"   # FRAME_DIR=... for a re-render
mkdir -p "$OUT_ROOT/logs" "$FRAME_DIR"

# Frames are whatever outer OBJs exist, in time order.
mapfile -t STAMPS < <(ls "$OUT_ROOT"/obj_fp_outer/fieldlines_*.obj | sed 's/.*fieldlines_//; s/\.obj//' | sort)
echo "cutaway shard ${RANK}/${NSHARDS}: ${#STAMPS[@]} frames total"
n=0
for ((i=RANK; i<${#STAMPS[@]}; i+=NSHARDS)); do
    TS="${STAMPS[$i]}"
    PNG="$FRAME_DIR/cutaway_${TS}.png"
    [[ -f "$PNG" ]] && continue
    [[ -f "$OUT_ROOT/obj_fp_inner/fieldlines_${TS}.obj" ]] || { echo "no inner OBJ for $TS -- skip"; continue; }
    "$PV_BATCH" "$SWMF_VR_ROOT/scripts/fp_cutaway.py" --ts "$TS" --out-root "$OUT_ROOT" --out "$PNG" \
        > "$OUT_ROOT/logs/cutaway_frame_${TS}.log" 2>&1 || echo "frame $TS failed"
    n=$((n + 1))
done
echo "cutaway shard ${RANK} done: $n frames rendered."
