#!/usr/bin/env bash
# Raw flux -> shaded GSM objs -> GEI objs + rotation matrices -> a GSM/GEI plot.
#
#   ./run.sh [START_IDX] [COUNT] [COLOR_BY]
#
# Does COUNT frames starting at START_IDX. Defaults: the 21:00 frame (480),
# colored by eqflux. Everything lands under run/.
#   ./run.sh                # one frame
#   ./run.sh 480 1 radius   # one frame, colored by radius
#   ./run.sh 0 1683         # the whole thing (slow on one process -- use slurm/)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/slurm/env.sh"

START="${1:-480}"; COUNT="${2:-1}"; COLOR_BY="${3:-eqflux}"
OUT="$ROOT/run"

echo "[1/4] seeds (frames $START..$((START+COUNT-1)))"
python3 "$ROOT/scripts/seeds.py" 0 1 --start "$START" --max-frames "$COUNT" --out-root "$OUT"

echo "[2/4] trace + shade ($COLOR_BY)"
"$PV_BATCH" "$ROOT/scripts/trace.py" 0 1 --color-by "$COLOR_BY" --out-root "$OUT"

echo "[3/4] GEI objs + rotation matrices"
python3 "$ROOT/scripts/gei.py" --obj-dir "$OUT/obj_inner" --out-dir "$OUT/obj_inner_gei"
python3 "$ROOT/scripts/gei.py" --obj-dir "$OUT/obj_max"   --out-dir "$OUT/obj_max_gei"
python3 "$ROOT/scripts/gei.py" --obj-dir "$OUT/obj_outer" --out-dir "$OUT/obj_outer_gei" \
    --transforms "$OUT/gsm_transforms.csv"

echo "[4/4] plot"
TS="$(ls "$OUT"/obj_outer/fieldlines_*.obj | head -1 | sed 's/.*fieldlines_//; s/\.obj//')"
python3 "$ROOT/scripts/plot.py" --ts "$TS" --gsm-root "$OUT" \
    --transforms "$OUT/gsm_transforms.csv" --out "$OUT/shell_renders/shells_${TS}.png"

echo "done -> $OUT/  (objs: obj_*/  GEI: obj_*_gei/  matrices: gsm_transforms.csv  plot: shell_renders/)"
