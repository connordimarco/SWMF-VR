#!/bin/bash
# Step 4-5 parity (pvbatch): the package Stage B must reproduce v6's colored OBJ
# geometry+shading (v/vt/l lines) and color assets, for one parity timestamp.
set -uo pipefail

CRA=/nfs/turbo/coe-tuija/shared/connor_austin/CRA
REPO="$CRA/SWMF-VR"
PV="$CRA/tools/ParaView-5.12.1-osmesa-MPI-Linux-Python3.10-x86_64/bin/pvbatch"
TS=20240510_210000
SCRATCH="$REPO/tests/parity/scratch_b"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
# Oracle = archived v6 (fall back to in-place v6 if not yet archived).
V6="$CRA/archive/v6"; [ -d "$V6" ] || V6="$CRA/v6"

echo "### 1. Regenerate v6 oracle (default eqflux_peak) for $TS"
cd "$V6"
"$PV" render_field_lines_v6.py --match "$TS" 2>&1 | grep -iE 'colored by|screenshot|FAILED' | sed 's/^/    /'

echo "### 2. Stage scratch seeds (copy v6 seeds for $TS)"
rm -rf "$SCRATCH"
for s in inner outer danger; do
  mkdir -p "$SCRATCH/out_seeds_$s"
  cp "$V6/out_seeds_$s/boundary_$TS.csv" "$SCRATCH/out_seeds_$s/"
done

echo "### 3. Run package Stage B into scratch (default eqflux_peak)"
cd "$REPO"
"$PV" scripts/stage_b_render_lines.py --match "$TS" --out-root "$SCRATCH" 2>&1 \
  | grep -iE 'cells ->|screenshot|FAILED|Color:' | sed 's/^/    /'

echo "### 4. Compare v/vt/l + assets per surface"
declare -A OBJDIR=( [inner]=obj_inner [outer]=obj_outer [danger]=obj_max )
fails=0
for s in inner outer danger; do
  od="${OBJDIR[$s]}"
  ref="$V6/$od/fieldlines_$TS.obj"
  new="$SCRATCH/$od/fieldlines_$TS.obj"
  if [[ ! -f "$new" ]]; then echo "  [FAIL] $s: package OBJ missing"; fails=$((fails+1)); continue; fi
  # geometry+color lines only (ignore the comment header which intentionally differs)
  if diff <(grep -E '^(v|vt|l) ' "$ref") <(grep -E '^(v|vt|l) ' "$new") >/dev/null; then
    echo "  [PASS] $s OBJ v/vt/l identical"
  else
    nv=$(diff <(grep -E '^(v|vt|l) ' "$ref") <(grep -E '^(v|vt|l) ' "$new") | grep -c '^[<>]')
    echo "  [FAIL] $s OBJ v/vt/l differ ($nv lines)"; fails=$((fails+1))
  fi
  # assets
  if diff "$V6/$od/material.mtl" "$SCRATCH/$od/material.mtl" >/dev/null; then
    echo "  [PASS] $s material.mtl identical"; else echo "  [FAIL] $s material.mtl differ"; fails=$((fails+1)); fi
  if diff "$V6/$od/colormap.json" "$SCRATCH/$od/colormap.json" >/dev/null; then
    echo "  [PASS] $s colormap.json identical"; else echo "  [FAIL] $s colormap.json differ"; fails=$((fails+1)); fi
  if [[ "$(md5sum < "$V6/$od/ramp.png")" == "$(md5sum < "$SCRATCH/$od/ramp.png")" ]]; then
    echo "  [PASS] $s ramp.png identical"; else echo "  [FAIL] $s ramp.png differ"; fails=$((fails+1)); fi
done

echo ""
if [[ "$fails" -eq 0 ]]; then echo "STEP 4-5 PARITY PASSED"; else echo "STEP 4-5 PARITY FAILED ($fails)"; fi
exit $fails
