#!/usr/bin/env bash
# Station-disjoint checks on the Section 13 result.
#
# The event-disjoint grid left one question open: 149 of 154 stations appeared in
# more than one split, so part of the network's margin could be per-station site
# response it had memorised rather than waveform shape.
#
#   G  --split-by station   site response cannot leak, but events now DO
#                           (one quake at a train station and a test station
#                           shares its source term) -- so this is not clean either
#   H  --split-by both      station-disjoint AND event-disjoint; neither term can
#                           leak. Costs ~4,000 val/test rows.
#
# H is run over three independent station partitions because the doubly-disjoint
# test set holds only 23 stations, few enough that a single draw could be lucky
# or unlucky on site response alone.
set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-$SCRIPT_DIR/../.venv/bin/python}"
cd "$SCRIPT_DIR/../src"

run () {
  local tag="$1"; shift
  echo ""
  echo "############ $tag ############"
  "$PY" cnn_groundmotion.py --out-csv "../experiment_results/groundmotion_cnn_${tag}.csv" "$@" 2>&1
  echo "############ $tag done (exit $?) ############"
}

run G_station    --target pgv_fwd --split-by station --seed-split 42
run H_both_s42   --target pgv_fwd --split-by both    --seed-split 42
run H_both_s43   --target pgv_fwd --split-by both    --seed-split 43
run H_both_s44   --target pgv_fwd --split-by both    --seed-split 44

echo ""
echo "ALL DISJOINT RUNS COMPLETE"
