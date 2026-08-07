#!/usr/bin/env bash
# Ground-motion experiment grid. Each configuration answers one question that
# was written down before it was run; results land in groundmotion_cnn_*.csv.
#
#   A  main            does shape beat the peak-amplitude floor at all?
#   B  --arch cnn      does the BiLSTM+attention earn its parameters?
#   C  --no-distance   given shape, is the distance scalar still needed?
#   D  waveform only   with no scalars, can the network recover amplitude
#                      itself? (input-norm none, or it has no amplitude to see)
#   E  pga_fwd         the paper's quantity on the honest, non-overlapping window
#   F  pga_full        the degenerate target, for contrast only -- NOT a
#                      like-for-like number, its window contains the input
set -u
PY="${PY:-/home/oguzb/Projects/sismokaos/data_downloader/.venv/bin/python}"
cd "$(dirname "$0")"

run () {
  local tag="$1"; shift
  echo ""
  echo "############ $tag ############"
  "$PY" cnn_groundmotion.py --out-csv "groundmotion_cnn_${tag}.csv" "$@" 2>&1
  echo "############ $tag done (exit $?) ############"
}

run A_main       --target pgv_fwd  --arch cnn_lstm
run B_nolstm     --target pgv_fwd  --arch cnn
run C_nodist     --target pgv_fwd  --arch cnn_lstm --no-distance
run D_waveonly   --target pgv_fwd  --arch cnn_lstm --no-aux --input-norm none
run E_pga        --target pga_fwd  --arch cnn_lstm
run F_pgafull    --target pga_full --arch cnn_lstm

echo ""
echo "ALL EXPERIMENTS COMPLETE"
