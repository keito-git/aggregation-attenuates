#!/bin/bash
# Dose-response runs (results/p3): aggregation strength on GoEmotions x Qwen3-32B.
# Strength = total number of interaction rounds L (the initial independent round included):
# debate and MoA with L in {1, 2, 4}, seeds 43/44/45. L = 3 is the main setting and is taken from the main
# runs (results/p1). Same scripts and arguments as the main runs; only --rounds differs.
#
# Usage:   bash scripts/run_dose_response.sh
# Environment (defaults in brackets): PY [python3]  HF_CLI [hf]  DATA_DIR [data]  REF_DIR [ref]
#   RESULTS_DIR [results]  LOG_DIR [logs]  MODEL_DIR [models]  QWEN_REVISION (optional)

set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-python3}
HF_CLI=${HF_CLI:-hf}
DATA_DIR=${DATA_DIR:-$ROOT/data}
REF_DIR=${REF_DIR:-$ROOT/ref}
RESULTS=${RESULTS_DIR:-$ROOT/results}/p3
LOGS=${LOG_DIR:-$ROOT/logs}/p3
MODEL_DIR=${MODEL_DIR:-$ROOT/models}
GEN=$ROOT/generation
QWEN_DIR=$MODEL_DIR/Qwen3-32B
GOE_DATA=$DATA_DIR/goe_test_prepared.json
SC_REF=$REF_DIR/C_sc_probs.npz

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export FLASHINFER_CUDA_ARCH_LIST=9.0
export VLLM_USE_FLASHINFER_SAMPLER=0

mkdir -p "$RESULTS" "$LOGS"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGS/run_dose_response.log"; }

run_job() {  # run_job <out> <script> [args...]; skips an existing output
    local out="$1"; shift
    local name; name=$(basename "$out" .npz)
    if [ -f "$out" ]; then log "  SKIP (exists): $name"; return 0; fi
    log "  RUN: $name"
    local t0; t0=$(date +%s); local rc=0
    "$PY" "$@" >> "$LOGS/$name.log" 2>&1 || rc=$?
    local dt=$(( $(date +%s) - t0 ))
    if [ $rc -ne 0 ]; then log "  ERROR (rc=$rc) in $name after ${dt}s"; return $rc; fi
    log "  DONE: $name in ${dt}s"
}

for f in "$GOE_DATA" "$SC_REF"; do [ -f "$f" ] || { log "missing $f (see data/README.md)"; exit 1; }; done

if [ ! -f "$QWEN_DIR/config.json" ]; then
    log "  downloading Qwen/Qwen3-32B -> $QWEN_DIR"
    "$HF_CLI" download Qwen/Qwen3-32B ${QWEN_REVISION:+--revision "$QWEN_REVISION"} --local-dir "$QWEN_DIR" \
        >> "$LOGS/download_qwen.log" 2>&1 || { log "  download failed"; exit 1; }
fi

for S in 43 44 45; do
    for L in 1 2 4; do
        run_job "$RESULTS/C_goe_qwen_debateF_L${L}_k5_v2_s${S}.npz" \
            "$GEN/goe_debate.py" --model "$QWEN_DIR" --data "$GOE_DATA" --sc_ref "$SC_REF" \
                --seed $S --rounds $L --out "$RESULTS/C_goe_qwen_debateF_L${L}_k5_v2_s${S}.npz"
        run_job "$RESULTS/C_goe_qwen_moaF_L${L}_k5_v2_s${S}.npz" \
            "$GEN/goe_moa.py" --model "$QWEN_DIR" --data "$GOE_DATA" --sc_ref "$SC_REF" \
                --seed $S --rounds $L --out "$RESULTS/C_goe_qwen_moaF_L${L}_k5_v2_s${S}.npz"
    done
done
log "=== dose-response runs done; outputs in $RESULTS ==="
