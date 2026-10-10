#!/bin/bash
# CPU analysis pipeline: data preparation -> reference file -> human reliability ceiling -> final analysis
# -> figures -> prompt appendix. All outputs go to $OUT_DIR; nothing is edited by hand afterwards.
#
# Usage:   bash scripts/run_analysis.sh
# Environment (defaults in brackets): PY [python3]  RAW_GOE_DIR [data/goemotions]  DATA_DIR [data]
#   REF_DIR [ref]  RESULTS_DIR [results]  OUT_DIR [output]

set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-python3}
DATA_DIR=${DATA_DIR:-$ROOT/data}
RAW_GOE_DIR=${RAW_GOE_DIR:-$DATA_DIR/goemotions}
REF_DIR=${REF_DIR:-$ROOT/ref}
export RESULTS_DIR=${RESULTS_DIR:-$ROOT/results}
export OUT_DIR=${OUT_DIR:-$ROOT/output}
export GOE_REF=${GOE_REF:-$DATA_DIR/goe_reference_1200.npz}
export CEILING_JSON=${CEILING_JSON:-$OUT_DIR/human_reliability_ceiling.json}
export MP_DATA=${MP_DATA:-$DATA_DIR/MP_test.json}
mkdir -p "$OUT_DIR"

if [ ! -f "$DATA_DIR/goe_test_prepared.json" ]; then
    "$PY" "$ROOT/data/prepare_goemotions.py" --raw_dir "$RAW_GOE_DIR" --out "$DATA_DIR/goe_test_prepared.json"
fi
if [ ! -f "$GOE_REF" ]; then
    "$PY" "$ROOT/analysis/build_goe_reference.py" --data "$DATA_DIR/goe_test_prepared.json" \
        --subset "$REF_DIR/C_sc_probs.npz" --out "$GOE_REF"
fi
if [ ! -f "$CEILING_JSON" ]; then
    "$PY" "$ROOT/analysis/human_reliability_ceiling.py" --raw_dir "$RAW_GOE_DIR" --reference "$GOE_REF" --out "$CEILING_JSON"
fi
"$PY" "$ROOT/analysis/inventory_results.py" > "$OUT_DIR/inventory.log" 2>&1 || true
tail -1 "$OUT_DIR/inventory.log"
"$PY" "$ROOT/analysis/final_analysis.py" > "$OUT_DIR/final_analysis.log" 2>&1
tail -1 "$OUT_DIR/final_analysis.log"
"$PY" "$ROOT/analysis/make_figures.py" "$OUT_DIR/figures" | tail -1
"$PY" "$ROOT/analysis/make_prompts_appendix.py" "$OUT_DIR/appendix_prompts_body.tex" | tail -1
