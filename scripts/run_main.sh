#!/bin/bash
# Main generation runs (results/p1): all conditions x seeds 43/44/45, one model family at a time.
#
#   Qwen3-32B      GoEmotions: plain-listing K=5, CoT K=5, debate (K=5, 3 rounds), MoA (K=5, 3 layers),
#                              SC (CoT K=43), plain-listing K=43; seed 43 only: listing K=1, Yes/No K=1,
#                              Yes/No K=43, clarification ensembling, zero-shot + CCD
#                  LeWiDi-MP : independent / debate / MoA
#   Mistral-Small-24B-Instruct-2501, Yi-1.5-34B-Chat
#                  GoEmotions: plain-listing K=5, CoT K=5, debate, MoA
#                  LeWiDi-MP : independent / debate / MoA
#
# Resumable: an output that exists and validates (loads, expected shape, finite, in [0,1], unique ids) is
# skipped; an output that does not validate is moved to $RESULTS_DIR/invalid_<time>/ and regenerated.
# Prompt-encoding gate (Mistral, Yi): before a family's first job, 5 LeWiDi items are generated and the log
# must report the token-id path with the expected BOS prefix ([1] for Mistral, [] for Yi); every later job
# log is checked for the same lines. A mismatch stops the run.
#
# Usage:   bash scripts/run_main.sh
#          DRY_RUN=1 bash scripts/run_main.sh        # list pending jobs only
# Environment (defaults in brackets):
#   PY [python3]  HF_CLI [hf]  DATA_DIR [data]  REF_DIR [ref]  RESULTS_DIR [results]  LOG_DIR [logs]
#   MODEL_DIR [models]  FAMILIES ["qwen mistral yi"]  GATE [1]  DELETE_WEIGHTS [0]  DRY_RUN [0]
#   QWEN_REVISION / MISTRAL_REVISION / YI_REVISION  (optional Hugging Face revisions; unset = main)

set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-python3}
HF_CLI=${HF_CLI:-hf}
DATA_DIR=${DATA_DIR:-$ROOT/data}
REF_DIR=${REF_DIR:-$ROOT/ref}
RESULTS=${RESULTS_DIR:-$ROOT/results}/p1
LOGS=${LOG_DIR:-$ROOT/logs}/p1
MODEL_DIR=${MODEL_DIR:-$ROOT/models}
GEN=$ROOT/generation
FAMILIES=${FAMILIES:-"qwen mistral yi"}
GATE=${GATE:-1}
DELETE_WEIGHTS=${DELETE_WEIGHTS:-0}
DRY_RUN=${DRY_RUN:-0}

GOE_DATA=$DATA_DIR/goe_test_prepared.json
MP_DATA=$DATA_DIR/MP_test.json
SC_REF=$REF_DIR/C_sc_probs.npz
GOE_ROWS=1200; GOE_COLS=28
MP_ROWS=3756;  MP_COLS=2
SEEDS="43 44 45"
SMOKE_N=5

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export FLASHINFER_CUDA_ARCH_LIST=9.0
export VLLM_USE_FLASHINFER_SAMPLER=0
export TOKENIZERS_PARALLELISM=false

repo_of() { case "$1" in qwen) echo Qwen/Qwen3-32B ;; mistral) echo mistralai/Mistral-Small-24B-Instruct-2501 ;; yi) echo 01-ai/Yi-1.5-34B-Chat ;; esac; }
dir_of()  { echo "$MODEL_DIR/$(basename "$(repo_of "$1")")"; }
rev_of()  { case "$1" in qwen) echo "${QWEN_REVISION:-}" ;; mistral) echo "${MISTRAL_REVISION:-}" ;; yi) echo "${YI_REVISION:-}" ;; esac; }
expected_bos() { case "$1" in mistral) echo "[1]" ;; yi) echo "[]" ;; *) echo "" ;; esac; }

if [ "$DRY_RUN" = 1 ]; then MAINLOG=/dev/null; else mkdir -p "$RESULTS" "$LOGS"; MAINLOG=$LOGS/run_main.log; fi
log() { echo "[$(date '+%F %T')] $*" | tee -a "$MAINLOG"; }
die() { log "FATAL: $*"; exit 1; }

# valid_npz <file> <rows> <cols> <key>
valid_npz() {
    "$PY" - "$1" "$2" "$3" "$4" <<'PYEOF' > /dev/null 2>&1
import sys
import numpy as np
f, rows, cols, key = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
with np.load(f) as z:
    p = z[key]
    assert p.shape == (rows, cols), p.shape
    assert np.isfinite(p).all() and p.min() >= 0 and p.max() <= 1
    if "ids" in z.files:
        assert len(z["ids"]) == rows and len(set(z["ids"].tolist())) == rows
PYEOF
}

# check_encoding <log-section> <family>: the job printed the expected prompt-encoding lines
check_encoding() {
    local sect="$1" exp
    exp=$(expected_bos "$2")
    [ -z "$exp" ] && return 0
    grep -qF "passing token ids encoded with tokenizer.json" "$sect" || return 1
    grep -qF "[prompt-encoding] BOS prefix on the token-id path: $exp" "$sect" || return 1
    if [ "$exp" = "[]" ]; then ! grep -qF "BOS prefix on the token-id path: [1]" "$sect"; else ! grep -qF "BOS prefix on the token-id path: []" "$sect"; fi
}

MODE=run; PENDING=0
# run_job <family> <out> <rows> <cols> <key> <script> [args...]
run_job() {
    local fam="$1" out="$2" rows="$3" cols="$4" key="$5"; shift 5
    local name; name=$(basename "$out" .npz)
    local jlog="$LOGS/$name.log"
    if [ -f "$out" ] && valid_npz "$out" "$rows" "$cols" "$key"; then
        [ "$MODE" = run ] && log "  SKIP (exists, valid): $name"
        return 0
    fi
    if [ "$MODE" = count ]; then
        PENDING=$((PENDING + 1)); [ "$DRY_RUN" = 1 ] && log "  [pending] $name"
        return 0
    fi
    if [ -f "$out" ]; then
        local q; q="$(dirname "$RESULTS")/invalid_$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$q"; mv "$out" "$q/"
        log "  WARN: existing $name did not validate; moved to $q/"
    fi
    log "  RUN: $name"
    touch "$jlog"
    local start; start=$(wc -l < "$jlog")
    local t0; t0=$(date +%s); local rc=0
    "$PY" "$@" >> "$jlog" 2>&1 || rc=$?
    local dt=$(( $(date +%s) - t0 ))
    [ $rc -eq 0 ] || die "job failed (rc=$rc) in $name after ${dt}s; see $jlog (rerun to resume)"
    valid_npz "$out" "$rows" "$cols" "$key" || die "$name finished but the output does not validate; see $jlog"
    local sect; sect=$(mktemp)
    tail -n +"$((start + 1))" "$jlog" > "$sect"
    if ! check_encoding "$sect" "$fam"; then
        local q="$(dirname "$RESULTS")/quarantine_encoding"
        mkdir -p "$q"; mv "$out" "$q/"; rm -f "$sect"
        die "$name: prompt-encoding lines are not the expected ones for $fam; output moved to $q/"
    fi
    log "  DONE: $name in ${dt}s  [$(grep -F '[sanity]' "$sect" | tail -1)]"
    rm -f "$sect"
}

ensure_model() {  # ensure_model <family>
    local repo dir rev
    repo=$(repo_of "$1"); dir=$(dir_of "$1"); rev=$(rev_of "$1")
    if [ "$DRY_RUN" = 1 ]; then log "  [plan] ensure $repo at $dir"; return 0; fi
    if [ -f "$dir/config.json" ] && [ -f "$dir/.download_complete" ]; then
        log "  $repo present at $dir"; return 0
    fi
    log "  downloading $repo${rev:+ (revision $rev)} -> $dir"
    "$HF_CLI" download "$repo" ${rev:+--revision "$rev"} --local-dir "$dir" >> "$LOGS/download_$(basename "$dir").log" 2>&1 \
        || die "download of $repo failed; see $LOGS/download_$(basename "$dir").log"
    [ -f "$dir/config.json" ] || die "$dir/config.json missing after download"
    date -u > "$dir/.download_complete"
}

bos_gate() {  # bos_gate <family>
    local fam="$1" mdir exp out glog
    mdir=$(dir_of "$fam"); exp=$(expected_bos "$fam")
    [ -z "$exp" ] && return 0
    if [ "$DRY_RUN" = 1 ]; then log "  [plan] encoding gate $fam: expect BOS prefix $exp"; return 0; fi
    out="$(dirname "$RESULTS")/smoke/C_mp_${fam}_plain_k5_v2_s43_limit${SMOKE_N}.npz"
    glog="$LOGS/smoke_${fam}.log"
    mkdir -p "$(dirname "$out")"
    "$PY" "$GEN/lewidi_independent.py" --model "$mdir" --data "$MP_DATA" --seed 43 --limit "$SMOKE_N" --out "$out" \
        > "$glog" 2>&1 || die "encoding gate run failed for $fam; see $glog"
    valid_npz "$out" "$SMOKE_N" "$MP_COLS" p || die "encoding gate output for $fam does not validate; see $glog"
    check_encoding "$glog" "$fam" || die "encoding gate FAILED for $fam (expected BOS prefix $exp); see $glog"
    log "  encoding gate ($fam) passed: $(grep -F 'BOS prefix' "$glog" | tail -1)"
}

jobs_family() {  # jobs_family <family>
    local fam="$1" M S
    M=$(dir_of "$fam")
    for S in $SEEDS; do
        run_job "$fam" "$RESULTS/C_goe_${fam}_plain_k5_v2_s${S}.npz" $GOE_ROWS $GOE_COLS p \
            "$GEN/goe_independent.py" --format plain --k 5 --model "$M" --data "$GOE_DATA" --sc_ref "$SC_REF" \
            --seed $S --out "$RESULTS/C_goe_${fam}_plain_k5_v2_s${S}.npz"
        run_job "$fam" "$RESULTS/C_goe_${fam}_cot_k5_v2_s${S}.npz" $GOE_ROWS $GOE_COLS p \
            "$GEN/goe_independent.py" --format cot --k 5 --model "$M" --data "$GOE_DATA" --sc_ref "$SC_REF" \
            --seed $S --out "$RESULTS/C_goe_${fam}_cot_k5_v2_s${S}.npz"
        run_job "$fam" "$RESULTS/C_goe_${fam}_debateF_k5_v2_s${S}.npz" $GOE_ROWS $GOE_COLS p \
            "$GEN/goe_debate.py" --model "$M" --data "$GOE_DATA" --sc_ref "$SC_REF" \
            --seed $S --out "$RESULTS/C_goe_${fam}_debateF_k5_v2_s${S}.npz"
        run_job "$fam" "$RESULTS/C_goe_${fam}_moaF_k5_v2_s${S}.npz" $GOE_ROWS $GOE_COLS p \
            "$GEN/goe_moa.py" --model "$M" --data "$GOE_DATA" --sc_ref "$SC_REF" \
            --seed $S --out "$RESULTS/C_goe_${fam}_moaF_k5_v2_s${S}.npz"
    done
    if [ "$fam" = qwen ]; then
        for S in $SEEDS; do
            run_job "$fam" "$RESULTS/C_goe_qwen_sc43_v2_s${S}.npz" $GOE_ROWS $GOE_COLS p \
                "$GEN/goe_independent.py" --format cot --k 43 --model "$M" --data "$GOE_DATA" --sc_ref "$SC_REF" \
                --seed $S --out "$RESULTS/C_goe_qwen_sc43_v2_s${S}.npz"
            run_job "$fam" "$RESULTS/C_goe_qwen_plain_k43_v2_s${S}.npz" $GOE_ROWS $GOE_COLS p \
                "$GEN/goe_independent.py" --format plain --k 43 --model "$M" --data "$GOE_DATA" --sc_ref "$SC_REF" \
                --seed $S --out "$RESULTS/C_goe_qwen_plain_k43_v2_s${S}.npz"
        done
        # 2x2 factorial (output format x K), seed 43
        run_job "$fam" "$RESULTS/C_goe_qwen_plain_k1_v2_s43.npz" $GOE_ROWS $GOE_COLS p \
            "$GEN/goe_independent.py" --format plain --k 1 --model "$M" --data "$GOE_DATA" --sc_ref "$SC_REF" \
            --seed 43 --out "$RESULTS/C_goe_qwen_plain_k1_v2_s43.npz"
        run_job "$fam" "$RESULTS/C_goe_qwen_yesno_k1_v2_s43.npz" $GOE_ROWS $GOE_COLS p \
            "$GEN/goe_independent.py" --format yesno --k 1 --model "$M" --data "$GOE_DATA" --sc_ref "$SC_REF" \
            --seed 43 --out "$RESULTS/C_goe_qwen_yesno_k1_v2_s43.npz"
        run_job "$fam" "$RESULTS/C_goe_qwen_yesno_k43_v2_s43.npz" $GOE_ROWS $GOE_COLS p \
            "$GEN/goe_independent.py" --format yesno --k 43 --model "$M" --data "$GOE_DATA" --sc_ref "$SC_REF" \
            --seed 43 --out "$RESULTS/C_goe_qwen_yesno_k43_v2_s43.npz"
        # context-generation methods (HuggingFace transformers), seed 43
        run_job "$fam" "$RESULTS/C_goe_qwen_hou_v2_s43.npz" $GOE_ROWS $GOE_COLS p_hou \
            "$GEN/goe_clarification.py" --model "$M" --data "$GOE_DATA" --sc_ref "$SC_REF" \
            --seed 43 --out "$RESULTS/C_goe_qwen_hou_v2_s43.npz"
        run_job "$fam" "$RESULTS/C_goe_qwen_ccd_v2_s43.npz" $GOE_ROWS $GOE_COLS p_ccd \
            "$GEN/goe_ccd.py" --model "$M" --data "$GOE_DATA" --sc_ref "$SC_REF" \
            --seed 43 --out "$RESULTS/C_goe_qwen_ccd_v2_s43.npz"
    fi
    for S in $SEEDS; do
        run_job "$fam" "$RESULTS/C_mp_${fam}_plain_k5_v2_s${S}.npz" $MP_ROWS $MP_COLS p \
            "$GEN/lewidi_independent.py" --model "$M" --data "$MP_DATA" \
            --seed $S --out "$RESULTS/C_mp_${fam}_plain_k5_v2_s${S}.npz"
        run_job "$fam" "$RESULTS/C_mp_${fam}_debateF_k5_v2_s${S}.npz" $MP_ROWS $MP_COLS p \
            "$GEN/lewidi_debate.py" --model "$M" --data "$MP_DATA" \
            --seed $S --out "$RESULTS/C_mp_${fam}_debateF_k5_v2_s${S}.npz"
        run_job "$fam" "$RESULTS/C_mp_${fam}_moaF_k5_v2_s${S}.npz" $MP_ROWS $MP_COLS p \
            "$GEN/lewidi_moa.py" --model "$M" --data "$MP_DATA" \
            --seed $S --out "$RESULTS/C_mp_${fam}_moaF_k5_v2_s${S}.npz"
    done
}

for f in "$GOE_DATA" "$MP_DATA" "$SC_REF"; do [ -f "$f" ] || die "missing $f (see data/README.md)"; done
if [ "$DRY_RUN" != 1 ]; then
    "$PY" -c "import vllm, torch, transformers; print('env:', vllm.__version__, torch.__version__, transformers.__version__)" | tee -a "$MAINLOG"
    "$PY" -c "import torch; assert torch.cuda.is_available(); p = torch.cuda.get_device_properties(0); print('GPU:', p.name)" | tee -a "$MAINLOG" \
        || die "torch cannot use the GPU"
fi

for fam in $FAMILIES; do
    log "=== $(repo_of "$fam") ==="
    PENDING=0; MODE=count; jobs_family "$fam"; MODE=run
    log "  pending jobs: $PENDING"
    [ "$PENDING" -gt 0 ] || continue
    ensure_model "$fam"
    [ "$GATE" = 1 ] && bos_gate "$fam"
    [ "$DRY_RUN" = 1 ] || jobs_family "$fam"
    if [ "$DELETE_WEIGHTS" = 1 ] && [ "$DRY_RUN" != 1 ]; then
        rm -rf "$(dir_of "$fam")"; log "  deleted weights of $(repo_of "$fam")"
    fi
done
log "=== all requested families done; outputs in $RESULTS ==="
