#!/bin/bash
# Build the GPU generation environment used for the paper (vLLM 0.30.0, torch 2.14.0+cu130,
# transformers 5.17.0; CPython 3.10). Idempotent: an existing complete environment is left unchanged.
#
#   1. creates a virtualenv at $VENV
#   2. installs torch / torchvision from the PyTorch cu130 index, then every other package of
#      requirements.txt pinned and with --no-deps (vllm 0.30.0 declares torch==2.13.0; the runs used 2.14.0)
#   3. optionally (APPLY_FLASHINFER_PATCH=1) patches flashinfer/jit/core.py for Blackwell (sm_120) GPUs:
#      skips flashinfer's architecture check and disables two nvcc flags that need nvcc >= 12.6
#   4. prints the library versions and whether CUDA is visible
#
# Usage:  VENV=$HOME/venvs/agg bash scripts/setup_env.sh
# Environment: VENV (required), PYTHON (default python3.10), APPLY_FLASHINFER_PATCH (default 0)

set -euo pipefail

: "${VENV:?set VENV to the virtualenv directory to create}"
PYTHON=${PYTHON:-python3.10}
APPLY_FLASHINFER_PATCH=${APPLY_FLASHINFER_PATCH:-0}
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
REQ=$ROOT/requirements.txt
PYPI=https://pypi.org/simple
CU130=https://download.pytorch.org/whl/cu130
export PIP_NO_CACHE_DIR=1

say() { echo "[setup $(date -u '+%F %T')] $*"; }
die() { say "FATAL: $*"; exit 1; }

"$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 10) else 1)' \
    || say "WARN: $PYTHON is not CPython 3.10; the pinned wheels are cp310"

venv_ok() {
    [ -x "$VENV/bin/python3" ] && "$VENV/bin/python3" - <<'PYEOF' 2>/dev/null
import vllm, torch, transformers
assert (vllm.__version__, torch.__version__, transformers.__version__) == ("0.30.0", "2.14.0+cu130", "5.17.0")
PYEOF
}

if venv_ok; then
    say "environment already complete at $VENV"
else
    [ -d "$VENV" ] && die "$VENV exists but is incomplete; remove it or choose another VENV"
    "$PYTHON" -m pip install --quiet virtualenv --index-url "$PYPI"
    "$PYTHON" -m virtualenv "$VENV"
    "$VENV/bin/pip" install --upgrade pip --index-url "$PYPI"
    say "installing torch 2.14.0+cu130 / torchvision 0.29.0+cu130"
    "$VENV/bin/pip" install torch==2.14.0+cu130 torchvision==0.29.0+cu130 --index-url "$CU130" --extra-index-url "$PYPI"
    REST=$(mktemp)
    grep -v -E '^(#|--|torch==|torchvision==)' "$REQ" | grep -v '^$' > "$REST"
    say "installing $(wc -l < "$REST") pinned packages with --no-deps"
    "$VENV/bin/pip" install --no-deps -r "$REST" --index-url "$PYPI" --extra-index-url "$CU130"
    rm -f "$REST"
    "$VENV/bin/pip" check || true
fi

if [ "$APPLY_FLASHINFER_PATCH" = 1 ]; then
    F=$("$VENV/bin/python3" -c "import flashinfer.jit.core as m; print(m.__file__)" 2>/dev/null | tail -1)
    [ -f "$F" ] || die "cannot locate flashinfer/jit/core.py"
    if grep -q "Patched: skip arch check" "$F"; then
        say "flashinfer already patched"
    else
        SITE=$(cd "$(dirname "$F")/../.." && pwd)
        cp "$F" "$F.orig"
        patch -p1 -d "$SITE" < "$HERE/flashinfer_sm120.patch"
        grep -q "Patched: skip arch check" "$F" || die "patch did not apply"
        say "flashinfer patched (original kept as $F.orig)"
    fi
fi

"$VENV/bin/python3" -c "import vllm, torch, transformers; print('vllm', vllm.__version__, 'torch', torch.__version__, 'transformers', transformers.__version__, 'cuda', torch.cuda.is_available())" 2>/dev/null | tail -1
"$VENV/bin/python3" -c "import torch; p = torch.cuda.get_device_properties(0); print('GPU:', p.name, 'CC:', p.major, p.minor)" 2>/dev/null | tail -1 \
    || say "WARN: torch could not open a GPU"
say "done"
