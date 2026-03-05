#!/usr/bin/env bash
# =============================================================================
# ci.sh — Wake Word Training Smoke Test
# =============================================================================
# Validates the full pipeline end-to-end in < 2 minutes:
#   1. Config loading validation
#   2. Synthetic dataset generation
#   3. Preprocessing (fast)
#   4. Training — fast_test preset, 10 steps only
#   5. Export to TFLite
#   6. ESPHome compatibility verification
#
# Exit codes:
#   0 — All checks passed
#   1 — One or more checks failed
#
# Usage:
#   bash scripts/ci.sh                    # Run all checks
#   bash scripts/ci.sh --skip-train       # Skip training step (fastest)
#   CI=true bash scripts/ci.sh            # Non-interactive CI mode
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# ── Colours ──────────────────────────────────────────────────────────────────
if [[ -t 1 && "${CI:-}" != "true" ]]; then
    GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'
else
    GREEN=''; RED=''; YELLOW=''; NC=''
fi

PASS="${GREEN}✓${NC}"
FAIL="${RED}✗${NC}"
WARN="${YELLOW}⚠${NC}"

# ── Argument parsing ──────────────────────────────────────────────────────────
SKIP_TRAIN=false
for arg in "$@"; do
    case "$arg" in
        --skip-train) SKIP_TRAIN=true ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

# ── Timing ───────────────────────────────────────────────────────────────────
START_TS=$(date +%s)
ERRORS=0

# ── Helper functions ─────────────────────────────────────────────────────────
step() { echo -e "\n${YELLOW}══ $1 ══${NC}"; }
ok()   { echo -e "$PASS $1"; }
fail() { echo -e "$FAIL $1"; ERRORS=$((ERRORS + 1)); }
warn() { echo -e "$WARN $1"; }

# ── Python executable ────────────────────────────────────────────────────────
PYTHON="${PYTHON:-python}"
if ! "$PYTHON" --version &>/dev/null; then
    PYTHON="python3"
fi
ok "Python: $($PYTHON --version 2>&1)"

# =============================================================================
# STEP 1 — Config validation
# =============================================================================
step "Config validation"

for preset in fast_test standard max_quality; do
    if "$PYTHON" -c "
import dataclasses, sys
from config.loader import load_full_config
try:
    cfg = load_full_config('$preset')
    assert cfg.model.architecture == 'mixednet', 'bad architecture'
    assert cfg.hardware.mel_bins == 40, 'bad mel_bins'
    assert cfg.hardware.sample_rate_hz == 16000, 'bad sample_rate'
    print('OK: $preset')
except Exception as e:
    print(f'FAIL: $preset: {e}', file=sys.stderr)
    sys.exit(1)
" 2>&1; then
        ok "Preset '$preset' valid"
    else
        fail "Preset '$preset' failed validation"
    fi
done

# =============================================================================
# STEP 2 — Synthetic dataset generation
# =============================================================================
step "Synthetic dataset generation"

CI_DATASET_DIR="$PROJECT_DIR/ci_dataset"
rm -rf "$CI_DATASET_DIR"

if "$PYTHON" scripts/generate_test_dataset.py 2>&1 | tail -5; then
    ok "Synthetic dataset generated"
else
    fail "Synthetic dataset generation failed"
fi

# =============================================================================
# STEP 3 — Imports smoke test (catch missing deps early)
# =============================================================================
step "Import smoke test"

IMPORTS=(
    "src.model.architecture"
    "src.data.dataset"
    "src.export.tflite"
    "src.evaluation.metrics"
    "src.training.trainer"
    "config.loader"
)

for mod in "${IMPORTS[@]}"; do
    if "$PYTHON" -c "import $mod; print('OK: $mod')" 2>/dev/null; then
        ok "import $mod"
    else
        fail "import $mod FAILED"
    fi
done

# =============================================================================
# STEP 4 — Training (fast_test, 10 steps)
# =============================================================================

# Build a minimal override that caps training at 10 steps total
TRAIN_OVERRIDE=$(mktemp /tmp/ci_override_XXXXXX.yaml)
cat > "$TRAIN_OVERRIDE" << 'YAML_EOF'
training:
  training_steps: [5, 5]
  eval_basic_step_interval: 5
  eval_advanced_step_interval: 10
  eval_checkpoints_interval: 10
  eval_log_every_step: false
  batch_size: 16
performance:
  tensorboard_enabled: false
  profiling: false
  tf_profile_start_step: 0
  gpu_memory_log_interval: 0
YAML_EOF

CHECKPOINT_PATH="$PROJECT_DIR/checkpoints/best_weights.weights.h5"
EXPORT_DIR="$PROJECT_DIR/ci_export"
MODEL_NAME="ci_wake_word"
TFLITE_PATH="$EXPORT_DIR/${MODEL_NAME}.tflite"

if [[ "$SKIP_TRAIN" == "true" ]]; then
    warn "Skipping training step (--skip-train)"
else
    step "Training (fast_test preset, 10 steps)"

    # Remove old checkpoints to get a fresh best_weights
    rm -f "$CHECKPOINT_PATH"

    if "$PYTHON" -m src.training.trainer \
            --config fast_test \
            --override "$TRAIN_OVERRIDE" \
            2>&1 | tail -20; then
        if [[ -f "$CHECKPOINT_PATH" ]]; then
            ok "Training completed — checkpoint at $CHECKPOINT_PATH"
        else
            # Accept any checkpoint in ./checkpoints/
            LATEST_CKPT=$(ls -t "$PROJECT_DIR"/checkpoints/*.weights.h5 2>/dev/null | head -1 || true)
            if [[ -n "$LATEST_CKPT" ]]; then
                CHECKPOINT_PATH="$LATEST_CKPT"
                ok "Training completed — checkpoint at $CHECKPOINT_PATH"
            else
                fail "Training completed but no checkpoint found"
            fi
        fi
    else
        fail "Training step failed"
    fi
fi

rm -f "$TRAIN_OVERRIDE"

# =============================================================================
# STEP 5 — Export to TFLite
# =============================================================================
step "TFLite export"

if [[ ! -f "$CHECKPOINT_PATH" ]]; then
    warn "No checkpoint available — skipping export and verification"
    ERRORS=$((ERRORS + 1))
else
    rm -rf "$EXPORT_DIR"
    if "$PYTHON" -m src.export.tflite \
            --checkpoint "$CHECKPOINT_PATH" \
            --config fast_test \
            --output "$EXPORT_DIR" \
            --model-name "$MODEL_NAME" \
            2>&1 | tail -20; then
        if [[ -f "$TFLITE_PATH" ]]; then
            ok "Export succeeded — $TFLITE_PATH"
        else
            # Accept any .tflite in export dir
            FOUND=$(ls "$EXPORT_DIR"/*.tflite 2>/dev/null | head -1 || true)
            if [[ -n "$FOUND" ]]; then
                TFLITE_PATH="$FOUND"
                ok "Export succeeded — $TFLITE_PATH"
            else
                fail "Export completed but no .tflite found in $EXPORT_DIR"
            fi
        fi
    else
        fail "Export step failed"
    fi

    # =========================================================================
    # STEP 6 — ESPHome compatibility verification
    # =========================================================================
    step "ESPHome compatibility verification"

    if [[ -f "$TFLITE_PATH" ]]; then
        if "$PYTHON" scripts/verify_esphome.py "$TFLITE_PATH" 2>&1; then
            ok "ESPHome verification passed"
        else
            fail "ESPHome verification failed"
        fi
    else
        warn "No .tflite to verify — skipping"
    fi
fi

# =============================================================================
# Summary
# =============================================================================
END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [[ "$ERRORS" -eq 0 ]]; then
    echo -e "$PASS All CI checks passed in ${ELAPSED}s"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    exit 0
else
    echo -e "$FAIL $ERRORS check(s) failed (elapsed: ${ELAPSED}s)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    exit 1
fi
