#!/usr/bin/env bash
# =============================================================================
# install.sh — One-Click Full Setup for microwakeword_trainer
# =============================================================================
#
# Run this ONCE after a clean Ubuntu WSL2 install and a fresh git clone.
# It does everything in order:
#
#   1. Install system packages (wget, curl, git, build-essential, ffmpeg, etc.)
#   2. Install Python 3.11 from deadsnakes PPA (system Python, NOT uv)
#   3. Install CUDA Toolkit 12.6 + cuDNN 9 via NVIDIA apt repo
#   4. Set CUDA environment variables in ~/.bashrc
#   5. Create ~/venvs/mww-tf  — install TF + all training deps
#   6. Create ~/venvs/mww-torch — install PyTorch + clustering deps
#   7. Install dev tools into mww-tf
#   8. Install the project (editable) into mww-tf
#   9. Write shell aliases to ~/.bashrc
#  10. Run verification (GPU, TF, CuPy, PyTorch)
#
# Usage (from project root):
#   chmod +x scripts/install.sh
#   ./scripts/install.sh
#
# WSL2 prerequisite:
#   Install NVIDIA GPU driver on Windows HOST (not inside WSL).
#   Download: https://www.nvidia.com/Download/index.aspx
#   Required driver version: 520+ (for CUDA 12 support inside WSL2)
#
# After completion, open a NEW terminal (or: source ~/.bashrc) then:
#   mww-tf      → activates TF env + cd to project
#   mww-torch   → activates PyTorch env + cd to project
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENVS_DIR="$HOME/venvs"
TF_VENV="$VENVS_DIR/mww-tf"
TORCH_VENV="$VENVS_DIR/mww-torch"
PYTHON_VERSION="3.11"   # ai-edge-litert 2.x supports 3.10 and 3.11 only
LOG_FILE="/tmp/mww_install_$(date +%Y%m%d_%H%M%S).log"

# ---- Color helpers -----------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

info()    { echo -e "${CYAN}[INFO]${NC}  $*" | tee -a "$LOG_FILE"; }
success() { echo -e "${GREEN}[OK]${NC}    $*" | tee -a "$LOG_FILE"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*" | tee -a "$LOG_FILE"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" | tee -a "$LOG_FILE" >&2; }
step()    { echo -e "\n${BOLD}${CYAN}━━━ $* ${NC}" | tee -a "$LOG_FILE"; }
die()     { error "$*"; error "Full log: $LOG_FILE"; exit 1; }

# ---- Redirect all output to log AND terminal ---------------------------------
exec > >(tee -a "$LOG_FILE") 2>&1

echo ""
echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${CYAN}║       microwakeword_trainer — Full Environment Setup      ║${NC}"
echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════════════╝${NC}"
echo -e "  Log file: ${LOG_FILE}"
echo ""

# =============================================================================
# STEP 0 — Pre-flight checks
# =============================================================================
step "STEP 0 — Pre-flight checks"

# Must NOT be root (breaks venv ownership)
if [[ "$EUID" -eq 0 ]]; then
    die "Do not run as root. Run as your normal user — sudo will be used internally where needed."
fi

# Check WSL2 + Windows NVIDIA driver
if grep -qi microsoft /proc/version 2>/dev/null; then
    info "WSL2 detected."
    if ! command -v nvidia-smi &>/dev/null; then
        die "nvidia-smi not found inside WSL2.\n  → Install NVIDIA driver on WINDOWS HOST (not inside WSL).\n  → Download: https://www.nvidia.com/Download/index.aspx\n  → Required: driver 520+ for CUDA 12 support."
    fi
    GPU_INFO=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1)
    success "Windows NVIDIA driver visible from WSL2: $GPU_INFO"
else
    info "Native Linux detected."
    if ! command -v nvidia-smi &>/dev/null; then
        warn "nvidia-smi not found. If you have a GPU, install the NVIDIA driver first."
        warn "Continuing anyway — GPU steps may fail."
    else
        GPU_INFO=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1)
        success "NVIDIA driver: $GPU_INFO"
    fi
fi

# Check internet connectivity
if ! curl -s --max-time 5 https://pypi.org > /dev/null; then
    die "No internet access. Check your network connection."
fi
success "Internet: OK"

# =============================================================================
# STEP 1 — System packages
# =============================================================================
step "STEP 1 — System packages"

info "Updating apt package lists..."
sudo apt-get update -qq

info "Installing build essentials + audio + git + wget..."
sudo apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential \
    git \
    wget \
    curl \
    ca-certificates \
    gnupg \
    lsb-release \
    pkg-config \
    libssl-dev \
    libffi-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    liblzma-dev \
    libncurses5-dev \
    libncursesw5-dev \
    libsndfile1 \
    libsndfile1-dev \
    ffmpeg \
    portaudio19-dev \
    unzip \
    htop

success "System packages installed."

# =============================================================================
# STEP 2 — Python 3.11 (system Python via deadsnakes PPA, NOT uv)
# =============================================================================
step "STEP 2 — Python ${PYTHON_VERSION} (system install, NOT uv)"

PYTHON_BIN="python${PYTHON_VERSION}"

if command -v "$PYTHON_BIN" &>/dev/null; then
    INSTALLED_VER=$("$PYTHON_BIN" --version 2>&1)
    success "Python ${PYTHON_VERSION} already installed: $INSTALLED_VER"
else
    info "Adding deadsnakes PPA for Python ${PYTHON_VERSION}..."
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update -qq

    info "Installing Python ${PYTHON_VERSION} and dev packages..."
    sudo apt-get install -y --no-install-recommends \
        "python${PYTHON_VERSION}" \
        "python${PYTHON_VERSION}-venv" \
        "python${PYTHON_VERSION}-dev" \
        "python${PYTHON_VERSION}-distutils" \
        "python${PYTHON_VERSION}-lib2to3"

    success "Python $("$PYTHON_BIN" --version) installed."
fi

# Verify venv module is available
"$PYTHON_BIN" -m venv --help > /dev/null 2>&1 || \
    die "python${PYTHON_VERSION}-venv not working. Run: sudo apt install python${PYTHON_VERSION}-venv"

success "Python venv module: OK"

# =============================================================================
# STEP 3 — CUDA Toolkit + cuDNN 8 + TensorRT 8.6
# =============================================================================
step "STEP 3 — CUDA Toolkit + cuDNN 8.9.7 + TensorRT 8.6.1"

if command -v nvcc &>/dev/null; then
    NVCC_VER=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release //' | sed 's/,.*//')
    success "nvcc already present (CUDA $NVCC_VER) — skipping toolkit install."
else
    # Detect Ubuntu version
    UBUNTU_VER=$(lsb_release -rs 2>/dev/null || echo "22.04")
    UBUNTU_MAJOR=$(echo "$UBUNTU_VER" | cut -d. -f1)
    UBUNTU_ID="ubuntu$(echo "$UBUNTU_VER" | tr -d '.')"  # e.g. ubuntu2204

    info "Ubuntu ${UBUNTU_VER} detected — installing CUDA 12.3 for ${UBUNTU_ID}..."

    # Add NVIDIA CUDA apt keyring
    KEYRING_DEB="cuda-keyring_1.1-1_all.deb"
    CUDA_REPO_URL="https://developer.download.nvidia.com/compute/cuda/repos/${UBUNTU_ID}/x86_64/${KEYRING_DEB}"

    info "Downloading CUDA keyring from: $CUDA_REPO_URL"
    wget -q "$CUDA_REPO_URL" -O "/tmp/$KEYRING_DEB" || \
        die "Failed to download CUDA keyring. Check Ubuntu version: $UBUNTU_VER (supported: 22.04, 24.04)"

    sudo dpkg -i "/tmp/$KEYRING_DEB"
    rm "/tmp/$KEYRING_DEB"

    sudo apt-get update -qq

    info "Installing cuda-toolkit-12-3..."
    sudo apt-get install -y --no-install-recommends cuda-toolkit-12-3

    success "CUDA Toolkit 12.3 installed."
fi

# cuDNN 8.9.7 via local deb
if dpkg -l libcudnn8 &>/dev/null; then
    success "cuDNN 8 already installed — skipping."
else
    CUDNN_DEB="cudnn-local-repo-ubuntu2204-8.9.7.29_1.0-1_amd64.deb"
    if [[ -f "$HOME/$CUDNN_DEB" ]]; then
        info "Installing cuDNN 8.9.7 from local deb..."
        sudo dpkg -i "$HOME/$CUDNN_DEB"
        sudo cp /var/cudnn-local-repo-ubuntu2204-8.9.7.29/cudnn-local-08A7D361-keyring.gpg /usr/share/keyrings/
        sudo apt-get update -qq
        sudo apt-get install -y --no-install-recommends libcudnn8 libcudnn8-dev
        success "cuDNN 8.9.7 installed."
    else
        warn "$HOME/$CUDNN_DEB not found — skipping cuDNN install."
    fi
fi

# TensorRT 8.6.1 via local deb
if dpkg -l tensorrt &>/dev/null; then
    success "TensorRT already installed — skipping."
else
    TRT_DEB="nv-tensorrt-local-repo-ubuntu2204-8.6.1-cuda-12.0_1.0-1_amd64.deb"
    if [[ -f "$HOME/$TRT_DEB" ]]; then
        info "Installing TensorRT 8.6.1 from local deb..."
        sudo dpkg -i "$HOME/$TRT_DEB"
        sudo cp /var/nv-tensorrt-local-repo-ubuntu2204-8.6.1-cuda-12.0/nv-tensorrt-local-42B2FC56-keyring.gpg /usr/share/keyrings/
        sudo apt-get update -qq
        sudo apt-get install -y tensorrt
        success "TensorRT 8.6.1 installed."
    else
        warn "$HOME/$TRT_DEB not found — skipping TensorRT install."
    fi
fi

# =============================================================================
# STEP 4 — CUDA environment variables in ~/.bashrc
# =============================================================================
step "STEP 4 — CUDA environment variables"

CUDA_PATH="/usr/local/cuda-12.3"
[[ ! -d "$CUDA_PATH" ]] && CUDA_PATH="/usr/local/cuda"
[[ ! -d "$CUDA_PATH" ]] && warn "CUDA not found at /usr/local/cuda* — PATH may be incomplete."

BASHRC="$HOME/.bashrc"
MARKER="# === microwakeword_trainer environment ==="

if grep -q "$MARKER" "$BASHRC" 2>/dev/null; then
    warn "~/.bashrc already has mww env block — skipping (remove manually to regenerate)."
else
    cat >> "$BASHRC" <<BASHRC_BLOCK

$MARKER
# CUDA 12 paths
export CUDA_HOME="${CUDA_PATH}"
export PATH="\${CUDA_HOME}/bin:\${PATH}"
export LD_LIBRARY_PATH="\${CUDA_HOME}/lib64\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"

# TensorFlow GPU best practices
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_ALLOCATOR=cuda_malloc_async

# microwakeword_trainer quick-switch aliases
alias mww-tf='source ${TF_VENV}/bin/activate && cd ${PROJECT_DIR}'
alias mww-torch='source ${TORCH_VENV}/bin/activate && cd ${PROJECT_DIR}'
# === end microwakeword_trainer environment ===
BASHRC_BLOCK

    success "Environment block written to ~/.bashrc"
fi

# Export for THIS shell session so pip installs work immediately
export CUDA_HOME="$CUDA_PATH"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# =============================================================================
# STEP 5 — TensorFlow venv (mww-tf)
# =============================================================================
step "STEP 5 — TensorFlow venv: ${TF_VENV}"

mkdir -p "$VENVS_DIR"

if [[ -d "$TF_VENV" ]]; then
    warn "Venv $TF_VENV already exists."
    warn "To fully rebuild: rm -rf $TF_VENV  then re-run this script."
else
    info "Creating venv with $($PYTHON_BIN --version)..."
    "$PYTHON_BIN" -m venv "$TF_VENV"
    success "Venv created: $TF_VENV"
fi

PIP_TF="$TF_VENV/bin/pip"

info "Upgrading pip + wheel + setuptools..."
"$PIP_TF" install --upgrade pip wheel setuptools

info "Installing TensorFlow environment (requirements.txt)..."
info "  → TF 2.16.x  (ESPHome-pinned, bundles CUDA/cuDNN via tensorflow[and-cuda])"
"$PIP_TF" install -r "$PROJECT_DIR/requirements.txt"

info "Installing development tools (requirements-dev.txt)..."
"$PIP_TF" install -r "$PROJECT_DIR/requirements-dev.txt"

info "Installing project in editable mode..."
"$PIP_TF" install -e "$PROJECT_DIR"

success "TF venv fully installed: $TF_VENV"

# =============================================================================
# STEP 6 — PyTorch venv (mww-torch)
# =============================================================================
step "STEP 6 — PyTorch venv: ${TORCH_VENV}"

if [[ -d "$TORCH_VENV" ]]; then
    warn "Venv $TORCH_VENV already exists."
    warn "To fully rebuild: rm -rf $TORCH_VENV  then re-run this script."
else
    info "Creating venv with $($PYTHON_BIN --version)..."
    "$PYTHON_BIN" -m venv "$TORCH_VENV"
    success "Venv created: $TORCH_VENV"
fi

PIP_TORCH="$TORCH_VENV/bin/pip"

info "Upgrading pip + wheel + setuptools..."
"$PIP_TORCH" install --upgrade pip wheel setuptools

# Install torch FIRST with the CUDA 12.4 wheel index before requirements-torch.txt
# (pip will skip re-installing if already present and satisfies version constraint)
info "Installing PyTorch + torchaudio with CUDA 12.4 wheels..."
info "  (cu124 is closest stable wheel to CUDA 12.6 runtime)"
"$PIP_TORCH" install \
    "torch<2.11" \
    "torchaudio<2.11" \
    --index-url https://download.pytorch.org/whl/cu126

info "Installing remaining PyTorch requirements (requirements-torch.txt)..."
"$PIP_TORCH" install -r "$PROJECT_DIR/requirements-torch.txt"

success "PyTorch venv fully installed: $TORCH_VENV"

# =============================================================================
# STEP 7 — HuggingFace login reminder (needed for SpeechBrain ECAPA-TDNN)
# =============================================================================
step "STEP 7 — HuggingFace login (for speaker clustering)"

info "Speaker clustering uses SpeechBrain ECAPA-TDNN from HuggingFace Hub."
info "You need a free HuggingFace account and must accept the model terms."
echo ""
echo -e "  ${YELLOW}TODO (one-time, manual):${NC}"
echo -e "  1. Create account at: ${CYAN}https://huggingface.co/join${NC}"
echo -e "  2. Accept model terms at: ${CYAN}https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb${NC}"
echo -e "  3. Then run: ${CYAN}source ${TORCH_VENV}/bin/activate && huggingface-cli login${NC}"
echo ""

# =============================================================================
# STEP 8 — Verification
# =============================================================================
step "STEP 8 — Verification"

VERIFY_FAILED=0

# --- TF + GPU ---
info "Verifying TensorFlow GPU..."
"$TF_VENV/bin/python" - <<'PYEOF' || { warn "TF GPU verification had issues — see output above."; VERIFY_FAILED=1; }
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(f"  TensorFlow {tf.__version__}")
if gpus:
    print(f"  GPU: {len(gpus)} device(s) — {gpus[0].name}")
else:
    print("  WARNING: No GPU visible to TensorFlow!")
    print("  (If driver is correct, try: export TF_FORCE_GPU_ALLOW_GROWTH=true)")
PYEOF

# --- CuPy ---
info "Verifying CuPy (SpecAugment GPU backend)..."
"$TF_VENV/bin/python" - <<'PYEOF' || { warn "CuPy verification failed — SpecAugment will not work."; VERIFY_FAILED=1; }
import cupy
n = cupy.cuda.runtime.getDeviceCount()
print(f"  CuPy {cupy.__version__}")
print(f"  CUDA devices visible: {n}")
PYEOF

# --- pymicro-features ---
info "Verifying pymicro-features..."
"$TF_VENV/bin/python" - <<'PYEOF' || { warn "pymicro-features verification failed."; VERIFY_FAILED=1; }
import pymicro_features
print(f"  pymicro-features: OK (version attr may not exist)")
PYEOF

# --- PyTorch ---
info "Verifying PyTorch..."
"$TORCH_VENV/bin/python" - <<'PYEOF' || { warn "PyTorch verification failed."; VERIFY_FAILED=1; }
import torch
print(f"  PyTorch {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
else:
    print("  WARNING: CUDA not available in PyTorch — check torch install index-url.")
PYEOF

# --- SpeechBrain ---
info "Verifying SpeechBrain..."
"$TORCH_VENV/bin/python" - <<'PYEOF' || { warn "SpeechBrain import failed."; VERIFY_FAILED=1; }
import speechbrain
print(f"  SpeechBrain {speechbrain.__version__}")
PYEOF

# --- Project import ---
info "Verifying project import..."
"$TF_VENV/bin/python" - <<'PYEOF' || { warn "Project import failed — check editable install."; VERIFY_FAILED=1; }
import src.model.architecture
import config.loader
print("  Project imports: OK")
PYEOF

# =============================================================================
# DONE
# =============================================================================
echo ""
if [[ $VERIFY_FAILED -eq 0 ]]; then
    echo -e "${BOLD}${GREEN}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}${GREEN}║   ✓  Setup complete — all verifications passed!           ║${NC}"
    echo -e "${BOLD}${GREEN}╚══════════════════════════════════════════════════════════╝${NC}"
else
    echo -e "${BOLD}${YELLOW}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}${YELLOW}║   ⚠  Setup complete — some verifications had warnings.    ║${NC}"
    echo -e "${BOLD}${YELLOW}║      Check the output above and the log file below.       ║${NC}"
    echo -e "${BOLD}${YELLOW}╚══════════════════════════════════════════════════════════╝${NC}"
fi
echo ""
echo -e "  Full install log: ${CYAN}${LOG_FILE}${NC}"
echo ""
echo -e "${BOLD}Next steps:${NC}"
echo -e "  ${CYAN}source ~/.bashrc${NC}   ← apply aliases (or open a new terminal)"
echo ""
echo -e "${BOLD}Train:${NC}"
echo -e "  ${CYAN}mww-tf${NC}"
echo -e "  ${CYAN}mww-train --config config/presets/standard.yaml${NC}"
echo ""
echo -e "${BOLD}Speaker clustering:${NC}"
echo -e "  ${CYAN}mww-torch${NC}"
echo -e "  ${CYAN}huggingface-cli login${NC}   ← first time only"
echo -e "  ${CYAN}python cluster-Test.py --config standard --dataset all${NC}"
echo ""
echo -e "${BOLD}Export:${NC}"
echo -e "  ${CYAN}mww-export --checkpoint checkpoints/best.ckpt --output models/exported/${NC}"
echo ""
