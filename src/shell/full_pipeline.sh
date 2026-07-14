#!/usr/bin/env bash
# ============================================================================
# full_pipeline.sh — video in, trained G1 policy out. One command.
#
#   export HF_TOKEN=<token>
#   bash full_pipeline.sh /workspace/videos/walking.mp4 walk_v1
#
# Encodes every fix verified in the 2026-07-12 session:
#   - Blackwell GPUs (sm_120) -> cu128 torch everywhere, with post-install guard
#   - venvs on LOCAL disk (/root) -> immune to NFS stale-handle corruption
#   - headless GL: libegl1/libosmesa6/xvfb installed; MUJOCO_GL=egl; GMR via xvfb-run
#   - MPLBACKEND=Agg, UV_LINK_MODE=copy, WANDB_MODE=offline
#   - GENMO undocumented body_model data files (6) fetched from private mirror
#   - GENMO empty-text bug patched (video-only inference crashes upstream)
#   - GENMO output path pattern: $OUT/<stem>_mix/smpl_params.pt
#   - mjlab csv_to_npz writes /tmp/motion.npz (NOT --output-name) -> rescued
#   - wandb artifact-link crash in offline mode tolerated (npz already saved)
#
# Idempotent: completed stages are skipped when their outputs exist.
# FORCE=1 re-runs everything. RUN_TRAIN=0 stops after the npz.
# ============================================================================
set -euo pipefail

VIDEO="${1:?usage: full_pipeline.sh <video.mp4> [run_name]}"
RUN="${2:-run_$(date +%Y%m%d_%H%M%S)}"

# ---- knobs -----------------------------------------------------------------
NUM_ENVS="${NUM_ENVS:-4096}"
ITERATIONS="${ITERATIONS:-10000}"
RUN_TRAIN="${RUN_TRAIN:-1}"
FORCE="${FORCE:-0}"

# ---- fixed layout ----------------------------------------------------------
export MODELS_DIR="${MODELS_DIR:-/workspace/models}"
DATA="/workspace/data/$RUN"
RUNS="/workspace/runs/$RUN"
REPO="/workspace/humanoid-robot"
GENMO_DIR="/workspace/GENMO"
GMR_DIR="/workspace/GMR"
VENV_GEM="/root/venv_gem"
VENV_GMR="/root/venv_gmr"
VENV_MJ="/root/venv_mjlab"
CU_INDEX="https://download.pytorch.org/whl/cu128"

# ---- session-verified environment insurance --------------------------------
export MPLBACKEND=Agg
export UV_LINK_MODE=copy
export WANDB_MODE=offline
export MUJOCO_GL=egl
: "${HF_TOKEN:?export HF_TOKEN=<hf token with read access to johan1031/smpl>}"

PHASE=""
trap 'echo; echo "!!! FAILED in phase: $PHASE"; echo "!!! Fix and re-run — completed stages will be skipped."' ERR

say() { echo; echo "=== [$1] $2"; PHASE="$1"; }
mkdir -p "$DATA" "$RUNS" /workspace/videos

# ============================================================ Phase 1: system
say "1/8" "system packages"
if ! command -v xvfb-run >/dev/null || ! dpkg -s libegl1 >/dev/null 2>&1; then
  apt-get update -qq
  apt-get install -y -qq libegl1 libgl1 libglvnd0 libosmesa6 xvfb ffmpeg git curl > /dev/null
fi
command -v uv >/dev/null || pip install -q uv
python3 -c "import huggingface_hub" 2>/dev/null || pip install -q huggingface_hub

# ============================================================ Phase 2: repos
say "2/8" "repositories"
[ -d "$REPO/.git" ]      || git clone -q https://github.com/wookkim1031/humanoid-robot.git "$REPO"
[ -d "$GENMO_DIR/.git" ] || git clone -q https://github.com/NVlabs/GENMO.git "$GENMO_DIR"
[ -d "$GMR_DIR/.git" ]   || git clone -q https://github.com/YanjieZe/GMR.git "$GMR_DIR"

# ============================================================ Phase 3: weights
say "3/8" "model weights -> $MODELS_DIR (idempotent)"
python3 - <<'PY'
import os
from huggingface_hub import hf_hub_download, snapshot_download
M   = os.environ["MODELS_DIR"]
tok = os.environ["HF_TOKEN"]

def need(p): return not os.path.exists(os.path.join(M, p))

if need("smplx/SMPLX_NEUTRAL.npz"):
    hf_hub_download(repo_id="johan1031/smpl", filename="SMPLX_NEUTRAL.npz",
                    local_dir=M + "/smplx", token=tok)
if need("gem/gem_smpl.ckpt"):
    snapshot_download(repo_id="nvidia/GEM-X",
                      revision="5ccf5ca3746c3620aa4016114f069a5f6ae399cd",
                      local_dir=M + "/gem",
                      allow_patterns=["gem_smpl.ckpt", "gem_smpl_config.json", "LICENSE"],
                      token=tok)
for spec in ["hmr2/epoch=10-step=25000.ckpt", "vitpose/vitpose-h-multi-coco.pth"]:
    if need(spec):
        hf_hub_download(repo_id="johan1031/smpl", filename=spec, local_dir=M, token=tok)
if need("body_models/coco_aug_dict.pth"):
    snapshot_download(repo_id="johan1031/smpl", allow_patterns=["body_models/*"],
                      local_dir=M, token=tok)

# hard verification footer: script may not succeed with anything missing
expected = ["smplx/SMPLX_NEUTRAL.npz", "gem/gem_smpl.ckpt",
            "hmr2/epoch=10-step=25000.ckpt", "vitpose/vitpose-h-multi-coco.pth"] + [
            f"body_models/{f}" for f in
            ["coco_aug_dict.pth", "smplx2smpl_sparse.pt", "smpl_coco17_J_regressor.pt",
             "smplx_verts437.pt", "smpl_neutral_J_regressor.pt",
             "smpl_3dpw14_J_regressor_sparse.pt"]]
missing = [p for p in expected if need(p)]
assert not missing, f"MISSING WEIGHTS: {missing} — upload to johan1031/smpl first"
print("[weights] all present")
PY

# ============================================================ Phase 4: venvs
make_venv() {  # $1=path  $2=marker-suffix  $3=install commands (function name)
  if [ "$FORCE" = "1" ] || [ ! -f "$1/.ok_$2" ]; then
    rm -rf "$1"; uv venv "$1" --python 3.10
    "$3" "$1"
    touch "$1/.ok_$2"
  fi
}
install_gem() {
  source "$1/bin/activate"
  uv pip install -q torch torchvision --index-url "$CU_INDEX"
  (cd "$GENMO_DIR" && bash scripts/install_env.sh)
  uv pip install -q --force-reinstall torch torchvision --index-url "$CU_INDEX"
  uv pip install -q huggingface_hub
  python -c "import torch; cc=torch.cuda.get_device_capability(0); assert cc>=(8,0), cc; \
             x=torch.randn(4,4,device='cuda'); (x@x).sum().item(); print('[gem venv] CUDA OK', cc)"
  deactivate
}
install_gmr() {
  source "$1/bin/activate"
  uv pip install -q -e "$GMR_DIR"
  python -c "import general_motion_retargeting; print('[gmr venv] import OK')"
  deactivate
}
install_mj() {
  source "$1/bin/activate"
  uv pip install -q torch --index-url "$CU_INDEX"
  uv pip install -q mjlab wandb
  uv pip install -q --no-deps -e "$REPO"
  python -c "import torch; cc=torch.cuda.get_device_capability(0); \
             import mjlab, zshumanoid; print('[mjlab venv] OK', cc)"
  deactivate
}
say "4/8" "virtual environments (local disk — NFS-safe)"
make_venv "$VENV_GEM" gem install_gem
make_venv "$VENV_GMR" gmr install_gmr
make_venv "$VENV_MJ"  mj  install_mj

# ============================================================ Phase 5: GENMO wiring
say "5/8" "GENMO wiring: symlinks + empty-text patch"
cd "$GENMO_DIR"
mkdir -p inputs/checkpoints/body_models inputs/pretrained
ln -sfn "$MODELS_DIR/hmr2"    inputs/checkpoints/hmr2
ln -sfn "$MODELS_DIR/vitpose" inputs/checkpoints/vitpose
ln -sfn "$MODELS_DIR/smplx"   inputs/checkpoints/body_models/smplx
ln -sf  "$MODELS_DIR/gem/gem_smpl.ckpt" inputs/pretrained/gem_smpl.ckpt
for f in "$MODELS_DIR"/body_models/*; do
  ln -sf "$f" "gem/utils/body_model/$(basename "$f")"
done
# idempotent patch: guard encode_text against empty caption list (video-only crash)
python3 - <<'PY'
import re, pathlib
p = pathlib.Path("gem/gem.py"); s = p.read_text()
old = 'if "text_embed" not in batch["meta"][0]["multi_text_data"]:'
new = ('if "text_embed" not in batch["meta"][0]["multi_text_data"] '
       'and len(batch["meta"][0]["multi_text_data"]["caption"]) > 0:')
if new in s:
    print("[patch] empty-text guard already applied")
elif old in s:
    p.write_text(s.replace(old, new)); print("[patch] empty-text guard applied")
else:
    raise SystemExit("[patch] anchor line not found — upstream changed; inspect gem/gem.py manually")
PY
source "$VENV_GEM/bin/activate"
python -c "from gem.utils.body_model.smplx_lite import SmplxLiteV437Coco17; print('[gate] GENMO imports OK')"
deactivate

# ============================================================ Phase 6: GEM inference
STEM="$(basename "$VIDEO")"; STEM="${STEM%.*}"
SMPL_PT="$DATA/${STEM}_mix/smpl_params.pt"
say "6/8" "Stage 1 — GEM: video -> smpl_params.pt"
if [ "$FORCE" = "1" ] || [ ! -f "$SMPL_PT" ]; then
  source "$VENV_GEM/bin/activate"
  cd "$GENMO_DIR"
  # rendering overlay segfaults headless without full EGL vendor config; the
  # .pt is saved BEFORE rendering, so tolerate a render-stage crash:
  python scripts/demo/demo_smpl.py --input_list "$VIDEO" \
      --ckpt_path "$MODELS_DIR/gem/gem_smpl.ckpt" --output_root "$DATA" || true
  deactivate
  [ -f "$SMPL_PT" ] || { echo "GEM produced no smpl_params.pt at $SMPL_PT"; exit 1; }
else
  echo "[skip] $SMPL_PT exists"
fi

# ============================================================ Phase 7: retarget -> npz
say "7/8" "Stages 2-4 — adapt -> GMR -> csv -> npz"
NPZ="$DATA/motion_g1.npz"
if [ "$FORCE" = "1" ] || [ ! -f "$NPZ" ]; then
  source "$VENV_GEM/bin/activate"        # adapter needs only torch
  python "$REPO/tools/gem_to_gmr.py" \
      --genmo "$SMPL_PT" --out "$DATA/gvhmr_style.pt" --frame global
  deactivate

  source "$VENV_GMR/bin/activate"
  cd "$GMR_DIR"
  mkdir -p assets/body_models
  ln -sfn "$MODELS_DIR/smplx" assets/body_models/smplx
  xvfb-run -a python scripts/gvhmr_to_robot.py \
      --gvhmr_pred_file "$DATA/gvhmr_style.pt" --robot unitree_g1 \
      --save_path "$DATA/motion_g1.pkl" --record_video
  deactivate
  [ -f "$DATA/motion_g1.pkl" ] || { echo "GMR produced no pkl"; exit 1; }
  cp -f videos/unitree_g1_gvhmr_style.mp4 "$DATA/retarget_check.mp4" 2>/dev/null || true

  source "$VENV_MJ/bin/activate"
  python "$REPO/tools/gmr_pkl_to_csv.py" "$DATA/motion_g1.pkl" "$DATA/motion_g1.csv"
  # NOTE: csv_to_npz saves to /tmp/motion.npz; --output-name is only the wandb
  # artifact label. Offline artifact-link crash is expected and tolerated.
  rm -f /tmp/motion.npz
  python -m mjlab.scripts.csv_to_npz \
      --input-file "$DATA/motion_g1.csv" --output-name "$NPZ" \
      --input-fps 30 --output-fps 50 --render True || true
  [ -f /tmp/motion.npz ] || { echo "csv_to_npz produced no /tmp/motion.npz"; exit 1; }
  cp -f /tmp/motion.npz "$NPZ"
  python - <<PY
import numpy as np
d = np.load("$NPZ")
keys = set(d.files)
need = {"joint_pos","joint_vel","body_pos_w","body_quat_w","body_lin_vel_w","body_ang_vel_w"}
assert need <= keys, f"npz missing {need-keys}"
assert d["joint_pos"].shape[1] == 29, d["joint_pos"].shape
print(f"[gate] npz OK: {d['joint_pos'].shape[0]} frames @50fps, 29 dof")
PY
  deactivate
else
  echo "[skip] $NPZ exists"
fi
echo ">>> retargeting check video: $DATA/retarget_check.mp4 — WATCH IT before long training runs"

# ============================================================ Phase 8: train
if [ "$RUN_TRAIN" != "1" ]; then
  say "8/8" "training skipped (RUN_TRAIN=0). Motion ready: $NPZ"
  exit 0
fi
say "8/8" "Stage 5 — PPO training on mjlab ($NUM_ENVS envs, $ITERATIONS iters)"
source "$VENV_MJ/bin/activate"
python "$REPO/src/train_g1_tracking.py" \
    --motion-file "$NPZ" \
    --num-envs "$NUM_ENVS" --iterations "$ITERATIONS" \
    --log-dir "$RUNS" 2>&1 | tee "$RUNS/train.log"

echo
echo "DONE. checkpoints: $RUNS | motion: $NPZ | retarget video: $DATA/retarget_check.mp4"