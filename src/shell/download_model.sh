set -euo pipefail

export MODELS_DIR="${MODELS_DIR:-/workspace/models}"
PY="${PY:-python3}"
mkdir -p "$MODELS_DIR"


if [ ! -f "$MODELS_DIR/smplx/SMPLX_NEUTRAL.npz" ]; then
    echo "[models] fetching SMPL-X from private HF mirror"
    mkdir -p "$MODELS_DIR/smplx"
    "$PY" - <<'PY'
import os 
from huggingface_hub import hf_hub_download
p = hf_hub_download(
    repo_id="johan1031/smpl",
    filename="SMPLX_NEUTRAL.npz", 
    local_dir=os.environ.get("MODELS_DIR", "/workspace/models") + "/smplx",
    token=os.environ["HF_TOKEN"],
)
print("download:", p)
PY
else 
    echo "[models] SMPL-X: cached"
fi

# GEM
if [ ! -f "$MODELS_DIR/gem/gem_smpl.ckpt" ]; then
  echo "[models] fetching GEM-X checkpoints (~12 GB, one-time)"
  "$PY" - <<'PY'
import os
from huggingface_hub import snapshot_download
p = snapshot_download(
    repo_id="nvidia/GEM-X",
    revision="5ccf5ca3746c3620aa4016114f069a5f6ae399cd",  # pin!
    local_dir=os.environ["MODELS_DIR"] + "/gem",
    allow_patterns=[
        "gem_smpl.ckpt", "gem_smpl_config.json"
    ],
    token=os.environ.get("HF_TOKEN"),
)
print("downloaded to:", p)
PY
else
  echo "[models] GEM-X: cached"
fi

# ------------------------------------- HMR2 + ViTPose (GEM demo prereqs) --
# Expected mirror layout: hmr2/epoch=10-step=25000.ckpt and
# vitpose/vitpose-h-multi-coco.pth. If your HF repo stores them elsewhere,
# adjust the two specs below (check with HfApi().list_repo_files).
for spec in "hmr2/epoch=10-step=25000.ckpt" "vitpose/vitpose-h-multi-coco.pth"; do
  if [ ! -f "$MODELS_DIR/$spec" ]; then
    echo "[models] fetching $spec from private HF mirror"
    SPEC="$spec" "$PY" - <<'PY'
import os
from huggingface_hub import hf_hub_download
p = hf_hub_download(
    repo_id="johan1031/smpl",
    filename=os.environ["SPEC"],
    local_dir=os.environ["MODELS_DIR"],
    token=os.environ["HF_TOKEN"],
)
print("downloaded:", p)
PY
  else
    echo "[models] $spec: cached"
  fi
done

echo "[models] all weights present on volume."
