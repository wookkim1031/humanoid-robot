set -euo pipefail

MODELS_DIR="${MODELS_DIR:-/workspace/models}"
mkdir -p "$MODELS_DIR"


if [! -f "$MODELS_DIR/simplx/SMPL_X_NEUTRAL.npz" ]; then 
    echo "[models] fetching SMPL-X from private HF mirror"
    mkdir -p "$MODELS_DIR/smplx"
    /venvs/genmo/bin/python - <<'PY'
import os 
from huggingface_hub import hf_hub_download
p = hf_hub_download(
    repo_id="johan1031/smpl",
    filename="SMPLX_NEUTRAL.npz", 
    local_dir=os.environ["MODELS_DIR] + "/smplx",
    token=os.environ["HF_TOKEN"],
)
print("download:", p)
PY
else 
    echo "[models] SMPL-X: cached"
fi

# GEM
GEM_DIR="$MODELS_DIR/gem"
if [ ! -f "$GEM_DIR/gem_smpl.ckpt" ]; then
  echo "[models] fetching GEM-X checkpoints (~12 GB, one-time)"
  /venvs/genmo/bin/python - <<'PY'
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
echo "[models] all weights present on volume."
