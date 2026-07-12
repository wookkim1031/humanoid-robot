#!/usr/bin/env bash
# run_pipeline.sh — video in, training out. One command on the server.
#   bash scripts/run_pipeline.sh /workspace/data/clip.mp4 walk_v1
set -euo pipefail

VIDEO="${1:?usage: run_pipeline.sh <video.mp4> [run_name]}"
RUN="${2:-run_$(date +%Y%m%d_%H%M%S)}"
DATA="${DATA_DIR:-/workspace/data}/$RUN"
RUNS="${RUNS_DIR:-/workspace/runs}/$RUN"
mkdir -p "$DATA" "$RUNS"

bash scripts/download_models.sh

echo "=== [1/5] GENMO: video -> smpl_params.pt"
/venvs/genmo/bin/python third_party/GENMO/demo_smpl.py \
    --video "$VIDEO" --output_dir "$DATA" \
    --checkpoint "$MODELS_DIR/genmo/genmo_release.ckpt"

echo "=== [2/5] adapt GENMO -> GVHMR-style .pt (GMR input)"
/venvs/genmo/bin/python scripts/adapt_genmo_to_gmr.py \
    --genmo "$DATA/smpl_params.pt" --out "$DATA/genmo_gvhmr_style.pt" \
    --frame global

echo "=== [3/5] GMR: retarget -> G1 pkl"
# NOTE: verify flag names against your pinned GMR commit
# (validated path: gvhmr_to_robot.py consuming the GVHMR-style .pt)
/venvs/gmr/bin/python third_party/GMR/scripts/gvhmr_to_robot.py \
    --gvhmr_pred_file "$DATA/genmo_gvhmr_style.pt" --robot unitree_g1 \
    --save_path "$DATA/motion_g1.pkl"

echo "=== [4/5] pkl -> csv -> npz (50 Hz)"
/venvs/gmr/bin/python scripts/gmr_pkl_to_csv.py \
    "$DATA/motion_g1.pkl" "$DATA/motion_g1.csv"
/venvs/mjlab/bin/python -m mjlab.scripts.csv_to_npz \
    --input-file "$DATA/motion_g1.csv" \
    --output-name "$DATA/motion_g1.npz" \
    --input-fps 30 --output-fps 50

if [ "${RUN_TRAIN:-1}" != "1" ]; then
  echo "=== [5/5] training skipped (RUN_TRAIN=0). Motion ready: $DATA/motion_g1.npz"
  exit 0
fi

echo "=== [5/5] PPO training (mjlab)"
/venvs/mjlab/bin/python train_g1_tracking.py \
    --motion-file "$DATA/motion_g1.npz" \
    --num-envs "${NUM_ENVS:-4096}" \
    --iterations "${ITERATIONS:-30000}" \
    --log-dir "$RUNS"

echo "done. checkpoints in $RUNS"
