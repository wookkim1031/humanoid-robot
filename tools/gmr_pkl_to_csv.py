"""
Convert GMR retargeting output (walking_g1.pkl) to mjlab's CSV format

CSV row (no header): root_pos(3), root_quat XYZW(4), dof_pos(29)
-> feed into:  python -m mjlab.scripts.csv_to_npz --input-file out.csv ...

Usage: 
    python gmr_pkl_to_csv.py walking_g1.pkl walking_g1.csv
"""

import pickle 
import sys

import numpy as np 

def load_gmr_pkl(path): 
    with open(path, "rb") as f: 
        data = pickle.load(f)
    
    print(f"pkl keys: {list(data.keys())}")

    # GMR key names have varied across versions
    def pick(*names): 
        for n in names: 
            if n in data: 
                return np.asarray(data[n])
        raise KeyError(f"none of {names} found in pkl; adjust key names "
                       f"(available: {list(data.keys())})")
        
    root_pos = pick("root_pos", "root_trans", "root_translation")
    root_rot = pick("root_rot", "root_quat", "root_orientation")
    dof_pos = pick("dof_pos", "joint_pos", "dof")
    fps = data.get("fps", data.get("motion_fps", 30))

    return root_pos, root_rot, dof_pos, fps

def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    in_pkl, out_csv = sys.argv[1], sys.argv[2]

    root_pos, root_rot, dof_pos, fps = load_gmr_pkl(in_pkl)

    T = root_pos.shape[0]
    assert root_pos.shape == (T,3), f"root_pos {root_pos.shape}"
    assert root_rot.shape == (T,4), f"root_ros {root_rot.shape}"
    assert dof_pos.shape[0] == T, f"dof_pos {dof_pos.shape}"
    if dof_pos.shape[1] != 29:
        print(f"wARNING: expected 29 dof for G1, got {dof_pos.shape[1]}")

    
    # Sanity: unit quaternions?
    norms = np.linalg.norm(root_rot, axis=-1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        print(f"WARNING: quaternions not unit norm "
              f"(min={norms.min():.4f}, max={norms.max():.4f})")

    # Heuristic convention check: for mostly-upright motion, xyzw has
    # |w| = |last component| large on average; wxyz has first large.
    if np.abs(root_rot[:, 3]).mean() < np.abs(root_rot[:, 0]).mean():
        print("WARNING: quaternion looks like wxyz (first component "
              "dominant). mjlab CSV needs xyzw — check your source!")

    rows = np.concatenate([root_pos, root_rot, dof_pos], axis=-1)
    np.savetxt(out_csv, rows, delimiter=",", fmt="%.8f")
    print(f"wrote {out_csv}: {T} frames x {rows.shape[1]} cols @ {fps} fps")
    print(f"-> python -m mjlab.scripts.csv_to_npz --input-file {out_csv} "
          f"--output-name motion.npz --input-fps {fps} --output-fps 50")


if __name__ == "__main__":
    main()