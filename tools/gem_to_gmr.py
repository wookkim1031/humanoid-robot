"""
Convert GEM's smpl_params.pt into GVHMR-style .pt that GMR's
gvhmr_to_robot.py expects. GEM and GMR both use SMPL-X with (N,63)
axis-angle body_pose, so this is a key-rename + betas tidy, not a real conversion.
"""

import argparse, torch

def adapt(gem_path, out_path, frame="global"):
    d = torch.load(gem_path, map_location="cpu", weights_only=False)
    key = f"body_params_{frame}"
    if key not in d: 
        key = "body_params_global" if "body_params_global" in d else "body_params_incam"
        print(f"[adapt] requested from missing; using {key}")
    bp = d[key]

    body_pose = bp["body_pose"].float()     # (N, 63) axis-angle
    global_orient = bp["global_orient"].float() # (N, 3) axis-angle
    transl = bp["transl"].float() # (N,3)
    N = body_pose.shape[0]

    # GMR does betas[0] then pads 10->16, so hand it a (1,10) tensor
    if bp.get("betas") is not None:
        betas = bp["betas"].float()
        if betas.ndim == 1:
            betas = betas[None]
        betas = betas[:1, :10]
    else: 
        betas = torch.zeros(1,10)

    # fail loud if the layer isn't what GMR feeds into smplx.create
    assert body_pose.ndim == 2 and body_pose.shape[1] == 63, \
        f"body_pose is {tuple(body_pose.shape)}, expected (N,63) SMPL-X axis-angle"
    assert global_orient.shape == (N, 3), f"global_orient {tuple(global_orient.shape)}"
    assert transl.shape == (N, 3),        f"transl {tuple(transl.shape)}"

    torch.save({"smpl_params_global": {
        "body_pose": body_pose,
        "global_orient": global_orient,
        "transl": transl,
        "betas": betas,
    }}, out_path)
    print(f"[adapt] {key}: {N} frames -> {out_path}")
    print(f"  body_pose {tuple(body_pose.shape)} | global_orient {tuple(global_orient.shape)} "
          f"| transl {tuple(transl.shape)} | betas {tuple(betas.shape)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--genmo", required=True, help="GENMO outputs/.../smpl_params.pt")
    ap.add_argument("--out", default="genmo_gvhmr_style.pt")
    ap.add_argument("--frame", choices=["global", "incam"], default="global",
                    help="global = world-grounded root trajectory (use this for retargeting)")
    args = ap.parse_args()
    adapt(args.genmo, args.out, args.frame)
