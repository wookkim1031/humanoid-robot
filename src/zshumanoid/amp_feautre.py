"""
amp_feature.py — the AMP feature source for mjlab.

THE PROBLEM THIS SOLVES
  PPOAMP needs a 65-dim per-frame feature from the *live env*, matching exactly
  what ExpertMotionBuffer computes from the *motion npz*. mjlab does not provide
  it. This module:
    1. defines the observation term (amp_feature)
    2. patches it into the G1 tracking env cfg as an "amp" observation group
    3. gives you a verification gate that hard-fails on expert/policy mismatch

FEATURE LAYOUT (must match ExpertMotionBuffer exactly)
    [ joint_pos(29) | joint_vel(29) | root_h(1) | lin_vel_b(3) | ang_vel_b(3) ]

CRITICAL — ABSOLUTE, NOT RELATIVE
  mjlab's built-in mdp.joint_pos_rel subtracts the default pose. The expert npz
  stores ABSOLUTE joint angles. Using joint_pos_rel here would offset every
  policy feature by the default pose -> the discriminator would separate expert
  from policy on that constant alone, style reward collapses to 0, and nothing
  looks obviously broken. Hence the custom term below reads joint_pos directly.

USAGE
    from amp_feature import add_amp_group, verify_amp_features
    env_cfg = unitree_g1_flat_tracking_env_cfg()
    add_amp_group(env_cfg)                    # <- adds obs["amp"]
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
    verify_amp_features(env, "/path/motion_g1.npz")   # <- gate; raises on mismatch

  Then in PPOAMP.collect(), the feature is in the OBS DICT, not extras:
        amp_now = next_obs["amp"]        # (N, 65)   <- replace extras["amp_obs"]
"""

from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

AMP_FEATURE_DIM = 65
_SLICES = {  # for the verification report
  "joint_pos": slice(0, 29), "joint_vel": slice(29, 58), "root_h": slice(58, 59),
  "lin_vel_b": slice(59, 62), "ang_vel_b": slice(62, 65),
}


def amp_feature(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """65-dim AMP feature. Mirrors ExpertMotionBuffer's per-frame layout."""
  asset: Entity = env.scene[asset_cfg.name]
  d = asset.data
  return torch.cat(
    [
      d.joint_pos,                      # (N,29) ABSOLUTE (not _rel!)
      d.joint_vel,                      # (N,29)
      d.root_link_pos_w[:, 2:3],        # (N,1)  pelvis height (no x,y: position-invariant)
      d.root_link_lin_vel_b,            # (N,3)  base frame => heading-invariant
      d.root_link_ang_vel_b,            # (N,3)
    ],
    dim=-1,
  )


def add_amp_group(env_cfg) -> None:
  """Add an 'amp' observation group to a tracking env cfg (in place).

  Imports are local so this module can be imported without mjlab present
  (e.g. when only running the expert-side code on a laptop).
  """
  from mjlab.managers.manager_term_config import (
    ObservationGroupCfg,
    ObservationTermCfg,
  )

  env_cfg.observations["amp"] = ObservationGroupCfg(
    terms={"amp": ObservationTermCfg(func=amp_feature)},
    concatenate_terms=True,
    enable_corruption=False,   # NO noise: the discriminator must see clean state
  )


@torch.no_grad()
def verify_amp_features(env, motion_npz: str, warmup: int = 50,
                        tol_sigma: float = 4.0) -> None:
  """Hard gate: do policy-side and expert-side features live in the same space?

  Rolls the env with zero actions, then compares each of the 65 dims against the
  expert buffer's range. Raises on gross mismatch (sign flips, unit errors,
  relative-vs-absolute offsets, joint reordering).

  This is the check that prevents a silent AMP failure: if the two feature
  computations disagree, the discriminator learns the artifact instead of style,
  r_style flatlines at ~0, and training looks 'fine' while learning nothing.
  """
  from amp_mvae import ExpertMotionBuffer  # local import: same package

  expert = ExpertMotionBuffer([motion_npz], device=str(env.device))
  E = expert.data[:, :AMP_FEATURE_DIM]            # (T-1, 65) single-frame half

  obs, _ = env.reset()
  if "amp" not in obs:
    raise RuntimeError(
      "obs has no 'amp' group — call add_amp_group(env_cfg) BEFORE constructing "
      f"the env. Groups present: {list(obs.keys())}"
    )
  zero = torch.zeros(env.num_envs, env.action_manager.total_action_dim,
                     device=env.device)
  for _ in range(warmup):                          # leave the reset pose
    obs, *_ = env.step(zero)
  P = obs["amp"]                                   # (N, 65)

  if P.shape[-1] != AMP_FEATURE_DIM:
    raise RuntimeError(f"policy feature is {P.shape[-1]}-dim, expert is "
                       f"{AMP_FEATURE_DIM}-dim")

  e_mu, e_sd = E.mean(0), E.std(0).clamp_min(1e-3)
  p_mu = P.mean(0)
  z = (p_mu - e_mu).abs() / e_sd                   # per-dim standardized gap

  print(f"{'dim':>4} {'block':<10} {'expert range':>22} {'policy mean':>12} {'z':>6}")
  bad = []
  for name, sl in _SLICES.items():
    for i in range(sl.start, sl.stop):
      flag = ""
      if z[i] > tol_sigma:
        flag, _ = " <-- MISMATCH", bad.append((i, name))
      print(f"{i:>4} {name:<10} [{E[:, i].min():>8.2f},{E[:, i].max():>8.2f}] "
            f"{p_mu[i]:>12.2f} {z[i]:>6.1f}{flag}")

  if bad:
    hints = {
      "joint_pos": "using joint_pos_rel instead of joint_pos? joint order differs?",
      "joint_vel": "unit mismatch (rad/s vs deg/s)? fps assumption wrong?",
      "root_h":    "different ground plane — apply the root-height offset "
                   "(ground-truthing) to the motion, or the terrain origin differs",
      "lin_vel_b": "expert velocities not rotated to base frame? quaternion "
                   "convention (wxyz vs xyzw) flipped in _world_to_base?",
      "ang_vel_b": "same as lin_vel_b",
    }
    blocks = sorted({n for _, n in bad})
    msg = "\n".join(f"  - {b}: {hints[b]}" for b in blocks)
    raise RuntimeError(
      f"AMP FEATURE MISMATCH in {len(bad)} dims across {blocks}.\n"
      f"Expert and policy features are NOT in the same space — the discriminator "
      f"would separate them on this artifact, not on style.\n{msg}"
    )
  print(f"\n[gate] AMP features OK — all {AMP_FEATURE_DIM} dims within "
        f"{tol_sigma}σ of expert distribution")