
"""
Train PPO on mjlab's G1 motion-tracking task 

Run on your pod (needs GPU — mujoco-warp requires CUDA):
    python train_g1_tracking.py --motion-file /path/to/walking_g1.npz

Prereqs:
    pip install mjlab            # or uv add mjlab
    # motion npz produced by:  python -m mjlab.scripts.csv_to_npz \
    #     --input-file walking_g1.csv --output-name walking_g1.npz \
    #     --input-fps 30 --output-fps 50
"""

import argparse
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.tracking.config.g1.env_cfgs import (
    unitree_g1_flat_tracking_env_cfg,
)

# johan.rl.ppo, amp_mvae in jupyter env
from zshumanoid.ppo import PPOModel, PPOConfig
from zshumanoid.amp_mvae import PPOAMP, ExpertMotionBuffer

@dataclass
class AMPConfig: 
    disc_lr: float = 1e-4
    grad_pen: float = 5.0       # R1 gradient-penalty weight
    w_task: float = 0.5
    w_style: float = 0.5
    disc_batch: int = 4096
    logit_reg: float = 0.05    

def build_env(motion_file: str, num_envs: int, device:str) -> ManagerBasedRlEnv: 
    env_cfg = unitree_g1_flat_tracking_env_cfg() # obs groups: "actor", "critic"
    env_cfg.scene.num_envs = num_envs
    env_cfg.commands["motion"].motion_file = motion_file
    return ManagerBasedRlEnv(cfg=env_cfg, device=device)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--motion-file", required=True, help="Path to motion .npz")
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--iterations", type=int, default=30_000)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--log-dir", default="runs/g1_tracking")
    p.add_argument("--save-every", type=int, default=500)
    args = p.parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    env = build_env(args.motion_file, args.num_envs, args.device)

    obs, _ = env.reset()
    actor_dim = obs["actor"].shape[-1]
    critic_dim = obs["critic"].shape[-1]
    act_dim = env.action_manager.total_action_dim
    print(f"[dims] actor={actor_dim} critic={critic_dim} act={act_dim} "
          f"envs={env.num_envs}")

    # Aligned with mjlab's tuned rsl-rl config for this task
    # (tasks/tracking/config/g1/rl_cfg.py), adapted to your two-optimizer design.
    cfg = PPOConfig(
        horizon=24,            # = num_steps_per_env
        gamma=0.99,
        lam=0.95,
        clip=0.2,
        epochs=5,              # mjlab uses 5, not 10
        num_minibatches=4,     # mjlab uses 4
        ent_coef=0.005,
        vf_coef=1.0,           # mjlab uses 1.0
        actor_lr=3e-4,         # mjlab: single lr 1e-3 with adaptive-KL schedule
        critic_lr=3e-4,
        max_grad_norm=1.0,
        init_log_std=0.0,      # mjlab init_std=1.0 -> log_std=0 (NOT -1: tracking
                               # with RSI needs wide exploration early)
        normalize_obs=True,
    )

    amp_cfg = AMPConfig()

    expert = ExpertMotionBuffer([args.motion_file], args.device)
    
    model = PPOAMP(env, cfg, amp_cfg, expert, actor_obs_dim=actor_dim,
                     critic_obs_dim=critic_dim, act_dim=act_dim,
                     device=args.device)

    t0 = time.time()
    for it in range(args.iterations):
        ep_stats = model.collect()
        stats = model.update()

        if it % 10 == 0:
            # Aggregate mjlab's per-reset log dicts (keys like
            # "Episode_Reward/motion_body_pos", "Metrics/motion/error_joint_pos").
            agg = defaultdict(list)
            for d in ep_stats:
                for k, v in d.items():
                    agg[k].append(float(v) if not torch.is_tensor(v) else v.item())
            fps = (it + 1) * cfg.horizon * env.num_envs / (time.time() - t0)
            line = f"it {it:6d} | fps {fps:9.0f} | kl {stats['approx_kl']:.4f} " \
                   f"| ent {stats['entropy']:.1f} | lr {stats['lr']:.2e}"
            line += (f" | d_exp {stats.get('d_expert', 0):+.2f}"
                     f" | d_pol {stats.get('d_policy', 0):+.2f}"
                     f" | gp {stats.get('grad_pen', 0):.3f}")
            for k in sorted(agg):
                if "error" in k or "Reward" in k:
                    line += f" | {k.split('/')[-1]} {sum(agg[k])/len(agg[k]):.3f}"
            print(line)

        if it % args.save_every == 0 and it > 0:
            ckpt = {
                "actor": model.actor.state_dict(),
                "critic": model.critic.state_dict(),
                "actor_rms": model.actor_rms.state_dict() if model.actor_rms else None,
                "critic_rms": model.critic_rms.state_dict() if model.critic_rms else None,
                "cfg": vars(cfg), "iteration": it,
            }
            torch.save(ckpt, log_dir / f"ckpt_{it}.pt")
            print(f"[saved] {log_dir}/ckpt_{it}.pt")


if __name__ == "__main__":
    main()
