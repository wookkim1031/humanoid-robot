"""
Smoke-test: does zshumanoid PPO learn on the point-mass task?

Pass criteria (checked at the end):
  - mean ep_return of last 10 iters  >  first 10 iters (learning happened)
  - final mean final_dist < 0.3      (actually reaches the target region)
"""
import time
import torch
from pointmass_env import PointMassEnv
from zshumanoid.ppo import PPOModel, PPOConfig

torch.manual_seed(0)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

cfg = PPOConfig(
    horizon=24,
    epochs=5,
    num_minibatches=4,
    actor_hidden=(64, 64),    # tiny task -> tiny nets (CPU-friendly)
    critic_hidden=(64, 64),
)

env = PointMassEnv(num_envs=64, device=device)
model = PPOModel(env, cfg, actor_obs_dim=4, critic_obs_dim=6,
                 act_dim=2, device=device)

history = []
t0 = time.time()
for it in range(300):
    ep_stats = model.collect()
    stats = model.update()

    if ep_stats:
        ret = sum(s["ep_return"] for s in ep_stats) / len(ep_stats)
        dist = sum(s["final_dist"] for s in ep_stats) / len(ep_stats)
        history.append((ret, dist))

    if it % 30 == 0 and history:
        ret, dist = history[-1]
        print(f"iter {it:4d} | ep_ret {ret:7.2f} | final_dist {dist:.3f} | "
              f"kl {stats['approx_kl']:.4f} | clip {stats['clip_frac']:.2f} | "
              f"ent {stats['entropy']:.2f}")

elapsed = time.time() - t0
first = sum(r for r, _ in history[:10]) / 10
last = sum(r for r, _ in history[-10:]) / 10
final_dist = sum(d for _, d in history[-10:]) / 10

print(f"\n{elapsed:.1f}s | ep_return first10={first:.2f} -> last10={last:.2f} | "
      f"final_dist={final_dist:.3f}")
print("PASS" if (last > first and final_dist < 0.3) else "FAIL")
