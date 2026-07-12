"""
Vectorized 2D point-mass env — a minimal stand-in for mjlab/Isaac Lab.

Interface contract (same shape as your PPOModel expects):
  env.num_envs
  env.reset() -> (obs_dict, info)          obs_dict = {"actor": (N,4), "critic": (N,6)}
  env.step(a) -> (obs_dict, rew, terminated, truncated, extras)
                 rew (N,) float, terminated/truncated (N,) bool
                 auto-resets done envs (Isaac Lab style)
                 extras["log"] populated when episodes finish

Task: point mass starts at random position in [-1,1]^2, action = acceleration
(clipped), reward = progress toward origin minus small action penalty.
  terminated: leaves [-3,3]^2 (failure — no bootstrap, like a G1 falling)
  truncated:  survives max_ep_len steps (timeout — bootstrap V)

Critic gets 2 extra "privileged" dims (distance, speed) to exercise
the asymmetric actor/critic path.
"""

import torch


class PointMassEnv:
    def __init__(self, num_envs=64, device="cpu", dt=0.1, max_ep_len=100):
        self.num_envs, self.device, self.dt = num_envs, device, dt
        self.max_ep_len = max_ep_len
        N = num_envs
        self.pos = torch.zeros(N, 2, device=device)
        self.vel = torch.zeros(N, 2, device=device)
        self.t = torch.zeros(N, device=device)
        self.ep_ret = torch.zeros(N, device=device)

    def _obs(self):
        actor = torch.cat([self.pos, self.vel], -1)                    # (N,4)
        dist = self.pos.norm(dim=-1, keepdim=True)                     # privileged
        speed = self.vel.norm(dim=-1, keepdim=True)
        critic = torch.cat([self.pos, self.vel, dist, speed], -1)      # (N,6)
        return {"actor": actor, "critic": critic}

    def _reset_idx(self, idx):
        n = idx.sum().item() if idx.dtype == torch.bool else len(idx)
        self.pos[idx] = torch.empty(n, 2, device=self.device).uniform_(-1, 1)
        self.vel[idx] = 0.0
        self.t[idx] = 0.0
        self.ep_ret[idx] = 0.0

    def reset(self):
        self._reset_idx(torch.ones(self.num_envs, dtype=torch.bool, device=self.device))
        return self._obs(), {}

    def step(self, action):
        a = action.clamp(-1, 1)
        old_dist = self.pos.norm(dim=-1)

        self.vel = self.vel + a * self.dt
        self.pos = self.pos + self.vel * self.dt
        self.t += 1

        new_dist = self.pos.norm(dim=-1)
        rew = (old_dist - new_dist) * 10.0 - 0.01 * a.pow(2).sum(-1)
        self.ep_ret += rew

        terminated = (self.pos.abs() > 3.0).any(-1)
        truncated = (self.t >= self.max_ep_len) & ~terminated
        done = terminated | truncated

        extras = {"log": {}}
        if done.any():
            extras["log"] = {
                "ep_return": self.ep_ret[done].mean().item(),
                "ep_length": self.t[done].mean().item(),
                "final_dist": new_dist[done].mean().item(),
            }
            self._reset_idx(done)  # auto-reset: next obs is fresh episode

        return self._obs(), rew, terminated, truncated, extras
