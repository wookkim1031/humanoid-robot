"""
Batched. PPO for vectorized GPU environments

* terminated vs truncated: truncations bootstrap V, terminations don't 
* separate actor/critic optimizers with different learning rates
"""

from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.distributions import Normal

@dataclass
class PPOConfig: 
    # Rollout: generating experience
    horizon: int = 24
    gamma: float = 0.99
    
    # Update: learning from that experience
    lam: float = 0.95
    clip: float = 0.2
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    adam_eps: float = 1e-5
    epochs: int = 10
    num_minibatches: int = 8
    ent_coef: float = 0.005
    vf_coef: float = 0.5
    max_grad_norm: float = 1.0
    normalize_obs: bool = True

    # Networks
    actor_hidden: tuple = (512, 256, 128)
    critic_hidden: tuple = (512, 256, 128)
    init_log_std: float = -1.0 

def _mlp(in_dim:int, hidden: tuple, out_dim: int, out_gain: float) -> nn.Sequential:
    layers, d = [], in_dim
    for h in hidden: 
        lin = nn.Linear(d, h)
        nn.init.orthogonal_(lin.weight, gain = 2 ** 0.5)
        nn.init.zeros_(lin.bias)
        layers += [lin, nn.ELU()]
        d = h
    head = nn.Linear(d, out_dim)
    nn.init.orthogonal_(head.weight, gain=out_gain)
    nn.init.zeros_(head.bias)
    layers.append(head)
    return nn.Sequential(*layers)

class Actor(nn.Module):
    """
    mean_net: outputs shape (N,A): one mean per action dimension per env
    log_std: log of the standard deviation
    """
    def __init__(self, obs_dim: int, act_dim:int, hidden: tuple = (512,256,128),
                 init_log_std: float=-1.0):
        super().__init__()
        self.mean_net = _mlp(obs_dim, hidden, act_dim, out_gain=0.01)
        self.log_std = nn.Parameter(torch.full((act_dim,), init_log_std))

    def dist(self, obs: torch.Tensor) -> Normal:
        return Normal(self.mean_net(obs), self.log_std.exp())
    

    @torch.no_grad()
    def act(self, obs: torch.Tensor):
        d = self.dist(obs)
        a = d.sample()
        return a, d.log_prob(a).sum(-1)
    
    def evaluate(self, obs:torch.Tensor, actions: torch.Tensor):
        d = self.dist(obs)
        return d.log_prob(actions).sum(-1), d.entropy().sum(-1)
    
class Critic(nn.Module):
    """
    One output dim V(s) is scalar per state 

    Returns shape (N,1) and uses squeeze(-1) to give (N,), because (N,1) - (N,) tensor
    gives (N,N) and its wrong
    """
    def __init__(self, obs_dim:int, hidden:tuple=(512,256,128)):
        super().__init__()
        self.v_net = _mlp(obs_dim, hidden, 1, out_gain=1.0)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.v_net(obs).squeeze(-1)

class RolloutBuffer: 
    """
    Fixed-size storage + GAE


    """

    def __init__(self, size:int, num_envs: int, actor_obs_dim:int, 
                 critic_obs_dim: int, act_dim: int, device:str): 
        self.size, self.N = size, num_envs
        d = dict(dtype=torch.float32, device=device)
        self.actor_obs = torch.zeros(size, num_envs, actor_obs_dim, **d)
        self.critic_obs = torch.zeros(size, num_envs, critic_obs_dim, **d)
        self.actions = torch.zeros(size, num_envs, act_dim, **d)
        self.logprobs = torch.zeros(size, num_envs, **d)
        self.rewards = torch.zeros(size, num_envs, **d)
        self.values = torch.zeros(size, num_envs, **d)
        self.dones = torch.zeros(size, num_envs, **d)
        self.advantages = torch.zeros(size, num_envs, **d)
        self.returns = torch.zeros(size, num_envs, **d)

    def reset(self):
        self.ptr = 0

    def add(self, actor_obs, critic_obs, action, logp, reward, value, done):
        i = self.ptr
        self.actor_obs[i] = actor_obs
        self.critic_obs[i] = critic_obs
        self.actions[i] = action
        self.logprobs[i] = logp
        self.rewards[i] = reward
        self.values[i] = value
        self.dones[i] = done
        self.ptr += 1

    """
    GAE: discounted, weighted sum of TD erros
    GAE: the advantage of action a_t is approximately the surprise at step t plus a decaying share of the surprises that followed

    delta: TD error = the discrepancy between what actually happened in one step (immediate reward plus discounted predicted future)
    and what the critic predicted before the step. 
    ** Positive delta: the world went better than the critic expected

    Advantages: learn how much better was this action than what the critic expected from this state. 

    Return: Advantages + Critics: reconstructing the regression target the critic will be trained on 

    GAE then exponentially averages these one-step estimates: 
        lambda = 0 gives pure TD (low variance, biased by critic errors)
        lambda = 1 gives Monte Carlo returns minus baseline (unbiased, high variance)
        lambda = 0.95 mostly trust nearby 

    GAE doesn't pick one k; it takes an exponentially weighted average of all of them

    GAE gives actor the lowest-noise-possible estimate of "was this action better than expected"
    by blending critic's fast-but-biased guesses with the slow-but-honest evidence of real rewards, with lambda as the dial.
    """
    @torch.no_grad()
    def compute_gae(self, gamma: float, lam: float, last_values:torch.Tensor):
        last_gae = torch.zeros(self.N, device=self.values.device)
        for t in reversed(range(self.size)):
            next_val = self.values[t+1] if t < self.size -1 else last_values
            nonterminal = 1.0 - self.dones[t] 
            # each delta is a little correction saying "the critic was wrong by this much at this step"
            delta = self.rewards[t] + gamma * next_val * nonterminal - self.values[t]
            last_gae = delta + gamma * lam * nonterminal * last_gae
            self.advantages[t] = last_gae # for actor 
        self.returns = self.advantages + self.values # for critic

class RunningMeanStd: 
    """
    Uses Welford online mean/var over batched observations

    Helps to normalize the values due to various variables (ex. joint positions, veloccities, base height)

    Adv: streaming, numerical stability, memory efficient 
    """
    def __init__(self, dim:int, device:str):
        self.mean = torch.zeros(dim, dtype=torch.float64, device=device)
        self.var = torch.ones(dim, dtype=torch.float64, device=device)
        self.count = 1e-4

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        bmean = x.mean(0).double() # columnwise mean, double for float.32 -> float.64
        bvar = x.var(0, unbiased=False).double()
        bcount = x.shape[0]
        delta = bmean - self.mean
        tot = bcount + self.count
        self.mean += delta * bcount / tot 
        m_a = self.var * self.count
        m_b = bvar * bcount
        self.var = (m_a + m_b + delta.pow(2) * self.count * bcount / tot) / tot
        self.count = tot

    def normalize(self, x: torch.Tensor) -> torch.Tensor: 
        return ((x - self.mean) / torch.sqrt(self.var + 1e-8)).float()
    
    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}
    
    def load_state_dict(self, sd):
        self.mean = sd["mean"].to(self.mean.device)
        self.var = sd["var"].to(self.var.device)
        self.count = sd["count"]

class PPOModel:
    def __init__(self, env, cfg: PPOConfig, actor_obs_dim:int,
                 critic_obs_dim: int, act_dim: int, device:str = "cuda:0"):
        self.env, self.cfg, self.device = env, cfg, device
        self.actor_obs_dim, self.critic_obs_dim, self.act_dim = (actor_obs_dim, critic_obs_dim, act_dim)

        N = env.num_envs

        self.actor = Actor(actor_obs_dim, act_dim, cfg.actor_hidden, cfg.init_log_std).to(device)
        self.critic = Critic(critic_obs_dim, cfg.critic_hidden).to(device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr,
                                            eps=cfg.adam_eps)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr,
                                           eps=cfg.adam_eps)
        self.buf = RolloutBuffer(cfg.horizon, N, actor_obs_dim, 
                                 critic_obs_dim, act_dim, device)
        self.actor_rms = RunningMeanStd(actor_obs_dim, device) if cfg.normalize_obs else None
        self.critic_rms = RunningMeanStd(critic_obs_dim, device) if cfg.normalize_obs else None

        obs, _ = env.reset()
        self._obs = obs

    
    def _norm(self,rms, x):
        return rms.normalize(x) if rms is not None else x
    
    """
    Ex. Isaaclab has *ManagerBasedRLEnv* which returns an observation. 
    RSL-RL wrapper maps policy -> actor obs and critic -> privileged critic obs 

    self.buf: the real payload
    self.obs: carried state
    self.actor_rms / self.critic_rms: updated statistics 
    """
    @torch.no_grad()
    def collect(self):
        self.buf.reset()
        ep_stats = []
        for _ in range(self.cfg.horizon):
            raw_a, raw_c = self._obs["actor"], self._obs["critic"]
            if self.actor_rms is not None: 
                self.actor_rms.update(raw_a)
                self.critic_rms.update(raw_c)
            a_obs = self._norm(self.actor_rms, raw_a)
            c_obs = self._norm(self.critic_rms, raw_c)
    
            value = self.critic(c_obs) # (N,)
            action, logp = self.actor.act(a_obs) # (N, A), (N,)

            next_obs, rew, terminated, truncated, extras = self.env.step(action)
            done = (terminated | truncated).float()
            rew = rew + self.cfg.gamma * value * truncated.float()

            self.buf.add(a_obs, c_obs, action, logp, rew, value, done)
            self._obs = next_obs
            if "log" in extras and extras["log"]:
                ep_stats.append(extras["log"])
        
        # run self critic one more time to run compute gae backwards
        last_values = self.critic(self._norm(self.critic_rms, self._obs["critic"]))
        self.buf.compute_gae(self.cfg.gamma, self.cfg.lam, last_values)
        return ep_stats

    """
    UPDATE
    
    """
    def update(self): 
        """
        reshape (100,4,17) with 100 steps x 4 envs x 17 dims into (400, 17) # 400 transitions
        """
        cfg = self.cfg
        a_obs = self.buf.actor_obs.reshape(-1, self.actor_obs_dim)
        c_obs = self.buf.critic_obs.reshape(-1, self.critic_obs_dim)
        actions = self.buf.actions.reshape(-1, self.act_dim)
        old_logp = self.buf.logprobs.reshape(-1)
        adv = self.buf.advantages.reshape(-1)
        ret = self.buf.returns.reshape(-1)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n = a_obs.shape[0]
        mb = max(1, n // cfg.num_minibatches)
        pi_l = v_l = ent_l = kl = 0.0 
        
        """
        new_logp: log probability of old actions under new policy 
        entropy: Entropy of current policy
        Entropy bonus (ent_coef) adds a small gradient pushing log_std upward, which counteracts the policy's natural tendency to collapse
        
        ratio = pi_new(a|s) / pi_old(a|s)
        """
        for _ in range(cfg.epochs):
            idx = torch.randperm(n, device=self.device)
            for s in range(0, n, mb):
                b = idx[s:s + mb]
                new_logp, entropy = self.actor.evaluate(a_obs[b], actions[b])
                ratio = (new_logp - old_logp[b]).exp()
                # Plain Policy Gradient objective
                s1 = ratio * adv[b]
                # Clamped objective
                s2 = torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv[b]
                pi_loss = -torch.min(s1,s2).mean()
                ent = entropy.mean()

                self.actor_opt.zero_grad()
                (pi_loss - cfg.ent_coef * ent).backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.max_grad_norm)
                self.actor_opt.step()

                v_loss = (self.critic(c_obs[b]) - ret[b]).pow(2).mean
                self.critic_opt.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_parameters(), cfg.max_grad_norm)
                self.critic_opt.step()

                with torch.no_grad():
                    kl = ((ratio - 1) - ratio.log()).mean().item()
                pi_l, v_l, ent_l = pi_loss.item(), v_loss.item(), ent.item()

        return {"pi_loss": pi_l, "v_loss": v_l, "entropy": ent_l, "approx_kl": kl}

    # ---------------------------------------------------------------------------------- train
    def train(self, iterations: int, log_every: int = 10): 
        for it in range(iterations):
            ep_stats = self.collect()
            log = self.update()
            if it % log_every == 0: 
                mean_rew = self.buf.rewards.mean().item()
                std = self.actor.log_std.exp().mean().item()
                print(f"iter {it:5d} | step_rew {mean_rew:8.4f} | "
                      f"pi {log['pi_loss']:+.3f} | v {log['v_loss']:.3f} | "
                      f"kl {log['approx_kl']:.4f} | std {std:.3f}")
                if ep_stats: 
                    # mjlab aggregates per-episode reward terms into extra["log"]
                    keys = sorted(ep_stats[-1].keys())[:6]
                    parts = []
                    for k in keys:
                        v = ep_stats[-1][k]
                        v = v.mean().item() if torch.is_tensor(v) else float(v)
                        parts.append(f"{k.split('/')[-1]}={v:.3f}")
                    print("        " + " | ".join(parts))

    # -------------------------------------------------------------------- io
    def save(self, path: str):
        torch.save({
            "cfg": self.cfg.to_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_rms": None if self.actor_rms is None else self.actor_rms.state_dict(),
            "critic_rms": None if self.critic_rms is None else self.critic_rms.state_dict(),
        }, path)

    def load(self, path: str):
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ck["actor"])
        self.critic.load_state_dict(ck["critic"])
        if self.actor_rms is not None and ck.get("actor_rms") is not None:
            self.actor_rms.load_state_dict(ck["actor_rms"])
        if self.critic_rms is not None and ck.get("critic_rms") is not None:
            self.critic_rms.load_state_dict(ck["critic_rms"])

