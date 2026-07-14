"""
Design decisions (fixed on purpose)
- LSGAN discriminator with R1-style gradient penalty on expert samples 
- style reward r_s = (max, 0,1 - 0.25 (D-1)^2) in [0,1]
- AMP features per transition (s, s'): [joint_pos, joint_vel, root_h, base-frame root lin/ang vel] x 2 frames
"""

from dataclasses import dataclass

import numpy as np 
import torch 
import torch.nn as nn

from zshumanoid.ppo import PPOModel, PPOConfig, _mlp 

class ExpertMotionBuffer: 
    """
    Holds expert transition features from motion npz files 

    Feature per frame: [joint_pos(29), joint_vel(29), root_h(1),
                        lin_vel_b(3), ang_vel_b(3)]  = 65 dims
    AMP input = concat(feat_t, feat_{t+1}) = 130 dims.
    """

    def __init__(self, npz_paths: list[str], device:str): 
        feats = []
        for p in npz_paths: 
            d = np.load(p)
            jp, jv = d["joint_pos"], d["joint_vel"] # (T,29)  T number of frames in motion clip and 29 degrees of freedom
            root_h = d["body_pos_w"][:, 0, 2:3]  # (:,14,3) all frames  14 tracked bodies with xyz position (0: pelvis)
            quat = d["body_quat_w"][:, 0]
            lin_w = d["body_lin_vel_w"][:, 0]
            ang_w = d["body_ang_vel_w"][:, 0]
            lin_b = _world_to_base(lin_w, quat)
            ang_b = _world_to_base(ang_w, quat)
            f = np.concatenate([jp, jv, root_h, lin_b, ang_b], -1) # (T,65)
            feats.append(np.concatenate([f[:-1], f[1:]], -1)) # AMP input concatenate two consecutive frames (T-1, T)
        self.data = torch.tensor(np.concatenate(feats, 0), 
                                 dtype=torch.float32, device=device)
        self.dim = self.data.shape[-1]

    def sample(self, n: int) -> torch.Tensor: 
        idx = torch.randint(0, self.data.shape[0], (n,), device=self.data.device)
        return self.data[idx]

"""
Ensures the discriminator learns style, not direction
"""
def _world_to_base(v_w: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray: 
    """Rotate world-frame vectors into the base
    heading invariant
    """
    w, x, y, z = (quat_wxyz[:, i] for i in range(4))
    # inverse rotation = conjugate for unit quats; apply q^-1 * v * q
    # via rotation-matrix rows (transposed R):
    R = np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y + w * z),     2 * (x * z - w * y),
        2 * (x * y - w * z),     1 - 2 * (x * x + z * z), 2 * (y * z + w * x),
        2 * (x * z + w * y),     2 * (y * z - w * x),     1 - 2 * (x * x + y * y),
    ], -1).reshape(-1, 3, 3)                              # R^T (world->base)
    return np.einsum("nij,nj->ni", R, v_w)

class AMPDiscriminator(nn.Module):
    def __init__(self, feat_dim: int, hidden=(1024, 512)):
        super().__init__()
        self.net = _mlp(feat_dim, hidden, 1, out_gain=1.0)
    
    def forward(self, x):
        return self.net(x).squeeze(-1)
    
    @torch.no_grad()
    def style_reward(self, x):
        d = self.forward(x)
        return torch.clamp(1.0 - 0.25 * (d - 1.0) ** 2, min = 0.0)
    
@dataclass
class AMPConfig: 
    disc_lr: float = 1e-4
    grad_pen: float = 5.0       # R1 gradient-penalty weight
    w_task: float = 0.5
    w_style: float = 0.5
    disc_batch: int = 4096
    logit_reg: float = 0.05     # keeps logits near [,1,1], stabilizes LSGAN

class PPOAMP(PPOModel):
    """
    env must expose amp_features in extras: extras["amp_obs]: (N, feat_dim/2) current-frame features
    """

    def __init__(self, env, cfg: PPOConfig, amp_cfg: AMPConfig, expert:ExpertMotionBuffer, actor_obs_dim, critic_obs_dim,
                 act_dim, device="cuda:0"): 
        super().__init__(env, cfg, actor_obs_dim, critic_obs_dim, act_dim, device)
        self.amp_cfg = amp_cfg
        self.expert = expert
        self.disc = AMPDiscriminator(expert.dim).to(device)
        self.disc_opt = torch.optim.Adam(self.disc.parameters(), lr=amp_cfg.disc_lr)
        self._prev_amp = None
        self._policy_trans = []

    @torch.no_grad()
    def collect(self):
        self.buf.reset()
        self._policy_trans = []
        ep_stats = []
        for _ in range(self.cfg.horizon):
            raw_a, raw_c = self._obs["actor"], self._obs["critic"]
            if self.actor_rms is not None: 
                self.actor_rms.update(raw_a); self.critic_rms.update(raw_c)
            a_obs = self._norm(self.actor_rms, raw_a)
            c_obs = self._norm(self.critic_rms, raw_c)
            value = self.critic(c_obs)
            action, logp = self.actor.act(a_obs)

            next_obs, rew, terminated, truncated, extras = self.env.step(action)
            done = (terminated | truncated).float()

            amp_now = extras["amp_obs"]  # (N,65)
            """
            Original AMP paper takes two frames as a sweet spot. 
            Feature dimensions grow linearly, the discriminator gets easier to overfit. 
            """
            if self._prev_amp is not None: 
                trans = torch.cat([self._prev_amp, amp_now], -1) # (N,130)
                style = self.disc.style_reward(trans)
                # rew here is what the mjlab's tracking task computes - the Deepmimic style imitation reward 
                rew = self.amp_cfg.w_task * rew + self.amp_cfg.w_style * style
                self._policy_trans.append(trans)
            #transition across a reset are invalid: 
            self._prev_amp = amp_now.clone()
            self._prev_amp[done.bool()] = float("nan")

            """
            AMP reward combine with the timeout bootstrap
            """
            rew = rew + self.cfg.gamma * value * truncated.float()
            self.buf.add(a_obs, c_obs, action, logp, rew, value, done)
            self._obs = next_obs
            if "log" in extras and extras["log"]: 
                ep_stats.append(extras["log"])
        
        last_values = self.critic(self._norm(self.critic_rms, self._obs["critic"]))
        self.buf.compute_gae(self.cfg.gamma, self.cfg.lam, last_values)
        return ep_stats

    def update_disc(self):
        if not self._policy_trans:
            return {}
        pol = torch.cat(self._policy_trans, 0)
        # keep values where no values in the row is NaN
        pol = pol[~torch.isnan(pol).any(-1)]  # drop cross-reset pairs

        n = min(self.amp_cfg.disc_batch, pol.shape[0])
        # random permutation fo all indices
        pol = pol[torch.randperm(pol.shape[0], device=pol.device)[:n]]
        # expert sampling (create n number of indices)
        exp = self.expert.sample(n).requires_grad_(True)

        # d_exp = expert (large dataset duplicates are rare)
        # d_pol = policy (limited to one rollout, want diversity)
        d_exp, d_pol = self.disc(exp), self.disc(pol)
        loss_lsgan = ((d_exp - 1) ** 2).mean() + ((d_pol + 1) ** 2).mean()
        grad = torch.autograd.grad(d_exp.sum(), exp, create_graph=True)[0]
        gp = grad.pow(2).sum(-1).mean() 
        logit_reg = d_exp.pow(2).mean() + d_pol.pow(2).mean()
        loss = loss_lsgan + 0.5 * self.amp_cfg.grad_pen * gp \
            + self.amp_cfg.logit_reg * logit_reg

        self.disc_opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.disc.parameters(), 1.0)
        self.disc_opt.step()
        return {"disc_loss": loss_lsgan.item(), "grad_pen": gp.item(),
                "d_expert": d_exp.mean().item(), "d_policy": d_pol.mean().item()}


# -------------------------- MVAE

class MoEDecoder(nn.Module):
    """
    Mixture-of-experts decoder (Ling 2020): gating net picks expert blend; 
    each expert maps (z, x_prev) -> x_next. MoE fights mean-pose collapse
    """

    def __init__(self, z_dim, x_dim, hidden=256, n_experts=4): 
        super().__init__()
        self.n = n_experts
        self.gate = nn.Sequential(nn.Linear(z_dim + x_dim, hidden), nn.ELU(), 
                                  nn.Linear(hidden, hidden), nn.ELU(), nn.Linear(hidden, n_experts))
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(z_dim + x_dim, hidden), nn.ELU(),
                          nn.Linear(hidden, hidden), nn.ELU(),
                          nn.Linear(hidden, x_dim))
        for _ in range(n_experts)])

    def forward(self, z, x_prev): 
        h = torch.cat([z, x_prev], -1)
        w = torch.softmax(self.gate(h), dim=-1)             # (B, n)
        outs = torch.stack([e(h) for e in self.experts], 1) # (B, n, x)
        return (w.unsqueeze(-1) * outs).sum(1)              # (B, x)
        
class MotionVAE(nn.Module): 
    """Autoregressive CVAE:  q(z | x_t, x_{t-1}),  p(x_t | z, x_{t-1})."""
    
    def __init__(self, x_dim:int, z_dim:int=32, hidden: int=256):
        super().__init__()
        self.z_dim = z_dim
        self.enc = nn.Sequential(nn.Linear(2 * x_dim, hidden), nn.ELU(),
                                 nn.Linear(hidden, hidden), nn.ELU())
        self.mu, self.logvar = nn.Linear(hidden, z_dim), nn.Linear(hidden, z_dim)
        self.dec = MoEDecoder(z_dim, x_dim, hidden)

    """
    q(z | x_t, x_{t-1}) - Encoder: infer latent z from current and previous frames 
    """
    def encode(self, x_t, x_prev):
        h = self.enc(torch.cat([x_t, x_prev], -1))
        return self.mu(h), self.logvar(h)
    
    def forward(self, x_t, x_prev): 
        mu, logvar = self.encode(x_t, x_prev)
        # sample z from the inferred distribution
        # teaches mu and logvar to encode information 
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        return self.dec(z, x_prev), mu, logvar
    
    @torch.no_grad()
    def step(self, z, x_prev): 
        """Policy facing API: latent action z -> nexxt pose feature"""
        return self.dec(z, x_prev)
    
def train_mvae(model: MotionVAE, clips: list[torch.Tensor], epochs=100, 
               batch=256, roll_len=8, lr=1e-4, beta=0.2, device="cuda:0",
               log_every=10):
    """clips: list of (T, x_dim) feature tensors (same 65-dim AMP feature works)
    
    Scheduled sampling: with prob p (annelead 0 -> 1) the decoder is conditioned on its own previosu prediction inside 
    a short rollout - without this the decoder drifts catastrophically at policy time.
    """
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    data = [c.to(device) for c in clips if c.shape[0] > roll_len+1]
    hist = []
    for ep in range(epochs): 
        p_own = min(1.0, ep / (0.6 * epochs)) # anneal 0 -> 1
        # sample windows 
        xs = []
        for _ in range(batch): 
            # Randomly pick a video clip
            c = data[np.random.randint(len(data))]
            # Random starting point
            t0 = np.random.randint(0, c.shape[0] - roll_len - 1)
            # Extract 9 window frames
            xs.append(c[t0:t0 + roll_len + 1])
        # Stack 256 windows into a batch
        x = torch.stack(xs)             # (B, L+1, D) ex. (256, 9, 65)

        x_prev = x[:, 0] # (256, 65)
        rec_loss = kl_loss = 0.0
        # For each time step (1 to 8)
        for t in range(1, roll_len + 1): 
            # Get the current frame 
            x_t = x[:, t]
            # x_hat: predicted current frame 
            # mu: latent mean 
            # logvar: latent log variance 
            # Predicted using the model 
            x_hat, mu, logvar = model(x_t, x_prev)
            # Compute loss between prediction and ground truth
            rec_loss = rec_loss + (x_hat - x_t).pow(2).mean()

            # pushes mu and logvar toward 0 
            kl_loss = kl_loss + (-0.5 * (1 + logvar - mu.pow(2)
                                         - logvar.exp()). sum(-1).mean())
            # scheduled sampling 
            # tries to take only ones that are < p_own
            use_own = (torch.rand(x.shape[0], 1, device=device) < p_own).float()
            # No gradients through x_hat -> x_prev is treated as constant flow 
            x_prev = use_own * x_hat.detach() + (1 - use_own) * x_t
        # beta chosen as 0.2 as stated on the paper
        # Reconstruction loss + Beta * KL-Divergence Loss
        loss = (rec_loss + beta * kl_loss) / roll_len
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        hist.append((rec_loss.item() / roll_len, kl_loss.item() / roll_len))
        if ep % log_every == 0: 
            print(f"mvae ep {ep:4d} | rec {hist[-1][0]:.4f} | kl {hist[-1][1]:.3f} "
                  f"| p_own {p_own:.2f}")
    return hist

# ---------------------------------------------------- latent space PPO === 

class LatentActionWrapper: 
    """
    Env wrapper: PPO acts in z-space; MVAE decodes z -> pose feature; the pose's joint_pos slice becomes the env action (PD targets)

    joint_slice: which slice of the MVAE feature is joint_pos (0:29 for the 65-dim
    feature above). Stage-2 usage: 
        env = LatentActionWrapper(mjlab_env, mvae, z_dim=32)
        model = PPOAMP(env, cfg, amp_cfg, expert, ..., act_dim=32)
    """

    def __init__(self, env, mvae: MotionVAE, joint_slice=slice(0,29)):
        self.env, self.mvae, self.joint_slice = env,mvae,joint_slice
        self.num_envs = env.num_envs
        self.x_dim = mvae.dec.experts[0][-1].out_features
        self._x_prev = None
    
    def reset(self):
        obs, info = self.env.reset()
        self._x_prev = None
        return obs, info 
    
    def step(self, z): 
        if self._x_prev is None:  # first step: hold zero action
            x = self.mvae.step(z, torch.zeros(
                z.shape[0], self.x_dim, 
                device=z.device
            ))
        else: 
            x = self.mvae.step(z, self._x_prev)
        obs, rew, term, trunc, extras = self.env.step(x[:, self.joint_slice])
        # next conditioning = the env's. REAL resulting feature 
        # (closes the loop, prevents decoder drift): env must expose it in extras["amp_obs"]
        self._x_prev = extras["amp_obs"]
        done = term | trunc
        if done.any(): 
            self._x_prev[done] = 0.0
        return obs, rew, term, trunc, extras