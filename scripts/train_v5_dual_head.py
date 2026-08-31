import sys, os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
from scripts.eval_vs_standard_ai import compute_metrics, metrics

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

class DualHeadGatedPolicy(nn.Module):
    def __init__(self, obs_dim=384, action_dim=11):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1)
        )
        self.actor_head = nn.Linear(64, action_dim)
        self.critic_head = nn.Linear(64, 1)

    def forward(self, x):
        feat = self.fc(x)
        return self.actor_head(feat), self.critic_head(feat)

    def get_action(self, obs, deterministic=True):
        with torch.no_grad():
            if isinstance(obs, np.ndarray):
                obs = torch.FloatTensor(obs).unsqueeze(0) if obs.ndim == 1 else torch.FloatTensor(obs)
            act, _ = self.forward(obs)
            act_np = act.cpu().numpy().squeeze(0)
            probs = np.exp(act_np) / np.sum(np.exp(act_np))
            return act_np, probs

class RealMarketV5Env(gym.Env):
    def __init__(self, price_df, max_assets=10):
        super().__init__()
        self.price_df = price_df
        self.prices = price_df.values[:, :max_assets]
        self.T, self.N = self.prices.shape
        self.obs_dim = 384
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.obs_dim,), np.float32)
        self.action_space = spaces.Box(-5.0, 5.0, (self.N + 1,), np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 30
        self.cash = 5000.0
        self.shares = (5000.0 / self.N) / self.prices[self.current_step]
        self.log_wealth = [10000.0]
        return np.zeros(self.obs_dim, dtype=np.float32), {}

    def step(self, action_tuple):
        self.current_step += 1
        done = self.current_step >= self.T - 1
        p = self.prices[min(self.current_step, self.T - 1)]
        w = self.cash + np.sum(self.shares * p)
        self.log_wealth.append(w)
        return np.zeros(self.obs_dim, dtype=np.float32), 0.0, done, False, {}
