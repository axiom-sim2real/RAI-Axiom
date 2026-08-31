"""
Axiom Multi-Seed Training for Kaggle (GPU T4 x2)
=================================================
Trains the Axiom model (AxiomNet architecture, AlphaSyntheticEnv)
across 10 seeds, then evaluates each checkpoint on real market data.

Axiom was previously called 'RAI v6 Alpha'. This script reproduces the exact
same training config but across multiple seeds for CI-verified results.

USAGE (Kaggle notebook):
  1. Paste this entire script into a single code cell
  2. Set Accelerator to GPU T4 x2
  3. Enable Internet (for yfinance downloads)
  4. Run the cell — takes ~30-60 min on dual T4

OUTPUT (download from Kaggle Output tab):
  - axiom_checkpoints.zip          — 10 .pt files with full metadata
  - axiom_per_seed_results.csv     — per-seed OOS + future holdout metrics
  - axiom_per_universe_summary.json — mean/std/min/max per universe
"""

import subprocess, sys
# Kaggle-only bootstrap. Guarded so that importing this module (e.g. from
# scripts/baseline_multiseed.py, to reuse the Axiom eval harness) does not
# trigger a pip network call. Behaviour when run as a Kaggle script is unchanged.
if __name__ == "__main__":
    subprocess.run([sys.executable, "-m", "pip", "install", "yfinance", "-q"], check=False)


import os, sys, time, json, zipfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# ══════════════════════════════════════════════════════════════════════════════
#  GPU SETUP
# ══════════════════════════════════════════════════════════════════════════════
if torch.cuda.is_available():
    DEVICES = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    print(f"  GPUs: {[torch.cuda.get_device_name(d) for d in DEVICES]}")
else:
    DEVICES = [torch.device('cpu')]
    print("  No GPU — running on CPU (will be slow)")

SEEDS = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]

# ══════════════════════════════════════════════════════════════════════════════
#  AXIOM ARCHITECTURE: AxiomNet -- NOT the same network as train_v6_alpha.py
#  (that file uses FastTradingNet: mean-pool, 51,703 params). The previous
#  "identical to train_v6_alpha.py" comment here was false.
# ══════════════════════════════════════════════════════════════════════════════
class AxiomNet(nn.Module):
    """Axiom's architecture. conv1 -> conv2 -> 1-layer Transformer ->
    flatten(embed_dim * history_len) -> fc_features + LayerNorm -> actor/critic.
    289,527 params at the default 30x22 -> 11 configuration.

    NOT the same network as FastTradingNet in scripts/train_v6_fast.py, which
    mean-pools over time instead of flattening and has 51,703 params. The two
    are not state_dict-compatible; until 2026-08-29 both carried the same
    (misleading) class name -- see docs/consolidation_report.md §15. state_dict
    keys here are conv1.*, conv2.*, fc_features.0/2, actor, critic -- matching
    checkpoints/axiom_multiseed/axiom_seed*.pt.
    """
    def __init__(self, history_len=30, features_per_step=22, action_dim=11, embed_dim=64):
        super().__init__()
        self.history_len = history_len
        self.features_per_step = features_per_step
        self.conv1 = nn.Conv1d(features_per_step, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, embed_dim, kernel_size=3, padding=1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=2, dim_feedforward=128,
            dropout=0.05, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.fc_features = nn.Sequential(
            nn.Linear(embed_dim * history_len, 128),
            nn.ReLU(),
            nn.LayerNorm(128)
        )
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)
        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.5)

    def forward(self, flat_obs):
        b = flat_obs.shape[0]
        x = flat_obs.reshape(b, self.history_len, self.features_per_step)
        x = x.transpose(1, 2)  # (B, features, time)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.transpose(1, 2)  # (B, time, embed)
        x = self.transformer(x)
        x = x.reshape(b, -1)
        features = self.fc_features(x)
        return self.actor(features), self.critic(features)

    def get_action(self, flat_obs, deterministic=True):
        with torch.no_grad():
            if isinstance(flat_obs, np.ndarray):
                flat_obs = torch.FloatTensor(flat_obs).unsqueeze(0)
            mean, _ = self.forward(flat_obs)
            if deterministic:
                return mean.squeeze(0).cpu().numpy()
            dist = Normal(mean, torch.exp(self.log_std))
            return dist.sample().squeeze(0).cpu().numpy()


# ══════════════════════════════════════════════════════════════════════════════
#  AXIOM SYNTHETIC ENVIRONMENT (identical to AlphaSyntheticEnv)
# ══════════════════════════════════════════════════════════════════════════════
import gymnasium as gym
from gymnasium import spaces

class AlphaSyntheticEnv(gym.Env):
    """Synthetic market env encouraging 90-100% stock participation in bull trends."""
    REGIMES = {
        'strong_bull': {'drift': (0.30, 0.70),  'vol': (0.08, 0.16)},
        'mild_bull':   {'drift': (0.15, 0.30),  'vol': (0.10, 0.18)},
        'sideways':    {'drift': (-0.02, 0.05), 'vol': (0.12, 0.20)},
        'bear_crash':  {'drift': (-0.50, -0.20),'vol': (0.25, 0.50)},
    }

    def __init__(self, num_assets=10, history_len=30, episode_len=504,
                 initial_cash=10000.0, fee=0.001):
        super().__init__()
        self.num_assets = num_assets
        self.history_len = history_len
        self.episode_len = episode_len
        self.initial_cash = initial_cash
        self.fee = fee
        self.features_per_step = 2 * num_assets + 2  # = 22
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(history_len * self.features_per_step,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-5.0, high=5.0, shape=(num_assets + 1,), dtype=np.float32
        )
        self.reset()

    def _generate_raw_prices(self):
        total_T = self.episode_len + self.history_len + 10
        n_seg = self.np_random.integers(3, 7)
        keys = list(self.REGIMES.keys())
        seq = [keys[self.np_random.integers(len(keys))] for _ in range(n_seg)]
        dur = self.np_random.dirichlet(np.ones(n_seg) * 2.0) * total_T
        dur = np.maximum(dur.astype(int), 30)
        dur[-1] = total_T - sum(dur[:-1])
        prices = np.zeros((total_T, self.num_assets), dtype=np.float64)
        for asset in range(self.num_assets):
            p = self.np_random.uniform(20.0, 300.0)
            series = [p]
            day = 0
            for reg, d in zip(seq, dur):
                params = self.REGIMES[reg]
                drift = self.np_random.uniform(*params['drift']) + self.np_random.uniform(-0.03, 0.03)
                vol = self.np_random.uniform(*params['vol']) * self.np_random.uniform(0.85, 1.15)
                mu = drift / 252.0; sigma = vol / np.sqrt(252.0)
                for _ in range(max(0, min(d, total_T - day - 1))):
                    p = max(0.01, p * np.exp((mu - 0.5*sigma**2) + sigma * self.np_random.standard_normal()))
                    series.append(p)
                    day += 1
                    if day >= total_T: break
                if day >= total_T: break
            while len(series) < total_T: series.append(series[-1])
            prices[:, asset] = series[:total_T]
        return prices

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.prices = self._generate_raw_prices()
        self.start = self.history_len
        self.current_step = self.start
        self.cash = self.initial_cash * 0.05
        init_p = self.prices[self.current_step]
        self.shares = (self.initial_cash * 0.95 / self.num_assets) / init_p
        self.peak_wealth = self.initial_cash
        self.last_wealth = self.initial_cash
        self.steps_done = 0
        self.obs_history = [self._obs_at(self.start - self.history_len + i) for i in range(self.history_len)]
        return self._flat_obs(), {}

    def _wealth(self):
        return self.cash + np.sum(self.shares * self.prices[self.current_step])

    def _obs_at(self, t):
        p = self.prices[t]; p_prev = self.prices[max(0, t-1)]
        w = max(1e-4, self.cash + np.sum(self.shares * p))
        norm_prices = p / self.prices[self.start]
        log_rets = np.log(p / np.maximum(1e-4, p_prev))
        cash_ratio = self.cash / w
        dd = np.clip((w - self.peak_wealth) / max(1e-4, self.peak_wealth), -1.0, 0.0)
        return np.concatenate([norm_prices, log_rets, [cash_ratio, dd]]).astype(np.float32)

    def _flat_obs(self):
        return np.concatenate(self.obs_history).astype(np.float32)

    def step(self, action):
        cash_logit = np.clip(action[0] - 2.5, -8.0, 3.0)
        target_cash_frac = 1.0 / (1.0 + np.exp(-cash_logit))
        stock_portion = 1.0 - target_cash_frac
        asset_logits = action[1:]
        exp_a = np.exp(asset_logits - np.max(asset_logits))
        target_asset_w = (exp_a / np.sum(exp_a)) * stock_portion
        prices = self.prices[self.current_step]
        wealth = max(1e-4, self._wealth())
        cur_asset_w = (self.shares * prices) / wealth
        cur_cash_frac = self.cash / wealth
        drift = abs(cur_cash_frac - target_cash_frac) + np.sum(np.abs(cur_asset_w - target_asset_w))
        if drift > 0.03:
            t_vol = abs(self.cash - wealth * target_cash_frac) + np.sum(np.abs(self.shares * prices - wealth * target_asset_w))
            net = max(1e-4, wealth - t_vol * self.fee)
            self.cash = net * target_cash_frac
            self.shares = (net * target_asset_w) / np.maximum(1e-4, prices)
        self.current_step += 1
        self.steps_done += 1
        new_wealth = self._wealth()
        self.peak_wealth = max(self.peak_wealth, new_wealth)
        daily_ret = (new_wealth - self.last_wealth) / max(1e-4, self.last_wealth)
        reward = daily_ret * 20.0
        done = self.current_step >= self.prices.shape[0] - 1 or self.steps_done >= self.episode_len
        self.last_wealth = new_wealth
        self.obs_history.pop(0)
        self.obs_history.append(self._obs_at(self.current_step))
        return self._flat_obs(), reward, done, False, {"portfolio_value": new_wealth, "cash_frac": target_cash_frac}


# ══════════════════════════════════════════════════════════════════════════════
#  PPO TRAINING (exact same hyperparameters as train_v6_alpha.py)
# ══════════════════════════════════════════════════════════════════════════════
def train_axiom(seed, device, steps=100_000):
    """Train one Axiom checkpoint with the given seed. Returns model on CPU."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = AlphaSyntheticEnv(num_assets=10, history_len=30, episode_len=504)
    model = AxiomNet(history_len=30, features_per_step=22, action_dim=11).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    BATCH = 64; ROLLOUT = 1024; EPOCHS = 4
    GAMMA = 0.99; LAM = 0.95; CLIP = 0.2

    obs, _ = env.reset(seed=seed)
    step = 0; t0 = time.time()

    while step < steps:
        obs_b, act_b, rew_b, val_b, logp_b = [], [], [], [], []
        for _ in range(ROLLOUT):
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                mean, val = model(obs_t)
                dist = Normal(mean, torch.exp(model.log_std))
                action = dist.sample()
                logp = dist.log_prob(action).sum(dim=-1)
            act_np = action.squeeze(0).cpu().numpy()
            nobs, rew, done, _, _ = env.step(act_np)
            obs_b.append(obs); act_b.append(act_np)
            rew_b.append(rew); val_b.append(val.item()); logp_b.append(logp.item())
            obs = nobs; step += 1
            if done: obs, _ = env.reset()

        with torch.no_grad():
            _, nval = model(torch.FloatTensor(obs).unsqueeze(0).to(device))
            nval = nval.item()

        r = np.array(rew_b); v = np.array(val_b + [nval])
        d = r + GAMMA * v[1:] - v[:-1]
        adv = np.zeros_like(r); gae = 0.0
        for t in reversed(range(len(r))):
            gae = d[t] + GAMMA * LAM * gae
            adv[t] = gae
        ret = adv + v[:-1]

        o_t = torch.FloatTensor(np.array(obs_b)).to(device)
        a_t = torch.FloatTensor(np.array(act_b)).to(device)
        adv_t = torch.FloatTensor(adv).to(device)
        ret_t = torch.FloatTensor(ret).to(device)
        old_t = torch.FloatTensor(np.array(logp_b)).to(device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        for _ in range(EPOCHS):
            idx = np.random.permutation(len(obs_b))
            for s in range(0, len(obs_b), BATCH):
                b_idx = idx[s:s+BATCH]
                mean, val = model(o_t[b_idx])
                dist = Normal(mean, torch.exp(model.log_std))
                new_logp = dist.log_prob(a_t[b_idx]).sum(dim=-1)
                ratio = torch.exp(new_logp - old_t[b_idx])
                surr1 = ratio * adv_t[b_idx]
                surr2 = torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * adv_t[b_idx]
                loss = -torch.min(surr1, surr2).mean() + 0.5 * F.mse_loss(val.squeeze(-1), ret_t[b_idx])
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

    elapsed = time.time() - t0
    print(f"    Seed {seed}: trained in {elapsed:.0f}s ({steps/elapsed:.0f} FPS)", flush=True)

    model.cpu().eval()
    return model


# ══════════════════════════════════════════════════════════════════════════════
#  EVALUATION ON REAL MARKET DATA
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_on_real_data(model, prices, fee_bps=5, slippage_pct=0.02, rebal_thresh=0.03):
    """Run frozen Axiom policy on real price array. Returns equity curve."""
    T, N = prices.shape
    history_len = 30
    if T <= history_len + 1:
        return np.ones(T) * 10000.0

    norm_prices = prices / prices[0]
    cash = 10000.0 * 0.05
    shares = (10000.0 * 0.95 / N) / prices[history_len]
    peak = 10000.0
    fee_rate = fee_bps / 10000.0 + slippage_pct / 100.0

    obs_history = []
    for i in range(history_len):
        p = norm_prices[i]; p_prev = norm_prices[max(0, i-1)]
        w = max(1e-4, cash + np.sum(shares * prices[i]))
        log_rets = np.log(p / np.maximum(1e-4, p_prev))
        cash_r = cash / w
        dd = np.clip((w - peak) / max(1e-4, peak), -1.0, 0.0)
        obs_history.append(np.concatenate([p, log_rets, [cash_r, dd]]).astype(np.float32))

    equity = [10000.0] * history_len

    for t in range(history_len, T):
        flat_obs = np.concatenate(obs_history).astype(np.float32)
        action = model.get_action(flat_obs, deterministic=True)

        cash_logit = np.clip(action[0] - 2.5, -8.0, 3.0)
        target_cash = 1.0 / (1.0 + np.exp(-cash_logit))
        stock_portion = 1.0 - target_cash
        asset_logits = action[1:N+1] if len(action) > N else action[1:]
        exp_a = np.exp(asset_logits - np.max(asset_logits))
        target_w = (exp_a / np.sum(exp_a)) * stock_portion

        wealth = max(1e-4, cash + np.sum(shares * prices[t]))
        cur_w = (shares * prices[t]) / wealth
        cur_cash = cash / wealth
        drift = abs(cur_cash - target_cash) + np.sum(np.abs(cur_w - target_w))

        if drift > rebal_thresh:
            t_vol = abs(cash - wealth * target_cash) + np.sum(np.abs(shares * prices[t] - wealth * target_w))
            net = max(1e-4, wealth - t_vol * fee_rate)
            cash = net * target_cash
            shares = (net * target_w) / np.maximum(1e-4, prices[t])

        wealth = cash + np.sum(shares * prices[t])
        peak = max(peak, wealth)
        equity.append(wealth)

        # Update obs
        p = norm_prices[t]; p_prev = norm_prices[max(0, t-1)]
        log_rets = np.log(p / np.maximum(1e-4, p_prev))
        cash_r = cash / max(1e-4, wealth)
        dd = np.clip((wealth - peak) / max(1e-4, peak), -1.0, 0.0)
        obs_history.pop(0)
        obs_history.append(np.concatenate([p, log_rets, [cash_r, dd]]).astype(np.float32))

    return np.array(equity)


def compute_metrics(equity):
    """Compute Sharpe, return, max drawdown from equity curve."""
    eq = np.array(equity, dtype=np.float64)
    ret = (eq[-1] / eq[0] - 1.0) * 100.0
    daily_rets = np.diff(eq) / np.maximum(1e-8, eq[:-1])
    sharpe = np.mean(daily_rets) / max(1e-8, np.std(daily_rets)) * np.sqrt(252) if len(daily_rets) > 1 else 0.0
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / np.maximum(1e-8, peak)
    max_dd = np.min(dd) * 100.0
    return {"return_pct": ret, "sharpe": sharpe, "max_dd": max_dd}


# ══════════════════════════════════════════════════════════════════════════════
#  UNIVERSES (same as canonical evaluation)
# ══════════════════════════════════════════════════════════════════════════════
UNIVERSES = {
    "US_ETFs": {
        "tickers": ["SPY", "QQQ", "EEM", "VNQ", "HYG", "TLT", "GLD", "USO", "UUP", "IWM"],
        "period": "10y",
    },
    "US_MegaCap_PIT": {
        "tickers": ["AAPL", "XOM", "MSFT", "GOOGL", "GE", "JNJ", "PG", "WFC", "JPM", "CVX"],
        "period": "10y",
    },
    "Global_Indices": {
        "tickers": ["SPY", "EWJ", "EWG", "EWU", "MCHI", "INDA", "EWZ", "EFA", "EEM", "FXI"],
        "period": "10y",
    },
    "India_Nifty_50": {
        "tickers": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
                     "BHARTIARTL.NS", "ITC.NS", "SBIN.NS", "LT.NS", "AXISBANK.NS"],
        "period": "5y",
    },
    "Forex_Commodities": {
        "tickers": ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X",
                     "GC=F", "CL=F", "SI=F", "HG=F", "NG=F"],
        "period": "5y",
    },
    "Crypto_PIT": {
        "tickers": ["BTC-USD", "ETH-USD", "XRP-USD", "BCH-USD", "LTC-USD",
                     "EOS-USD", "BNB-USD", "XTZ-USD", "LINK-USD", "TRX-USD"],
        "period": "5y",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN: TRAIN + EVALUATE
# ══════════════════════════════════════════════════════════════════════════════
def main():
    import yfinance as yf

    print("=" * 100)
    print("  AXIOM MULTI-SEED TRAINING (10 seeds, AxiomNet)")
    print("  Training config: lr=3e-4, batch=64, rollout=1024, steps=100k, epochs=4")
    print("  Environment: AlphaSyntheticEnv (4 regimes, cash_logit_offset=-2.5)")
    print(f"  GPUs: {len(DEVICES)}, Seeds: {SEEDS}")
    print("=" * 100)

    # ── Phase 1: Train all seeds ──────────────────────────────────────────
    os.makedirs("axiom_checkpoints", exist_ok=True)
    models = {}

    for i, seed in enumerate(SEEDS):
        device = DEVICES[i % len(DEVICES)]
        print(f"\n  [{i+1}/{len(SEEDS)}] Training seed {seed} on {device}...", flush=True)
        model = train_axiom(seed, device, steps=100_000)

        # Save checkpoint with metadata
        ckpt = {
            "model_state_dict": model.state_dict(),
            "architecture": "AxiomNet",
            "model_name": "Axiom",
            "version": "v0.9",
            "seed": seed,
            "training_config": {
                "lr": 3e-4, "batch": 64, "rollout": 1024,
                "steps": 100000, "epochs": 4, "gamma": 0.99, "lam": 0.95,
                "clip": 0.2, "grad_clip": 0.5,
                "env": "AlphaSyntheticEnv", "episode_len": 504,
                "cash_logit_offset": -2.5, "reward_scale": 20.0,
            },
        }
        path = f"axiom_checkpoints/axiom_seed{seed}.pt"
        torch.save(ckpt, path)
        models[seed] = model
        print(f"    Saved: {path}", flush=True)

    # ── Phase 2: Evaluate on all universes ────────────────────────────────
    print("\n" + "=" * 100)
    print("  EVALUATING ON REAL MARKET DATA (6 universes)")
    print("=" * 100)

    all_rows = []

    for uname, uinfo in UNIVERSES.items():
        print(f"\n  --- {uname} ---", flush=True)

        # Download data
        try:
            df = yf.download(uinfo["tickers"], period=uinfo["period"],
                           auto_adjust=True, progress=False)["Close"]
            df = df.dropna()
            prices = df.values
            print(f"    Downloaded {prices.shape[0]} days x {prices.shape[1]} assets", flush=True)
        except Exception as e:
            print(f"    ERROR downloading {uname}: {e}", flush=True)
            continue

        if prices.shape[0] < 100:
            print(f"    SKIP: too few data points ({prices.shape[0]})", flush=True)
            continue

        # Split: 60% train, 20% OOS, 20% future holdout
        T = prices.shape[0]
        train_end = int(T * 0.6)
        oos_end = int(T * 0.8)

        oos_prices = prices[train_end:oos_end]
        future_prices = prices[oos_end:]

        for seed in SEEDS:
            model = models[seed]

            # OOS evaluation
            oos_eq = evaluate_on_real_data(model, oos_prices)
            oos_m = compute_metrics(oos_eq)

            # Future holdout evaluation
            fut_eq = evaluate_on_real_data(model, future_prices)
            fut_m = compute_metrics(fut_eq)

            row = {
                "Universe": uname,
                "Model": "Axiom v0.9",
                "Seed": seed,
                "OOS Return (%)": oos_m["return_pct"],
                "OOS Sharpe": oos_m["sharpe"],
                "OOS Max DD (%)": oos_m["max_dd"],
                "Future Return (%)": fut_m["return_pct"],
                "Future Sharpe": fut_m["sharpe"],
                "Future Max DD (%)": fut_m["max_dd"],
            }
            all_rows.append(row)
            print(f"    Seed {seed}: OOS Sharpe={oos_m['sharpe']:+.3f}, Future Sharpe={fut_m['sharpe']:+.3f}", flush=True)

    # ── Phase 3: Save results ─────────────────────────────────────────────
    results_df = pd.DataFrame(all_rows)
    results_df.to_csv("axiom_per_seed_results.csv", index=False)
    print(f"\n  Per-seed results saved to 'axiom_per_seed_results.csv' ({len(results_df)} rows)")

    # Summary statistics
    summary = {}
    for uname in results_df["Universe"].unique():
        udf = results_df[results_df["Universe"] == uname]
        summary[uname] = {
            "n_seeds": len(udf),
            "OOS_Sharpe_mean": float(udf["OOS Sharpe"].mean()),
            "OOS_Sharpe_std": float(udf["OOS Sharpe"].std()),
            "Future_Sharpe_mean": float(udf["Future Sharpe"].mean()),
            "Future_Sharpe_std": float(udf["Future Sharpe"].std()),
            "Future_Return_mean": float(udf["Future Return (%)"].mean()),
            "Future_MaxDD_mean": float(udf["Future Max DD (%)"].mean()),
        }
    with open("axiom_per_universe_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved to 'axiom_per_universe_summary.json'")

    # Zip checkpoints
    with zipfile.ZipFile("axiom_checkpoints.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in sorted(os.listdir("axiom_checkpoints")):
            if fname.endswith(".pt"):
                zf.write(f"axiom_checkpoints/{fname}", fname)
    print(f"  Checkpoints saved to 'axiom_checkpoints.zip'")

    # ── Phase 4: Print summary table ──────────────────────────────────────
    print("\n" + "=" * 100)
    print("  AXIOM v0.9 — MULTI-SEED SUMMARY (10 seeds)")
    print("=" * 100)
    print(f"{'Universe':>25s} | {'OOS Sharpe':>20s} | {'Future Sharpe':>20s} | {'Future Return':>15s}")
    print("-" * 90)
    for uname, s in summary.items():
        print(f"{uname:>25s} | {s['OOS_Sharpe_mean']:+.3f} +/- {s['OOS_Sharpe_std']:.3f}     | {s['Future_Sharpe_mean']:+.3f} +/- {s['Future_Sharpe_std']:.3f}     | {s['Future_Return_mean']:+.1f}%")

    print("\n" + "=" * 100)
    print("  DONE. Download from Kaggle Output tab:")
    print("    - axiom_checkpoints.zip")
    print("    - axiom_per_seed_results.csv")
    print("    - axiom_per_universe_summary.json")
    print("  Then place .pt files into RAI/checkpoints/axiom_multiseed/")
    print("=" * 100)


if __name__ == "__main__":
    main()
