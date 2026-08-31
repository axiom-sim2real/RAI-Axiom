import os, sys
import torch
import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rai.core.agent import Agent
from rai.core.entity import Entity
from rai.core.world import World
from rai.core.events import EventLogger
from scripts.train_v6_fast import RawPriceSyntheticEnv, FastTradingNet
from scripts.download_data import ensure_real_market_checkpoints

def test_agent_inventory():
    agent = Agent(id=1)
    entity = Entity(id=101)
    agent.inventory[entity] = 15.0
    assert agent.inventory[entity] == 15.0

def test_raw_price_synthetic_env():
    env = RawPriceSyntheticEnv(num_assets=10, history_len=30, episode_len=50)
    obs, info = env.reset(seed=42)
    assert obs.shape == (30 * 22,)
    
    action = np.zeros(11, dtype=np.float32)
    nobs, reward, done, truncated, info = env.step(action)
    assert nobs.shape == (30 * 22,)
    assert isinstance(reward, (float, np.floating))
    assert isinstance(done, bool)

def test_deep_end_to_end_trading_net():
    net = FastTradingNet(history_len=30, features_per_step=22, action_dim=11)
    obs = torch.randn(2, 660)
    actor_out, critic_out = net(obs)
    assert actor_out.shape == (2, 11)
    assert critic_out.shape == (2, 1)
    
    action = net.get_action(np.zeros(660, dtype=np.float32))
    assert action.shape == (11,)

def test_event_logger(tmp_path):
    log_file = tmp_path / "events.jsonl"
    logger = EventLogger(str(log_file))
    logger.log_event({"event": "EXCHANGE", "agent_1": 1, "agent_2": 2})
    assert log_file.exists()
    content = log_file.read_text(encoding='utf-8')
    assert "EXCHANGE" in content

def test_ensure_real_market_checkpoints():
    ensure_real_market_checkpoints()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_path = os.path.join(project_root, "data", "real_market_checkpoints", "test_prices.csv")
    assert os.path.exists(test_path)
