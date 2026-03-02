"""
Glue code: makes gym_togyzkumalak env look like AlphaZero game API.

We rely on the existing environment:
- env.available_action() -> list[0/1] for 9 actions
- env.observation() -> np.array shape (1,128) or (1,1,128) depending; we'll flatten robustly
- env.step(action) -> (obs, reward_white, done, info) where reward is from WHITE perspective (+1/-1/0)
- env.unwrapped.board.run.name tells who is to move
"""
from __future__ import annotations

import numpy as np
import copy
import gym
import gym_togyzkumalak  # registers env

def make_env():
    env = gym.make("Togyzkumalak-v0")
    env.reset()          # обязательно инициализировать board
    return env.unwrapped # убрать OrderEnforcing/EnvChecker wrappers

def player_to_move(env) -> int:
    # 0 = white, 1 = black
    b = env.unwrapped.board
    return 0 if b.run.name == "white" else 1

def legal_mask(env) -> np.ndarray:
    mask = np.array(env.available_action(), dtype=np.float32)
    if mask.shape != (9,):
        mask = mask.reshape(9,)
    return mask

def canonical_obs(env) -> np.ndarray:
    """
    Convert env observation to canonical: player-to-move features first.
    Original encoding: [white63, black63, turn2].
    If black to move, we swap halves and flip turn bits to [1,0].
    """
    obs = env.observation()
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    if obs.shape[0] != 128:
        raise ValueError(f"Expected 128 obs, got {obs.shape}")
    turn = obs[-2:]
    is_black = (turn[0] < 0.5 and turn[1] > 0.5)
    if is_black:
        a = obs[:63].copy()
        b = obs[63:126].copy()
        obs[:63] = b
        obs[63:126] = a
        obs[-2] = 1.0
        obs[-1] = 0.0
    # else already canonical with turn=[1,0]
    return obs

def state_key(env) -> bytes:
    """
    Stable key for caching inside MCTS.
    Use canonical obs bytes + current player id.
    """
    o = canonical_obs(env)
    p = player_to_move(env)
    return o.tobytes() + bytes([p])

def step_env(env, action: int):
    _, r, done, _info = env.step(action)
    return float(r), bool(done)

def clone_env(env):
    return copy.deepcopy(env)
