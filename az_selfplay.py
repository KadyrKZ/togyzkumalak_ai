import os
import time
import random
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

import torch

from az_mcts import AlphaZeroMCTS
from az_togyz_adapter import make_env, canonical_obs, state_key, legal_mask, step_env, player_to_move
from az_model import AZNet

@dataclass
class Sample:
    obs: np.ndarray      # (128,)
    pi: np.ndarray       # (9,)
    z: float             # in [-1,1], from perspective of player-to-move in obs

def temperature_schedule(move_idx: int, temp_moves: int = 20) -> float:
    return 1.0 if move_idx < temp_moves else 0.0

def choose_action(pi: np.ndarray, tau: float) -> int:
    if tau <= 0.0:
        return int(np.argmax(pi))
    # sample
    probs = pi ** (1.0 / max(tau, 1e-6))
    probs = probs / probs.sum()
    return int(np.random.choice(len(pi), p=probs))

def play_one_game(net: AZNet, device: str, sims: int = 400, c_puct: float = 1.25) -> List[Sample]:
    env = make_env()
    mcts = AlphaZeroMCTS(net, device=device, c_puct=c_puct)

    history: List[Tuple[np.ndarray, np.ndarray, int]] = []  # (obs, pi, player_id)
    done = False
    move_idx = 0
    last_r_white = 0.0

    while not done and move_idx < 2000:
        obs = canonical_obs(env)
        p = player_to_move(env)
        mask = legal_mask(env)

        pi = mcts.search(
            env,
            canonical_obs_fn=canonical_obs,
            key_fn=state_key,
            mask_fn=legal_mask,
            step_fn=step_env,
            terminal_fn=lambda e: False,
            sims=sims,
            add_dirichlet=True,
        )

        # ensure illegal moves are zero
        pi = pi * mask
        if pi.sum() <= 0:
            legal = np.where(mask > 0)[0]
            pi = np.zeros(9, dtype=np.float32)
            pi[legal] = 1.0 / len(legal)
        else:
            pi = pi / pi.sum()

        history.append((obs, pi.astype(np.float32), p))

        tau = temperature_schedule(move_idx)
        a = choose_action(pi, tau=tau)
        r_white, done = step_env(env, a)
        last_r_white = r_white
        move_idx += 1

    # terminal outcome: from WHITE perspective (env reward)
    # Convert to z for each stored state (player-to-move perspective)
    samples: List[Sample] = []
    for obs, pi, p in history:
        z = last_r_white if p == 0 else -last_r_white
        samples.append(Sample(obs=obs, pi=pi, z=float(z)))
    return samples

def selfplay_batch(net_path: str, out_path: str, games: int = 10, sims: int = 200, device: str = "cpu"):
    net = AZNet()
    if os.path.exists(net_path):
        net.load_state_dict(torch.load(net_path, map_location=device))
    net.to(device).eval()

    all_samples: List[Sample] = []
    for g in range(games):
        s = play_one_game(net, device=device, sims=sims)
        all_samples.extend(s)
        print(f"game {g+1}/{games}: {len(s)} samples, total={len(all_samples)}")

    # save as npz
    obs = np.stack([x.obs for x in all_samples]).astype(np.float32)
    pi = np.stack([x.pi for x in all_samples]).astype(np.float32)
    z = np.array([x.z for x in all_samples], dtype=np.float32)
    np.savez_compressed(out_path, obs=obs, pi=pi, z=z)
    print("saved", out_path, "samples:", len(all_samples))
