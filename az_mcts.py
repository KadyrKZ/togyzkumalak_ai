import math
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List

def softmax_masked(logits: np.ndarray, mask: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Masked softmax over actions."""
    x = logits.astype(np.float64)
    x = x - np.max(x)
    ex = np.exp(x) * mask
    s = ex.sum()
    if s < eps:
        # if all masked (shouldn't happen), fallback uniform over legal
        legal = np.where(mask > 0)[0]
        out = np.zeros_like(mask, dtype=np.float64)
        if len(legal) > 0:
            out[legal] = 1.0 / len(legal)
        return out
    return (ex / s).astype(np.float32)

@dataclass
class EdgeStats:
    P: float
    N: int = 0
    W: float = 0.0

    @property
    def Q(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0

@dataclass
class Node:
    key: bytes
    priors: np.ndarray = field(default_factory=lambda: np.zeros(9, dtype=np.float32))
    edges: Dict[int, EdgeStats] = field(default_factory=dict)
    expanded: bool = False

class AlphaZeroMCTS:
    def __init__(self, net, device="cpu", c_puct: float = 1.25, dirichlet_alpha: float = 0.3, dirichlet_eps: float = 0.25):
        self.net = net
        self.device = device
        self.c_puct = c_puct
        self.dir_alpha = dirichlet_alpha
        self.dir_eps = dirichlet_eps
        self.tree: Dict[bytes, Node] = {}

    def _get_node(self, key: bytes) -> Node:
        if key not in self.tree:
            self.tree[key] = Node(key=key)
        return self.tree[key]

    def _puct_select(self, node: Node, mask: np.ndarray) -> int:
        total_N = sum(e.N for e in node.edges.values()) + 1
        best_a, best_score = None, -1e18
        for a in range(9):
            if mask[a] == 0:
                continue
            e = node.edges.get(a)
            if e is None:
                # unexpanded edge: treat as zero stats but has prior
                P = float(node.priors[a])
                Q = 0.0
                N = 0
            else:
                P = float(e.P)
                Q = float(e.Q)
                N = e.N
            U = self.c_puct * P * math.sqrt(total_N) / (1 + N)
            score = Q + U
            if score > best_score:
                best_score = score
                best_a = a
        return int(best_a)

    def _expand_and_eval(self, node: Node, obs128: np.ndarray, mask: np.ndarray, add_dirichlet: bool) -> float:
        """
        Expand node with priors from network and return value v for current player (canonical).
        """
        with torch.no_grad():
            x = torch.tensor(obs128[None, :], dtype=torch.float32, device=self.device)
            logits, v = self.net(x)
            logits = logits.squeeze(0).detach().cpu().numpy()
            v = float(v.item())

        priors = softmax_masked(logits, mask.astype(np.float32))

        if add_dirichlet:
            legal = np.where(mask > 0)[0]
            if len(legal) > 0:
                noise = np.random.dirichlet([self.dir_alpha] * len(legal)).astype(np.float32)
                priors2 = priors.copy()
                priors2[legal] = (1 - self.dir_eps) * priors2[legal] + self.dir_eps * noise
                priors = priors2

        node.priors = priors.astype(np.float32)
        # initialize edges with priors
        node.edges = {a: EdgeStats(P=float(priors[a])) for a in range(9) if mask[a] > 0}
        node.expanded = True
        return v

    def search(self, root_env, canonical_obs_fn, key_fn, mask_fn, step_fn, terminal_fn, sims: int = 400, add_dirichlet: bool = True) -> np.ndarray:
        """
        Run MCTS sims from root state. Returns visit-count policy pi (length 9, sums to 1).
        - canonical_obs_fn(env) -> obs128 float32
        - key_fn(env) -> bytes key for caching
        - mask_fn(env) -> np.array(9) of 0/1 legal
        - step_fn(env, action) -> (reward_white, done)
        - terminal_fn(env) -> bool (optional, can be done via step)
        """
        root_key = key_fn(root_env)
        root_node = self._get_node(root_key)
        root_mask = mask_fn(root_env)

        # if root not expanded, expand with dirichlet
        if not root_node.expanded:
            obs = canonical_obs_fn(root_env)
            _ = self._expand_and_eval(root_node, obs, root_mask, add_dirichlet=add_dirichlet)

        for _ in range(sims):
            env = root_env.__deepcopy__({}) if hasattr(root_env, "__deepcopy__") else None
            # fallback: caller should deepcopy; we handle with numpy/pickle? keep simple:
            import copy
            env = copy.deepcopy(root_env)

            path: List[Tuple[Node, int]] = []
            invert = 1.0  # value sign flips each ply (because player switches)

            while True:
                key = key_fn(env)
                node = self._get_node(key)
                mask = mask_fn(env)

                if not node.expanded:
                    obs = canonical_obs_fn(env)
                    v = self._expand_and_eval(node, obs, mask, add_dirichlet=False)
                    # backprop leaf value
                    leaf_v = invert * v
                    for n, a in reversed(path):
                        e = n.edges[a]
                        e.N += 1
                        e.W += leaf_v
                        leaf_v = -leaf_v
                    break

                # selection
                a = self._puct_select(node, mask)
                path.append((node, a))

                # step
                reward_white, done = step_fn(env, a)
                if done:
                    # convert terminal reward (white perspective) to current-player perspective at this node
                    # canonical_obs_fn always makes "to-move" perspective, so from leaf to-move at terminal we need z in [-1,1]
                    # If white wins reward_white=+1. If current player at terminal (to move after step) is ???.
                    # Easiest: terminal value from perspective of player-to-move at *current node* before stepping:
                    # invert tracks flips from root; after stepping, turn flips, so terminal value for current node is invert * z_current
                    # We compute z_current for player who JUST moved? AlphaZero usually defines z for player-to-move at state.
                    # We'll define terminal value for next-to-move state: if reward_white=+1 and next-to-move is white then z=+1 else -1.
                    # Here we already flipped turn by env.step; so we can compute z_next_to_move and use invert * z_next_to_move.
                    from .az_togyz_adapter import player_to_move  # local import to avoid circular
                    p = player_to_move(env)  # 0 white,1 black
                    z_next = float(reward_white) if p == 0 else float(-reward_white)
                    leaf_v = invert * z_next
                    for n, a2 in reversed(path):
                        e = n.edges[a2]
                        e.N += 1
                        e.W += leaf_v
                        leaf_v = -leaf_v
                    break

                invert = -invert

        # build pi from visits at root
        counts = np.zeros(9, dtype=np.float32)
        for a, e in root_node.edges.items():
            counts[a] = float(e.N)
        if counts.sum() <= 0:
            # fallback uniform legal
            legal = np.where(root_mask > 0)[0]
            if len(legal) > 0:
                counts[legal] = 1.0
        pi = counts / counts.sum()
        return pi
