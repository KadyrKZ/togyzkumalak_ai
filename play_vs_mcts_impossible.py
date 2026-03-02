# play_vs_mcts.py
# IMPOSSIBLE MODE: Human vs (very) strong MCTS bot for gym-togyzkumalak
#
# Run:
#   cd ~/gym-togyzkumalak
#   source .venv/bin/activate
#   python play_vs_mcts.py
#
# Controls:
#   - Enter pit number 1..9 on your turn
#
# Strength knobs (near bottom):
#   TIME_PER_MOVE, UCT_C, ROLLOUT_MAX_DEPTH
#
# This bot is NOT a trained neural net — it is a heavy search bot:
# - Time-limited MCTS
# - Deterministic greedy rollouts (epsilon = 0)
# - Position evaluation from real game state:
#   kazan diff + tuzdyk ownership + tuzdyk threat + mobility
# - Transposition (evaluation) cache keyed by board.state() bytes
#
# Note: Because deepcopy() is used, higher time/depth costs CPU/RAM.
# On MacBook Air: start with TIME_PER_MOVE=8..12, then raise.

import copy
import math
import random
import time
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple, Any

import gym
import gym_togyzkumalak


# ----------------------------
# Helpers
# ----------------------------
def legal_actions(env) -> List[int]:
    """Return legal actions 0..8 using env.available_action() mask."""
    mask = env.available_action()
    return [i for i, m in enumerate(mask) if m == 1]


def current_player_by_step(step_i: int) -> int:
    """0 = WHITE on even steps, 1 = BLACK on odd steps (env alternates turns)."""
    return 0 if (step_i % 2 == 0) else 1


def ask_human_move(legal: List[int]) -> int:
    while True:
        s = input(f"Твой ход. Выбери лунку 1-9 (легальные: {[x+1 for x in legal]}): ").strip()
        if not s.isdigit():
            print("Введи число 1..9")
            continue
        a = int(s) - 1
        if a in legal:
            return a
        print("Нелегальный ход. Попробуй снова.")


def unwrap_env(env):
    try:
        return env.unwrapped
    except Exception:
        return env


def get_board(env) -> Any:
    base = unwrap_env(env)
    if hasattr(base, "board"):
        return base.board
    raise AttributeError("Can't find env.unwrapped.board (board not found).")


# ----------------------------
# Exact signals from THIS repo
# ----------------------------
def extract_kazans(env) -> Tuple[int, int]:
    """Return (white_kazan, black_kazan) using env.unwrapped.board.gamers[...]"""
    board = get_board(env)
    wk = int(board.gamers["white"].kazan.score)
    bk = int(board.gamers["black"].kazan.score)
    return wk, bk


def extract_tuzdyk_indices(env) -> Tuple[Optional[int], Optional[int]]:
    """
    Return (white_tuzdyk_on_black_side, black_tuzdyk_on_white_side) as 0..8 or None.
    In this repo:
      - If WHITE has a tuzdyk, it is marked on BLACK's home (black_home[*].tuzduk==True)
      - If BLACK has a tuzdyk, it is marked on WHITE's home (white_home[*].tuzduk==True)
    """
    board = get_board(env)
    white_home = board.gamers["white"].home  # dict 0..8 -> Otau
    black_home = board.gamers["black"].home

    white_tuz = None
    for i, o in black_home.items():
        if getattr(o, "tuzduk", False):
            white_tuz = i
            break

    black_tuz = None
    for i, o in white_home.items():
        if getattr(o, "tuzduk", False):
            black_tuz = i
            break

    return white_tuz, black_tuz


def count_tuzdyk_threats(env) -> Tuple[int, int]:
    """
    Threat = opponent-side pits with exactly 2 stones (where a '2->3' landing could create tuzdyk),
    excluding pit 9 (index 8) and excluding already-tuzdyk pits.
    Returns (white_threats_on_black_side, black_threats_on_white_side)
    """
    board = get_board(env)
    white_home = board.gamers["white"].home
    black_home = board.gamers["black"].home

    # White can create tuzdyk only on BLACK side (black_home), pits 0..7
    w_threat = 0
    for i, o in black_home.items():
        if i == 8:
            continue
        if getattr(o, "tuzduk", False):
            continue
        if int(getattr(o, "kumalaks", 0)) == 2:
            w_threat += 1

    # Black can create tuzdyk only on WHITE side (white_home), pits 0..7
    b_threat = 0
    for i, o in white_home.items():
        if i == 8:
            continue
        if getattr(o, "tuzduk", False):
            continue
        if int(getattr(o, "kumalaks", 0)) == 2:
            b_threat += 1

    return w_threat, b_threat


def mobility(env) -> int:
    return len(legal_actions(env))


def state_key(env, step_i: int) -> bytes:
    """
    Robust hashable key for caching.
    Tries, in order:
      1) board.state()        -> numpy array (best)
      2) board.observation()  -> numpy array (also good)
      3) pickle(board)        -> fallback
      4) repr(board.__dict__) -> last resort
    """
    import pickle

    board = get_board(env)
    if board is None:
        return (str(id(env)) + "_" + str(step_i % 2)).encode()

    # 1) board.state()
    if hasattr(board, "state"):
        try:
            s = board.state()
            # include side-to-move parity
            return s.tobytes() + bytes([step_i % 2])
        except Exception:
            pass

    # 2) board.observation()
    if hasattr(board, "observation"):
        try:
            s = board.observation()
            return s.tobytes() + bytes([step_i % 2])
        except Exception:
            pass

    # 3) pickle fallback
    try:
        blob = pickle.dumps(board, protocol=4)
        return blob + bytes([step_i % 2])
    except Exception:
        pass

    # 4) last resort
    try:
        return (repr(board.__dict__) + "|" + str(step_i % 2)).encode()
    except Exception:
        return (str(id(board)) + "_" + str(step_i % 2)).encode()


# ----------------------------
# Heuristic evaluation (WHITE perspective)
# ----------------------------
def heuristic_value(env, step_i: int) -> float:
    """
    Value in [-1,1] from WHITE perspective.
    Strong signals:
      - kazan difference (main)
      - tuzdyk ownership (big)
      - tuzdyk threats (medium)
      - mobility (small)
    """
    wk, bk = extract_kazans(env)
    v = (wk - bk) / 82.0  # normalize by winning threshold

    white_tuz, black_tuz = extract_tuzdyk_indices(env)
    # Tuzdyk ownership bonus (big)
    if white_tuz is not None:
        v += 0.25
    if black_tuz is not None:
        v -= 0.25

    # Threats (medium)
    w_th, b_th = count_tuzdyk_threats(env)
    v += 0.04 * w_th
    v -= 0.04 * b_th

    # Mobility (small): prefer current-player mobility
    cur = current_player_by_step(step_i)
    m = mobility(env)
    if cur == 0:
        v += (m - 4.5) * 0.01
    else:
        v -= (m - 4.5) * 0.01

    # Clip
    if v > 1.0:
        v = 1.0
    if v < -1.0:
        v = -1.0
    return float(v)


# ----------------------------
# MCTS Node
# ----------------------------
@dataclass
class Node:
    env: Any
    step_i: int
    parent: Optional["Node"] = None
    parent_action: Optional[int] = None
    children: Dict[int, "Node"] = None
    visits: int = 0
    value_sum: float = 0.0
    untried: List[int] = None

    def __post_init__(self):
        self.children = {}
        self.untried = legal_actions(self.env)

    def q(self) -> float:
        return self.value_sum / self.visits if self.visits > 0 else 0.0


def uct_select(node: Node, c: float) -> Node:
    best_score = -1e18
    best_child = None
    logN = math.log(node.visits + 1.0)

    for _a, ch in node.children.items():
        exploit = ch.q()
        explore = c * math.sqrt(logN / (ch.visits + 1e-9))
        score = exploit + explore
        if score > best_score:
            best_score = score
            best_child = ch

    return best_child


def expand(node: Node) -> Node:
    # Randomize expansion choice a little to avoid deterministic bias.
    idx = random.randrange(len(node.untried))
    a = node.untried.pop(idx)
    env2 = copy.deepcopy(node.env)
    env2.step(a)
    child = Node(env=env2, step_i=node.step_i + 1, parent=node, parent_action=a)
    node.children[a] = child
    return child


def backprop(node: Node, value: float):
    cur = node
    while cur is not None:
        cur.visits += 1
        cur.value_sum += value
        cur = cur.parent


# ----------------------------
# Deterministic greedy rollout
# ----------------------------
def best_one_step_action(env, step_i: int, acts: List[int], eval_cache: Dict[bytes, float]) -> int:
    """
    Choose the best action by 1-ply lookahead using:
      - terminal reward (huge)
      - heuristic after the move (big)
      - opponent mobility after the move (medium)
    """
    best_a = acts[0]
    best_score = -1e18

    for a in acts:
        e2 = copy.deepcopy(env)
        _obs2, r2, d2, _info2 = e2.step(a)

        if d2:
            score = float(r2) * 10000.0
        else:
            # After move, step_i+1
            k = state_key(e2, step_i + 1)
            if k in eval_cache:
                hv = eval_cache[k]
            else:
                hv = heuristic_value(e2, step_i + 1)
                eval_cache[k] = hv

            score = hv * 200.0

            # reduce opponent mobility
            try:
                score += (9 - mobility(e2)) * 1.0
            except Exception:
                pass

        if score > best_score:
            best_score = score
            best_a = a

    return best_a


def rollout(env, step_i: int, eval_cache: Dict[bytes, float],
            max_depth: int) -> float:
    """
    Rollout policy: always greedy one-step action (epsilon = 0).
    Return:
      - terminal reward if finished
      - else heuristic_value at cutoff
    """
    done = False
    last_reward = 0.0
    depth = 0

    while not done and depth < max_depth:
        acts = legal_actions(env)
        if not acts:
            break

        a = best_one_step_action(env, step_i, acts, eval_cache)
        _obs, r, done, _info = env.step(a)
        last_reward = r
        step_i += 1
        depth += 1

    if done:
        return float(last_reward)

    k = state_key(env, step_i)
    if k in eval_cache:
        return float(eval_cache[k])

    v = heuristic_value(env, step_i)
    eval_cache[k] = v
    return float(v)


# ----------------------------
# Time-based MCTS (IMPOSSIBLE MODE)
# ----------------------------
def mcts_search_time(
    root_env,
    root_step_i: int,
    time_limit_s: float,
    c: float,
    rollout_max_depth: int,
) -> int:
    root = Node(env=copy.deepcopy(root_env), step_i=root_step_i)

    if len(root.untried) == 1:
        return root.untried[0]

    eval_cache: Dict[bytes, float] = {}

    t0 = time.time()
    while time.time() - t0 < time_limit_s:
        node = root

        # 1) Selection
        while not node.untried and node.children:
            node = uct_select(node, c=c)

        # 2) Expansion
        if node.untried:
            node = expand(node)

        # 3) Simulation
        env_roll = copy.deepcopy(node.env)
        value = rollout(env_roll, node.step_i, eval_cache, max_depth=rollout_max_depth)

        # 4) Backprop
        backprop(node, value)

    # Choose robustly: top by visits, then best q() among top.
    items = list(root.children.items())
    items.sort(key=lambda kv: kv[1].visits, reverse=True)
    top = items[: min(7, len(items))]
    best_a, _best_ch = max(top, key=lambda kv: kv[1].q())
    return best_a


# ----------------------------
# Main
# ----------------------------
def main():
    env = gym.make("Togyzkumalak-v0")
    env.reset()

    # Human side: 0=WHITE, 1=BLACK
    # To make it harder, play BLACK so AI starts.
    human_player = 1

    # -------- IMPOSSIBLE MODE knobs --------
    TIME_PER_MOVE = 25.0      # 10 strong, 15 very strong, 25 "almost unfair", 40 insane
    UCT_C = 0.90              # lower = greedier; with strong eval, this is often stronger
    ROLLOUT_MAX_DEPTH = 2500  # deeper rollouts => stronger (but slower)
    # --------------------------------------

    print("\n=== Human vs IMPOSSIBLE MCTS AI ===")
    print(f"Ты играешь за {'BLACK' if human_player==1 else 'WHITE'}")
    print(f"AI time per move = {TIME_PER_MOVE:.1f}s | UCT c = {UCT_C} | rollout depth = {ROLLOUT_MAX_DEPTH}")
    print("Ввод: номер лунки 1..9\n")

    step_i = 0
    done = False
    env.render()

    while not done and step_i < 20000:
        player = current_player_by_step(step_i)
        acts = legal_actions(env)
        if not acts:
            print("Нет легальных ходов. Конец.")
            break

        if player == human_player:
            a = ask_human_move(acts)
            print(f"\nТы: {a+1}\n")
        else:
            print("\nAI думает...\n")
            a = mcts_search_time(
                env,
                step_i,
                time_limit_s=TIME_PER_MOVE,
                c=UCT_C,
                rollout_max_depth=ROLLOUT_MAX_DEPTH,
            )
            print(f"AI: {a+1}\n")

        _obs, r, done, _info = env.step(a)
        env.render()

        if done:
            print("\n=== Игра завершена ===")
            print("Reward:", r)
            if r > 0:
                print("WHITE выиграл.")
            elif r < 0:
                print("BLACK выиграл.")
            else:
                print("Ничья.")
            break

        step_i += 1

    env.close()


if __name__ == "__main__":
    main()
