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
    """Get base env (unwrapped)."""
    try:
        return env.unwrapped
    except Exception:
        return env


# ----------------------------
# Exact evaluation from this repo internals (VERY IMPORTANT)
# ----------------------------
def get_board(env) -> Optional[Any]:
    base = unwrap_env(env)
    if hasattr(base, "board"):
        return getattr(base, "board")
    return None


def extract_kazans_exact(env) -> Optional[Tuple[int, int]]:
    """
    This repo: env.unwrapped.board.gamers['white'].kazan.score and ['black']...
    Returns (white_kazan, black_kazan) or None if structure differs.
    """
    board = get_board(env)
    if board is None:
        return None
    try:
        wk = int(board.gamers["white"].kazan.score)
        bk = int(board.gamers["black"].kazan.score)
        return wk, bk
    except Exception:
        return None


def extract_tuzdyk_exact(env) -> Optional[Tuple[Optional[int], Optional[int]]]:
    """
    Returns (white_tuzdyk_index, black_tuzdyk_index) as 0..8 or None if no tuzdyk.
    We infer tuzdyk by scanning opponent home:
      - If black has a tuzdyk, it is on white home (white otau has .tuzduk True)
      - If white has a tuzdyk, it is on black home (black otau has .tuzduk True)
    """
    board = get_board(env)
    if board is None:
        return None
    try:
        white_home = board.gamers["white"].home  # dict 0..8 -> Otau
        black_home = board.gamers["black"].home

        # black_tuzdyk lives on white side
        black_tuz = None
        for i, o in white_home.items():
            if getattr(o, "tuzduk", False):
                black_tuz = i
                break

        # white_tuzdyk lives on black side
        white_tuz = None
        for i, o in black_home.items():
            if getattr(o, "tuzduk", False):
                white_tuz = i
                break

        return white_tuz, black_tuz
    except Exception:
        return None


def mobility(env) -> int:
    """How many legal moves current player has."""
    return len(legal_actions(env))


def state_key(env, step_i: int) -> bytes:
    """
    Hashable key for caching evaluation.
    We use board.state() (128,1) bytes + current player parity.
    """
    board = get_board(env)
    if board is None:
        # fallback (rare)
        return (str(id(env)) + "_" + str(step_i % 2)).encode()

    try:
        s = board.state()
        # include player-to-move parity
        return s.tobytes() + bytes([step_i % 2])
    except Exception:
        return (str(id(env)) + "_" + str(step_i % 2)).encode()


def heuristic_value(env, step_i: int) -> float:
    """
    Strong heuristic in [-1,1] from WHITE perspective.
    Uses:
      - kazan difference (main signal)
      - tuzdyk bonus (who has it)
      - mobility (reduce opponent moves)
    """
    kb = extract_kazans_exact(env)
    if kb is None:
        return 0.0
    wk, bk = kb
    v = (wk - bk) / 82.0  # normalize

    # Tuzdyk bonus (small but important)
    tz = extract_tuzdyk_exact(env)
    if tz is not None:
        white_tuz, black_tuz = tz
        if white_tuz is not None:
            v += 0.10
        if black_tuz is not None:
            v -= 0.10

    # Mobility: prefer positions where opponent has fewer moves
    # We can approximate by looking at current player's legal count:
    # If it's WHITE to move (step even): more mobility for WHITE is good.
    cur_player = current_player_by_step(step_i)
    m = mobility(env)
    if cur_player == 0:
        v += (m - 4.5) * 0.01
    else:
        v -= (m - 4.5) * 0.01

    # clip
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


def uct_select(node: Node, c: float = 1.10) -> Node:
    """
    UCT selection. Lower c => greedier, often stronger with good eval.
    """
    best_score = -1e18
    best_child = None
    logN = math.log(node.visits + 1)

    for _a, ch in node.children.items():
        exploit = ch.q()
        explore = c * math.sqrt(logN / (ch.visits + 1e-9))
        score = exploit + explore
        if score > best_score:
            best_score = score
            best_child = ch
    return best_child


def expand(node: Node) -> Node:
    # choose a random untried to diversify expansion (slightly better)
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
# Strong rollout (biased + eval at cutoff) + caching
# ----------------------------
def best_one_step_action(env, step_i: int, acts: List[int]) -> int:
    """
    Stronger than random: try each action, pick the one with best immediate outcome.
    - If terminal -> prioritize reward
    - Else -> prioritize heuristic_value after move AND reduce opponent mobility
    """
    best_a = None
    best_score = -1e18

    for a in acts:
        e2 = copy.deepcopy(env)
        _obs2, r2, d2, _info2 = e2.step(a)

        if d2:
            score = float(r2) * 1000.0
        else:
            # After our move, opponent to move:
            score = heuristic_value(e2, step_i + 1) * 50.0

            # also prefer reducing opponent mobility
            try:
                opp_m = mobility(e2)
                score += (9 - opp_m) * 0.6
            except Exception:
                pass

        if score > best_score:
            best_score = score
            best_a = a

    return best_a if best_a is not None else random.choice(acts)


def rollout(env, step_i: int, eval_cache: Dict[bytes, float],
            max_depth: int = 2500, epsilon: float = 0.06) -> float:
    """
    Rollout returns value from WHITE perspective.
    - Terminal => env reward (+1/-1/0)
    - Depth cutoff => heuristic_value
    Uses eval_cache for fast repeated evaluations.
    """
    done = False
    last_reward = 0.0
    depth = 0

    while not done and depth < max_depth:
        acts = legal_actions(env)
        if not acts:
            break

        # epsilon-greedy: mostly best_one_step_action, rarely random for exploration
        if random.random() < epsilon:
            a = random.choice(acts)
        else:
            a = best_one_step_action(env, step_i, acts)

        _obs, r, done, _info = env.step(a)
        last_reward = r
        step_i += 1
        depth += 1

    if done:
        return float(last_reward)

    # depth cutoff -> cached eval
    k = state_key(env, step_i)
    if k in eval_cache:
        return eval_cache[k]

    v = heuristic_value(env, step_i)
    eval_cache[k] = v
    return v


# ----------------------------
# Time-based MCTS (very strong)
# ----------------------------
def mcts_search_time(
    root_env,
    root_step_i: int,
    time_limit_s: float = 8.0,
    c: float = 1.10,
) -> int:
    """
    Run MCTS for time_limit_s seconds and return best action.
    Uses:
      - strong heuristic (kazans + tuzdyk + mobility)
      - strong rollout (biased 1-step + caching)
    """
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
        value = rollout(env_roll, node.step_i, eval_cache=eval_cache)

        # 4) Backprop
        backprop(node, value)

    if not root.children:
        acts = legal_actions(root_env)
        return random.choice(acts)

    # Robust choice: pick among top visits then best Q
    items = list(root.children.items())
    items.sort(key=lambda kv: kv[1].visits, reverse=True)
    topK = items[: min(5, len(items))]
    best_a, _best_ch = max(topK, key=lambda kv: kv[1].q())
    return best_a


# ----------------------------
# Main: Human vs Very Strong MCTS
# ----------------------------
def main():
    env = gym.make("Togyzkumalak-v0")
    env.reset()

    # 0=WHITE, 1=BLACK
    # Чтобы бот давил с первого хода — играй за BLACK
    human_player = 1

    # СИЛА:
    # 4-6 сек — очень сильно
    # 8-12 сек — почти “нечестно”
    TIME_PER_MOVE = 16.0

    # UCT: ниже => более жадный (с хорошей оценкой обычно сильнее)
    UCT_C = 0.9

    print("\n=== Human vs VERY STRONG MCTS AI ===")
    print(f"Ты играешь за {'BLACK' if human_player==1 else 'WHITE'}")
    print(f"AI thinking time per move = {TIME_PER_MOVE:.1f}s | UCT c = {UCT_C}\n")
    print("Подсказка: вводи номер лунки 1..9\n")

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
            a = mcts_search_time(env, step_i, time_limit_s=TIME_PER_MOVE, c=UCT_C)
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