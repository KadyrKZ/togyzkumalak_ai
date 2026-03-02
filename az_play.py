"""
Play Human vs AlphaZero MCTS using the trained model.

Run:
  python az_play.py --ckpt checkpoints/az_latest.pt --sims 800 --device cpu
"""
import argparse
import numpy as np
import torch

from az_model import AZNet
from az_mcts import AlphaZeroMCTS
from az_togyz_adapter import make_env, canonical_obs, state_key, legal_mask, step_env, player_to_move

def ask_human(mask):
    legal = [i for i,m in enumerate(mask) if m > 0.5]
    while True:
        s = input(f"Твой ход (1-9), легальные: {[x+1 for x in legal]}: ").strip()
        if not s.isdigit():
            continue
        a = int(s) - 1
        if a in legal:
            return a

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="checkpoints/az_latest.pt")
    ap.add_argument("--sims", type=int, default=800)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--human", type=str, default="black", choices=["white","black"])
    args = ap.parse_args()

    net = AZNet()
    net.load_state_dict(torch.load(args.ckpt, map_location=args.device))
    net.to(args.device).eval()
    mcts = AlphaZeroMCTS(net, device=args.device, c_puct=1.25)

    env = make_env()
    env.reset()
    env.render()

    human_is_white = (args.human == "white")

    done = False
    last_r = 0.0
    ply = 0
    while not done and ply < 5000:
        p = player_to_move(env)  # 0 white 1 black
        mask = legal_mask(env)

        if (p == 0 and human_is_white) or (p == 1 and not human_is_white):
            a = ask_human(mask)
            print("YOU:", a+1)
        else:
            pi = mcts.search(
                env,
                canonical_obs_fn=canonical_obs,
                key_fn=state_key,
                mask_fn=legal_mask,
                step_fn=step_env,
                terminal_fn=lambda e: False,
                sims=args.sims,
                add_dirichlet=False,
            )
            # choose best
            a = int(np.argmax(pi))
            print("AI :", a+1)

        last_r, done = step_env(env, a)
        env.render()
        ply += 1

    print("Game ended. reward_white =", last_r)

if __name__ == "__main__":
    main()
