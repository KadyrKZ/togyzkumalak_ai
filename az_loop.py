"""
End-to-end AlphaZero loop (minimal):

1) Self-play to generate data into replay/*.npz
2) Train network on replay data
Repeat.

Run examples:
  python az_loop.py --iters 20 --games 20 --sims 300 --device cpu

You can start small; strength grows with more iterations + sims.
"""
import os
import argparse
import glob
import time

from az_selfplay import selfplay_batch
from az_train import train

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--replay_dir", type=str, default="replay")
    ap.add_argument("--ckpt", type=str, default="checkpoints/az_latest.pt")
    args = ap.parse_args()

    os.makedirs(args.replay_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.ckpt), exist_ok=True)

    for it in range(1, args.iters + 1):
        out_npz = os.path.join(args.replay_dir, f"it{it:03d}.npz")
        print(f"\n=== Iteration {it}/{args.iters}: self-play ===")
        selfplay_batch(net_path=args.ckpt, out_path=out_npz, games=args.games, sims=args.sims, device=args.device)

        print(f"\n=== Iteration {it}/{args.iters}: train ===")
        train(data_glob=os.path.join(args.replay_dir, "*.npz"), out_model=args.ckpt, device=args.device, epochs=2)

    print("\nDone. Model:", args.ckpt)

if __name__ == "__main__":
    main()
