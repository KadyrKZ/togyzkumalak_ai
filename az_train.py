import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from az_model import AZNet

def load_npz_files(pattern: str):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files match: {pattern}")
    obs_list, pi_list, z_list = [], [], []
    for f in files:
        data = np.load(f)
        obs_list.append(data["obs"])
        pi_list.append(data["pi"])
        z_list.append(data["z"])
    obs = np.concatenate(obs_list, axis=0)
    pi = np.concatenate(pi_list, axis=0)
    z = np.concatenate(z_list, axis=0)
    return obs, pi, z

def train(
    data_glob: str = "replay/*.npz",
    out_model: str = "checkpoints/az_latest.pt",
    device: str = "cpu",
    batch_size: int = 256,
    epochs: int = 3,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
):
    os.makedirs(os.path.dirname(out_model), exist_ok=True)

    obs, pi, z = load_npz_files(data_glob)

    x = torch.tensor(obs, dtype=torch.float32)
    t_pi = torch.tensor(pi, dtype=torch.float32)
    t_z = torch.tensor(z, dtype=torch.float32)

    ds = TensorDataset(x, t_pi, t_z)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    net = AZNet().to(device)
    if os.path.exists(out_model):
        net.load_state_dict(torch.load(out_model, map_location=device))
        print("loaded", out_model)

    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)

    net.train()
    for ep in range(1, epochs + 1):
        total = 0.0
        for xb, pib, zb in dl:
            xb = xb.to(device)
            pib = pib.to(device)
            zb = zb.to(device)

            logits, v = net(xb)
            # policy loss: cross-entropy with target distribution
            logp = F.log_softmax(logits, dim=-1)
            pol_loss = -(pib * logp).sum(dim=-1).mean()
            val_loss = F.mse_loss(v, zb)
            loss = pol_loss + val_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()

            total += float(loss.item())

        print(f"epoch {ep}/{epochs}: loss={total/len(dl):.4f}")

    torch.save(net.state_dict(), out_model)
    print("saved model:", out_model)

if __name__ == "__main__":
    train()
