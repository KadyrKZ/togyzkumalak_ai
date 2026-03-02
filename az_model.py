import torch
import torch.nn as nn
import torch.nn.functional as F

class AZNet(nn.Module):
    """
    AlphaZero-style network for Togyzkumalak.
    Input: 128-dim float vector (canonicalized: player-to-move features first; last two bits fixed to [1,0])
    Outputs:
      - policy logits over 9 actions
      - value in [-1, 1]
    """
    def __init__(self, in_dim: int = 128, hidden: int = 256, blocks: int = 4):
        super().__init__()
        self.fc_in = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList([nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        ) for _ in range(blocks)])
        self.fc_pol = nn.Linear(hidden, 9)
        self.fc_val1 = nn.Linear(hidden, hidden // 2)
        self.fc_val2 = nn.Linear(hidden // 2, 1)

    def forward(self, x):
        # x: (B, 128)
        h = F.relu(self.fc_in(x))
        for b in self.blocks:
            r = h
            h = b(h)
            h = F.relu(h + r)  # residual-ish
        policy_logits = self.fc_pol(h)
        v = torch.tanh(self.fc_val2(F.relu(self.fc_val1(h))))
        return policy_logits, v.squeeze(-1)
