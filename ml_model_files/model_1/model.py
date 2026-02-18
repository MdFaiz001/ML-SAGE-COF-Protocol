import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, drop=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(drop)
        )

    def forward(self, x):
        return self.net(x)


class COFModel1(nn.Module):

    def __init__(self, dims, hidden=128):
        super().__init__()

        self.topo_enc = MLP(dims['topo'], hidden)
        self.node_enc = MLP(dims['node'], hidden)
        self.linker_enc = MLP(dims['linker'], hidden)
        self.base_enc = MLP(dims['base'], hidden)
        self.misc_enc = MLP(dims['misc'], hidden // 2)

        self.regressor = nn.Sequential(
            nn.Linear(hidden * 4 + hidden // 2, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 2)
        )

    def forward(self, batch):
        x = torch.cat([
            self.topo_enc(batch['topo']),
            self.node_enc(batch['node']),
            self.linker_enc(batch['linker']),
            self.base_enc(batch['base']),
            self.misc_enc(batch['misc'])
        ], dim=1)
        return self.regressor(x)

