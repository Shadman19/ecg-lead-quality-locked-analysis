"""Models matching the manuscript's Table II, plus baselines and a lead-preserving head."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def block(cin, cout, stride, k=7):
    return nn.Sequential(nn.Conv1d(cin, cout, k, stride, k // 2, bias=False), nn.BatchNorm1d(cout), nn.GELU())


class Encoder(nn.Module):
    """6 Conv1D+BN+GELU blocks: 12->64(/2)->64->128(/2)->128->256(/2)->256, dropout 0.15."""

    def __init__(self, cin=12, p=0.15):
        super().__init__()
        self.net = nn.Sequential(block(cin, 64, 2), block(64, 64, 1), block(64, 128, 2),
                                 block(128, 128, 1), block(128, 256, 2), block(256, 256, 1), nn.Dropout(p))

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose1d(256, 128, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose1d(128, 64, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose1d(64, 64, 4, 2, 1), nn.GELU(),
            nn.Conv1d(64, 12, 7, 1, 3))

    def forward(self, h, t):
        y = self.net(h)
        return F.interpolate(y, size=t, mode="linear", align_corners=False) if y.shape[-1] != t else y


class LeadwiseHead(nn.Module):
    """Lead-preserving per-lead head: grouped convs keep one feature stream per lead,
    so lead identity is explicit (contingency fix E7)."""

    def __init__(self, width=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(12, 12 * width, 7, 2, 3, groups=12), nn.GELU(),
            nn.Conv1d(12 * width, 12 * width, 7, 2, 3, groups=12), nn.GELU(),
            nn.Conv1d(12 * width, 12 * width, 7, 2, 3, groups=12), nn.GELU())
        self.width = width
        self.out = nn.Linear(width * 2, 1)

    def features(self, x):
        h = self.net(x)                                   # [B, 12*w, T']
        b, _, t = h.shape
        h = h.view(b, 12, self.width, t)
        return torch.cat([h.mean(-1), h.std(-1)], dim=-1)  # [B, 12, 2w]

    def forward(self, x):
        return self.out(self.features(x)).squeeze(-1)      # [B, 12] logits


class MultiTask(nn.Module):
    def __init__(self, head="cnn", tasks=("diag", "lead", "rec"), lead_head="pooled"):
        super().__init__()
        self.tasks = set(tasks)
        self.enc = Encoder()
        self.head_type = head
        if head == "cnn":
            din = 256 + (12 * 32 if lead_head == "fused" else 0)
            self.diag = nn.Sequential(nn.Linear(din, 128), nn.GELU(), nn.Linear(128, 5))
        else:
            layer = nn.TransformerEncoderLayer(256, 4, 512, 0.1, batch_first=True, activation="gelu")
            self.tx = nn.TransformerEncoder(layer, 3)
            self.diag = nn.Linear(256, 5)
        self.lead_head_type = lead_head
        if "lead" in self.tasks:
            if lead_head == "pooled":
                self.lead = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 12))
            else:                                          # "leadwise" or "fused"
                self.lead = LeadwiseHead()
        if "rec" in self.tasks:
            self.dec = Decoder()

    def forward(self, x):
        h = self.enc(x)
        z = h.mean(-1)
        out = {}
        if self.lead_head_type == "fused":
            # Quality-gated fusion (added after the RE-SCOPE decision): the lead-preserving head's
            # per-lead features are weighted by the predicted intact probability and passed to the
            # diagnosis head together with the pooled encoder features, so lead-quality supervision
            # can shape the diagnostic input directly.
            f = self.lead.features(x)                              # [B, 12, 32]
            lead_logit = self.lead.out(f).squeeze(-1)              # [B, 12]
            g = torch.sigmoid(lead_logit).unsqueeze(-1)            # [B, 12, 1]
            out["diag"] = self.diag(torch.cat([z, (f * g).flatten(1)], dim=1))
            out["lead"] = lead_logit
            if "rec" in self.tasks:
                out["rec"] = self.dec(h, x.shape[-1])
            return out
        if self.head_type == "cnn":
            out["diag"] = self.diag(z)
        else:
            out["diag"] = self.diag(self.tx(h.transpose(1, 2)).mean(1))
        if "lead" in self.tasks:
            out["lead"] = self.lead(z) if self.lead_head_type == "pooled" else self.lead(x)
        if "rec" in self.tasks:
            out["rec"] = self.dec(h, x.shape[-1])
        return out


class ResBlockWang(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.c = nn.Sequential(
            nn.Conv1d(cin, cout, 7, padding=3, bias=False), nn.BatchNorm1d(cout), nn.ReLU(),
            nn.Conv1d(cout, cout, 5, padding=2, bias=False), nn.BatchNorm1d(cout), nn.ReLU(),
            nn.Conv1d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm1d(cout))
        self.s = nn.Sequential(nn.Conv1d(cin, cout, 1, bias=False), nn.BatchNorm1d(cout))

    def forward(self, x):
        return F.relu(self.c(x) + self.s(x))


class ResNet1DWang(nn.Module):
    """Wang et al. (2017) style: 3 residual blocks (64,128,128), GAP, linear."""

    def __init__(self):
        super().__init__()
        self.b = nn.Sequential(ResBlockWang(12, 64), ResBlockWang(64, 128), ResBlockWang(128, 128))
        self.fc = nn.Linear(128, 5)

    def forward(self, x):
        return {"diag": self.fc(self.b(x).mean(-1))}


class InceptionBlock(nn.Module):
    def __init__(self, cin, nf=32):
        super().__init__()
        self.bott = nn.Conv1d(cin, nf, 1, bias=False)
        self.convs = nn.ModuleList([nn.Conv1d(nf, nf, k, padding=k // 2, bias=False) for k in (39, 19, 9)])
        self.pool = nn.Sequential(nn.MaxPool1d(3, 1, 1), nn.Conv1d(cin, nf, 1, bias=False))
        self.bn = nn.BatchNorm1d(4 * nf)

    def forward(self, x):
        b = self.bott(x)
        return F.relu(self.bn(torch.cat([c(b) for c in self.convs] + [self.pool(x)], 1)))


class InceptionTimeLite(nn.Module):
    def __init__(self, depth=3):
        super().__init__()
        layers, cin = [], 12
        for _ in range(depth):
            layers.append(InceptionBlock(cin)); cin = 128
        self.b = nn.Sequential(*layers)
        self.fc = nn.Linear(128, 5)

    def forward(self, x):
        return {"diag": self.fc(self.b(x).mean(-1))}


def build(arch: str, tasks, lead_head="pooled"):
    if arch in ("cnn", "tx"):
        return MultiTask(arch, tasks, lead_head)
    if arch == "resnet_wang":
        return ResNet1DWang()
    if arch == "inception":
        return InceptionTimeLite()
    raise ValueError(arch)
