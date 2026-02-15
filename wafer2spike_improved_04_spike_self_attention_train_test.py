"""04) Improved + Spike-driven self-attention hybrid architecture."""
import torch
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli
from wafer2spike_improved_train_test import (
    build_dataloaders, training, FastSigmoidSurrogate,
    CurrentBasedGLIF, CurrentBasedGLIFWithDropout,
)


class Wafer2SpikeAttention(nn.Module):
    def __init__(self, numClasses, dropout_fc, spike_ts, device, params):
        super().__init__()
        self.device = device
        self.spike_ts = spike_ts
        self.dropout_fc = dropout_fc

        self.conv0 = CurrentBasedGLIF(nn.Conv2d(1, 64, 7, stride=1, bias=True), nn.GroupNorm(8, 64), FastSigmoidSurrogate, params)
        self.conv1 = CurrentBasedGLIF(nn.Conv2d(64, 64, 7, stride=2, bias=True), nn.GroupNorm(8, 64), FastSigmoidSurrogate, params)
        self.conv2 = CurrentBasedGLIF(nn.Conv2d(64, 64, 7, stride=2, bias=True), nn.GroupNorm(8, 64), FastSigmoidSurrogate, params)
        self.attn = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)

        self.fc = CurrentBasedGLIFWithDropout(nn.Linear(64 * 9, 256 * 9, bias=True), nn.GroupNorm(32, 256 * 9), FastSigmoidSurrogate, params)
        self.head = nn.Linear(256 * 9, numClasses)
        self.w_t = nn.Parameter(torch.ones(spike_ts, device=device) / spike_ts)

    def _z(self, b, d):
        return tuple(torch.zeros(b, *d, device=self.device) for _ in range(4))

    def forward(self, x):
        b = x.size(0)
        s0, s1, s2, sfc = self._z(b, (64, 30, 30)), self._z(b, (64, 12, 12)), self._z(b, (64, 3, 3)), self._z(b, (256 * 9,))
        mask_fc = Bernoulli(torch.full((b, 256 * 9), 1 - self.dropout_fc, device=self.device)).sample() / (1 - self.dropout_fc)
        outs = []
        for t in range(self.spike_ts):
            h0, s0 = self.conv0(x, s0)
            h1, s1 = self.conv1(h0, s1)
            h2, s2 = self.conv2(h1, s2)
            tok = h2.flatten(2).transpose(1, 2)
            tok2, _ = self.attn(tok, tok, tok)
            feat = tok2.transpose(1, 2).reshape(b, -1)
            hf, sfc = self.fc(feat, sfc, mask_fc, self.training)
            outs.append(self.head(hf) * self.w_t[t])
        return torch.stack(outs).sum(0)


class CurrentBasedSNNAttn(nn.Module):
    def __init__(self, numClasses, dropout_fc, spike_ts, device, params):
        super().__init__()
        self.net = Wafer2SpikeAttention(numClasses, dropout_fc, spike_ts, device, params)

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    dataloaders = build_dataloaders()
    training(
        network=CurrentBasedSNNAttn,
        params=[0.05, 0.10, 0.08, 5.0],
        dataloaders=dataloaders,
        spike_ts=10,
        batch_size=256,
        epochs=20,
        lr=1e-4,
        dropout_fc=0.20,
    )
