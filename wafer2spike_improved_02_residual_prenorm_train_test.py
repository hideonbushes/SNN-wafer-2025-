"""02) Improved + Spiking Residual/Pre-Norm style architecture."""
import torch
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli
from wafer2spike_improved_train_test import (
    build_dataloaders, training, FastSigmoidSurrogate,
    CurrentBasedGLIF, CurrentBasedGLIFWithDropout,
)


class Wafer2SpikeResidual(nn.Module):
    def __init__(self, numClasses, dropout_fc, spike_ts, device, params):
        super().__init__()
        self.device = device
        self.spike_ts = spike_ts
        self.dropout_fc = dropout_fc
        self.scdecay, self.vdecay, self.vth, self.alpha = params

        self.stem = CurrentBasedGLIF(nn.Conv2d(1, 64, 7, stride=1, bias=True), nn.GroupNorm(8, 64), FastSigmoidSurrogate, params)
        self.block1 = CurrentBasedGLIF(nn.Conv2d(64, 64, 7, stride=2, bias=True), nn.GroupNorm(8, 64), FastSigmoidSurrogate, params)
        self.block2 = CurrentBasedGLIF(nn.Conv2d(64, 64, 7, stride=2, bias=True), nn.GroupNorm(8, 64), FastSigmoidSurrogate, params)

        self.skip1 = nn.Conv2d(64, 64, 1, stride=2, bias=False)
        self.skip2 = nn.Conv2d(64, 64, 1, stride=2, bias=False)

        self.fc = CurrentBasedGLIFWithDropout(nn.Linear(64 * 9, 256 * 9, bias=True), nn.GroupNorm(32, 256 * 9), FastSigmoidSurrogate, params)
        self.head = nn.Linear(256 * 9, numClasses)
        self.w_t = nn.Parameter(torch.ones(spike_ts, device=device) / spike_ts)

    def _zero_state(self, b, d):
        return tuple(torch.zeros(b, *d, device=self.device) for _ in range(4))

    def forward(self, x):
        b = x.size(0)
        s_stem = self._zero_state(b, (64, 30, 30))
        s_b1 = self._zero_state(b, (64, 12, 12))
        s_b2 = self._zero_state(b, (64, 3, 3))
        s_fc = self._zero_state(b, (256 * 9,))

        mask_fc = Bernoulli(torch.full((b, 256 * 9), 1 - self.dropout_fc, device=self.device)).sample() / (1 - self.dropout_fc)
        outs = []
        for t in range(self.spike_ts):
            h0, s_stem = self.stem(x, s_stem)
            h1, s_b1 = self.block1(h0, s_b1)
            h1 = h1 + self.skip1(h0)
            h2, s_b2 = self.block2(h1, s_b2)
            h2 = h2 + self.skip2(h1)
            flat = h2.view(b, -1)
            hf, s_fc = self.fc(flat, s_fc, mask_fc, self.training)
            outs.append(self.head(hf) * self.w_t[t])
        return torch.stack(outs).sum(0)


class CurrentBasedSNNResidual(nn.Module):
    def __init__(self, numClasses, dropout_fc, spike_ts, device, params):
        super().__init__()
        self.net = Wafer2SpikeResidual(numClasses, dropout_fc, spike_ts, device, params)

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    dataloaders = build_dataloaders()
    training(
        network=CurrentBasedSNNResidual,
        params=[0.05, 0.10, 0.08, 5.0],
        dataloaders=dataloaders,
        spike_ts=10,
        batch_size=256,
        epochs=20,
        lr=1e-4,
        dropout_fc=0.20,
    )
