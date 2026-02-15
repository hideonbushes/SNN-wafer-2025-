"""03) Original + Recurrent spatio-temporal Conv-GLIF blocks."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli
from wafer2spike_original_train_test import (
    build_dataloaders, training, PseudoGradSpike, PseudoGradSpikeWithDropout,
    CurrentBasedGLIF, CurrentBasedGLIFWithDropout,
)


class RecurrentSpikingBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride, params):
        super().__init__()
        pseudo = PseudoGradSpike.apply
        self.ff = CurrentBasedGLIF(nn.Conv2d(in_ch, out_ch, 7, stride=stride, bias=True), pseudo, params)
        self.rec = nn.Conv2d(out_ch, in_ch, 3, stride=1, padding=1, bias=False)

    def forward(self, x, state, h_prev):
        rec = self.rec(h_prev)
        if rec.shape[-2:] != x.shape[-2:]:
            rec = F.adaptive_avg_pool2d(rec, x.shape[-2:])
        x = x + rec
        spk, state = self.ff(x, state)
        return spk, state, spk


class Wafer2SpikeRecurrent(nn.Module):
    def __init__(self, numClasses, dropout_fc, spike_ts, device, params):
        super().__init__()
        self.device = device
        self.spike_ts = spike_ts
        self.dropout_fc = dropout_fc

        self.b0 = RecurrentSpikingBlock(1, 64, 1, params)
        self.b1 = RecurrentSpikingBlock(64, 64, 2, params)
        self.b2 = RecurrentSpikingBlock(64, 64, 2, params)

        pseudo_do = PseudoGradSpikeWithDropout.apply
        self.fc = CurrentBasedGLIFWithDropout(nn.Linear(64 * 9, 256 * 9, bias=True), pseudo_do, params)
        self.head = nn.Linear(256 * 9, numClasses)
        self.w_t = nn.Parameter(torch.ones(spike_ts, device=device) / spike_ts)

    def _z(self, b, d):
        return tuple(torch.zeros(b, *d, device=self.device) for _ in range(4))

    def forward(self, x):
        b = x.size(0)
        s0, s1, s2, sfc = self._z(b, (64, 30, 30)), self._z(b, (64, 12, 12)), self._z(b, (64, 3, 3)), self._z(b, (256 * 9,))
        h0 = torch.zeros(b, 64, 30, 30, device=self.device)
        h1 = torch.zeros(b, 64, 12, 12, device=self.device)
        h2 = torch.zeros(b, 64, 3, 3, device=self.device)
        mask_fc = Bernoulli(torch.full((b, 256 * 9), 1 - self.dropout_fc, device=self.device)).sample() / (1 - self.dropout_fc)
        outs = []
        for t in range(self.spike_ts):
            y0, s0, h0 = self.b0(x, s0, h0)
            y1, s1, h1 = self.b1(y0, s1, h1)
            y2, s2, h2 = self.b2(y1, s2, h2)
            yf, sfc = self.fc(y2.view(b, -1), sfc, mask_fc, self.training)
            outs.append(self.head(yf) * self.w_t[t])
        return torch.stack(outs).sum(0)


class CurrentBasedSNNRecurrent(nn.Module):
    def __init__(self, numClasses, dropout_fc, spike_ts, device, params):
        super().__init__()
        self.net = Wafer2SpikeRecurrent(numClasses, dropout_fc, spike_ts, device, params)

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    dataloaders = build_dataloaders()
    training(network=CurrentBasedSNNRecurrent, params=[0.05, 0.10, 0.08, 0.30], dataloaders=dataloaders)
