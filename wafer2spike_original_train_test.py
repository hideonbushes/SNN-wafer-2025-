# -*- coding: utf-8 -*-
"""Original Wafer2Spike-style training/testing script.

전처리 파이프라인은 기존 노트북 코드와 동일하게 유지.
"""

import os, random
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from torch.distributions.bernoulli import Bernoulli


# =========================
# 1) Original Wafer2Spike-style block
# =========================
Cg = 0.3  # Coefficient Gain for surrogate gradient


class PseudoGradSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, vth, cw):
        ctx.save_for_backward(input)
        ctx.vth = vth
        ctx.cw = cw
        return input.gt(vth).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        vth = ctx.vth
        cw = ctx.cw
        grad_input = grad_output.clone()
        spike_pseudo_grad = abs(input - vth) < cw
        return Cg * grad_input * spike_pseudo_grad.float(), None, None


class PseudoGradSpikeWithDropout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, vth, cw, mask):
        ctx.save_for_backward(input)
        ctx.vth = vth
        ctx.cw = cw
        ctx.mask = mask
        return input.gt(vth).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        vth = ctx.vth
        cw = ctx.cw
        mask = ctx.mask
        grad_input = grad_output.clone()
        spike_pseudo_grad = abs(input - vth) < cw
        spike_pseudo_grad[mask == 0] = 0
        return Cg * grad_input * spike_pseudo_grad.float(), None, None, None


class CurrentBasedGLIF(nn.Module):
    """GLIF/ALIF neuron: learnable multi-timescale current + adaptive threshold."""
    def __init__(self, func_v, pseudo_grad_ops, param):
        super(CurrentBasedGLIF, self).__init__()
        self.func_v = func_v
        self.pseudo_grad_ops = pseudo_grad_ops
        self.w_scdecay, self.w_vdecay, self.vth, self.cw = param
        self.w_adapt_decay = nn.Parameter(torch.tensor(0.8))
        self.w_adapt_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, input_data, state):
        pre_spike, pre_current, pre_volt, pre_adapt = state
        current = self.w_scdecay * pre_current + self.func_v(input_data)
        adapt = torch.sigmoid(self.w_adapt_decay) * pre_adapt + pre_spike
        vth_eff = self.vth + F.softplus(self.w_adapt_scale) * adapt
        volt = self.w_vdecay * pre_volt * (1. - pre_spike) + current
        output = self.pseudo_grad_ops(volt, vth_eff, self.cw)
        return output, (output, current, volt, adapt)


class CurrentBasedGLIFWithDropout(nn.Module):
    def __init__(self, func_v, pseudo_grad_ops, param):
        super(CurrentBasedGLIFWithDropout, self).__init__()
        self.func_v = func_v
        self.pseudo_grad_ops = pseudo_grad_ops
        self.w_scdecay, self.w_vdecay, self.vth, self.cw = param
        self.w_adapt_decay = nn.Parameter(torch.tensor(0.8))
        self.w_adapt_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, input_data, state, mask, train):
        pre_spike, pre_current, pre_volt, pre_adapt = state
        current = self.w_scdecay * pre_current + self.func_v(input_data)
        if train is True:
            current = current * mask
        adapt = torch.sigmoid(self.w_adapt_decay) * pre_adapt + pre_spike
        vth_eff = self.vth + F.softplus(self.w_adapt_scale) * adapt
        volt = self.w_vdecay * pre_volt * (1. - pre_spike) + current
        output = self.pseudo_grad_ops(volt, vth_eff, self.cw, mask)
        return output, (output, current, volt, adapt)


class Wafer2Spike(nn.Module):
    def __init__(self, numClasses, dropout_fc, spike_ts, device, params):
        super(Wafer2Spike, self).__init__()
        self.device = device
        self.spike_ts = spike_ts
        self.dropout_fc = dropout_fc
        self.scdecay, self.vdecay, self.vth, self.cw = params

        pseudo_grad_ops = PseudoGradSpike.apply
        pseudo_grad_ops_with_dropout = PseudoGradSpikeWithDropout.apply

        self.conv_spk_enc_w_vdecay = nn.Parameter(torch.ones(1, 64, 30, 30, device=self.device) * self.vdecay)
        self.conv_spk_enc_w_scdecay = nn.Parameter(torch.ones(1, 64, 30, 30, device=self.device) * self.scdecay)

        self.Spk_conv1_w_vdecay = nn.Parameter(torch.ones(1, 64, 12, 12, device=self.device) * self.vdecay)
        self.Spk_conv1_w_scdecay = nn.Parameter(torch.ones(1, 64, 12, 12, device=self.device) * self.scdecay)

        self.Spk_conv2_w_vdecay = nn.Parameter(torch.ones(1, 64, 3, 3, device=self.device) * self.vdecay)
        self.Spk_conv2_w_scdecay = nn.Parameter(torch.ones(1, 64, 3, 3, device=self.device) * self.scdecay)

        self.Spk_fc_w_vdecay = nn.Parameter(torch.ones(1, 256 * 9, device=self.device) * self.vdecay)
        self.Spk_fc_w_scdecay = nn.Parameter(torch.ones(1, 256 * 9, device=self.device) * self.scdecay)

        self.w_t = nn.Parameter(torch.ones((self.spike_ts), device=self.device) / self.spike_ts)

        self.conv_spk_enc = CurrentBasedGLIF(nn.Conv2d(1, 64, (7, 7), stride=1, bias=True), pseudo_grad_ops,
                                            [self.conv_spk_enc_w_scdecay, self.conv_spk_enc_w_vdecay, self.vth, self.cw])

        self.Spk_conv1 = CurrentBasedGLIF(nn.Conv2d(64, 64, (7, 7), stride=2, bias=True), pseudo_grad_ops,
                                         [self.Spk_conv1_w_scdecay, self.Spk_conv1_w_vdecay, self.vth, self.cw])

        self.Spk_conv2 = CurrentBasedGLIF(nn.Conv2d(64, 64, (7, 7), stride=2, bias=True), pseudo_grad_ops,
                                         [self.Spk_conv2_w_scdecay, self.Spk_conv2_w_vdecay, self.vth, self.cw])

        self.Spk_fc = CurrentBasedGLIFWithDropout(nn.Linear(64 * 9, 256 * 9, bias=True), pseudo_grad_ops_with_dropout,
                                                 [self.Spk_fc_w_scdecay, self.Spk_fc_w_vdecay, self.vth, self.cw])

        self.nonSpk_fc = nn.Linear(256 * 9, numClasses)

    def forward(self, input_data, states):
        batch_size = input_data.shape[0]
        output_spikes = []

        conv_spk_enc_state, Spk_conv1_state, Spk_conv2_state, Spk_fc_state = states[0], states[1], states[2], states[3]

        mask_fc = Bernoulli(
            torch.full_like(torch.zeros(batch_size, 256 * 9, device=self.device), 1 - self.dropout_fc)
        ).sample() / (1 - self.dropout_fc)

        for step in range(self.spike_ts):
            input_spike = input_data
            conv_spk_enc_spike, conv_spk_enc_state = self.conv_spk_enc(input_spike, conv_spk_enc_state)
            Spk_conv1_spike, Spk_conv1_state = self.Spk_conv1(conv_spk_enc_spike, Spk_conv1_state)
            Spk_conv2_spike, Spk_conv2_state = self.Spk_conv2(Spk_conv1_spike, Spk_conv2_state)

            flattened_spike = Spk_conv2_spike.view(batch_size, -1)
            Spk_fc_spike, Spk_fc_state = self.Spk_fc(flattened_spike, Spk_fc_state, mask_fc, self.training)
            nonSpk_fc_output = self.nonSpk_fc(Spk_fc_spike)
            output_spikes += [nonSpk_fc_output * self.w_t[step]]

        return torch.stack(output_spikes).sum(dim=0)


class CurrentBasedSNN(nn.Module):
    def __init__(self, numClasses, dropout_fc, spike_ts, device, params):
        super(CurrentBasedSNN, self).__init__()
        self.device = device
        self.wafer2spike = Wafer2Spike(numClasses, dropout_fc, spike_ts, device, params)

    def forward(self, input_data):
        batch_size = input_data.shape[0]

        conv_spk_enc_state = tuple(torch.zeros(batch_size, 64, 30, 30, device=self.device) for _ in range(4))
        Spk_conv1_state = tuple(torch.zeros(batch_size, 64, 12, 12, device=self.device) for _ in range(4))
        Spk_conv2_state = tuple(torch.zeros(batch_size, 64, 3, 3, device=self.device) for _ in range(4))
        Spk_fc_state = tuple(torch.zeros(batch_size, 256 * 9, device=self.device) for _ in range(4))

        states = (conv_spk_enc_state, Spk_conv1_state, Spk_conv2_state, Spk_fc_state)
        return self.wafer2spike(input_data, states)


# =========================
# 2) Preprocessing block (기존 코드와 동일 유지)
# =========================
TRAIN_RATIO = 0.6
VAL_RATIO = 0.1
TEST_RATIO = 0.3

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def build_dataloaders(data_path="/content/drive/MyDrive/WM-811k/LSWMD.pkl", batch_size=256):
    df = pd.read_pickle(data_path)

    trte = []
    for j in df["trianTestLabel"]:
        try:
            trte.append(j[0][0])
        except Exception:
            trte.append(np.nan)
    df["trianTestLabel"] = trte

    ft = []
    for j in df["failureType"]:
        try:
            ft.append(j[0][0])
        except Exception:
            ft.append(np.nan)
    df["failureType"] = ft

    map_type = {
        'Center': 0, 'Donut': 1, 'Edge-Loc': 2, 'Edge-Ring': 3,
        'Loc': 4, 'Random': 5, 'Scratch': 6, 'Near-full': 7, 'none': 8
    }
    map_tt = {'Training': 0, 'Test': 1}
    df["failureNum"] = df["failureType"].map(map_type)
    df["trainTestNum"] = df["trianTestLabel"].map(map_tt)

    df = df[df["failureNum"].notna() & df["trainTestNum"].notna()].reset_index(drop=True)
    df["failureNum"] = df["failureNum"].astype(int)
    df["trainTestNum"] = df["trainTestNum"].astype(int)

    df_train = df[df["trainTestNum"] == 0].reset_index(drop=True)

    test_size1 = 1.0 - TRAIN_RATIO
    X = df_train["waferMap"].values
    y = df_train["failureNum"].values
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size1, stratify=y, random_state=seed
    )

    test_size2 = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_size2, stratify=y_temp, random_state=seed
    )

    def rot_aug(img):
        ang = random.randint(0, 360)
        img_n = img / img.max()
        M = cv2.getRotationMatrix2D((18, 18), ang, 1.0)
        return cv2.warpAffine(img_n, M, (36, 36))

    X_train_aug = list(X_train)
    y_train_aug = list(y_train)
    idxs3 = np.where(np.array(y_train) == 3)[0]
    random.shuffle(idxs3)
    for ix in idxs3[:3884]:
        for a in np.unique([rot_aug(X_train[ix])], axis=0):
            X_train_aug.append(a)
            y_train_aug.append(3)

    def preprocess_images(arrays):
        out = []
        for im in arrays:
            im_n = im / im.max()
            im_r = cv2.resize(im_n, (36, 36), interpolation=cv2.INTER_CUBIC)
            out.append(im_r)
        return np.stack(out).astype("float32")

    wafer_tr_np = preprocess_images(X_train_aug)
    wafer_val_np = preprocess_images(X_val)
    wafer_te_np = preprocess_images(X_test)

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2999,), (0.19235,))
    ])
    wafer_tr_data = tf(wafer_tr_np).permute(1, 0, 2)[:, None, :, :]
    wafer_val_data = tf(wafer_val_np).permute(1, 0, 2)[:, None, :, :]
    wafer_te_data = tf(wafer_te_np).permute(1, 0, 2)[:, None, :, :]

    train_labels = torch.tensor(y_train_aug, dtype=torch.long)
    val_labels = torch.tensor(y_val, dtype=torch.long)
    te_labels = torch.tensor(y_test, dtype=torch.long)

    class WaferDataset(Dataset):
        def __init__(self, data, labels):
            self.data, self.labels = data, labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]

    train_loader = DataLoader(WaferDataset(wafer_tr_data, train_labels), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(WaferDataset(wafer_val_data, val_labels), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(WaferDataset(wafer_te_data, te_labels), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# =========================
# 3) Training/Test loop (Original)
# =========================
def test_accuracy(model, loader, criterion, device, phase="Validation"):
    model.eval()
    loss_sum, correct = 0.0, 0
    with torch.no_grad():
        for data, label in loader:
            data, label = data.to(device), label.to(device)
            out = model(data)
            loss_sum += criterion(out, label).item() * data.size(0)
            correct += (out.argmax(1) == label).sum().item()
    avg_loss = loss_sum / len(loader.dataset)
    acc = correct / len(loader.dataset)
    print(f"{phase} Loss: {avg_loss:.4f}, {phase} Acc: {acc:.4f}")
    model.train()
    return avg_loss, acc


def training(network, params,
             batch_size=256, epochs=10, lr=1e-4,
             dataloaders=None, numClasses=9,
             spike_ts=10, dropout_fc=0.3,
             return_metrics=False):
    train_loader, val_loader, test_loader = dataloaders
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wafer2spike_snn = network(numClasses, dropout_fc, spike_ts, device, params=params)
    model = nn.DataParallel(wafer2spike_snn.to(device))

    criterion = nn.CrossEntropyLoss()
    decays = [
        'module.wafer2spike.conv_spk_enc_w_vdecay',
        'module.wafer2spike.conv_spk_enc_w_scdecay',
        'module.wafer2spike.Spk_conv1_w_vdecay',
        'module.wafer2spike.Spk_conv1_w_scdecay',
        'module.wafer2spike.Spk_conv2_w_vdecay',
        'module.wafer2spike.Spk_conv2_w_scdecay',
        'module.wafer2spike.Spk_fc_w_vdecay',
        'module.wafer2spike.Spk_fc_w_scdecay'
    ]
    weights_ts = ['module.wafer2spike.w_t']

    decay_params = [p for n, p in model.named_parameters() if n in decays]
    params_ts = [p for n, p in model.named_parameters() if n in weights_ts]
    weights = [p for n, p in model.named_parameters() if n not in decays + weights_ts]

    optimizer = torch.optim.Adam(
        [{'params': weights}, {'params': decay_params}, {'params': params_ts}],
        lr=lr
    )

    train_loss_history, train_acc_history = [], []
    val_loss_history, val_acc_history = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum, correct = 0.0, 0
        for data, label in train_loader:
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * data.size(0)
            correct += (out.argmax(1) == label).sum().item()

        for p in decay_params:
            p.data.clamp_(min=1e-7)

        train_loss = loss_sum / len(train_loader.dataset)
        train_acc = correct / len(train_loader.dataset)
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        print(f"Epoch {epoch}/{epochs} — Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        val_loss, val_acc = test_accuracy(model, val_loader, criterion, device, phase="Validation")
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

    test_loss, test_acc = test_accuracy(model, test_loader, criterion, device, phase="Test")

    preds, trues = [], []
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            preds.extend(model(data).argmax(1).cpu().tolist())
            trues.extend(label.cpu().tolist())

    print("\nConfusion Matrix:")
    cm = confusion_matrix(trues, preds)
    print(cm)
    print(classification_report(trues, preds))

    if return_metrics:
        return {
            "model": model,
            "preds": preds,
            "trues": trues,
            "confusion_matrix": cm,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "train_loss_history": train_loss_history,
            "train_acc_history": train_acc_history,
            "val_loss_history": val_loss_history,
            "val_acc_history": val_acc_history,
        }
    return model


if __name__ == "__main__":
    dataloaders = build_dataloaders()
    trained_model = training(
        network=CurrentBasedSNN,
        params=[0.05, 0.1, 0.08, 0.3],
        dataloaders=dataloaders
    )
