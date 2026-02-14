# -*- coding: utf-8 -*-
"""Improved Wafer2Spike (GroupNorm + Fast-Sigmoid) training/testing script.

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
from torch.optim.lr_scheduler import ReduceLROnPlateau


# =========================
# 1) Improved block (GroupNorm + Fast-Sigmoid)
# =========================
class FastSigmoidSurrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, vth, alpha=10.0):
        ctx.save_for_backward(input)
        ctx.vth = vth
        ctx.alpha = alpha
        return (input > vth).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        vth = ctx.vth
        alpha = ctx.alpha
        x = input - vth
        expm = torch.exp(-alpha * x)
        grad = alpha * expm / (1 + expm) ** 2
        return grad_output * grad, None, None


class CurrentBasedGLIF(nn.Module):
    """GLIF/ALIF neuron with adaptive threshold in GroupNorm-based block."""
    def __init__(self, conv_or_linear, bn_layer, surrogate, param):
        super(CurrentBasedGLIF, self).__init__()
        self.layer = conv_or_linear
        self.bn = bn_layer
        self.surrogate = surrogate.apply
        self.w_scdecay, self.w_vdecay, self.vth, self.alpha = param
        self.w_adapt_decay = nn.Parameter(torch.tensor(0.8))
        self.w_adapt_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, input_data, state):
        pre_spike, pre_current, pre_volt, pre_adapt = state
        x = self.layer(input_data)
        x = self.bn(x)
        x = torch.relu(x)
        current = self.w_scdecay * pre_current + x
        adapt = torch.sigmoid(self.w_adapt_decay) * pre_adapt + pre_spike
        vth_eff = self.vth + F.softplus(self.w_adapt_scale) * adapt
        volt = self.w_vdecay * pre_volt * (1. - pre_spike) + current
        output = self.surrogate(volt, vth_eff, self.alpha)
        return output, (output, current, volt, adapt)


class CurrentBasedGLIFWithDropout(nn.Module):
    def __init__(self, linear, bn_layer, surrogate, param):
        super(CurrentBasedGLIFWithDropout, self).__init__()
        self.layer = linear
        self.bn = bn_layer
        self.surrogate = surrogate.apply
        self.w_scdecay, self.w_vdecay, self.vth, self.alpha = param
        self.w_adapt_decay = nn.Parameter(torch.tensor(0.8))
        self.w_adapt_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, input_data, state, mask, train):
        pre_spike, pre_current, pre_volt, pre_adapt = state
        x = self.layer(input_data)
        x = self.bn(x)
        x = torch.relu(x)
        if train:
            x = x * mask
        current = self.w_scdecay * pre_current + x
        adapt = torch.sigmoid(self.w_adapt_decay) * pre_adapt + pre_spike
        vth_eff = self.vth + F.softplus(self.w_adapt_scale) * adapt
        volt = self.w_vdecay * pre_volt * (1. - pre_spike) + current
        output = self.surrogate(volt, vth_eff, self.alpha)
        return output, (output, current, volt, adapt)


class Wafer2Spike(nn.Module):
    def __init__(self, numClasses, dropout_fc, spike_ts, device, params):
        super(Wafer2Spike, self).__init__()
        self.device = device
        self.spike_ts = spike_ts
        self.dropout_fc = dropout_fc
        self.scdecay, self.vdecay, self.vth, self.alpha = params

        conv_enc = nn.Conv2d(1, 64, 7, stride=1, bias=True).to(device)
        gn_enc = nn.GroupNorm(num_groups=8, num_channels=64).to(device)

        conv1 = nn.Conv2d(64, 64, 7, stride=2, bias=True).to(device)
        gn1 = nn.GroupNorm(num_groups=8, num_channels=64).to(device)

        conv2 = nn.Conv2d(64, 64, 7, stride=2, bias=True).to(device)
        gn2 = nn.GroupNorm(num_groups=8, num_channels=64).to(device)

        fc_lin = nn.Linear(64 * 9, 256 * 9, bias=True).to(device)
        gn_fc = nn.GroupNorm(num_groups=32, num_channels=256 * 9).to(device)

        self.conv_spk_enc = CurrentBasedGLIF(conv_or_linear=conv_enc, bn_layer=gn_enc,
                                            surrogate=FastSigmoidSurrogate, param=params)
        self.Spk_conv1 = CurrentBasedGLIF(conv_or_linear=conv1, bn_layer=gn1,
                                         surrogate=FastSigmoidSurrogate, param=params)
        self.Spk_conv2 = CurrentBasedGLIF(conv_or_linear=conv2, bn_layer=gn2,
                                         surrogate=FastSigmoidSurrogate, param=params)
        self.Spk_fc = CurrentBasedGLIFWithDropout(linear=fc_lin, bn_layer=gn_fc,
                                                 surrogate=FastSigmoidSurrogate, param=params)

        self.w_t = nn.Parameter(torch.ones(self.spike_ts, device=device) / self.spike_ts)
        self.nonSpk_fc = nn.Linear(256 * 9, numClasses).to(device)

    def forward(self, input_data, states=None):
        batch = input_data.size(0)
        if states is None:
            states = []
            for dims in [(64, 30, 30), (64, 12, 12), (64, 3, 3), (256 * 9,)]:
                states.append(tuple(torch.zeros(batch, *dims, device=self.device) for _ in range(4)))

        mask_fc = Bernoulli(torch.full((batch, 256 * 9), 1 - self.dropout_fc, device=self.device)).sample() / (1 - self.dropout_fc)

        conv_s, c1_s, c2_s, fc_s = states
        outputs = []
        for t in range(self.spike_ts):
            x, conv_s = self.conv_spk_enc(input_data, conv_s)
            x, c1_s = self.Spk_conv1(x, c1_s)
            x, c2_s = self.Spk_conv2(x, c2_s)
            flat = x.view(batch, -1)
            x, fc_s = self.Spk_fc(flat, fc_s, mask_fc, self.training)
            out = self.nonSpk_fc(x) * self.w_t[t]
            outputs.append(out)

        return torch.stack(outputs).sum(0)


class CurrentBasedSNN(nn.Module):
    def __init__(self, numClasses, dropout_fc, spike_ts, device, params):
        super(CurrentBasedSNN, self).__init__()
        self.wafer2spike = Wafer2Spike(numClasses, dropout_fc, spike_ts, device, params)

    def forward(self, input_data):
        return self.wafer2spike(input_data)


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
# 3) Training/Test loop (Improved)
# =========================
def test_accuracy(model, loader, criterion, device, phase="Validation"):
    model.eval()
    loss_sum, correct = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss_sum += criterion(out, y).item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
    avg_loss = loss_sum / len(loader.dataset)
    acc = correct / len(loader.dataset)
    print(f"{phase} Loss: {avg_loss:.4f}, {phase} Acc: {acc:.4f}")
    model.train()
    return acc


def training(network,
             params,
             dataloaders,
             spike_ts=10,
             batch_size=256,
             epochs=20,
             lr=1e-4,
             dropout_fc=0.20,
             weight_decay=1e-5,
             alpha_wd=0.0,
             patience_lr=2,
             patience_es=3):
    train_loader, val_loader, test_loader = dataloaders
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = network(9, dropout_fc, spike_ts, device, params)
    model = nn.DataParallel(model.to(device))

    all_labels = torch.cat([y for _, y in train_loader], dim=0).cpu().numpy()
    counts = np.bincount(all_labels, minlength=9)
    wts = 1.0 / counts
    wts = wts / wts.sum() * 9
    weight_tensor = torch.tensor(wts, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    thr_params = [p for n, p in model.named_parameters() if 'w_t' in n]
    other_params = [p for n, p in model.named_parameters() if 'w_t' not in n]
    optimizer = torch.optim.Adam([
        {'params': other_params, 'weight_decay': weight_decay},
        {'params': thr_params, 'weight_decay': alpha_wd}
    ], lr=lr)

    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=patience_lr, verbose=True)
    best_val_acc, es_count = 0.0, 0

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss, tr_corr = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * x.size(0)
            tr_corr += (out.argmax(1) == y).sum().item()

        tr_loss /= len(train_loader.dataset)
        tr_acc = tr_corr / len(train_loader.dataset)
        print(f"Epoch {epoch}/{epochs} — Train Loss: {tr_loss:.4f}, Train Acc: {tr_acc:.4f}")

        val_acc = test_accuracy(model, val_loader, criterion, device, phase="Validation")
        scheduler.step(val_acc)
        if val_acc > best_val_acc:
            best_val_acc, es_count = val_acc, 0
        else:
            es_count += 1
            if es_count >= patience_es:
                print(f"Early stopping at epoch {epoch}, best val_acc={best_val_acc:.4f}")
                break

    _ = test_accuracy(model, test_loader, criterion, device, phase="Test")

    all_preds, all_trues = [], []
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(1)
            all_preds.extend(preds.cpu().tolist())
            all_trues.extend(y.cpu().tolist())

    print("\nConfusion Matrix:")
    print(confusion_matrix(all_trues, all_preds))
    print("\nClassification Report:")
    print(classification_report(all_trues, all_preds, digits=4))

    return model


if __name__ == "__main__":
    dataloaders = build_dataloaders()
    trained_model = training(
        network=CurrentBasedSNN,
        params=[0.05, 0.10, 0.08, 5.0],
        dataloaders=dataloaders,
        spike_ts=10,
        batch_size=256,
        epochs=20,
        lr=1e-4,
        dropout_fc=0.20,
        weight_decay=1e-5,
        alpha_wd=0.0,
        patience_lr=2,
        patience_es=3
    )
