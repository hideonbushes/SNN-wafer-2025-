# -*- coding: utf-8 -*-
"""Original Wafer2Spike + latency encoding preprocessing."""

import random
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

from wafer2spike_original_train_test import CurrentBasedSNN, training


TRAIN_RATIO = 0.6
VAL_RATIO = 0.1
TEST_RATIO = 0.3
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def _latency_encode_image(image_norm: np.ndarray, num_steps: int = 10) -> np.ndarray:
    """Convert normalized intensity map to latency-coded map (early spike -> stronger value)."""
    image_norm = np.clip(image_norm, 0.0, 1.0)
    latency_step = np.floor((1.0 - image_norm) * (num_steps - 1)).astype(np.float32)
    encoded = 1.0 - (latency_step / max(1, num_steps - 1))
    encoded[image_norm <= 0.0] = 0.0
    return encoded.astype(np.float32)


def build_dataloaders(data_path="/content/drive/MyDrive/WM-811k/LSWMD.pkl", batch_size=256, latency_steps=10):
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

    X = df_train["waferMap"].values
    y = df_train["failureNum"].values
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1.0 - TRAIN_RATIO), stratify=y, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(TEST_RATIO / (VAL_RATIO + TEST_RATIO)), stratify=y_temp, random_state=seed
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
            im_e = _latency_encode_image(im_r, num_steps=latency_steps)
            out.append(im_e)
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

    class WaferDataset(Dataset):
        def __init__(self, data, labels):
            self.data, self.labels = data, labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]

    train_loader = DataLoader(WaferDataset(wafer_tr_data, torch.tensor(y_train_aug, dtype=torch.long)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(WaferDataset(wafer_val_data, torch.tensor(y_val, dtype=torch.long)), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(WaferDataset(wafer_te_data, torch.tensor(y_test, dtype=torch.long)), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    dataloaders = build_dataloaders()
    training(network=CurrentBasedSNN, params=[0.05, 0.1, 0.08, 0.3], dataloaders=dataloaders)
