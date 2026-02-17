"""Create a visual example of wafer preprocessing (normalize/resize/rate/latency)."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import cv2


def _rate_encode_image(image_norm: np.ndarray, num_steps: int = 10) -> np.ndarray:
    image_norm = np.clip(image_norm, 0.0, 1.0)
    spike_train = (np.random.rand(num_steps, *image_norm.shape) < image_norm[None, :, :]).astype(np.float32)
    return spike_train.mean(axis=0)


def _latency_encode_image(image_norm: np.ndarray, num_steps: int = 10) -> np.ndarray:
    image_norm = np.clip(image_norm, 0.0, 1.0)
    latency_step = np.floor((1.0 - image_norm) * (num_steps - 1)).astype(np.float32)
    encoded = 1.0 - (latency_step / max(1, num_steps - 1))
    encoded[image_norm <= 0.0] = 0.0
    return encoded.astype(np.float32)


def make_example_wafer(size: int = 36) -> np.ndarray:
    y, x = np.ogrid[:size, :size]
    cx = cy = (size - 1) / 2.0
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    wafer = np.zeros((size, size), dtype=np.float32)
    wafer[r <= 16] = 2.0

    wafer[(x - cx) ** 2 / 36 + (y - cy - 1.5) ** 2 / 9 <= 1] = 1.0

    rng = np.random.default_rng(42)
    noise_mask = (rng.random((size, size)) < 0.08) & (r <= 16)
    wafer[noise_mask] = 1.0
    return wafer


def main() -> None:
    np.random.seed(42)

    raw = make_example_wafer(36)
    raw_norm = raw / raw.max()
    resized = cv2.resize(raw_norm, (36, 36), interpolation=cv2.INTER_CUBIC)

    rate = _rate_encode_image(resized, num_steps=10)
    latency = _latency_encode_image(resized, num_steps=10)

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
    items = [
        (raw, "Original waferMap\n(values: 0/1/2)", "viridis", 0, 2),
        (resized, "Normalize + Resize\n(range: 0~1)", "viridis", 0, 1),
        (rate, "Rate encoded\n(mean spike rate)", "magma", 0, 1),
        (latency, "Latency encoded\n(early spike -> high)", "magma", 0, 1),
    ]

    for ax, (arr, title, cmap, vmin, vmax) in zip(axes, items):
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Wafer preprocessing example used by current rate/latency scripts", fontsize=11)
    fig.tight_layout()
    out_path = "docs/images/preprocessing_example.png"
    fig.savefig(out_path, dpi=150)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
