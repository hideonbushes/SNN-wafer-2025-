# SNN Wafer Defect Classification (Wafer2Spike)

## 1) Project Overview

This repository centers on **spiking neural network (SNN)-based wafer map defect classification** using a Wafer2Spike-style current-based LIF model.

The main training script (exported from Colab) loads WM-811K wafer maps, preprocesses them into 36×36 tensors, and trains a multi-class classifier (9 classes) over multiple SNN time steps. The code includes:

- an original Wafer2Spike-like implementation with pseudo-gradient surrogate spikes,
- an improved variant using GroupNorm + Fast-Sigmoid surrogate gradients,
- end-to-end training/evaluation loops and class-level report outputs.

References in source:
- Dataset loading from `LSWMD.pkl` and initial context. (`wafer2spike_structured_notebook_ipynb(05_22_최종_제출본).py`)
- Original SNN and improved SNN blocks are both present in the same script.

---

## 2) File Structure

```text
SNN-wafer-2025-/
├── README.md
├── wafer2spike_structured_notebook_ipynb(05_22_최종_제출본).py
├── wafer2spike_original_train_test.py
├── wafer2spike_improved_train_test.py
```

### File roles

- **`README.md`**
  - Comprehensive technical documentation for this repository.

- **`wafer2spike_structured_notebook_ipynb(05_22_최종_제출본).py`**
  - Primary SNN project code (exported Colab notebook).
  - Contains:
    - WM-811K DataFrame loading,
    - label extraction and mapping,
    - train/val/test stratified splitting,
    - preprocessing & augmentation,
    - `Dataset`/`DataLoader` creation,
    - Wafer2Spike model definitions,
    - training/validation/test logic, confusion matrix/report,
    - experimental blocks for alternate architecture/hyperparameters.


- **`wafer2spike_original_train_test.py`**
  - Original Wafer2Spike-style block을 독립 실행 가능하게 분리한 파일.
  - 기존 전처리 + original 모델/학습/테스트 루프를 단독으로 포함.

- **`wafer2spike_improved_train_test.py`**
  - GroupNorm + Fast-Sigmoid 개선 블록을 독립 실행 가능하게 분리한 파일.
  - 기존 전처리 + improved 모델/학습/테스트 루프를 단독으로 포함.

---

## 3) Data Pipeline

The wafer data path in code is:

1. **Load raw dataset** from pickle (`/content/drive/MyDrive/WM-811k/LSWMD.pkl`).
2. **Extract labels safely** from nested fields (`trianTestLabel`, `failureType`) and map to numeric IDs.
3. **Filter valid samples** and keep `Training` subset.
4. **Stratified split** into Train/Val/Test using configured ratios (`0.6/0.1/0.3`).
5. **Class-specific augmentation**:
   - for class ID `3`, random rotation augmentation is applied to a subset (`idxs3[:3884]`).
6. **Image preprocessing**:
   - normalize each wafer map (`im / im.max()`),
   - resize to `36×36` using OpenCV cubic interpolation,
   - convert to `float32`.
7. **Tensor transform**:
   - `ToTensor()` + `Normalize(mean=0.2999, std=0.19235)`,
   - reshape/permutation and channel insertion to produce model input shape.
8. **DataLoader construction** with batch size `256`.

### Spike encoding method (important)

This code does **not** use an explicit Poisson/rate input spike encoder. Instead, the same preprocessed frame is fed each time step (`input_spike = input_data`), and spike events are generated **inside LIF layers** via thresholding + surrogate gradients. So temporal spiking behavior is produced by recurrent membrane/current state dynamics rather than stochastic input encoding.

---

## 4) Model Architecture

The script contains two major SNN variants.

### A. Original Wafer2Spike-style block

- **Neuron core**: current-based LIF modules (`CurrentBasedLIF`, `CurrentBasedLIFWithDropout`).
- **Spike function**: custom surrogate (`PseudoGradSpike`, `PseudoGradSpikeWithDropout`) with coefficient `Cg=0.3` and window `cw`.
- **Main stack**:
  1. `Conv2d(1→64, k=7, s=1)` + LIF
  2. `Conv2d(64→64, k=7, s=2)` + LIF
  3. `Conv2d(64→64, k=7, s=2)` + LIF
  4. flatten
  5. `Linear(64*9 → 256*9)` + dropout-aware LIF
  6. `Linear(256*9 → numClasses)`
- **Temporal aggregation**:
  - outputs from each time step are weighted by learnable `w_t` and summed.

### B. Proposed/improved block in the notebook

- **Surrogate gradient**: `FastSigmoidSurrogate`.
- **Normalization**: GroupNorm inserted in conv/FC paths (code comments mention replacing BatchNorm concerns for SNN stability).
- **Layer pattern**: layer → GroupNorm → ReLU → LIF state update → surrogate spike.
- **State handling**:
  - if state not provided, the model initializes per-layer spike/current/voltage states internally.

---

## 5) Training Strategy

Two training pipelines appear in the script (baseline and improved).

### Baseline training block

- **Loss**: `CrossEntropyLoss`.
- **Optimizer**: `Adam` with parameter groups:
  - regular weights,
  - decay parameters (`w_vdecay`, `w_scdecay` across layers),
  - temporal weights `w_t`.
- **Key defaults**:
  - batch size `256`, epochs `10`, learning rate `1e-4`, `spike_ts=10`, `dropout_fc=0.3`.
- **Stability handling**:
  - decay parameters are clamped to minimum `1e-7` after each epoch.

### Improved training block

- **Loss**: class-weighted `CrossEntropyLoss` (inverse-frequency class weights from training labels).
- **Optimizer**: `Adam` with separate weight decay policies:
  - non-`w_t` params use `weight_decay=1e-5`,
  - `w_t` params use `alpha_wd` (default `0.0`).
- **Scheduler**: `ReduceLROnPlateau` on validation accuracy (`factor=0.5`, `patience=2`).
- **Early stopping**: patience-based stop (`patience_es=3`).
- **Example hyperparameters**:
  - `params=[0.05, 0.10, 0.08, 5.0]` interpreted as `[scDecay, vDecay, vTh, alpha]`,
  - `spike_ts=10`, `epochs=20`, `lr=1e-4`, `dropout_fc=0.20`.
- **Evaluation outputs**:
  - validation/test accuracy,
  - confusion matrix,
  - classification report.

---

## Practical Notes

- The main script is an exported notebook and includes multiple experimental blocks sequentially; it is not yet modularized as a package.
- Paths are Colab/Drive-specific and should be parameterized for local or production runs.
