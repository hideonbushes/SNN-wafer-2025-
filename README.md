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
├── wafer2spike_original_01_glif_alif_train_test.py
├── wafer2spike_improved_01_glif_alif_train_test.py
├── wafer2spike_original_02_residual_prenorm_train_test.py
├── wafer2spike_improved_02_residual_prenorm_train_test.py
├── wafer2spike_original_03_recurrent_conv_train_test.py
├── wafer2spike_improved_03_recurrent_conv_train_test.py
├── wafer2spike_original_04_spike_self_attention_train_test.py
├── wafer2spike_improved_04_spike_self_attention_train_test.py
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


- **`wafer2spike_original_01_glif_alif_train_test.py` / `wafer2spike_improved_01_glif_alif_train_test.py`**
  - 개선안 #1(GLIF/ALIF adaptive threshold) 실험 실행 파일.

- **`wafer2spike_original_02_residual_prenorm_train_test.py` / `wafer2spike_improved_02_residual_prenorm_train_test.py`**
  - 개선안 #2(Spiking Residual + Pre-Norm) 실험 실행 파일.

- **`wafer2spike_original_03_recurrent_conv_train_test.py` / `wafer2spike_improved_03_recurrent_conv_train_test.py`**
  - 개선안 #3(Recurrent spatio-temporal block) 실험 실행 파일.

- **`wafer2spike_original_04_spike_self_attention_train_test.py` / `wafer2spike_improved_04_spike_self_attention_train_test.py`**
  - 개선안 #4(Spike-driven self-attention hybrid) 실험 실행 파일.

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

- **Neuron core**: ALIF/GLIF 기반 current-state neuron (`CurrentBasedGLIF`, `CurrentBasedGLIFWithDropout`).
- **State update**: current/voltage state에 더해 adaptive-threshold 상태(`adapt`)를 추가하여 동적 임계치(`vth_eff`)를 형성.
- **Spike function**: custom surrogate (`PseudoGradSpike`, `PseudoGradSpikeWithDropout`) with coefficient `Cg=0.3` and window `cw`.
- **Main stack**:
  1. `Conv2d(1→64, k=7, s=1)` + GLIF
  2. `Conv2d(64→64, k=7, s=2)` + GLIF
  3. `Conv2d(64→64, k=7, s=2)` + GLIF
  4. flatten
  5. `Linear(64*9 → 256*9)` + dropout-aware GLIF
  6. `Linear(256*9 → numClasses)`
- **Temporal aggregation**:
  - outputs from each time step are weighted by learnable `w_t` and summed.

### B. Proposed/improved block in the notebook

- **Surrogate gradient**: `FastSigmoidSurrogate`.
- **Normalization**: GroupNorm inserted in conv/FC paths (code comments mention replacing BatchNorm concerns for SNN stability).
- **Layer pattern**: layer → GroupNorm → ReLU → GLIF/ALIF state update(적응 임계치 포함) → surrogate spike.
- **State handling**:
  - if state not provided, the model initializes per-layer spike/current/voltage/adaptation states internally.

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

---

## 6) Local Git Bash Quick Commands (처음 설정용 + 매번 업데이트용)

아래 명령은 **Windows 로컬 Git Bash** 기준으로, 바로 복붙해서 사용할 수 있는 최소 세트입니다.

### A. 처음 1회 설정 (GitHub 연동)

```bash
# (0) Git 사용자 정보 설정 (최초 1회)
git config --global user.name "YOUR_NAME"
git config --global user.email "YOUR_EMAIL"

# (1) 작업 폴더로 이동
cd /c/path/to/your/workspace

# (2) 원격 저장소 clone
git clone https://github.com/<YOUR_ID>/SNN-wafer-2025-.git
cd SNN-wafer-2025-

# (3) 원격 확인
git remote -v

# (4) 기본 브랜치 최신화
git checkout main
git pull origin main
```

> SSH를 쓸 경우 `git clone git@github.com:<YOUR_ID>/SNN-wafer-2025-.git` 형태를 사용.

### B. 매번 코드 수정 후 업데이트 (반복 루틴)

```bash
# (1) 저장소 폴더로 이동
cd /c/path/to/your/workspace/SNN-wafer-2025-

# (2) 최신 코드 당기기
git checkout main
git pull origin main

# (3) 작업 브랜치 생성 (권장)
git checkout -b feat/temporal-encoding-update

# (4) 코드 수정 후 상태 확인
git status

# (5) 변경 파일 add + commit
git add .
git commit -m "feat: update temporal encoding experiment code"

# (6) GitHub로 push
git push -u origin feat/temporal-encoding-update
```

### C. Colab에서 최신 코드 실행

```bash
# Colab 셀 기준 (최초 1회)
!git clone https://github.com/<YOUR_ID>/SNN-wafer-2025-.git
%cd /content/SNN-wafer-2025-

# 이후 세션/재실행 시
!git pull origin main

# 실행 예시
!python wafer2spike_improved_train_test.py
```

실무에서는 로컬에서 commit/push를 먼저 완료하고, Colab에서는 clone/pull로 동기화한 뒤 실행하는 흐름을 권장합니다.
