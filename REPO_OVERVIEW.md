# Repository Overview

## 파일 구성

- `wafer2spike_structured_notebook_ipynb(05_22_최종_제출본).py`
  - Colab 노트북을 Python 스크립트로 내보낸 버전.
  - Wafer2Spike 기반 SNN 모델 정의(기존 surrogate gradient 버전, GroupNorm + Fast-Sigmoid 개선 버전),
    wafer 데이터 전처리/증강, 학습/평가, GPU 전력 측정, 하이퍼파라미터 실험 코드가 한 파일에 순차적으로 포함됨.

- `나노소자_project2_Q2_.ipynb`
  - 소자 시뮬레이션 CSV를 읽어 Vth/DIBL을 계산하고 모델 피팅하는 실험 노트북.
  - Bulk, DG, UTB-SOI 구조별로 sweep 데이터를 처리하고 DIBL-산화막 두께 관계를 비교/시각화함.
