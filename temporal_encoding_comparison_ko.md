# Temporal Encoding 구현 설명 (원래 코드 대비)

## 1) 왜 변경했나
기존 `original_residual` 실험은 **모든 timestep에 동일한 입력 `x`를 그대로 반복 주입**했습니다.  
이번 변경은 이 지점에 선택 가능한 인코딩 단계를 넣어, timestep마다 입력을 확률적으로 샘플링하는 **Bernoulli temporal encoding**을 사용할 수 있게 한 것입니다.

---

## 2) 원래 코드(기존 방식: constant)
기존 동작의 핵심은 아래처럼 볼 수 있습니다.

- timestep 루프 안에서 입력 `x`를 그대로 사용
- 즉, `t=0..T-1` 동안 입력 텐서는 동일
- 시간축 변화는 입력이 아니라, GLIF 내부 상태(state) 누적/감쇠에서만 발생

개념적으로는 다음과 같았습니다.

```python
for t in range(spike_ts):
    h0, s_stem = stem(x, s_stem)  # 같은 x를 반복 주입
    ...
```

---

## 3) 현재 코드(변경 후)
현재 `Wafer2SpikeResidual`에는 `input_encoding` 인자가 추가되어 있고, 기본값은 `"constant"`입니다.

### (1) 인코딩 모드 파라미터
- `input_encoding="constant"`: 기존과 동일하게 입력을 그대로 사용
- `input_encoding="bernoulli"`: `sigmoid(x)`를 확률로 사용해 `torch.bernoulli` 샘플링

### (2) 인코딩 함수 추가
`_encode_temporal_input` 함수가 입력 모드에 따라 아래처럼 동작합니다.

- Bernoulli 모드: `prob = torch.sigmoid(x)` → `torch.bernoulli(prob)`
- Constant 모드: `x` 그대로 반환

### (3) timestep 루프 반영
매 timestep마다 `x_t = self._encode_temporal_input(x)`를 만든 뒤, `stem`에 `x_t`를 넣습니다.

즉, Bernoulli 모드에서는 timestep마다 서로 다른 샘플이 들어가고, constant 모드에서는 이전과 동일한 입력이 반복됩니다.

---

## 4) 원래 코드 대비 핵심 차이 한 줄 요약
- **원래:** `x`를 timestep마다 그대로 반복 주입
- **현재:** `x`를 timestep마다 `constant` 또는 `bernoulli`로 인코딩한 `x_t`로 주입

---

## 5) 실험 실행 관점에서의 변화
`run_all_experiments.py`에서 `original_residual`이 두 엔트리로 분리되어, Colab에서 `--only`로 직접 모드를 선택할 수 있습니다.

- `original_02_residual_prenorm_constant`
- `original_02_residual_prenorm_bernoulli`

예시:

```bash
python run_all_experiments.py --only original_02_residual_prenorm_constant
python run_all_experiments.py --only original_02_residual_prenorm_bernoulli
```

이 구조 덕분에 전처리/데이터로더는 공유하면서, 입력 인코딩 방식만 분리 비교할 수 있습니다.
