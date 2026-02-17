# Rate/Latency 전처리 예시 (현재 코드 기준)

아래 그림은 현재 구현(`wafer2spike_original_rate_encoding_train_test.py`, `wafer2spike_original_latency_encoding_train_test.py`)이
단일 wafer 이미지를 어떻게 변환하는지 보여줍니다.

![preprocessing example](images/preprocessing_example.png)

## 1) Normalize + Resize
- 각 wafer map `im`에 대해 `im_n = im / im.max()` 수행
- 이후 `cv2.resize(im_n, (36, 36), interpolation=cv2.INTER_CUBIC)` 적용

## 2) Rate encoding
현재 함수:

```python
spike_train = (np.random.rand(num_steps, *image_norm.shape) < image_norm[None, :, :]).astype(np.float32)
rate_map = spike_train.mean(axis=0)
```

- 픽셀 intensity를 발화확률로 사용해 `num_steps`번 Bernoulli 샘플링
- 시간축 평균(`mean(axis=0)`)으로 최종 2D rate map 생성

## 3) Latency encoding
현재 함수:

```python
latency_step = np.floor((1.0 - image_norm) * (num_steps - 1)).astype(np.float32)
encoded = 1.0 - (latency_step / max(1, num_steps - 1))
encoded[image_norm <= 0.0] = 0.0
```

- 밝을수록 `latency_step`이 작아짐(더 이른 발화 시점)
- 최종 출력은 0~1 범위의 2D encoded map

## 4) Tensor 변환
- `transforms.ToTensor()` 후 `Normalize((0.2999,), (0.19235,))` 적용
- 즉, 현재 구현은 **시간축 spike tensor를 직접 입력하지 않고**, 전처리 결과인 **정적 2D map**을 텐서화해 사용

## 재현 명령
```bash
python scripts/visualize_preprocessing_example.py
```
