# Original Residual Temporal Encoding 비교 보고서

## 1. 보고서 목적
본 문서는 `original_02_residual_prenorm` 계열에서 사용하는 두 입력 시간 인코딩 모드,
- **Constant 입력 반복 주입**
- **Bernoulli 기반 시간 인코딩**
을 같은 관점에서 비교하기 위한 보고서다.

핵심 질문은 다음과 같다.
1) 두 방식은 timestep별 입력 신호를 어떻게 다르게 만드는가?  
2) 학습 안정성/표현력/재현성 관점에서 어떤 트레이드오프가 있는가?  
3) 실험에서는 어떤 지표로 공정하게 비교해야 하는가?

---

## 2. 비교 대상 정의

### 2.1 Constant 모드
- 각 timestep에서 동일한 정규화 입력 `x`를 반복 주입한다.
- 입력의 시간 변동은 없고, 시간적 변화는 GLIF 내부 상태(전류/전압/스파이크 상태 누적)에서만 발생한다.


a. 직관: “같은 프레임을 10번 보여주고 뉴런 내부 동역학으로 시간 정보를 만든다.”

### 2.2 Bernoulli 모드
- 각 timestep에서 `p = sigmoid(x)`를 계산하고, `Bernoulli(p)` 샘플링으로 `x_t`를 생성한다.
- 같은 원본 샘플이어도 timestep마다 서로 다른 0/1 이벤트 패턴이 입력으로 들어간다.

b. 직관: “원본 입력을 확률화해 매 timestep마다 다른 스파이크 이벤트로 보여준다.”

---

## 3. 구현 차이 요약

| 항목 | Constant | Bernoulli |
|---|---|---|
| timestep 입력 | 항상 동일한 `x` | `x_t = Bernoulli(sigmoid(x))` |
| 시간축 확률성 | 없음(결정적) | 있음(확률적 샘플링) |
| 입력 분산 | 낮음 | 높음 |
| 재현성 | 상대적으로 높음 | seed 고정 필요 |
| 노이즈 정규화 효과 | 제한적 | 가능(데이터 증강 유사 효과) |

---

## 4. 장단점 비교

### 4.1 Constant의 장점
- 실험 재현성이 좋고 결과 변동폭이 작다.
- 입력이 고정이라 학습 초기에 수렴이 안정적인 편이다.
- 베이스라인으로 쓰기 좋다.

### 4.2 Constant의 단점
- timestep별 입력 다양성이 없어서 temporal encoding 효과가 제한적일 수 있다.
- 내부 동역학 의존도가 커져 입력 기반 시간 정보 학습이 약할 수 있다.

### 4.3 Bernoulli의 장점
- 입력 단계에서 시간축 다양성을 제공한다.
- 확률적 이벤트 주입으로 과적합 완화에 도움될 수 있다.
- 스파이킹 입력 특성에 더 가까운 실험이 가능하다.

### 4.4 Bernoulli의 단점
- 샘플링 노이즈로 인해 학습/평가 분산이 커질 수 있다.
- seed 미고정 시 재현성 저하.
- 확률 매핑(`sigmoid`)의 스케일 민감도가 있어 데이터 정규화 상태에 성능이 좌우될 수 있다.

---

## 5. 공정 비교를 위한 실험 프로토콜 제안

두 모드 비교 시 아래를 동일하게 맞춘다.

1. **동일 split/dataloader**: 같은 train/val/test 분할 사용  
2. **동일 하이퍼파라미터**: LR, epoch, batch size, `spike_ts`, dropout 동일  
3. **동일 seed 세트**: 예) `[0,1,2,3,4]` 다중 seed 평균 비교  
4. **동일 평가 지표**: Accuracy, Macro-F1, confusion matrix, 학습시간  
5. **통계 요약**: 평균 ± 표준편차를 함께 보고

권장 보고 형식:
- 단일 최고 점수보다 **평균 성능 + 분산** 중심 비교
- 클래스 불균형이 있으므로 Accuracy 단독 해석 지양

---

## 6. 실행 방법(Colab / CLI)

아래처럼 `run_all_experiments.py --only`로 모드를 직접 선택해 실행한다.

```bash
python run_all_experiments.py --only original_02_residual_prenorm_constant
python run_all_experiments.py --only original_02_residual_prenorm_bernoulli
```

동일 조건 비교를 위해 두 실험을 같은 세션/같은 환경에서 연속 실행하는 것을 권장한다.

---

## 7. 해석 가이드

- **Bernoulli > Constant**: 입력 단계의 temporal encoding이 성능에 기여한 신호로 해석 가능
- **Bernoulli ≈ Constant**: 현재 스케일/노이즈 수준에서는 내부 GLIF 동역학만으로 충분할 가능성
- **Bernoulli < Constant**: 샘플링 노이즈 과다, sigmoid 확률 맵 불일치, seed 분산 증가 가능성 점검

추가 점검 포인트:
- sigmoid 입력 분포(포화 여부)
- epoch별 train/val gap
- 클래스별 성능 저하 패턴(특정 클래스만 흔들리는지)

---

## 8. 결론

실무 관점 권장 순서는 다음과 같다.
1. Constant를 안정 베이스라인으로 확보  
2. Bernoulli를 동일 조건/다중 seed로 비교  
3. 평균 성능 + 분산이 모두 개선될 때 채택

즉, Bernoulli는 “항상 더 좋다”가 아니라, **표현력 증가와 분산 증가의 교환관계**를 갖는 옵션으로 보는 것이 타당하다.
