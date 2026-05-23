# Label 정책

## 공식 문제 정의

주가 방향 예측 문제는 예측일 `t` 장마감 이후에 사용할 수 있는 정보로
향후 5거래일 수익률 방향을 예측하는 3-class 분류로 정의한다.

```text
prediction_date = t
future_5d_return = close[t+5] / close[t] - 1

target_5d_3class:
  0 = down
  1 = neutral
  2 = up
```

`close[t]`를 label 계산의 기준 가격으로 사용하므로, prediction timestamp는
반드시 `t` 장마감 이후로 해석한다. 장중 또는 장시작 전 예측 문제로 바꾸면
label 기준도 다시 정의해야 한다.

## Threshold 정책

고정 `±2%` threshold는 종목별 변동성 차이를 반영하지 못할 수 있으므로
공식 target에는 사용하지 않는다. 대신 종목별 train split의
`future_5d_return` 분포에서 threshold를 계산한다.

```text
down threshold = train future_5d_return 30% quantile
up threshold   = train future_5d_return 70% quantile

down     if future_5d_return <= down threshold
neutral  if down threshold < future_5d_return < up threshold
up       if future_5d_return >= up threshold
```

threshold 계산에는 validation, test, recent regime 데이터를 사용하지 않는다.
따라서 평가 구간의 미래 수익률 분포가 train threshold에 새어 들어가지 않는다.

## Split 정책

5거래일 미래 수익률 label은 인접 row끼리 label window가 겹친다. 이를 줄이기 위해
2018~2024 모델 구간은 시간 순서로 나누고, split 사이에는 5거래일 embargo를 둔다.

검증 조건은 다음과 같다.

```text
max(train future_5d_date) < min(validation date)
max(validation future_5d_date) < min(test date)
```

random split과 shuffled cross validation은 사용하지 않는다.

## Pending target

각 종목의 마지막 5개 거래일은 `close[t+5]`가 아직 없기 때문에
`target_5d_3class`를 비워 둔다. 이 row는 supervised 학습과 지표 계산에서 제외한다.
필요하면 별도의 live-style 예측 결과로만 해석한다.

## 모델 평가와 해석

3-class target에서는 accuracy만으로 성능을 판단하지 않는다. 기본 산출물에는
accuracy, balanced accuracy, macro F1, weighted F1, class별 precision/recall,
multiclass log loss, 예측 class별 평균 향후 5거래일 수익률, up/down long-short spread를
포함한다.

SHAP은 class별로 해석한다. 예를 들어 up class의 양수 SHAP 값은 해당 feature가
모델 출력을 up class 쪽으로 높였다는 뜻이지, 현실의 인과 효과를 증명한다는 뜻은 아니다.
