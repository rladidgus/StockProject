# Technical + extended XGBoost 성능 탐색

## 목적

Core macro-only와 단순 extended ablation만으로는 성능 개선이 제한적이어서,
기술적 지표 baseline에 extended 피처를 붙이고 XGBoost 파라미터까지 함께 탐색한다.

이 단계의 목적은 좋은 결과만 고르는 것이 아니라,
어떤 피처군이 실제 out-of-time/test 성능으로 이어지는지 확인하는 것이다.

## 실행

```bash
python3 -m src.technical_extended_search
```

## 탐색 범위

### Feature recipe

| recipe | 구성 |
|---|---|
| `technical_only` | 기술적 지표만 |
| `technical_core` | 기술적 지표 + core macro |
| `technical_rates` | 기술적 지표 + core macro + 금리 |
| `technical_dollar` | 기술적 지표 + core macro + broad dollar |
| `technical_semi_etf` | 기술적 지표 + core macro + SMH/SOXX |
| `technical_monthly_industry` | 기술적 지표 + core macro + 월간 업황 |
| `technical_domestic_alpha` | 기술적 지표 + core macro + 국내 수급 |
| `technical_daily_extended` | 기술적 지표 + core macro + 금리 + 달러 + 반도체 ETF |
| `technical_all_guarded` | 위 후보군 전체, lag/as-of 안전장치 적용 |

### Hyperparameter

가중치 적용 모델과 unweighted 모델을 함께 비교한다.
3-class에서 sample weight는 macro F1을 안정화할 수 있지만 accuracy를 낮출 수 있으므로
둘 다 탐색 대상으로 둔다.

## 산출물

```text
outputs/results/technical_extended_search/model_metrics.csv
outputs/results/technical_extended_search/best_by_recipe.csv
outputs/results/technical_extended_search/feature_sets.csv
outputs/results/technical_extended_search/search_summary.csv
outputs/results/technical_extended_search/selected_by_ticker.csv
```

## 결과 요약

Validation 기준 종목별 선택 결과는 다음과 같다.

| ticker | selected recipe | selected params | validation macro F1 |
|---|---|---|---:|
| `NVDA` | `technical_monthly_industry` | `baseline_depth3_unweighted` | 0.3498 |
| `005930` | `technical_only` | `depth3_regularized_unweighted` | 0.3158 |
| `000660` | `technical_only` | `depth3_regularized` | 0.3546 |

Test 기준으로 보면 extended를 붙인 모델이 항상 개선되지는 않았다.
특히 삼성전자와 SK하이닉스는 여전히 technical-only 계열이 강하다.
NVDA는 기존 기술적 baseline보다 tuned technical-only가 개선되었다.

기존 `technical_baseline_xgboost` test macro F1과 비교하면:

| ticker | 기존 technical baseline | 이번 탐색 best test |
|---|---:|---:|
| `NVDA` | 0.2563 | 0.2871 |
| `005930` | 0.4111 | 0.3944 |
| `000660` | 0.3572 | 0.3488 |

따라서 현재까지의 결론은 다음과 같다.

- Extended 피처를 무작정 붙인다고 test 성능이 오르지 않는다.
- 기술적 지표가 가장 강한 기준선이다.
- Extended는 validation에서는 좋아 보일 수 있지만 test/recent regime에서 쉽게 무너진다.
- 성능 개선은 단순 피처 추가보다 target 재정의, feature engineering, 모델/검증 설계 쪽에서 더 가능성이 크다.

## 다음 성능 개선 후보

1. `target_5d_3class` 외에 binary up/down 또는 neutral 제외 실험을 별도 산출물로 만든다.
2. 기술적 지표를 더 늘린다.
   - MACD
   - stochastic oscillator
   - ATR
   - volume moving average / volume shock
   - 5/10/20일 momentum
3. 종목별 feature recipe를 분리한다.
   - NVDA는 월간 업황 후보가 validation에서 강했다.
   - 국내 종목은 technical-only가 더 안정적이다.
4. XGBoost 외 모델을 비교한다.
   - RandomForest
   - ExtraTrees
   - LogisticRegression with scaling
5. 최근 regime은 별도 fine-tuning 대상이 아니라 out-of-time 검증으로 계속 유지한다.

## 도메인 안전장치

- 모델 선택은 validation 기준으로 수행한다.
- test/recent regime은 선택 이후 평가로만 사용한다.
- 월간 지표는 월말 이후 5영업일 뒤 known date로 보수 처리한다.
- 국내 수급은 lag feature로만 사용한다.
- test 기준 최고 조합은 oracle 참고용이며 최종 모델 선택 기준으로 쓰지 않는다.
