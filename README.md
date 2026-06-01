# BDA 10기 수료 예측 — 데이터 분석 & 모델링 파이프라인

> **한 줄 요약:** 9기(과거) 데이터로 학습해 10기(미래) 수강생의 **수료 여부**를 예측하는 이진 분류 문제.
> 핵심 난관은 **9기↔10기 분포 차이(Distribution Shift)** 였고, 이를 정면으로 다루는 *Adversarial Validation* 기반 파이프라인으로 해결했습니다.

---

## 📌 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [데이터 설명](#2-데이터-설명)
3. [핵심 문제: Distribution Shift](#3-핵심-문제-distribution-shift)
4. [전체 파이프라인 한눈에 보기](#4-전체-파이프라인-한눈에-보기)
5. [단계별 상세 설명](#5-단계별-상세-설명)
6. [모델 구성](#6-모델-구성)
7. [실행 방법](#7-실행-방법)
8. [결과물(Output)](#8-결과물output)
9. [파일 구조](#9-파일-구조)

---

## 1. 프로젝트 개요

비어플(BDA) 데이터 분석 학회의 **수료 예측** 대회 코드입니다.
중도 탈퇴가 아니라, **학습 과정을 끝까지 완료하여 '수료'에 도달한 학습자**를 맞히는 것이 목표입니다.

| 항목 | 내용 |
|------|------|
| **문제 유형** | 이진 분류 (Binary Classification) |
| **타깃 변수** | `completed` (0 = 미수료, 1 = 수료) |
| **평가 지표** | **Binary F1 Score** |
| **학습 데이터** | 748명 (9기) |
| **예측 대상** | 814명 (10기) |
| **핵심 챌린지** | 9기와 10기의 데이터 분포가 크게 다름 (Adversarial AUC ≈ 0.999) |

> 평가지표가 Accuracy가 아니라 **F1**이기 때문에, 단순 정확도보다 **양성(수료) 클래스의 정밀도/재현율 균형**과 **임계값(threshold) 최적화**가 성능을 크게 좌우합니다.

---

## 2. 데이터 설명

원본 데이터는 학회 가입 시점에 받은 **설문 응답** 기반입니다. (`train.csv`, `test.csv`)

> ⚠️ 데이터에 한글 + 콤마가 섞인 멀티셀렉트(다중 선택) 응답이 많아, Excel로 열면 깨집니다. 반드시 pandas 등으로 읽어야 합니다. (`encoding='utf-8-sig'`)

### 컬럼 분류

원본 컬럼은 의미상 6개 그룹으로 나뉩니다.

| 그룹 | 대표 컬럼 | 설명 |
|------|-----------|------|
| **① 학업/전공** | `school1`, `major type`, `major1_1`, `major1_2`, `major_data`, `major_field`, `completed_semester` | 대학, 전공, 데이터 전공자 여부, 이수 학기 |
| **② 참여 여건** | `job`, `time_input`, `re_registration`, `project_type`, `nationality`, `hope_for_group` | 직무, 하루 투입 가능 시간, 재등록 여부, 팀/개인 선호, 조별활동 희망 |
| **③ 진로/동기** | `inflow_route`, `whyBDA`, `what_to_gain`, `desired_career_path`, `desired_job` | 유입 경로, 가입 동기, 희망 진로/직무 |
| **④ 자격/준비도** | `certificate_acquisition`, `desired_certificate`, `desired_job_except_data` | 보유 자격증, 희망 자격증 |
| **⑤ 프로그램 기대** | `incumbents_level`, `incumbents_lecture`, `incumbents_company_level`, `incumbents_lecture_type`, `incumbents_lecture_scale`, `incumbents_lecture_scale_reason` | 현직자 강연에 대한 기대(연차/주제/회사 규모/온·오프라인/규모) |
| **⑥ 관심 분야** | `interested_company`, `expected_domain`, `onedayclass_topic` | 관심 기업, 희망 도메인, 원데이 클래스 주제 |

### 주요 특이사항

- **멀티셀렉트 컬럼**: `desired_job`, `certificate_acquisition`, `onedayclass_topic` 등은 `"A, B, C"` 형태의 콤마 구분 문자열 → 개수/플래그 피처로 분해 필요.
- **거의 비어있는 컬럼**: `contest_award`, `contest_participation`, `idea_contest`, `class3`, `class4`, `previous_class_3~9` 등은 대부분 결측 → 제거 또는 파생 피처 생성에만 활용.
- **자유 서술형 컬럼**: `incumbents_lecture_scale_reason`, `interested_company` → 길이·키워드 기반 피처로 변환.

---

## 3. 핵심 문제: Distribution Shift

이 프로젝트의 **가장 중요한 인사이트**입니다.

학습 데이터(9기)와 평가 데이터(10기)를 섞어놓고 "이 샘플이 9기냐 10기냐"를 맞히는 분류기를 학습시켜 보면 **AUC ≈ 0.999** 가 나옵니다.
즉, **9기와 10기는 거의 완벽히 구분될 만큼 분포가 다릅니다.**

이게 왜 위험한가?
- 모델이 9기 데이터에 최적화되면, 9기에서만 통하는 패턴(예: 특정 학교, 특정 전공 분포)을 학습 → 10기에서는 무력화.
- 9기/10기를 구분하는 데 유용한 피처일수록 **수료 예측에는 오히려 노이즈**.

그래서 파이프라인 전반에 **Adversarial Validation** 기법을 적용합니다.

1. **Adversarial Feature Removal** — 9기/10기를 너무 잘 구분하는 피처를 찾아 **제거**.
2. **Adversarial Sample Weighting** — 9기 샘플 중 **10기와 닮은 샘플에 더 큰 가중치**를 줘서, 모델이 "미래(10기)와 비슷한 과거"에 집중하도록 유도.

---

## 4. 전체 파이프라인 한눈에 보기

```
 train.csv (9기, 748명)              test.csv (10기, 814명)
        │                                   │
        └──────────────┬────────────────────┘
                       ▼
   [1] 전처리 & 피처 엔지니어링  (멀티셀렉트 분해, 결측/이상치 처리, 교차피처)
                       ▼
   [2] 인코딩  (Label Encoding + OOF Target Encoding)
                       ▼
   [2-0] Adversarial Feature Removal  ← 9기/10기 구분 피처 제거
                       ▼
   [2-1] 피처 선택  (분산 / 상관관계 / 중요도 / Permutation Importance)
                       ▼
   [2-2] Adversarial Sample Weighting  ← 10기와 닮은 9기 샘플 가중
                       ▼
   [3] Optuna 하이퍼파라미터 튜닝  (LGBM / XGB / CatBoost, 각 50 trials)
                       ▼
   [4] 멀티 시드 앙상블 학습  (5 seeds × 5 folds × 7 models)
                       ▼
   [5] 앙상블 최적화 & Threshold 탐색  (단순/가중/Top-3/Top-4/Stacking 비교)
                       ▼
   [5-2] Probability Calibration  (Isotonic Regression)
                       ▼
   [6] 최종 예측 + Positive Rate Cap → submission.csv
                       ▼
   [7] Feature Importance 리포트
```

---

## 5. 단계별 상세 설명

### [1] 전처리 & 피처 엔지니어링 (`feature_engineering`)

원본 설문 데이터를 모델이 학습 가능한 수치형으로 변환합니다. train/test에 **동일 함수**를 적용해 일관성을 보장합니다.

- **재등록자 파생**: `previous_class_*` 결측 개수를 세어 `prev_gen_count` (과거 참여 기수 수) 생성.
- **결측·이상치 처리**:
  - `completed_semester` → 14로 클리핑, 결측은 6.0으로 채움.
  - `time_input` → 10으로 클리핑, 결측은 2.0으로 채움.
- **Binary 변환**: `re_registration`(예/아니요), `major_data`, `is_foreign`(외국인 여부).
- **멀티셀렉트 → 카운트 + 플래그**: 콤마 구분 문자열을 분해해
  - 선택 **개수**(`desired_job_count`, `cert_count`, `topic_count` 등),
  - 특정 항목 **포함 여부 플래그**(`want_analyst`, `has_ADsP`, `topic_ml`, `field_it` 등) 생성.
- **자유 서술형 → 텍스트 통계**:
  - `interested_company` → 빅테크/대기업/글로벌 기업 키워드 플래그, 텍스트 길이, 기업 수.
  - `incumbents_lecture_scale_reason` → 글자 수, 단어 수, "긴 답변" 플래그(적극성 대리 지표).
- **인터랙션/교차 피처** (실험으로 효과 검증된 것들):
  - `re_reg_x_satisfied` (재등록 × 만족도), `time_x_rereg` (투입시간 × 재등록),
  - `lecture_type_online_flag` (온라인 선호 — 온라인 37% vs 오프라인 25% 수료율 차이),
  - `re_reg_x_online`, `time_x_online`, `cert_has_x_want` 등.

### [2] 인코딩

- **타깃 분리**: `completed`를 `y`로 분리.
- **Label Encoding**: 순서 무관 카테고리 컬럼(`major type`, `job`, `whyBDA` 등)을 train+test 합쳐서 fit → 미지의 카테고리 문제 방지.
- **Target Encoding (OOF 방식)**: 카디널리티가 높은 `school1`, `class1`은 타깃 평균으로 인코딩.
  - **CV leak 방지**를 위해 StratifiedKFold 기반 **Out-Of-Fold** 인코딩 + smoothing 적용.
  - test는 전체 train 통계 사용.

### [2-0] Adversarial Feature Removal

- `GradientBoostingClassifier`로 "9기 vs 10기" 분류기를 학습.
- Feature Importance가 높은 피처 = 두 기수를 잘 가르는 피처 = **수료 예측엔 해로움** → 제거.
- 추가로 분석에서 문제로 확인된 피처(`onedayclass_topic`, `major1_*`, `school1` 파생 등)도 제거.
- 단, 유용성이 검증된 **Target Encoding 피처(`school1_te`, `class1_te`)는 유지**.

### [2-1] 피처 선택 (4단계 필터링)

1. **분산 필터** — 거의 상수인 피처(variance < 1e-6) 제거.
2. **상관관계 필터** — 상관계수 0.95 초과 중복 피처 제거.
3. **중요도 필터** — 빠른 LightGBM으로 중요도 0인 피처 제거.
4. **Permutation Importance 필터** — RandomForest 기반으로 중요도가 음수/극히 낮은 **노이즈 피처** 제거.

### [2-2] Adversarial Sample Weighting

- 피처 선택 후 다시 "9기 vs 10기" 분류기를 학습.
- 각 9기 샘플이 **10기일 확률**을 계산 → `weight = p / (1-p)` 형태로 변환.
- `[0.1, 10.0]` 범위로 클리핑 후 정규화 → **10기와 닮은 9기 샘플일수록 큰 가중치**.
- 이후 모든 모델 학습 시 `sample_weight`로 전달.

### [3] Optuna 하이퍼파라미터 튜닝

- `LightGBM`, `XGBoost`, `CatBoost` 각각에 대해 **TPE Sampler로 50 trials** 탐색.
- 목적 함수: 5-fold CV의 **OOF F1 Score** (임계값 최적화 포함).
- 클래스 불균형 대응: `scale_pos_weight` / `auto_class_weights='Balanced'` 탐색 범위에 포함.

### [4] 멀티 시드 앙상블 학습

- **5개 시드 × 5-fold CV** 로 안정성 확보.
- **7개 모델** 각각에 대해 OOF 예측 + Test 예측 생성, 시드 평균으로 분산 감소.
- 모든 모델에 [2-2]의 **Adversarial Sample Weight** 적용.

### [5] 앙상블 최적화 & Threshold 탐색

여러 앙상블 전략을 만들어 **OOF F1 기준으로 최고를 자동 선택**합니다.

| 전략 | 설명 |
|------|------|
| `simple_avg` | 7개 모델 단순 평균 |
| `weighted_perf` | 모델별 F1에 비례한 가중 평균 |
| `top3` | 성능 상위 3개 모델 평균 |
| `weighted_opt` | 상위 4개 모델 가중치 Grid Search |
| `stacking` | **2-Level Stacking** (7개 모델 OOF → LogisticRegression 메타 모델) |

- **Threshold 최적화**: F1을 최대화하는 임계값을 탐색하되, 예측 양성 비율이 과도하게 높아지지 않도록 **Positive Rate Cap(0.70)** 제약을 둡니다.

### [5-2] Probability Calibration

- **Isotonic Regression**으로 예측 확률을 실제 수료율에 가깝게 보정.
- 보정 전/후 OOF F1을 비교해 **더 좋은 쪽만 채택**.

### [6] 최종 예측 & 제출

- 최종 확률에 OOF 최적 threshold 적용.
- 예측 양성 비율이 0.70을 넘으면 **확률 상위 70%만 1로 강제**(`apply_positive_rate_cap`).
- `submission.csv`(0/1) + `submission_prob.csv`(확률) 생성.

### [7] Feature Importance

- 마지막 LightGBM 모델의 중요도 상위 25개 피처 출력 → 어떤 변수가 수료를 가르는지 해석.

---

## 6. 모델 구성

| 모델 | 역할 | 특징 |
|------|------|------|
| **LightGBM ×3** | 주력 | 기본(Optuna) / 단순(과적합 방지) / DART(복잡·정규화) 3종 다양화 |
| **XGBoost** | 주력 | early stopping + scale_pos_weight |
| **CatBoost** | 주력 | `auto_class_weights='Balanced'` |
| **ExtraTrees** | 다양성 | 높은 랜덤성으로 앙상블 다양성 기여 |
| **LogisticRegression** | 다양성 | 선형 관점 + Level-2 메타 모델로도 사용 |

> **다양성(diversity)** 을 의도적으로 확보한 것이 포인트입니다. 비슷한 모델만 모으면 앙상블 효과가 적어, 트리/선형/부스팅 계열을 섞고 LGBM 내부에서도 성격이 다른 3종을 두었습니다.

---

## 7. 실행 방법

### 의존 패키지

```bash
pip install pandas numpy scikit-learn lightgbm xgboost catboost optuna
```

### 경로 설정

`0.44.py` 상단의 `BASE_PATH`를 본인 환경에 맞게 수정하세요.

```python
BASE_PATH = '/path/to/your/data'   # train.csv, test.csv, sample_submission.csv 가 있는 폴더
```

### 실행

```bash
python 0.44.py
```

> ⏱️ Optuna 튜닝(3개 모델 × 50 trials) + 멀티시드 앙상블(5×5×7) 때문에 실행에 시간이 다소 걸립니다.

---

## 8. 결과물(Output)

| 파일 | 내용 |
|------|------|
| `0.44_submission.csv` | 최종 제출 파일 (`ID`, `completed` 0/1) |
| `0.44_submission_prob.csv` | 확률 버전 (디버깅·분석용) |
| 콘솔 로그 | 단계별 F1, 선택된 앙상블 방법, threshold, Feature Importance Top 25 |

---

## 9. 파일 구조

```
.
├── train.csv                 # 원본 학습 데이터 (9기, 748명)
├── test.csv                  # 원본 평가 데이터 (10기, 814명)
├── train_info.txt            # 컬럼 정의서
├── 0.44.py                   # 메인 파이프라인 (이 README가 설명하는 코드)
├── 0.44_submission.csv       # 최종 예측 결과
└── 0.44_submission_prob.csv  # 확률 예측 결과
```

---

## 💡 이 프로젝트의 핵심 교훈

1. **분포 차이를 먼저 진단하기.** Adversarial Validation으로 train/test가 얼마나 다른지 측정하는 것이 출발점.
2. **해로운 피처는 강한 피처일 수 있다.** 9기/10기를 잘 가르는 피처는 오히려 일반화를 해친다.
3. **F1 대회는 threshold가 절반이다.** 좋은 확률을 뽑는 것만큼 임계값을 제대로 고르는 것이 중요.
4. **다양성 있는 앙상블 + 보정**이 소수 데이터에서 안정성을 만든다.
