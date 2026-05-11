"""
================================================================================
BDA 10기 수료 예측 파이프라인 (Distribution Shift 대응 버전)
================================================================================

개요
- 평가지표: Binary F1 Score
- 핵심 문제: 9기→10기 Distribution Shift (Adversarial AUC = 0.999)
- 데이터: Train 748명 (9기), Test 814명 (10기)

주요 기법
1. Adversarial Feature Detection & Removal (9기/10기 구분 피처 제거)
2. Adversarial Sample Weighting (10기와 유사한 9기 샘플에 높은 가중치)
3. 피처 선택 (분산, 상관관계, 중요도, Permutation Importance)
4. 전처리 개선 (lecture_type, company_count, 교차 피처)
5. 2-Level Stacking (7개 모델 → LR 메타 모델)
6. 가중치 앙상블 최적화
7. Probability Calibration

모델 구성
- LightGBM × 3 (기본, 단순, DART)
- XGBoost
- CatBoost
- ExtraTrees
- LogisticRegression

실행 흐름
0. 데이터 로드
1. 전처리 & 피처 엔지니어링
2. 인코딩 (Label Encoding, Target Encoding)
2-0. Adversarial Feature Detection & Removal
2-1. 피처 선택
2-2. Adversarial Sample Weighting
3. Optuna 하이퍼파라미터 튜닝
4. 멀티 시드 앙상블 학습 (5 seeds × 5 folds)
5. 최종 앙상블 & Threshold 최적화
6. 최종 예측 & 제출 파일 생성
7. Feature Importance 출력
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============================================================
# 0. 데이터 로드
# ============================================================
# 목적: train.csv, test.csv, sample_submission.csv 로드
# 출력: 데이터 shape, 수료율 확인

BASE_PATH = '/Users/kimsoyeong/Desktop/수료예측'

train = pd.read_csv(f'{BASE_PATH}/train.csv', encoding='utf-8-sig')
test = pd.read_csv(f'{BASE_PATH}/test.csv', encoding='utf-8-sig')
try:
    sample = pd.read_csv(f'{BASE_PATH}/sample_submission.csv', encoding='utf-8-sig')
except FileNotFoundError:
    sample = pd.read_csv(f'{BASE_PATH}/submission.csv', encoding='utf-8-sig')

print(f"Train: {train.shape}, Test: {test.shape}")
print(f"수료율: {train['completed'].mean():.3f}")

# ============================================================
# 1. 전처리 & 피처 엔지니어링
# ============================================================
# 목적: 원본 데이터를 모델이 학습할 수 있는 형태로 변환
# 주요 작업:
#   - 불필요 컬럼 제거
#   - 수치형 이상치 클리핑 & 결측 처리
#   - Binary 변환
#   - 멀티셀렉트 컬럼 → 파생 피처 (count, flag 등)
#   - 텍스트 피처 → 카테고리화
#   - 인터랙션/조합 피처 생성
#   - lecture_type 활용 (온라인 선호도)
#   - company_count 추가
#   - 교차 피처 생성

def feature_engineering(df):
    """
    train/test 공통 피처 엔지니어링
    
    Args:
        df: 입력 데이터프레임 (train 또는 test)
    
    Returns:
        전처리된 데이터프레임
    """
    df = df.copy()

    # ----------------------------------------------------------
    # 1-1. 재등록자 파생 피처 (제거 전 생성)
    # ----------------------------------------------------------
    prev_cols = [c for c in df.columns if c.startswith('previous_class_')]
    df['prev_gen_count'] = df[prev_cols].notna().sum(axis=1)

    # ----------------------------------------------------------
    # 1-2. 거의 전부 결측인 컬럼 제거
    # ----------------------------------------------------------
    drop_cols = ['ID', 'generation', 'contest_award', 'contest_participation',
                 'idea_contest', 'class3', 'class4'] + prev_cols
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    # ----------------------------------------------------------
    # 1-3. 수치형 이상치 클리핑 & 결측 처리
    # ----------------------------------------------------------
    if 'completed_semester' in df.columns:
        df['completed_semester'] = pd.to_numeric(df['completed_semester'], errors='coerce')
        df['completed_semester'] = df['completed_semester'].clip(upper=14)
        df['completed_semester'] = df['completed_semester'].fillna(6.0)

    if 'time_input' in df.columns:
        df['time_input'] = df['time_input'].clip(upper=10)
        df['time_input'] = df['time_input'].fillna(2.0)

    # class2: 값 유무 + 결측 처리
    if 'class2' in df.columns:
        df['has_class2'] = df['class2'].notna().astype(int)
        df['class2'] = df['class2'].fillna(0)

    # ----------------------------------------------------------
    # 1-4. Binary 변환
    # ----------------------------------------------------------
    df['re_registration'] = (df['re_registration'] == '예').astype(int)
    df['major_data'] = df['major_data'].astype(int)
    df['is_foreign'] = (df['nationality'] == '외국인').astype(int)

    # ----------------------------------------------------------
    # 1-5. 멀티셀렉트 컬럼 → 파생 피처
    # ----------------------------------------------------------

    # desired_job
    dj = df['desired_job'].fillna('')
    df['desired_job_count'] = dj.apply(lambda x: len([i for i in x.split(',') if i.strip()]) if x else 0)
    df['want_analyst'] = dj.str.contains('데이터 분석가', na=False).astype(int)
    df['want_scientist'] = dj.str.contains('데이터 사이언티스트', na=False).astype(int)
    df['want_engineer'] = dj.str.contains('데이터 엔지니어', na=False).astype(int)
    df['want_ai'] = dj.str.contains('인공지능', na=False).astype(int)
    df['want_marketer'] = dj.str.contains('마케터', na=False).astype(int)
    df['want_pm'] = dj.str.contains('PM|기획', na=False).astype(int)
    df['want_only_analyst'] = ((dj.str.strip() == 'B. 데이터 분석가')).astype(int)

    # certificate_acquisition
    ca = df['certificate_acquisition'].fillna('')
    df['cert_count'] = ca.apply(lambda x: len([i for i in x.split(',') if i.strip()]) if x and x != '없음' else 0)
    df['has_ADsP'] = ca.str.contains('ADsP', na=False).astype(int)
    df['has_SQLD'] = ca.str.contains('SQLD', na=False).astype(int)
    df['has_no_cert'] = (ca == '없음').astype(int)
    df['has_bigdata_cert'] = ca.str.contains('빅데이터', na=False).astype(int)
    df['has_info_cert'] = ca.str.contains('정보처리', na=False).astype(int)

    # desired_certificate
    dc = df['desired_certificate'].fillna('')
    df['desired_cert_count'] = dc.apply(lambda x: len([i for i in x.split(',') if i.strip()]) if x else 0)
    df['desire_ADsP'] = dc.str.contains('ADsP', na=False).astype(int)
    df['desire_SQLD'] = dc.str.contains('SQLD', na=False).astype(int)
    df['desire_bigdata'] = dc.str.contains('빅데이터', na=False).astype(int)

    # desired_job_except_data
    dje = df['desired_job_except_data'].fillna('')
    df['except_data_count'] = dje.apply(lambda x: len([i for i in x.split(',') if i.strip()]) if x else 0)
    df['want_finance'] = dje.str.contains('금융', na=False).astype(int)
    df['want_sw_dev'] = dje.str.contains('소프트웨어', na=False).astype(int)
    df['want_planning'] = dje.str.contains('기획|전략|경영', na=False).astype(int)
    df['want_research'] = dje.str.contains('연구', na=False).astype(int)

    # expected_domain
    ed = df['expected_domain'].fillna('')
    df['domain_count'] = ed.apply(lambda x: len([i for i in x.split(',') if i.strip()]) if x else 0)
    df['domain_it'] = ed.str.contains('정보통신', na=False).astype(int)
    df['domain_finance'] = ed.str.contains('금융', na=False).astype(int)
    df['domain_manufacture'] = ed.str.contains('제조', na=False).astype(int)
    df['domain_public'] = ed.str.contains('공공', na=False).astype(int)

    # onedayclass_topic
    oc = df['onedayclass_topic'].fillna('')
    df['topic_count'] = oc.apply(lambda x: len([i for i in x.split(',') if i.strip()]) if x else 0)
    df['topic_ml'] = oc.str.contains('머신러닝|딥러닝', na=False).astype(int)
    df['topic_python'] = oc.str.contains('Python', na=False).astype(int)
    df['topic_sql'] = oc.str.contains('SQL', na=False).astype(int)
    df['topic_viz'] = oc.str.contains('시각화', na=False).astype(int)
    df['topic_crawl'] = oc.str.contains('크롤링', na=False).astype(int)

    # major_field
    mf = df['major_field'].fillna('')
    df['field_count'] = mf.apply(lambda x: len([i for i in x.split(',') if i.strip()]) if x else 0)
    df['field_it'] = mf.str.contains('IT|컴퓨터', na=False).astype(int)
    df['field_engineering'] = mf.str.contains('공학', na=False).astype(int)
    df['field_business'] = mf.str.contains('경영', na=False).astype(int)
    df['field_natural_sci'] = mf.str.contains('자연과학', na=False).astype(int)
    df['field_social_sci'] = mf.str.contains('사회과학', na=False).astype(int)

    # ----------------------------------------------------------
    # 1-6. 텍스트 피처 → 간단 처리
    # ----------------------------------------------------------

    # interested_company 카테고리화
    ic = df['interested_company'].fillna('')
    df['company_bigtech_kr'] = ic.str.contains(
        '네이버|카카오|토스|라인|쿠팡|배달|당근', na=False).astype(int)
    df['company_big_corp_kr'] = ic.str.contains(
        '삼성|LG|SK|현대|롯데|포스코|CJ|한화', na=False).astype(int)
    df['company_global'] = ic.str.contains(
        '구글|Google|아마존|Amazon|마이크로|Microsoft|애플|Apple|메타|Meta|테슬라|엔비디아',
        na=False).astype(int)
    df['company_text_len'] = ic.str.len()
    df['company_none'] = ic.str.contains('없|모르|잘|^\\.$|^$', na=False).astype(int)
    # company_count 추가 (실험 결과 효과 큼)
    df['company_count'] = ic.apply(lambda x: len([i for i in x.split(',') if i.strip()]) if x and x not in ['없음', '없습니다', ''] else 0)

    # incumbents_lecture_scale_reason 텍스트 길이
    reason = df['incumbents_lecture_scale_reason'].fillna('')
    df['reason_len'] = reason.str.len()
    df['reason_word_count'] = reason.apply(lambda x: len(x.split()) if x else 0)
    # 긴 답변 = 적극적인 참여자?
    df['reason_long'] = (df['reason_len'] > 30).astype(int)

    # ----------------------------------------------------------
    # 1-7. 인터랙션/조합 피처
    # ----------------------------------------------------------
    df['re_reg_x_satisfied'] = (
        df['re_registration'] *
        (df.get('whyBDA', pd.Series([''] * len(df))) == '이전 기수에 매우 만족해서').astype(int)
    )
    df['total_select_count'] = (
        df['desired_job_count'] + df['desired_cert_count'] +
        df['except_data_count'] + df['domain_count'] + df['topic_count']
    )
    # 자격증 보유 수 대비 희망 수 비율
    df['cert_ratio'] = df['cert_count'] / (df['desired_cert_count'] + 1)
    # 투입 시간 * 재등록 = 강한 의지
    df['time_x_rereg'] = df['time_input'] * df['re_registration']
    # 오프라인 참여 의지
    df['offline_preference'] = (
        (df.get('hope_for_group', pd.Series([''] * len(df))) == '네. 오프라인으로 참여하고 싶어요').astype(int)
    )

    # ----------------------------------------------------------
    # 1-8. incumbents_lecture_type 활용 (실험 결과 효과 큼: +0.015)
    # 온라인 37% vs 오프라인 25% 수료율 차이
    # ----------------------------------------------------------
    if 'incumbents_lecture_type' in df.columns:
        ilt = df['incumbents_lecture_type'].fillna('')
        df['lecture_type_online'] = ilt.str.contains('온라인', na=False).astype(int)
        df['lecture_type_offline'] = ilt.str.contains('오프라인', na=False).astype(int)
        df['lecture_type_hybrid'] = ilt.str.contains('하이브리드|혼합', na=False).astype(int)
        # 온라인 선호도가 수료율과 높은 상관관계
        df['lecture_type_online_flag'] = (ilt.str.contains('온라인', na=False)).astype(int)

    # ----------------------------------------------------------
    # 1-9. 교차 피처 추가 (실험 결과 효과: +0.006)
    # ----------------------------------------------------------
    # 재등록 × 강의 타입
    if 'lecture_type_online_flag' in df.columns:
        df['re_reg_x_online'] = df['re_registration'] * df['lecture_type_online_flag']
    
    # 재등록 × 자격증 보유
    df['re_reg_x_cert'] = df['re_registration'] * (df['cert_count'] > 0).astype(int)
    
    # 시간 투입 × 강의 타입
    if 'lecture_type_online_flag' in df.columns:
        df['time_x_online'] = df['time_input'] * df['lecture_type_online_flag']
    
    # 자격증 보유 × 희망 자격증
    df['cert_has_x_want'] = (df['cert_count'] > 0).astype(int) * (df['desired_cert_count'] > 0).astype(int)

    return df


# 적용
train_ids = train['ID'].values
test_ids = test['ID'].values

train_fe = feature_engineering(train)
test_fe = feature_engineering(test)

# ============================================================
# 2. 인코딩
# ============================================================
# 목적: 카테고리 변수를 수치형으로 변환
# 주요 작업:
#   - 타겟 변수 분리
#   - Label Encoding (카테고리 변수)
#   - Target Encoding (school1, class1) - CV leak 방지
#   - 결측치 처리

# 타겟 분리
y = train_fe['completed'].values
if 'completed' in train_fe.columns:
    train_fe = train_fe.drop(columns=['completed'])
if 'completed' in test_fe.columns:
    test_fe = test_fe.drop(columns=['completed'])

# Label Encoding 할 카테고리 컬럼
cat_cols_label = [
    'major type', 'major1_1', 'major1_2', 'job',
    'inflow_route', 'whyBDA', 'what_to_gain', 'hope_for_group',
    'desired_career_path', 'project_type',
    'incumbents_level', 'incumbents_lecture',
    'incumbents_company_level', 'incumbents_lecture_type',
    'incumbents_lecture_scale'
]

# 원본 텍스트 컬럼 제거 (파생 피처로 대체했으므로)
# 주의: incumbents_lecture_type은 파생 피처를 만들었으므로 원본은 제거 가능
# 하지만 Label Encoding도 함께 사용할 수 있으므로 선택적으로 유지
text_cols_to_drop = [
    'nationality',  # is_foreign으로 대체
    'desired_job', 'certificate_acquisition', 'desired_certificate',
    'desired_job_except_data', 'expected_domain', 'onedayclass_topic',
    'interested_company', 'major_field',
    'incumbents_lecture_scale_reason'  # reason_len, reason_word_count로 대체
]
train_fe = train_fe.drop(columns=[c for c in text_cols_to_drop if c in train_fe.columns])
test_fe = test_fe.drop(columns=[c for c in text_cols_to_drop if c in test_fe.columns])

# Label Encoding (train + test 합쳐서 fit)
label_encoders = {}
for col in cat_cols_label:
    if col not in train_fe.columns:
        continue
    le = LabelEncoder()
    combined = pd.concat([train_fe[col], test_fe[col]], axis=0).fillna('__MISSING__').astype(str)
    le.fit(combined)
    train_fe[col] = le.transform(train_fe[col].fillna('__MISSING__').astype(str))
    test_fe[col] = le.transform(test_fe[col].fillna('__MISSING__').astype(str))
    label_encoders[col] = le

# Target Encoding with smoothing (CV leak 방지를 위해 OOF 방식)
def target_encode_cv(train_col, test_col, target, n_splits=5, smoothing=10, seed=42):
    """CV 기반 Target Encoding (train은 OOF, test는 전체 통계 사용)"""
    global_mean = target.mean()
    train_encoded = np.full(len(train_col), global_mean)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for tr_idx, val_idx in skf.split(train_col, target):
        stats = pd.DataFrame({'col': train_col.iloc[tr_idx], 'target': target[tr_idx]})
        agg = stats.groupby('col')['target'].agg(['mean', 'count'])
        smooth = (agg['count'] * agg['mean'] + smoothing * global_mean) / (agg['count'] + smoothing)
        train_encoded[val_idx] = train_col.iloc[val_idx].map(smooth).fillna(global_mean).values

    # test: 전체 train 통계 사용
    stats_all = pd.DataFrame({'col': train_col, 'target': target})
    agg_all = stats_all.groupby('col')['target'].agg(['mean', 'count'])
    smooth_all = (agg_all['count'] * agg_all['mean'] + smoothing * global_mean) / (agg_all['count'] + smoothing)
    test_encoded = test_col.map(smooth_all).fillna(global_mean)

    return train_encoded, test_encoded

# school1, class1 Target Encoding
train_fe['school1_te'], test_fe['school1_te'] = target_encode_cv(
    train_fe['school1'], test_fe['school1'], y
)
train_fe['class1_te'], test_fe['class1_te'] = target_encode_cv(
    train_fe['class1'], test_fe['class1'], y
)

# 남은 결측치 처리
train_fe = train_fe.fillna(-1)
test_fe = test_fe.fillna(-1)

# 최종 피처
feature_names = list(train_fe.columns)
X = train_fe.values.astype(np.float32)
X_test = test_fe.values.astype(np.float32)

print(f"\n초기 피처 수: {len(feature_names)}")

# ============================================================
# 2-0. Adversarial Feature Detection & Removal
# ============================================================
# 목적: 9기와 10기를 구분하는 피처 제거 (수료 예측에 방해됨)
# 방법: GradientBoosting으로 train/test 구분 모델 학습
#       → Importance가 높은 피처 = 9기/10기 구분에만 유용한 피처
#       → 이런 피처는 제거하여 일반화 성능 향상

print("\n" + "="*60)
print("Adversarial Feature Detection (9기↔10기 구분 피처)")
print("="*60)

# 9기와 10기를 구분하는 피처 찾기 (이런 피처는 수료 예측에 방해됨)
# GradientBoostingClassifier는 이미 import됨

X_combined = np.vstack([X, X_test])
y_is_test = np.array([0]*len(X) + [1]*len(X_test))

adv_model = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42, verbose=0)
adv_model.fit(X_combined, y_is_test)

# Adversarial Importance 계산
adv_importance = pd.DataFrame({
    'feature': feature_names,
    'adv_importance': adv_model.feature_importances_
}).sort_values('adv_importance', ascending=False)

print("Adversarial Importance Top 10:")
print(adv_importance.head(10).to_string(index=False))

# Adversarial Importance가 높은 피처 제거 (9기/10기 구분에만 유용)
# 상위 5% 제거 또는 threshold 기반
adv_threshold = adv_importance['adv_importance'].quantile(0.95)
high_adv_features = adv_importance[adv_importance['adv_importance'] > adv_threshold]['feature'].tolist()

# 문제 피처 강제 제거 (분석에서 확인된 피처들)
problem_features = ['onedayclass_topic', 'major1_1', 'major1_2', 'school1']
problem_features = [f for f in problem_features if f in feature_names]

# 파생 피처 중에서도 문제가 될 수 있는 것들
problem_derived = [f for f in feature_names if any(pf in f for pf in ['topic_', 'major1_', 'school1'])]
# school1_te, class1_te는 유지 (Target Encoding은 유용)

# 제거할 피처 합치기
to_remove = list(set(high_adv_features[:5] + problem_derived))  # 상위 5개 + 파생 피처
to_remove = [f for f in to_remove if f not in ['school1_te', 'class1_te']]  # TE는 유지

if to_remove:
    print(f"\n제거할 Adversarial 피처: {len(to_remove)}개")
    print(f"  {to_remove[:10]}")
    keep_features = [f for f in feature_names if f not in to_remove]
    feature_names = keep_features
    X = train_fe[feature_names].values.astype(np.float32)
    X_test = test_fe[feature_names].values.astype(np.float32)
    print(f"제거 후 피처 수: {len(feature_names)}")

# ============================================================
# 2-1. 피처 선택
# ============================================================
# 목적: 노이즈 피처 제거, 일반화 성능 향상
# 방법:
#   1. 분산 낮은 피처 제거 (거의 상수)
#   2. 높은 상관관계 피처 제거 (중복 제거)
#   3. Feature Importance 기반 제거 (중요도 0)
#   4. Permutation Importance 기반 제거 (음수 또는 매우 낮은 중요도)

print("\n" + "="*60)
print("피처 선택")
print("="*60)

# 1. 분산이 매우 낮은 피처 제거 (거의 상수)
feature_variance = pd.DataFrame({
    'feature': feature_names,
    'variance': np.var(X, axis=0)
})
low_var_features = feature_variance[feature_variance['variance'] < 1e-6]['feature'].tolist()
if low_var_features:
    print(f"분산 낮은 피처 제거: {len(low_var_features)}개")
    keep_features = [f for f in feature_names if f not in low_var_features]
    feature_names = keep_features
    X = train_fe[feature_names].values.astype(np.float32)
    X_test = test_fe[feature_names].values.astype(np.float32)

# 2. 높은 상관관계 피처 제거
print("상관관계 기반 피처 제거 중...")
train_df = pd.DataFrame(X, columns=feature_names)
corr_matrix = train_df.corr().abs()
upper_triangle = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)
high_corr_pairs = []
for col in upper_triangle.columns:
    high_corr = upper_triangle.index[upper_triangle[col] > 0.95].tolist()
    if high_corr:
        for hc in high_corr:
            high_corr_pairs.append((col, hc))

# 상관관계 높은 피처 중 하나 제거 (중요도 낮은 것)
if high_corr_pairs:
    print(f"높은 상관관계 피처 쌍 발견: {len(high_corr_pairs)}개")
    # 간단히 첫 번째 피처 제거 (실제로는 중요도 기반으로 선택)
    to_remove = set()
    for f1, f2 in high_corr_pairs[:10]:  # 최대 10개만 제거
        to_remove.add(f2)
    if to_remove:
        keep_features = [f for f in feature_names if f not in to_remove]
        feature_names = keep_features
        X = train_fe[feature_names].values.astype(np.float32)
        X_test = test_fe[feature_names].values.astype(np.float32)
        print(f"제거된 피처: {len(to_remove)}개")

# 3. Feature Importance 기반 피처 선택 (빠른 LightGBM으로)
print("Feature Importance 기반 피처 선택 중...")
quick_lgb = lgb.LGBMClassifier(
    n_estimators=100, learning_rate=0.1, random_state=42, verbosity=-1
)
quick_lgb.fit(X, y)
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': quick_lgb.feature_importances_
}).sort_values('importance', ascending=False)

# 중요도가 0인 피처 제거
zero_importance = feature_importance[feature_importance['importance'] == 0]['feature'].tolist()
if zero_importance:
    keep_features = [f for f in feature_names if f not in zero_importance]
    feature_names = keep_features
    X = train_fe[feature_names].values.astype(np.float32)
    X_test = test_fe[feature_names].values.astype(np.float32)
    print(f"중요도 0인 피처 제거: {len(zero_importance)}개")

# 4. Permutation Importance로 음수 피처 제거 (노이즈 제거)
print("Permutation Importance 계산 중... (노이즈 피처 제거)")
try:
    # 빠른 RandomForest로 Permutation Importance 계산
    rf_perm = RandomForestClassifier(
        n_estimators=100, max_depth=5, class_weight='balanced',
        random_state=42, n_jobs=-1
    )
    rf_perm.fit(X, y)
    
    # Permutation Importance 계산 (시간이 걸리므로 샘플링)
    perm_result = permutation_importance(
        rf_perm, X[:500], y[:500], n_repeats=5, random_state=42, n_jobs=-1
    )
    
    perm_importance = pd.DataFrame({
        'feature': feature_names,
        'perm_importance': perm_result.importances_mean
    }).sort_values('perm_importance', ascending=False)
    
    # 음수 또는 매우 낮은 중요도 피처 제거
    negative_features = perm_importance[perm_importance['perm_importance'] < 0]['feature'].tolist()
    very_low_features = perm_importance[perm_importance['perm_importance'] < 0.001]['feature'].tolist()
    noise_features = list(set(negative_features + very_low_features))
    
    if noise_features:
        print(f"노이즈 피처 제거: {len(noise_features)}개 (음수 또는 매우 낮은 중요도)")
        print(f"  예시: {noise_features[:5]}")
        keep_features = [f for f in feature_names if f not in noise_features]
        feature_names = keep_features
        X = train_fe[feature_names].values.astype(np.float32)
        X_test = test_fe[feature_names].values.astype(np.float32)
        print(f"제거 후 피처 수: {len(feature_names)}")
except Exception as e:
    print(f"Permutation Importance 계산 실패 (스킵): {e}")

print(f"최종 피처 수: {len(feature_names)}")

# ============================================================
# 2-2. Adversarial Sample Weighting (피처 선택 후)
# ============================================================
# 목적: 10기와 유사한 특성을 가진 9기 샘플에 높은 가중치 부여
# 효과: Distribution Shift 대응, 일반화 성능 향상
# 방법:
#   1. 선택된 피처로 9기/10기 구분 모델 재학습
#   2. 각 9기 샘플이 10기일 확률 계산
#   3. 확률이 높은 샘플(10기와 유사)에 높은 가중치 부여
#   4. 모든 모델 학습 시 sample_weight 적용

print("\n" + "="*60)
print("Adversarial Sample Weighting (10기와 유사한 9기 샘플에 높은 가중치)")
print("="*60)

# 선택된 피처로 9기/10기 구분 모델 재학습
X_combined_final = np.vstack([X, X_test])
y_is_test = np.array([0]*len(X) + [1]*len(X_test))

adv_model2 = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42, verbose=0)
adv_model2.fit(X_combined_final, y_is_test)
train_proba = adv_model2.predict_proba(X)[:, 1]  # 각 9기 샘플이 10기일 확률

# test와 유사한 train 샘플에 높은 가중치
sample_weights = train_proba / (1 - train_proba + 1e-6)
sample_weights = np.clip(sample_weights, 0.1, 10.0)  # 극단값 제한
sample_weights = sample_weights / sample_weights.mean()  # 정규화

print(f"Sample weights 통계: min={sample_weights.min():.3f}, max={sample_weights.max():.3f}, mean={sample_weights.mean():.3f}")

# ============================================================
# 3. Optuna 하이퍼파라미터 튜닝
# ============================================================
# 목적: 각 모델의 최적 하이퍼파라미터 탐색
# 방법: Optuna TPE Sampler로 50 trials 탐색
# 평가: 5-fold CV의 OOF F1 Score 최대화
# 모델: LightGBM, XGBoost, CatBoost

print("\n" + "="*60)
print("Optuna 하이퍼파라미터 튜닝")
print("="*60)

N_SPLITS = 5  # 교차 검증 fold 수

def find_best_threshold(y_true, y_prob, pos_cap=0.70, step=0.005):
    """
    F1 최적화 + Positive Rate Cap 제약
    pos_cap: 예측 1 비율 상한 (기본 0.70)
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    
    best_f1, best_t = -1.0, 0.5
    best_pos = None
    
    for t in np.arange(0.0, 1.0 + 1e-12, step):
        y_pred = (y_prob >= t).astype(int)
        pos_rate = float(y_pred.mean())
        
        # pos_cap 제약 확인
        if pos_rate > pos_cap:
            continue
        
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if (f1 > best_f1) or (f1 == best_f1 and (best_pos is None or pos_rate < best_pos)):
            best_f1 = f1
            best_t = float(t)
            best_pos = pos_rate
    
    # cap 조건을 만족하는 thr이 없으면 cap 무시하고 최고 F1 선택
    if best_f1 < 0:
        best_t = 0.5
        best_f1 = f1_score(y_true, (y_prob >= 0.5).astype(int), zero_division=0)
        best_pos = float((y_prob >= 0.5).mean())
    
    return best_t, best_f1, best_pos

def apply_positive_rate_cap(y_prob, base_thr, pos_cap=0.70):
    """
    base_thr로 예측했을 때 1 비율이 pos_cap 초과하면,
    확률 상위 pos_cap 비율만 1로 강제
    """
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= base_thr).astype(int)
    pos_rate = float(y_pred.mean())
    
    if pos_rate <= pos_cap:
        return y_pred, base_thr, pos_rate
    
    n = len(y_prob)
    k = int(np.floor(pos_cap * n))
    
    order = np.argsort(-y_prob)  # descending
    y_pred2 = np.zeros(n, dtype=int)
    if k > 0:
        y_pred2[order[:k]] = 1
    
    effective_thr = float(y_prob[order[k-1]]) if k > 0 else 1.0
    return y_pred2, effective_thr, float(y_pred2.mean())


# --- LightGBM 튜닝 ---
def lgb_objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'n_estimators': 2000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'num_leaves': trial.suggest_int('num_leaves', 8, 64),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 60),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 10, log=True),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.5, 4.0),
        'random_state': 42,
    }
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    oof = np.zeros(len(X))
    for tr_idx, val_idx in skf.split(X, y):
        model = lgb.LGBMClassifier(**params)
        model.fit(X[tr_idx], y[tr_idx], eval_set=[(X[val_idx], y[val_idx])],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        oof[val_idx] = model.predict_proba(X[val_idx])[:, 1]
    _, best_f1, _ = find_best_threshold(y, oof, pos_cap=0.70)
    return best_f1

print("LightGBM 튜닝 중...")
lgb_study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
lgb_study.optimize(lgb_objective, n_trials=50, show_progress_bar=False)
print(f"  Best F1: {lgb_study.best_value:.4f}")

# --- XGBoost 튜닝 ---
def xgb_objective(trial):
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'n_estimators': 2000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 10, log=True),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.5, 4.0),
        'random_state': 42,
        'verbosity': 0,
        'early_stopping_rounds': 50,
    }
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    oof = np.zeros(len(X))
    for tr_idx, val_idx in skf.split(X, y):
        model = xgb.XGBClassifier(**params)
        model.fit(X[tr_idx], y[tr_idx], eval_set=[(X[val_idx], y[val_idx])], verbose=False)
        oof[val_idx] = model.predict_proba(X[val_idx])[:, 1]
    _, best_f1, _ = find_best_threshold(y, oof, pos_cap=0.70)
    return best_f1

print("XGBoost 튜닝 중...")
xgb_study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
xgb_study.optimize(xgb_objective, n_trials=50, show_progress_bar=False)
print(f"  Best F1: {xgb_study.best_value:.4f}")

# --- CatBoost 튜닝 ---
def cat_objective(trial):
    params = {
        'iterations': 2000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15),
        'depth': trial.suggest_int('depth', 4, 8),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 10, log=True),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 50),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 3),
        'random_strength': trial.suggest_float('random_strength', 0.1, 10, log=True),
        'auto_class_weights': 'Balanced',
        'eval_metric': 'Logloss',
        'random_seed': 42,
        'verbose': 0,
        'early_stopping_rounds': 50,
    }
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    oof = np.zeros(len(X))
    for tr_idx, val_idx in skf.split(X, y):
        model = CatBoostClassifier(**params)
        model.fit(X[tr_idx], y[tr_idx], eval_set=(X[val_idx], y[val_idx]), verbose=0)
        oof[val_idx] = model.predict_proba(X[val_idx])[:, 1]
    _, best_f1, _ = find_best_threshold(y, oof, pos_cap=0.70)
    return best_f1

print("CatBoost 튜닝 중...")
cat_study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
cat_study.optimize(cat_objective, n_trials=50, show_progress_bar=False)
print(f"  Best F1: {cat_study.best_value:.4f}")

# ============================================================
# 4. 멀티 시드 앙상블 학습 (다양한 모델)
# ============================================================
# 목적: 다양한 모델과 시드로 안정적인 앙상블 구축
# 구성:
#   - 5개 시드 × 5-fold CV = 25개 모델 per 타입
#   - 7개 모델 타입 (LGB×3, XGB, CB, ET, LR)
#   - 각 모델에 Adversarial Sample Weighting 적용
# 출력: 각 모델의 OOF 예측값과 Test 예측값

print("\n" + "="*60)
print("멀티 시드 앙상블 학습 (다양한 모델)")
print("="*60)

SEEDS = [42, 123, 456, 789, 2024]  # 다양한 시드로 안정성 확보

# 다양한 모델의 OOF와 예측 저장
all_oof_models = {
    'lgb1': np.zeros(len(X)),  # 기본 LGB
    'lgb2': np.zeros(len(X)),  # 단순 LGB
    'lgb3': np.zeros(len(X)),  # 복잡 LGB
    'xgb': np.zeros(len(X)),
    'cat': np.zeros(len(X)),
    'et': np.zeros(len(X)),    # ExtraTrees
    'lr': np.zeros(len(X))     # LogisticRegression
}
all_pred_models = {k: np.zeros(len(X_test)) for k in all_oof_models.keys()}

# Feature Importance를 위한 마지막 모델 저장
last_lgb_model = None

for seed_idx, seed in enumerate(SEEDS):
    print(f"\n=== Seed {seed} ({seed_idx+1}/{len(SEEDS)}) ===")
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

    oof_models = {k: np.zeros(len(X)) for k in all_oof_models.keys()}
    pred_models = {k: np.zeros(len(X_test)) for k in all_oof_models.keys()}

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        w_tr = sample_weights[tr_idx]  # Adversarial Sample Weighting

        # LightGBM 1 (기본 - Optuna 최적화)
        lgb_params1 = {
            'objective': 'binary', 'metric': 'binary_logloss', 'verbosity': -1,
            'n_estimators': 2000, 'random_state': seed,
            **lgb_study.best_params
        }
        m_lgb1 = lgb.LGBMClassifier(**lgb_params1)
        m_lgb1.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_val, y_val)],
                   callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        oof_models['lgb1'][val_idx] = m_lgb1.predict_proba(X_val)[:, 1]
        pred_models['lgb1'] += m_lgb1.predict_proba(X_test)[:, 1] / N_SPLITS

        # LightGBM 2 (단순 - 과적합 방지)
        lgb_params2 = {
            'objective': 'binary', 'metric': 'binary_logloss', 'verbosity': -1,
            'n_estimators': 2000, 'random_state': seed,
            'num_leaves': 15, 'max_depth': 4, 'learning_rate': 0.05,
            'min_child_samples': 30, 'subsample': 0.7, 'colsample_bytree': 0.7,
            'reg_alpha': 1.0, 'reg_lambda': 1.0
        }
        m_lgb2 = lgb.LGBMClassifier(**lgb_params2)
        m_lgb2.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_val, y_val)],
                   callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        oof_models['lgb2'][val_idx] = m_lgb2.predict_proba(X_val)[:, 1]
        pred_models['lgb2'] += m_lgb2.predict_proba(X_test)[:, 1] / N_SPLITS

        # LightGBM 3 (복잡 - DART)
        lgb_params3 = {
            'objective': 'binary', 'metric': 'binary_logloss', 'verbosity': -1,
            'boosting_type': 'dart', 'n_estimators': 2000, 'random_state': seed,
            'num_leaves': 63, 'max_depth': 8, 'learning_rate': 0.01,
            'min_child_samples': 10, 'subsample': 0.8, 'colsample_bytree': 0.8,
            'reg_alpha': 0.1, 'reg_lambda': 0.1, 'drop_rate': 0.1
        }
        m_lgb3 = lgb.LGBMClassifier(**lgb_params3)
        m_lgb3.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_val, y_val)],
                   callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        oof_models['lgb3'][val_idx] = m_lgb3.predict_proba(X_val)[:, 1]
        pred_models['lgb3'] += m_lgb3.predict_proba(X_test)[:, 1] / N_SPLITS

        # XGBoost
        xgb_params = {
            'objective': 'binary:logistic', 'eval_metric': 'logloss',
            'n_estimators': 2000, 'random_state': seed, 'verbosity': 0,
            'early_stopping_rounds': 50,
            **xgb_study.best_params
        }
        m_xgb = xgb.XGBClassifier(**xgb_params)
        m_xgb.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_val, y_val)], verbose=False)
        oof_models['xgb'][val_idx] = m_xgb.predict_proba(X_val)[:, 1]
        pred_models['xgb'] += m_xgb.predict_proba(X_test)[:, 1] / N_SPLITS

        # CatBoost
        cat_params = {
            'iterations': 2000, 'auto_class_weights': 'Balanced',
            'eval_metric': 'Logloss', 'random_seed': seed, 'verbose': 0,
            'early_stopping_rounds': 50,
            **cat_study.best_params
        }
        m_cat = CatBoostClassifier(**cat_params)
        m_cat.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=(X_val, y_val), verbose=0)
        oof_models['cat'][val_idx] = m_cat.predict_proba(X_val)[:, 1]
        pred_models['cat'] += m_cat.predict_proba(X_test)[:, 1] / N_SPLITS

        # ExtraTrees (랜덤성 높음)
        m_et = ExtraTreesClassifier(
            n_estimators=200, max_depth=8, min_samples_split=10,
            min_samples_leaf=5, class_weight='balanced', random_state=seed, n_jobs=-1
        )
        m_et.fit(X_tr, y_tr, sample_weight=w_tr)
        oof_models['et'][val_idx] = m_et.predict_proba(X_val)[:, 1]
        pred_models['et'] += m_et.predict_proba(X_test)[:, 1] / N_SPLITS

        # LogisticRegression (선형 모델)
        m_lr = LogisticRegression(
            C=0.1, max_iter=1000, class_weight='balanced', random_state=seed, solver='lbfgs'
        )
        m_lr.fit(X_tr, y_tr, sample_weight=w_tr)
        oof_models['lr'][val_idx] = m_lr.predict_proba(X_val)[:, 1]
        pred_models['lr'] += m_lr.predict_proba(X_test)[:, 1] / N_SPLITS

        # 마지막 모델 저장 (Feature Importance용)
        if seed_idx == len(SEEDS) - 1 and fold == N_SPLITS - 1:
            last_lgb_model = m_lgb1

    # Seed별 평균 누적
    for k in all_oof_models.keys():
        all_oof_models[k] += oof_models[k] / len(SEEDS)
        all_pred_models[k] += pred_models[k] / len(SEEDS)

    # Seed별 결과
    for name, oof in oof_models.items():
        _, f1, _ = find_best_threshold(y, oof, pos_cap=0.70)
        print(f"  {name.upper():4s} F1: {f1:.4f}")

# ============================================================
# 5. 최종 앙상블 & Threshold 최적화
# ============================================================
# 목적: 여러 앙상블 방법 중 최고 성능 선택
# 방법:
#   1. 단순 평균 (모든 모델)
#   2. 성능 기반 가중치 (F1 점수에 비례)
#   3. 상위 3개 모델 평균
#   4. 최적 가중치 탐색 (Grid Search, 상위 4개 모델)
#   5. 2-Level Stacking (Level 1: 7개 모델 → Level 2: LR)
# 평가: OOF F1 Score 기준 최고 성능 선택

print("\n" + "="*60)
print("최종 앙상블 최적화")
print("="*60)

# 개별 모델 결과
print("\n개별 모델 성능:")
for name, oof in all_oof_models.items():
    t, f1, pos_rate = find_best_threshold(y, oof, pos_cap=0.70, step=0.005)
    print(f"  {name:6s} | F1={f1:.4f} @ threshold={t:.3f} (pos_rate={pos_rate:.3f})")

# 단순 평균 앙상블 (모든 모델)
oof_simple = np.mean(list(all_oof_models.values()), axis=0)
t_simple, f1_simple, pos_simple = find_best_threshold(y, oof_simple, pos_cap=0.70, step=0.01)
print(f"\n단순 평균 (모든 모델): F1={f1_simple:.4f} @ threshold={t_simple:.3f} (pos_rate={pos_simple:.3f})")

# ============================================================
# 5-0. 가중치 앙상블 최적화 (성능 기반)
# ============================================================
# 목적: 모델별 성능에 따라 가중치 부여
# 방법:
#   1. 성능 기반 가중치 (F1 점수에 비례)
#   2. 상위 3개 모델만 앙상블
#   3. 최적 가중치 Grid Search (상위 4개 모델)

print("\n" + "="*60)
print("가중치 앙상블 최적화")
print("="*60)

# 모델별 성능 계산
model_performances = {}
for name, oof in all_oof_models.items():
    t, f1, _ = find_best_threshold(y, oof, pos_cap=0.70, step=0.01)
    model_performances[name] = f1

# 성능 기반 가중치 (F1 점수에 비례)
perf_weights = np.array([model_performances[k] for k in all_oof_models.keys()])
perf_weights = perf_weights / perf_weights.sum()  # 정규화

print("성능 기반 가중치:")
for name, w in zip(all_oof_models.keys(), perf_weights):
    print(f"  {name:6s}: {w:.3f} (F1={model_performances[name]:.4f})")

# 성능 기반 가중치 앙상블
oof_weighted_perf = np.zeros(len(y))
pred_weighted_perf = np.zeros(len(X_test))
for i, (name, oof) in enumerate(all_oof_models.items()):
    oof_weighted_perf += oof * perf_weights[i]
    pred_weighted_perf += all_pred_models[name] * perf_weights[i]

t_weighted_perf, f1_weighted_perf, pos_weighted_perf = find_best_threshold(
    y, oof_weighted_perf, pos_cap=0.70, step=0.01
)
print(f"\n성능 기반 가중치 앙상블: F1={f1_weighted_perf:.4f} @ threshold={t_weighted_perf:.3f} (pos_rate={pos_weighted_perf:.3f})")

# 상위 3개 모델만 앙상블
top3_models = sorted(model_performances.items(), key=lambda x: x[1], reverse=True)[:3]
top3_names = [name for name, _ in top3_models]
print(f"\n상위 3개 모델: {top3_names}")

oof_top3 = np.mean([all_oof_models[name] for name in top3_names], axis=0)
pred_top3 = np.mean([all_pred_models[name] for name in top3_names], axis=0)
t_top3, f1_top3, pos_top3 = find_best_threshold(y, oof_top3, pos_cap=0.70, step=0.01)
print(f"상위 3개 모델 평균: F1={f1_top3:.4f} @ threshold={t_top3:.3f} (pos_rate={pos_top3:.3f})")

# 최적 가중치 탐색 (상위 모델만 사용하여 간소화)
print("\n최적 가중치 탐색 중... (상위 4개 모델만)")
best_f1_weighted = 0
best_weights = None
best_threshold_weighted = 0.5

# 상위 4개 모델만 선택
top4_models = sorted(model_performances.items(), key=lambda x: x[1], reverse=True)[:4]
top4_names = [name for name, _ in top4_models]

# 상위 4개 모델로만 가중치 탐색
for w1 in np.arange(0.0, 1.01, 0.15):
    for w2 in np.arange(0.0, 1.01 - w1, 0.15):
        for w3 in np.arange(0.0, 1.01 - w1 - w2, 0.15):
            w4 = 1.0 - w1 - w2 - w3
            if w4 < -0.01:
                continue
            w4 = max(w4, 0)
            weights = np.array([w1, w2, w3, w4])
            weights = weights / weights.sum()  # 정규화
            
            oof_ens = np.zeros(len(y))
            for i, name in enumerate(top4_names):
                oof_ens += all_oof_models[name] * weights[i]
            
            t, f1, _ = find_best_threshold(y, oof_ens, pos_cap=0.70, step=0.01)
            if f1 > best_f1_weighted:
                best_f1_weighted = f1
                best_weights = {name: weights[i] for i, name in enumerate(top4_names)}
                best_threshold_weighted = t

if best_weights is not None:
    print(f"\n최적 가중치 앙상블 (상위 4개): F1={best_f1_weighted:.4f} @ threshold={best_threshold_weighted:.3f}")
    print("가중치:")
    for name, w in best_weights.items():
        print(f"  {name:6s}: {w:.3f}")
    
    pred_weighted_opt = np.zeros(len(X_test))
    for name, w in best_weights.items():
        pred_weighted_opt += all_pred_models[name] * w
    
    # 전체 모델에 대해 확장 (나머지는 0)
    oof_weighted_opt = np.zeros(len(y))
    for name in all_oof_models.keys():
        if name in best_weights:
            oof_weighted_opt += all_oof_models[name] * best_weights[name]
        # 나머지 모델은 가중치 0
else:
    pred_weighted_opt = None
    oof_weighted_opt = None


# ============================================================
# 5-1. 2-Level Stacking (다양한 모델 활용)
# ============================================================
# 목적: Level 1 모델들의 예측을 피처로 사용하는 메타 모델 학습
# 구조:
#   - Level 1: 7개 모델의 OOF 예측값 (다양성 확보)
#   - Level 2: LogisticRegression 메타 모델 (과적합 방지)
# 효과: 모델 간 상호작용 포착, 일반화 성능 향상

print("\n" + "="*60)
print("2-Level Stacking (Level 1: 다양한 모델 → Level 2: LR/Ridge)")
print("="*60)

# Level 1 모델들의 OOF 예측을 피처로 사용
meta_features_train = np.column_stack(list(all_oof_models.values()))
meta_features_test = np.column_stack(list(all_pred_models.values()))

# 교차 검증으로 메타 모델 학습
skf_meta = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
meta_oof = np.zeros(len(y))
meta_pred = np.zeros(len(X_test))

# Level 2: LogisticRegression 또는 Ridge
for tr_idx, val_idx in skf_meta.split(meta_features_train, y):
    # LogisticRegression 메타 모델
    meta_model = LogisticRegression(
        C=0.1, max_iter=1000, random_state=42, solver='lbfgs', class_weight='balanced'
    )
    meta_model.fit(meta_features_train[tr_idx], y[tr_idx])
    meta_oof[val_idx] = meta_model.predict_proba(meta_features_train[val_idx])[:, 1]
    meta_pred += meta_model.predict_proba(meta_features_test)[:, 1] / 5

# 메타 모델 결과
t_meta, f1_meta, pos_meta = find_best_threshold(y, meta_oof, pos_cap=0.70, step=0.01)
print(f"Stacking (LR meta): F1={f1_meta:.4f} @ threshold={t_meta:.3f} (pos_rate={pos_meta:.3f})")

# 최종 선택: 모든 앙상블 방법 비교
ensemble_results = {
    'simple_avg': (f1_simple, t_simple, oof_simple, np.mean(list(all_pred_models.values()), axis=0)),
    'weighted_perf': (f1_weighted_perf, t_weighted_perf, oof_weighted_perf, pred_weighted_perf),
    'top3': (f1_top3, t_top3, oof_top3, pred_top3),
    'stacking': (f1_meta, t_meta, meta_oof, meta_pred),
}

if best_weights is not None and oof_weighted_opt is not None:
    ensemble_results['weighted_opt'] = (
        best_f1_weighted, best_threshold_weighted,
        oof_weighted_opt,
        pred_weighted_opt
    )

print("\n" + "="*60)
print("앙상블 방법 비교")
print("="*60)
for method, (f1, t, _, _) in ensemble_results.items():
    print(f"  {method:15s}: F1={f1:.4f} @ threshold={t:.3f}")

# 최고 성능 선택
best_method = max(ensemble_results.items(), key=lambda x: x[1][0])
best_f1_final, best_t_final, base_oof, base_pred_prob = best_method[1]

print(f"\n최종 선택: {best_method[0]} (F1: {best_f1_final:.4f})")
use_stacking = (best_method[0] == 'stacking')
base_threshold = best_t_final

# ============================================================
# 5-2. Probability Calibration (확률 보정)
# ============================================================
# 목적: 모델의 확률 예측을 보정하여 threshold 탐색 정확도 향상
# 방법: Isotonic Regression으로 확률 보정
# 효과: 확률이 실제 수료율과 더 잘 일치하도록 조정

print("\n" + "="*60)
print("Probability Calibration (확률 보정)")
print("="*60)

# Calibration을 위한 베이스 모델 (단순한 모델 사용)
try:
    # Level 1 모델들의 평균을 베이스로 사용
    calibrated_oof = np.zeros(len(y))
    calibrated_pred = np.zeros(len(X_test))
    
    skf_cal = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for tr_idx, val_idx in skf_cal.split(base_oof.reshape(-1, 1), y):
        # Isotonic Regression으로 확률 보정
        from sklearn.isotonic import IsotonicRegression
        
        iso_reg = IsotonicRegression(out_of_bounds='clip')
        iso_reg.fit(base_oof[tr_idx], y[tr_idx])
        
        calibrated_oof[val_idx] = iso_reg.transform(base_oof[val_idx])
        calibrated_pred += iso_reg.transform(base_pred_prob) / 5
    
    # Calibration 후 성능 비교
    t_cal, f1_cal, pos_cal = find_best_threshold(y, calibrated_oof, pos_cap=0.70, step=0.01)
    t_base, f1_base, pos_base = find_best_threshold(y, base_oof, pos_cap=0.70, step=0.01)
    
    print(f"Calibration 전: F1={f1_base:.4f} @ threshold={t_base:.3f}")
    print(f"Calibration 후: F1={f1_cal:.4f} @ threshold={t_cal:.3f}")
    
    if f1_cal > f1_base:
        print(f"Calibration 사용 (F1: {f1_cal:.4f} > {f1_base:.4f})")
        use_calibration = True
        final_oof = calibrated_oof
        final_pred_prob = calibrated_pred
        final_threshold = t_cal
    else:
        print(f"Calibration 미사용 (F1: {f1_base:.4f} >= {f1_cal:.4f})")
        use_calibration = False
        final_oof = base_oof
        final_pred_prob = base_pred_prob
        final_threshold = base_threshold
except Exception as e:
    print(f"Calibration 실패 (스킵): {e}")
    use_calibration = False
    final_oof = base_oof
    final_pred_prob = base_pred_prob
    final_threshold = base_threshold

# 최종 threshold 최적화 (더 넓은 범위 탐색)
final_oof_thr, final_oof_f1, final_oof_pos = find_best_threshold(y, final_oof, pos_cap=0.70, step=0.005)
print(f"\n최종 OOF F1: {final_oof_f1:.4f} @ threshold={final_oof_thr:.3f} (pos_rate={final_oof_pos:.3f})")

# Threshold 조정 옵션
# OOF 최적 threshold를 기본으로 사용하되, 예측 수료 비율이 너무 낮으면 조정
use_oof_threshold = True  # OOF 최적 threshold 사용
if use_oof_threshold:
    test_threshold_base = final_oof_thr
    print(f"OOF 최적 Threshold 사용: {test_threshold_base:.3f}")
else:
    # 보수적 threshold (CV-PB 차이 줄이기)
    test_threshold_base = min(final_oof_thr + 0.01, 0.50)
    print(f"보수적 Threshold: {test_threshold_base:.3f} (원래 threshold + 0.01)")

# ============================================================
# 6. 최종 예측 & 제출 파일 생성 (Positive Rate Cap 적용)
# ============================================================
# 목적: 최종 예측 생성 및 제출 파일 저장
# 과정:
#   1. OOF 최적 threshold 적용
#   2. Positive Rate Cap (0.70) 적용 (초과 시 상위 70%만 선택)
#   3. submission.csv 생성
#   4. submission_prob.csv 생성 (확률 값, 디버깅용)

print("\n" + "="*60)
print("최종 예측 생성 (pos_cap=0.70 적용)")
print("="*60)

# 1) Threshold 적용
test_threshold = test_threshold_base if 'test_threshold_base' in locals() else final_oof_thr
print(f"사용할 threshold: {test_threshold:.3f}")

test_pred_raw = (final_pred_prob >= test_threshold).astype(int)
raw_pos_rate = float(test_pred_raw.mean())

# 2) pos_rate가 0.70 초과하면 강제 cap
test_pred, effective_thr, final_pos_rate = apply_positive_rate_cap(
    final_pred_prob, base_thr=test_threshold, pos_cap=0.70
)

print(f"raw_pos_rate@threshold={raw_pos_rate:.4f}")
print(f"final_pos_rate(after cap)={final_pos_rate:.4f} | effective_thr≈{effective_thr:.6f}")
print(f"예측 수료 인원: {test_pred.sum()} / {len(test_pred)}")

# 제출 파일 생성
submission = sample.copy()
submission['completed'] = test_pred
submission.to_csv(f'{BASE_PATH}/0.44_submission.csv', index=False)
print(f"\n제출 파일 저장: {BASE_PATH}/0.44_submission.csv")
print(submission['completed'].value_counts())

# 확률 버전도 저장 (디버깅용)
sub_prob = sample.copy()
sub_prob['completed'] = final_pred_prob
sub_prob.to_csv(f'{BASE_PATH}/0.44_submission_prob.csv', index=False)
print(f"확률 파일 저장: {BASE_PATH}/0.44_submission_prob.csv")

# 앙상블 방법 저장
ensemble_info = {
    'method': best_method[0] if 'best_method' in locals() else ('stacking' if use_stacking else 'simple_avg'),
    'calibration': use_calibration if 'use_calibration' in locals() else False,
    'threshold': final_threshold,
    'effective_threshold': effective_thr,
    'oof_f1': final_oof_f1,
    'final_pos_rate': final_pos_rate,
    'num_models': len(all_oof_models)
}
method_str = ensemble_info['method']
if ensemble_info['calibration']:
    method_str += ' + calibration'
print(f"\n앙상블 방법: {method_str} ({ensemble_info['num_models']}개 모델)")

# ============================================================
# 7. Feature Importance
# ============================================================
# 목적: 모델이 중요하게 생각하는 피처 확인
# 방법: 마지막 LightGBM 모델의 feature_importances_ 사용
# 출력: 상위 25개 피처의 중요도

print("\n" + "="*60)
print("Feature Importance (마지막 LightGBM)")
print("="*60)

# Feature Importance (마지막 모델 사용)
if last_lgb_model is not None:
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': last_lgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
else:
    # 폴백: 전체 데이터로 빠른 모델 학습
    quick_model = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbosity=-1)
    quick_model.fit(X, y)
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': quick_model.feature_importances_
    }).sort_values('importance', ascending=False)

print(importance.head(25).to_string(index=False))

print("\n🎉 완료!")