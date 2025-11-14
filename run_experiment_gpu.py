# SINDy-증강 설명가능 AI 프레임워크: 사기 탐지 실험
# 제안된 방법론을 구현하고 평가하기 위한 파이썬 스크립트 (GPU 가속 및 SMOTE 방법론 적용 최종 수정본)

# ==============================================================================
# STEP 0: SETUP - 라이브러리 임포트
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import shap
import pysindy as ps
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, average_precision_score
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
import time
import os
import warnings

# pandas와 tqdm 연동 (진행률 표시)
tqdm.pandas()

# 체크포인트 파일 이름 정의
CHECKPOINT_TRAIN_PATH = 'df_train_checkpoint.pkl'
CHECKPOINT_TEST_PATH = 'df_test_checkpoint.pkl'

print("STEP 0: 라이브러리 임포트 완료.")
# 라이브러리 버전 확인
try:
    print(f"PySINDy Version: {ps.__version__}")
    print(f"SHAP Version: {shap.__version__}")
except Exception as e:
    print(f"라이브러리 버전 확인 중 오류: {e}")


# ==============================================================================
# STEP 1: 데이터 로딩 및 전처리 (PaySim 맞춤)
# ==============================================================================
print("\nSTEP 1: 데이터 로딩 및 전처리를 시작합니다...")
start_time = time.time()

try:
    df = pd.read_csv('PS_20174392719_1491204439457_log.csv')
    print(f"데이터셋 로드 성공. Shape: {df.shape}")
except FileNotFoundError:
    print("치명적 오류: 'PS_20174392719_1491204439457_log.csv' 파일을 찾을 수 없습니다.")
    exit()

# --- 기본 전처리 (PaySim 맞춤) ---
df = df[df['type'].isin(['CASH_OUT', 'TRANSFER'])]
df = pd.get_dummies(df, columns=['type'], prefix='type', drop_first=True)
df = df.drop(['nameDest', 'isFlaggedFraud'], axis=1)
df = df.rename(columns={'nameOrig': 'userID', 'isFraud': 'Class'})
print("기본 전처리 완료.")

# --- 특성 스케일링 ---
numerical_cols = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig']
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
print("수치형 특성 스케일링 완료.")

# --- 훈련/테스트 데이터 분할 ---
X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

df_train = X_train.copy(); df_train['Class'] = y_train
df_test = X_test.copy(); df_test['Class'] = y_test

print("훈련/테스트 데이터 분할 완료.")
print(f"STEP 1 완료. 소요 시간: {time.time() - start_time:.2f}초")


# ==============================================================================
# STEP 2: Baseline 모델 훈련
# ==============================================================================
print("\nSTEP 2: Baseline LightGBM 모델 훈련을 시작합니다 (GPU 사용)...")
start_time = time.time()

X_train_base = X_train.drop('userID', axis=1)
X_test_base = X_test.drop('userID', axis=1)
scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
params = {
    'objective': 'binary', 'scale_pos_weight': scale_pos_weight, 'n_estimators': 1000,
    'learning_rate': 0.05, 'random_state': 42, 'n_jobs': -1, 'device': 'gpu',
    'gpu_platform_id': 0, 'gpu_device_id': 0, 'min_child_samples': 20, 'min_split_gain': 0.001
}
model_base = lgb.LGBMClassifier(**params)
callbacks = [lgb.early_stopping(stopping_rounds=100, verbose=False)]
model_base.fit(X_train_base, y_train, eval_set=[(X_test_base, y_test)], eval_metric='aucpr', callbacks=callbacks)
print("Baseline 모델 훈련 완료.")
print(f"STEP 2 완료. 소요 시간: {time.time() - start_time:.2f}초")


# ==============================================================================
# STEP 3: 동적 특성 공학 (SINDy) - SMOTE 방법론 적용
# ==============================================================================
print("\nSTEP 3: SINDy를 이용한 동적 특성 공학을 시작합니다...")
start_time = time.time()

# 방법론이 변경되었으므로, 이전 체크포인트가 있다면 반드시 삭제하고 새로 시작
if os.path.exists(CHECKPOINT_TRAIN_PATH):
    print(f"Warning: 방법론 변경으로 인해 기존 체크포인트 파일 '{CHECKPOINT_TRAIN_PATH}'을 삭제합니다.")
    os.remove(CHECKPOINT_TRAIN_PATH)
if os.path.exists(CHECKPOINT_TEST_PATH):
    print(f"Warning: 방법론 변경으로 인해 기존 체크포인트 파일 '{CHECKPOINT_TEST_PATH}'을 삭제합니다.")
    os.remove(CHECKPOINT_TEST_PATH)

if os.path.exists(CHECKPOINT_TRAIN_PATH) and os.path.exists(CHECKPOINT_TEST_PATH):
    print("체크포인트 파일 발견! STEP 3를 건너뛰고 저장된 데이터를 로드합니다...")
    df_train = pd.read_pickle(CHECKPOINT_TRAIN_PATH)
    df_test = pd.read_pickle(CHECKPOINT_TEST_PATH)
    print("데이터 로드 완료.")

else:
    print("체크포인트 파일 없음. STEP 3를 처음부터 실행합니다...")
    
    # --- 3a. 동적 상태 변수 설계 (원본 데이터 대상) ---
    def engineer_state_variables(df_user):
        df_user = df_user.sort_values('step')
        df_user['x1_balance_exhaust'] = df_user['amount'] / (df_user['oldbalanceOrg'] + 1e-6)
        df_user['x2_tx_freq'] = df_user['step'].rolling(window=3, min_periods=1).apply(lambda x: len(x) / (x.max() - x.min() + 1e-6), raw=False)
        df_user['x3_time_since_last'] = df_user['step'].diff().fillna(0)
        return df_user

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        df_train = df_train.groupby('userID').progress_apply(engineer_state_variables).reset_index(drop=True)
        df_test = df_test.groupby('userID').progress_apply(engineer_state_variables).reset_index(drop=True)

    state_vars = ['x1_balance_exhaust', 'x2_tx_freq', 'x3_time_since_last']
    print("동적 상태 변수 설계 완료.")
    
    # --- 3b. SINDy 훈련용 데이터 생성 (SMOTE 적용) ---
    print("SINDy 훈련을 위해 SMOTE로 사기 데이터 증강을 시작합니다...")
    sindy_training_features = ['amount', 'oldbalanceOrg', 'newbalanceOrig'] + state_vars
    X_sindy_train_orig = df_train[sindy_training_features]
    y_sindy_train_orig = df_train['Class']
    
    X_sindy_train_orig = X_sindy_train_orig.replace([np.inf, -np.inf], np.nan).fillna(0)

    smote = SMOTE(random_state=42, sampling_strategy=0.5) # 정상:사기 = 2:1 비율
    X_res, y_res = smote.fit_resample(X_sindy_train_orig, y_sindy_train_orig)
    
    df_sindy_train_balanced = pd.DataFrame(X_res, columns=sindy_training_features)
    df_sindy_train_balanced['Class'] = y_res
    print(f"SMOTE 적용 완료. SINDy 훈련용 데이터 Shape: {df_sindy_train_balanced.shape}")
    
    # --- 3c. 분리된 SINDy 모델 훈련 (증강된 데이터 사용) ---
    df_sindy_train_balanced['group'] = np.arange(len(df_sindy_train_balanced)) // 10 # 10개씩 묶어 가상의 trajectory로 간주
    
    def prepare_sindy_data(df_source, class_label):
        ts_data_list, t_data_list = [], []
        scaler = MinMaxScaler()
        
        df_class = df_source[df_source['Class'] == class_label]
        for _, group in tqdm(df_class.groupby('group'), desc=f"SINDy {'Fraud' if class_label else 'Normal'} 데이터 준비 중"):
            if len(group) > 1:
                traj = group[state_vars].values
                traj_scaled = scaler.fit_transform(traj)
                ts_data_list.append(traj_scaled)
                t_data_list.append(np.arange(len(traj_scaled)))
        return ts_data_list, t_data_list

    normal_ts, normal_t = prepare_sindy_data(df_sindy_train_balanced, 0)
    fraud_ts, fraud_t = prepare_sindy_data(df_sindy_train_balanced, 1)

    if not normal_ts or not fraud_ts:
        print("치명적 오류: SINDy 모델을 훈련시키기에 충분한 데이터를 수집하지 못했습니다.")
        exit()

    SINDY_THRESHOLD = 0.1
    sindy_optimizer = ps.STLSQ(threshold=SINDY_THRESHOLD)
    differentiation_method = ps.FiniteDifference(order=1)
    
    model_sindy_normal = ps.SINDy(optimizer=sindy_optimizer, feature_library=ps.PolynomialLibrary(degree=2), differentiation_method=differentiation_method)
    model_sindy_fraud = ps.SINDy(optimizer=sindy_optimizer, feature_library=ps.PolynomialLibrary(degree=2), differentiation_method=differentiation_method)

    print("SINDy 정상 동역학 모델 훈련 중...")
    model_sindy_normal.fit(normal_ts, t=normal_t)
    print("SINDy 사기 동역학 모델 훈련 중...")
    model_sindy_fraud.fit(fraud_ts, t=fraud_t)

    print("\n--- SINDy 정상 동역학 모델 ---")
    model_sindy_normal.print(feature_names=state_vars)
    print("\n--- SINDy 사기 동역학 모델 ---")
    model_sindy_fraud.print(feature_names=state_vars)

    # --- 3d. Dynamic_Score 생성 (원본 데이터 대상) ---
    def calculate_dynamic_score(df_source, model_normal, model_fraud):
        scores = []
        scaler = MinMaxScaler()
        for user, group in tqdm(df_source.groupby('userID'), desc="Dynamic_Score 계산 중"):
            user_ts = group[state_vars].values
            if len(user_ts) > 1:
                try:
                    user_ts = np.nan_to_num(user_ts, nan=0.0, posinf=1e5, neginf=-1e5)
                    user_ts_scaled = scaler.fit_transform(user_ts)
                    t = np.arange(len(user_ts_scaled))
                    actual_derivative = ps.FiniteDifference(order=1)._differentiate(user_ts_scaled, t)
                    actual_derivative = np.nan_to_num(actual_derivative, nan=0.0, posinf=1e5, neginf=-1e5)
                    error_normal = np.mean((model_normal.predict(user_ts_scaled, t=t) - actual_derivative)**2)
                    error_fraud = np.mean((model_fraud.predict(user_ts_scaled, t=t) - actual_derivative)**2)
                    score = np.log((error_normal + 1e-6) / (error_fraud + 1e-6))
                except Exception:
                    score = 0
            else:
                score = 0
            scores.extend([score] * len(group))
        return scores

    df_train['Dynamic_Score'] = calculate_dynamic_score(df_train, model_sindy_normal, model_sindy_fraud)
    df_test['Dynamic_Score'] = calculate_dynamic_score(df_test, model_sindy_normal, model_sindy_fraud)
    print("Dynamic_Score 생성 완료.")
    
    print("STEP 3 완료. 결과를 체크포인트 파일로 저장합니다...")
    df_train.to_pickle(CHECKPOINT_TRAIN_PATH)
    df_test.to_pickle(CHECKPOINT_TEST_PATH)
    print("체크포인트 저장 완료.")

print(f"STEP 3 완료. 소요 시간: {time.time() - start_time:.2f}초")


# ==============================================================================
# STEP 4: SINDy-증강 모델 훈련
# ==============================================================================
print("\nSTEP 4: SINDy-증강 LightGBM 모델 훈련을 시작합니다 (GPU 사용)...")
start_time = time.time()

X_train_aug = df_train.drop(['userID', 'Class'], axis=1)
X_test_aug = df_test.drop(['userID', 'Class'], axis=1)
y_train_aug = df_train['Class']
y_test_aug = df_test['Class']

model_aug = lgb.LGBMClassifier(**params)
model_aug.fit(X_train_aug, y_train_aug, 
              eval_set=[(X_test_aug, y_test_aug)], 
              eval_metric='aucpr',
              callbacks=callbacks)

print("SINDy-증강 모델 훈련 완료.")
print(f"STEP 4 완료. 소요 시간: {time.time() - start_time:.2f}초")


# ==============================================================================
# STEP 5: 성능 비교 및 분석
# ==============================================================================
print("\nSTEP 5: 모델 성능 비교를 시작합니다...")
start_time = time.time()

y_pred_proba_base = model_base.predict_proba(X_test_base)[:, 1]
auc_pr_base = average_precision_score(y_test, y_pred_proba_base)

y_pred_proba_aug = model_aug.predict_proba(X_test_aug)[:, 1]
auc_pr_aug = average_precision_score(y_test_aug, y_pred_proba_aug)

print("\n--- 성능 비교 결과 ---")
print(f"Baseline LightGBM AUC-PR: {auc_pr_base:.4f}")
print(f"SINDy-Augmented LightGBM AUC-PR: {auc_pr_aug:.4f}")
print(f"성능 향상률: {(auc_pr_aug - auc_pr_base) / auc_pr_base * 100:.2f}%")

print("\n--- Baseline 모델 상세 리포트 ---")
print(classification_report(y_test, (y_pred_proba_base > 0.5).astype(int), digits=4))
print("\n--- SINDy-증강 모델 상세 리포트 ---")
print(classification_report(y_test_aug, (y_pred_proba_aug > 0.5).astype(int), digits=4))
print(f"STEP 5 완료. 소요 시간: {time.time() - start_time:.2f}초")


# ==============================================================================
# STEP 6: 증강 모델에 대한 SHAP 분석
# ==============================================================================
print("\nSTEP 6: 증강 모델에 대한 SHAP 분석을 시작합니다...")
start_time = time.time()

explainer_aug = shap.TreeExplainer(model_aug)
X_test_sample = X_test_aug.sample(n=min(5000, len(X_test_aug)), random_state=42)
shap_values_aug = explainer_aug.shap_values(X_test_sample)

if isinstance(shap_values_aug, list):
    shap_values_for_plot = shap_values_aug[1]
else:
    shap_values_for_plot = shap_values_aug

plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values_for_plot, X_test_sample, show=False)
plt.title("SHAP Summary Plot for SINDy-Augmented Model", fontsize=16)
plt.tight_layout()
plt.savefig('shap_summary_augmented_model.png', dpi=150)
print("SHAP 요약 플롯이 'shap_summary_augmented_model.png' 파일로 저장되었습니다.")

print(f"STEP 6 완료. 소요 시간: {time.time() - start_time:.2f}초")

print("\n--- 모든 실험이 성공적으로 종료되었습니다. ---")

