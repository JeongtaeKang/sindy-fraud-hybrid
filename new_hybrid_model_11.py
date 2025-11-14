#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SINDy-증강 설명가능 AI 프레임워크 (v4_HYBRID, EBM/RuleFit 제거) + 체크포인트/재시작:
- 1단계: SINDy 동역학 비교기 (정상/사기 모델 MSE 계산)
- 2단계: LightGBM/XGBoost/로지스틱/결정트리 최종 분류기 (원본 특징 + SINDy 동적 특징 결합)
- 비교 실험 자동화: SINDy 단독(점수/이진), Logistic(SINDy전용), DT/LightGBM/XGBoost(전체)
- artifacts/로깅/재시작 지원 + 대용량 안전화
"""

# === ANCHOR-IMPORTS: AFTER ===
import json
import time
import joblib
import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict
from pathlib import Path

from tqdm import tqdm
tqdm.pandas()
from collections import Counter
import sys, platform
import sklearn  # 버전 문자열 읽기용(이미 세부 모듈은 사용 중)

# === Safe pickle helpers (for SINDy) ===
try:
    import dill  # pip install dill==0.3.8
    import gzip
    def dump_safely(obj, path):
        with gzip.open(str(path) + ".gz", "wb") as f:
            dill.dump(obj, f)
    def load_safely(path):
        # path에 .gz를 붙여서 로드 (세이브와 통일)
        gz_path = str(path)
        if not gz_path.endswith(".gz"):
            gz_path += ".gz"
        with gzip.open(gz_path, "rb") as f:
            return dill.load(f)
    HAS_DILL = True
except Exception:
    import joblib
    def dump_safely(obj, path):
        joblib.dump(obj, path)
    def load_safely(path):
        return joblib.load(path)
    HAS_DILL = False

# Sklearn / Imbalanced
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    f1_score,
    confusion_matrix,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Advanced Classifiers
import lightgbm as lgb
import xgboost as xgb

# SHAP / SINDy / Plot
import shap
import pysindy as ps
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# === [ADD] PDP/LIME ===
from sklearn.inspection import PartialDependenceDisplay
import importlib
try:
    LimeTabularExplainer = importlib.import_module("lime.lime_tabular").LimeTabularExplainer  # type: ignore[attr-defined]
except Exception:
    LimeTabularExplainer = None  # lime 미설치/로딩 실패 시 안전 폴백

import matplotlib
matplotlib.rcParams['figure.max_open_warning'] = 0

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 파일 상단 (전역, 한 번만 정의)
PLOT_DPI = 600                      # ← MDPI 안전권장: 600dpi
PLOT_FIGSIZE_STD  = (9, 6)
PLOT_FIGSIZE_WIDE = (10, 6)
PLOT_FIGSIZE_TALL = (8, 10)

def xgb_has_gpu() -> bool:
    """현재 설치된 xgboost가 GPU(CUDA) 빌드인지 점검"""
    try:
        info = xgb.build_info()  # 1.7+ 에서 dict, 구버전은 str일 수 있음
        if isinstance(info, dict):
            flag = str(info.get("USE_CUDA", "")).upper()
            return flag in ("ON", "TRUE", "1")
        return "CUDA" in str(info).upper() or "GPU" in str(info).upper()
    except Exception:
        return False

def dump_environment_snapshot():
    info = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__ if 'sklearn' in sys.modules else None,
        "lightgbm": getattr(lgb, "__version__", None),
        "xgboost": getattr(xgb, "__version__", None),
        "pysindy": getattr(ps, "__version__", None),
        "shap": getattr(shap, "__version__", None),
        "use_gpu": bool(CFG.use_gpu),
        "xgb_gpu_build": bool(xgb_has_gpu()),
        "random_state": CFG.random_state,
    }
    (ART / "env_snapshot.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    # 가급적 pip freeze도 남김
    try:
        import subprocess
        out = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True, timeout=60)
        (ART / "pip_freeze.txt").write_text(out, encoding="utf-8")
    except Exception as e:
        log(f"[WARN] pip freeze 실패: {e}")

# === Add this helper ===
def ensure_unique_columns(df, where=""):
    dup = df.columns[df.columns.duplicated()].tolist()
    if dup:
        print(f"[FIX][{where}] Dropping duplicated columns: {sorted(set(dup))}")
        df = df.loc[:, ~df.columns.duplicated(keep="last")]
    return df

# === Patch your save_df ===
def save_df(df, path):
    df = ensure_unique_columns(df, where=f"pre-save:{path}")
    try:
        df.to_parquet(path, index=False)
    except Exception as e:
        print(f"[WARN] to_parquet failed ({e}); trying engine='pyarrow'")
        df = ensure_unique_columns(df, where=f"pre-save-retry:{path}")
        try:
            df.to_parquet(path, index=False, engine="pyarrow")
        except Exception as e2:
            print(f"[WARN] pyarrow도 실패({e2}); CSV로 폴백")
            csv_path = str(path).rsplit(".", 1)[0] + ".csv"
            df.to_csv(csv_path, index=False)

# --- NUMERIC HELPERS: expit with SciPy fallback ---
try:
    from scipy.special import expit  # logistic sigmoid
except Exception:
    def expit(x):
        x = np.asarray(x, dtype=float)
        return 1.0 / (1.0 + np.exp(-x))

def _make_sr3(thr: float):
    """
    pysindy 버전별 SR3 시그니처 차이를 흡수하기 위한 팩토리.
    - 더 강한 희소화(큰 nu, 선택적 l0_penalty) 우선 시도
    - threshold/thresh, max_iter/max_iterations 등 호환 시도
    실패 시 None 반환 (STLSQ로 폴백)
    """
    # 1) l0_penalty 지원 버전이면 가장 먼저 시도
    prefer = [
        dict(threshold=thr, nu=5e-1, tol=1e-5, max_iter=30000, l0_penalty=10.0),
        dict(thresh=thr,    nu=5e-1, tol=1e-5, max_iter=30000, l0_penalty=10.0),
        dict(threshold=thr, nu=5e-1, tol=1e-5, max_iterations=30000, l0_penalty=10.0),
        dict(thresh=thr,    nu=5e-1, tol=1e-5, max_iterations=30000, l0_penalty=10.0),
    ]
    for kwargs in prefer:
        try:
            return ps.SR3(**kwargs)
        except TypeError:
            continue
        except Exception:
            continue

    # 2) 일반 SR3 (nu 크게)
    tried = [
        dict(threshold=thr, nu=5e-2, tol=1e-5, max_iter=20000),
        dict(thresh=thr,    nu=5e-2, tol=1e-5, max_iter=20000),
        dict(threshold=thr, nu=5e-2, tol=1e-5, max_iterations=20000),
        dict(thresh=thr,    nu=5e-2, tol=1e-5, max_iterations=20000),
        dict(threshold=thr, nu=1e-1, tol=1e-5, max_iter=20000),
        dict(thresh=thr,    nu=1e-1, tol=1e-5, max_iter=20000),
    ]
    for kwargs in tried:
        try:
            return ps.SR3(**kwargs)
        except TypeError:
            continue
        except Exception:
            continue
    return None

def _make_stlsq(thr: float):
    """
    pysindy 버전별 STLSQ 시그니처 차이를 흡수하기 위한 팩토리.
    - threshold/thresh, alpha/ridge_alpha 조합을 시도
    실패 시 STRidge, 마지막으로 최소 인자 STLSQ로 폴백
    """
    tried = [
        dict(threshold=thr, alpha=1e-2),
        dict(threshold=thr, ridge_alpha=1e-1),
        dict(thresh=thr,    alpha=1e-2),
        dict(thresh=thr,    ridge_alpha=1e-1),
        dict(threshold=thr, alpha=1e-1),       # ← 추가 (조금 더 강한 릿지)
        dict(threshold=thr),         # 최소 인자
        dict(thresh=thr),            # 최소 인자 (별칭)
    ]
    for kwargs in tried:
        try:
            return ps.STLSQ(**kwargs)
        except TypeError:
            continue
        except Exception:
            continue
    # STLSQ가 모두 실패하면 STRidge로 최후 폴백
    try:
        return ps.STRidge(threshold=thr, alpha=1e-5)
    except Exception:
        # 정말 최후의 수단: 인자 없는 STLSQ 시도
        return ps.STLSQ()

# 금지 컬럼(사후정보/파생) - 전역 상수
FORBID_COLS = {"newbalanceOrig", "oldbalanceDest", "newbalanceDest"}

def log(msg: str): print(f"[SINDY-PRO] {msg}", flush=True)

LABEL_COL = 'Class'

@dataclass
class CCSStats:
    mean0: pd.Series; std0: pd.Series; mean1: pd.Series; std1: pd.Series; cols: list
def fit_ccs_stats(train_df: pd.DataFrame, feature_cols: list, label_col: str = LABEL_COL) -> CCSStats:
    g0 = train_df[train_df[label_col] == 0][feature_cols]; g1 = train_df[train_df[label_col] == 1][feature_cols]
    mean0, std0 = g0.mean(), g0.std(ddof=0).replace(0, 1.0)
    mean1, std1 = g1.mean(), g1.std(ddof=0).replace(0, 1.0)
    return CCSStats(mean0, std0, mean1, std1, feature_cols)
def apply_ccs(df: pd.DataFrame, stats: CCSStats, label_col: str = LABEL_COL) -> pd.DataFrame:
    df = df.copy(); mask1 = (df[label_col] == 1)
    df.loc[~mask1, stats.cols] = (df.loc[~mask1, stats.cols] - stats.mean0) / stats.std0
    df.loc[mask1, stats.cols] = (df.loc[mask1, stats.cols] - stats.mean1) / stats.std1
    df[stats.cols] = df[stats.cols].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return df
def apply_ccs_inference(df: pd.DataFrame, stats: CCSStats) -> pd.DataFrame:
    df = df.copy()
    df[stats.cols] = (df[stats.cols] - stats.mean0) / stats.std0
    df[stats.cols] = df[stats.cols].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return df

# === 사용자 분리 + 윈도우 기반 궤적 수집 ===
def _select_users_by_class(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    user_has_fraud = df.groupby(CFG.user_col)["Class"].max()
    fraud_users  = user_has_fraud[user_has_fraud == 1].index.to_numpy()
    normal_users = user_has_fraud[user_has_fraud == 0].index.to_numpy()
    rng = np.random.RandomState(CFG.random_state)
    rng.shuffle(fraud_users); rng.shuffle(normal_users)
    return normal_users, fraud_users

def make_strictly_increasing(t: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    t = np.asarray(t, dtype=float).copy()
    for i in range(1, len(t)):
        if t[i] <= t[i-1]: t[i] = t[i-1] + eps
    return t

class _IdentityScaler:
    def transform(self, X): return np.asarray(X, dtype=float)

def build_window_labeled_trajectories(df_ccs: pd.DataFrame,
                                      state_cols: list,
                                      window: int = 7,
                                      stride: int = 2,
                                      max_traj_per_class: int = 5000) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], int, int]:
    normal_users, fraud_users = _select_users_by_class(df_ccs)
    scaler = _IdentityScaler()

    groups = {u: g.sort_values("step_raw") for u, g in df_ccs.groupby(CFG.user_col)}
    MIN_PTS = 3

    def collect(w: int, s: int):
        n_ts, n_t, f_ts, f_t = [], [], [], []

        # FRAUD: y==1 주변
        for idx_u, u in enumerate(fraud_users):
            g = groups.get(u)
            if g is None or len(g) < MIN_PTS:
                continue
            y = g["Class"].to_numpy()
            sig = g[state_cols].to_numpy()
            t = g["step_raw"].to_numpy()
            idxs = np.flatnonzero(y == 1)
            for i in idxs:
                for offset in range(-(w-1), 1, s):
                    a = max(0, i+offset); b = min(len(g), a+w)
                    if b - a < MIN_PTS:
                        continue
                    X = scaler.transform(sig[a:b])
                    tt = make_strictly_increasing(t[a:b] - t[a])
                    if len(tt) >= 2:
                        dt_med = float(np.median(np.diff(tt)))
                        if dt_med > 0:
                            tt = tt / dt_med
                    f_ts.append(X); f_t.append(tt)
                    if len(f_ts) >= max_traj_per_class:
                        break
                if len(f_ts) >= max_traj_per_class: break
            if len(f_ts) >= max_traj_per_class: break
            if idx_u % 1000 == 0:
                log(f"[collect/fraud] users processed={idx_u}, collected={len(f_ts)}")

        # NORMAL 1차
        target_norm = min(max_traj_per_class, max(1, len(f_ts)))
        for idx_u, u in enumerate(normal_users):
            if len(n_ts) >= target_norm: break
            g = groups.get(u)
            if g is None or len(g) < MIN_PTS:
                continue
            sig = g[state_cols].to_numpy()
            t = g["step_raw"].to_numpy()
            start = 0
            while start < len(g) and len(n_ts) < target_norm:
                end = min(start+w, len(g))
                if end - start >= MIN_PTS:
                    X = scaler.transform(sig[start:end])
                    tt = make_strictly_increasing(t[start:end] - t[start])
                    if len(tt) >= 2:
                        dt_med = float(np.median(np.diff(tt)))
                        if dt_med > 0:
                            tt = tt / dt_med
                    n_ts.append(X); n_t.append(tt)
                start += max(1, s)
            if idx_u % 1000 == 0:
                log(f"[collect/normal] users processed={idx_u}, collected={len(n_ts)}/{target_norm}")

        # NORMAL 2차: fraud 사용자 중 y==0 구간
        if len(n_ts) < target_norm:
            need = target_norm - len(n_ts)
            for idx_u, u in enumerate(fraud_users):
                if need <= 0: break
                g = groups.get(u)
                if g is None or len(g) < MIN_PTS: continue
                g0 = g[g["Class"] == 0]
                if len(g0) < MIN_PTS: continue
                sig = g0[state_cols].to_numpy()
                t = g0["step_raw"].to_numpy()
                start = 0
                while start < len(g0) and need > 0:
                    end = min(start+w, len(g0))
                    if end - start >= MIN_PTS:
                        X = scaler.transform(sig[start:end])
                        tt = make_strictly_increasing(t[start:end] - t[start])
                        if len(tt) >= 2:
                            dt_med = float(np.median(np.diff(tt)))
                            if dt_med > 0:
                                tt = tt / dt_med
                        n_ts.append(X); n_t.append(tt); need -= 1
                    start += max(1, s)
            log(f"[collect/normal backfill] total_normal={len(n_ts)}/{target_norm}")

        return n_ts, n_t, f_ts, f_t

    n_ts, n_t, f_ts, f_t = collect(window, stride)

    def _keep_valid(X_list, t_list):
        keep = []
        for i, (X, tt) in enumerate(zip(X_list, t_list)):
            if getattr(X, "shape", (0,))[0] >= MIN_PTS and len(tt) >= MIN_PTS and X.shape[0] == len(tt):
                keep.append(i)
        return keep

    keep_n = _keep_valid(n_ts, n_t)
    keep_f = _keep_valid(f_ts, f_t)
    n_ts = [n_ts[i] for i in keep_n]; n_t = [n_t[i] for i in keep_n]
    f_ts = [f_ts[i] for i in keep_f]; f_t = [f_t[i] for i in keep_f]

    while (len(f_ts) == 0 or len(n_ts) == 0) and window > 3:
        window = max(3, window-1); stride = max(1, stride//2)
        log(f"trajectory 부족 → 폴백 (window={window}, stride={stride})")
        n_ts, n_t, f_ts, f_t = collect(window, stride)
        keep_n = _keep_valid(n_ts, n_t)
        keep_f = _keep_valid(f_ts, f_t)
        n_ts = [n_ts[i] for i in keep_n]; n_t = [n_t[i] for i in keep_n]
        f_ts = [f_ts[i] for i in keep_f]; f_t = [f_t[i] for i in keep_f]

    k = min(len(n_ts), len(f_ts))
    if k == 0:
        raise RuntimeError("No valid trajectories (len>=3) after fallback. Consider switching group key or features.")

    bal_k = min(k, CFG.max_traj_per_class)
    n_ts, n_t = n_ts[:bal_k], n_t[:bal_k]
    f_ts, f_t = f_ts[:bal_k], f_t[:bal_k]
    log(f"[collect/done] normal={len(n_ts)}, fraud={len(f_ts)} (window={window}, stride={stride})")
    return n_ts, n_t, f_ts, f_t, window, stride

# =================
# Config & helpers
# =================
@dataclass
class Config:
    random_state: int = 42
    test_size: float = 0.2
    raw_csv_path: str = "PS_20174392719_1491204439457_log.csv"
    target_col: str = "isFraud"; user_col_raw: str = "nameOrig"; user_col: str = "userID"
    force_user_group_key: bool = False
    drop_cols: Tuple[str, ...] = ("isFlaggedFraud",)
    numeric_cols: Tuple[str, ...] = ("step", "amount", "oldbalanceOrg")
    storage: str = "parquet"; artifacts_dir: str = "artifacts_sindy_v4_hybrid"
    smote_classifier_sampling: float = 0.1
    use_gpu: bool = False
    sindy_retry_cos_tol: float = 0.85
    sindy_fraud_subsample_seq: Tuple[float, ...] = (0.8, 0.6, 0.5)
    run_shap: bool = True; shap_sample: int = 4000
    sindy_window: int = 5
    sindy_stride: int = 2
    max_traj_per_class: int = 10000
    temporal_holdout_enable: bool = True
    temporal_cut_step: int = 600
    sindy_degree: int = 2
    sindy_include_interaction: bool = True
    sindy_include_bias: bool = False
    sindy_threshold: float = 0.3
    resume_from_run: Optional[str] = "latest"
        # === [ADD] XAI 확장 플래그 ===
    run_pdp: bool = True
    pdp_features_topk: int = 8            # PDP 그릴 상위 특징 개수
    pdp_grid_resolution: int = 20         # PDP 격자 해상도
    run_lime: bool = True
    lime_samples: int = 3                # LIME 생성할 샘플 수


CFG = Config()

def _get_run_dir(run_id: str) -> Path:
    return Path(CFG.artifacts_dir) / "runs" / run_id

def _resolve_resume_source_dir():
    if not CFG.resume_from_run:
        return None
    run_id = CFG.resume_from_run
    if run_id == "latest":
        latest_file = Path(CFG.artifacts_dir) / "runs" / "latest.txt"
        if latest_file.exists():
            run_id = latest_file.read_text(encoding="utf-8").strip()
        else:
            log("resume_from_run='latest' 이지만 latest.txt가 없어 재시작을 건너뜁니다.")
            return None
    src = _get_run_dir(run_id)
    if not src.exists():
        log(f"지정한 재시작 소스가 존재하지 않습니다: {src}")
        return None
    log(f"재시작 소스 런 디렉터리: {src}")
    return src

# === Tee 로깅 부트스트랩 ===
def _bootstrap_logging(artifacts_dir: str):
    import sys, io, atexit, time as _time
    from pathlib import Path as _Path
    global RUN_ID, ART, RUNLOG_PATH

    RUN_ID = _time.strftime("%Y%m%d_%H%M%S")
    ART = _Path(artifacts_dir) / "runs" / RUN_ID
    ART.mkdir(parents=True, exist_ok=True)
    RUNLOG_PATH = ART / f"run_{_time.strftime('%Y%m%d_%H%M%S')}.log"
    _runlog_fp = open(RUNLOG_PATH, "w", encoding="utf-8")

    class _Tee(io.TextIOBase):
        def __init__(self, fp):
            self.fp = fp
            self.console_out = sys.__stdout__
        def write(self, s):
            self.console_out.write(s); self.fp.write(s)
        def flush(self):
            self.console_out.flush(); self.fp.flush()

    sys.stdout = _Tee(_runlog_fp)
    sys.stderr = _Tee(_runlog_fp)
    atexit.register(_runlog_fp.close)

_bootstrap_logging(CFG.artifacts_dir)
log("로깅 부트스트랩 완료. 재시작 소스 해석을 시작합니다.")
RESUME_SRC = _resolve_resume_source_dir()
(Path(CFG.artifacts_dir) / "runs" / "latest.txt").write_text(RUN_ID, encoding="utf-8")

class _NpEncoder(json.JSONEncoder):
    def default(self, o):
        import numpy as _np
        if isinstance(o, (_np.integer,)):  return int(o)
        if isinstance(o, (_np.floating,)): return float(o)
        if isinstance(o, _np.ndarray):     return o.tolist()
        return super().default(o)

def save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, cls=_NpEncoder)

def load_df(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception as e:
        log(f"[WARN] read_parquet failed ({e}); trying engine='pyarrow'")
        try:
            return pd.read_parquet(path, engine="pyarrow")
        except Exception as e2:
            # CSV 폴백 시도
            csv_path = Path(str(path).rsplit(".", 1)[0] + ".csv")
            if csv_path.exists():
                log(f"[WARN] parquet 불가 → CSV 로드 시도: {csv_path.name}")
                return pd.read_csv(csv_path)
            raise

def eval_from_proba(y_true, y_prob, model_name, fixed_thr: float | None = None):
    ap = average_precision_score(y_true, y_prob)
    roc = roc_auc_score(y_true, y_prob)
    if fixed_thr is None:
        prec, rec, thr = precision_recall_curve(y_true, y_prob)
        f1s = 2*(prec*rec)/(prec+rec+1e-12)
        best_idx = int(np.nanargmax(f1s)) if len(f1s) > 0 else 0
        best_thr = float(thr[max(0, min(best_idx-1, len(thr)-1))]) if len(thr)>0 else 0.5
    else:
        best_thr = float(fixed_thr)
    y_pred = (y_prob >= best_thr).astype(int)
    f1 = float(f1_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred).tolist()
    rep = classification_report(y_true, y_pred, digits=4, output_dict=True)
    metrics = {"model": model_name, "roc_auc": roc, "avg_precision": ap, "best_threshold": best_thr, "f1": f1, "confusion_matrix": cm, "report": rep}
    log(f"[{model_name}] AP={ap:.4f}, ROC-AUC={roc:.4f}, F1={f1:.4f} @ thr={best_thr:.3f}")
    save_json(metrics, ART / f"metrics_{model_name}.json")
    return metrics

def _tau_max_f1(y_true: np.ndarray, proba: np.ndarray) -> float:
    p, r, thr = precision_recall_curve(y_true, proba)
    f1 = 2 * p * r / (p + r + 1e-12)
    k = int(np.nanargmax(f1))
    # precision_recall_curve는 마지막 점에서 threshold가 없을 수 있음 → 보정
    return float(thr[min(k, len(thr)-1)]) if len(thr) > 0 else 0.5

def _dump_tau(name: str, tau: float, path: Path):
    rec = {"model": name, "tau": float(tau)}
    path.write_text(json.dumps(rec, indent=2), encoding="utf-8")

def _align_columns(df_like, cols_ref):
    """df_like 을 학습때의 열 순서 cols_ref로 재인덱싱 (누락=0.0)."""
    return (df_like.reindex(columns=cols_ref, fill_value=0.0)
                  .astype(np.float64))

def sanity_check_shuffle(y_true, y_prob, model_name):
    rng = np.random.RandomState(CFG.random_state)
    y_shuf = rng.permutation(y_true)
    try:
        ap_s = average_precision_score(y_shuf, y_prob)
        roc_s = roc_auc_score(y_shuf, y_prob)
        if ap_s > 0.7 or roc_s > 0.7:
            log(f"[LEAK?] {model_name}: shuffled-label AP={ap_s:.3f}, ROC={roc_s:.3f} → 데이터 누수/과적합 의심")
    except Exception:
        pass

def make_lgb_params(scale_pos_weight: float, use_gpu: bool) -> dict:
    params = {
        "objective": "binary",
        "n_estimators": 1500,
        "learning_rate": 0.05,
        "random_state": CFG.random_state,
        "n_jobs": -1,
        "min_split_gain": 1e-2,
        "num_leaves": 31,
        "feature_fraction": 0.5,
        "bagging_fraction": 0.7,
        "lambda_l2": 40.0,
        "min_data_in_leaf": 800,
        "feature_pre_filter": True,
        "min_sum_hessian_in_leaf": 1e-3,
        "max_bin": 255,
        "bagging_freq": 1,
        "scale_pos_weight": float(scale_pos_weight),
        "metric": "average_precision",
        "deterministic": True,
        "force_col_wise": True,
        "verbose": -1,
        "extra_trees": True,
        "seed": CFG.random_state,
        "bagging_seed": CFG.random_state,
        "feature_fraction_seed": CFG.random_state,
    }
    # LGBM 4.x에서만 안전
    try:
        from packaging.version import Version
        if Version(lgb.__version__) >= Version("4.0"):
            params["deterministic"] = True
    except Exception:
        pass

    if use_gpu:
        try: 
            params.update({"device": "gpu"})
            # 추가: 새 버전 호환
            params.update({"device_type": "gpu"})
        except Exception: log("[WARN] LightGBM GPU 설정 실패 → CPU로 폴백")
    return params

def save_feature_list(name: str, cols: List[str]):
    (ART / f"{name}_cols.json").write_text(json.dumps(cols, indent=2), encoding="utf-8")
def align_features(df: pd.DataFrame, required_cols: List[str]) -> pd.DataFrame:
    X = df.copy()
    for c in required_cols:
        if c not in X.columns: X[c] = 0.0
    return X[required_cols]

def boost_sindy_columns(df: pd.DataFrame, factor: float = 3.0) -> pd.DataFrame:
    out = df.copy()
    sindy_cols = [c for c in out.columns if c.startswith("sindy_")]
    if sindy_cols:
        out[sindy_cols] = out[sindy_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0) * float(factor)
    return out

def to_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    return (df.apply(pd.to_numeric, errors="coerce")
              .replace([np.inf, -np.inf], np.nan)
              .fillna(0.0)
              .astype(np.float64))

def detect_data_leakage(train_df: pd.DataFrame, test_df: pd.DataFrame, id_col: str, time_col: str, label_col: str = LABEL_COL):
    issues = []
    inter_users = set(train_df[id_col].unique()) & set(test_df[id_col].unique())
    if len(inter_users) > 0:
        issues.append(f"ID overlap between train/test: {len(inter_users)}")
    if (train_df[time_col].max() > test_df[time_col].min()):
        issues.append("Temporal leakage risk: train.max(step_raw) > test.min(step_raw)")
    bad_cols = [c for c in train_df.columns if c.lower() in {"class","label","target"} and c != label_col]
    if bad_cols:
        issues.append(f"Label-like columns present in features: {bad_cols}")
    if issues:
        log("[LEAKAGE] " + " | ".join(issues))
    else:
        log("[LEAKAGE] No obvious leakage detected.")

def _split_stats(df: pd.DataFrame, user_col: str = "nameOrig", step_col: str = "step_raw") -> dict:
    if step_col not in df.columns:
        raise KeyError(f"'{step_col}' column not found")
    if user_col not in df.columns:
        raise KeyError(f"'{user_col}' column not found")
    n_users = int(df[user_col].nunique())
    n_tx    = int(len(df))
    step_min = int(df[step_col].min())
    step_max = int(df[step_col].max())
    return {"users": n_users, "tx": n_tx, "step_min": step_min, "step_max": step_max}

def _emit_split_stats_artifacts(train_stats: dict, test_stats: dict, outdir: Path):
    # 콘솔/로그에 한 줄 요약
    log(f"[SPLIT] Train: users={train_stats['users']:,}, tx={train_stats['tx']:,}, step=[{train_stats['step_min']}, {train_stats['step_max']}]")
    log(f"[SPLIT] Test : users={test_stats['users']:,}, tx={test_stats['tx']:,}, step=[{test_stats['step_min']}, {test_stats['step_max']}]")
    # CSV/JSON 저장(논문 표 작성용)
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"split": "Train", **train_stats}, {"split": "Test", **test_stats}]
    ).to_csv(outdir / "split_stats.csv", index=False)
    with open(outdir / "split_stats.json", "w", encoding="utf-8") as f:
        json.dump({"train": train_stats, "test": test_stats}, f, ensure_ascii=False, indent=2)

def fit_transform_scalers(train_df: pd.DataFrame, test_df: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    scaler = StandardScaler().fit(train_df[cols].values)
    train_df = train_df.copy(); test_df = test_df.copy()
    train_df[cols] = scaler.transform(train_df[cols].values)
    test_df[cols]  = scaler.transform(test_df[cols].values)
    return train_df, test_df, scaler

def reliability_plot(y_true: np.ndarray, y_prob: np.ndarray, out_png: Path, n_bins: int = 10, title: str = "Reliability Diagram"):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_prob, bins) - 1
    bin_true = []; bin_pred = []; bin_cnt = []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0: continue
        bin_true.append(y_true[m].mean())
        bin_pred.append(y_prob[m].mean())
        bin_cnt.append(int(m.sum()))
    plt.figure(figsize=PLOT_FIGSIZE_STD, dpi=PLOT_DPI)
    plt.plot([0,1],[0,1], linestyle="--")
    plt.scatter(bin_pred, bin_true, s=np.array(bin_cnt)/np.max(bin_cnt)*200)
    plt.xlabel("Predicted probability"); plt.ylabel("Empirical fraud rate")
    plt.title(title); plt.tight_layout(); plt.savefig(out_png, dpi=PLOT_DPI, bbox_inches='tight'); plt.close()

def permutation_importance_safely(model, X: pd.DataFrame, y: np.ndarray, out_csv: Path, n_repeats: int = 5):
    try:
        r = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=CFG.random_state, n_jobs=-1, scoring="average_precision")
        imp = pd.DataFrame({"feature": X.columns, "importance_mean": r.importances_mean, "importance_std": r.importances_std})
        imp.sort_values("importance_mean", ascending=False).to_csv(out_csv, index=False)
        log(f"Permutation importance saved -> {out_csv.name}")
    except Exception as e:
        log(f"[WARN] permutation_importance failed: {e}")

def _states_exist_and_valid(path: Path, state_cols: List[str]) -> bool:
    if not path or not path.exists():
        return False
    try:
        df = load_df(path)
        needed = set(state_cols) | {CFG.user_col, "step_raw", "Class"}
        return needed.issubset(set(df.columns)) and len(df) > 0
    except Exception:
        return False

# ======================================================================
# 안전 스케일링/미분/Δt 헬퍼
# ======================================================================
def _safe_scale(X: np.ndarray, scaler_or_stats) -> np.ndarray:
    try:
        return scaler_or_stats.transform(X)
    except Exception:
        mean_ = getattr(scaler_or_stats, "mean_", None)
        scale_ = getattr(scaler_or_stats, "scale_", None)
        if mean_ is not None and scale_ is not None:
            return (X - mean_) / scale_
        if hasattr(scaler_or_stats, "mean0") and hasattr(scaler_or_stats, "std0"):
            return (X - scaler_or_stats.mean0.values) / scaler_or_stats.std0.values
        return X

def _safe_fd(X: np.ndarray, t: np.ndarray) -> np.ndarray:
    fd = ps.FiniteDifference()
    try:
        return fd(X, t)  # pysindy 2.x
    except Exception:
        return fd._differentiate(X, t)  # fallback

def _avg_dt(df_with_states: pd.DataFrame, user_col: str = None) -> float:
    if user_col is None:
        s = np.diff(np.sort(df_with_states['step_raw'].values))
        diffs = s if len(s) else np.array([1.0])
    else:
        series = df_with_states.groupby(user_col)['step_raw'].apply(
            lambda x: np.diff(np.sort(x.values))
        )
        diffs = np.concatenate([d for d in series if len(d) > 0]) if len(series) > 0 else np.array([1.0])
    dt = float(np.clip(np.nanmedian(diffs), 1e-6, None))
    return dt

def _choose_group_key_for_sequences(df: pd.DataFrame,
                                    min_len: int = 2,
                                    user_key: str = "userID",
                                    dest_key: str = "nameDest") -> str:
    def score(key: str) -> int:
        gsize = df.groupby(key).size()
        gfraud = df.groupby(key)["Class"].max()
        return int(((gfraud == 1) & (gsize >= min_len)).sum())
    s_user = score(user_key)
    s_dest = score(dest_key)
    picked = dest_key if s_dest > s_user else user_key
    log(f"[group-key] user={s_user} vs dest={s_dest} -> picked='{picked}' (min_len={min_len})")
    return picked

# ======================================================================
# STEP 1, 2
# ======================================================================
def step1_load_and_prepare() -> Path:
    log("STEP1: 데이터 로딩/전처리...")
    df = pd.read_csv(CFG.raw_csv_path)
    df = df[df["type"].isin(["CASH_OUT", "TRANSFER"])].copy()
    df["step_raw"] = df["step"].astype(float)

    for c in ["amount", "oldbalanceOrg"]:
        if c in df.columns:
            df[f"{c}_raw"] = df[c].astype(float)
    if "newbalanceOrig" in df.columns:
        df.drop(columns=["newbalanceOrig"], inplace=True)

    leak_cols = ["oldbalanceDest", "newbalanceDest"]
    df.drop(columns=[c for c in leak_cols if c in df.columns], inplace=True)

    df = pd.get_dummies(df, columns=["type"], prefix="type", drop_first=True)
    df.rename(columns={CFG.user_col_raw: CFG.user_col, CFG.target_col: "Class"}, inplace=True)
    df.drop(columns=[c for c in CFG.drop_cols if c in df.columns], inplace=True)

    df.replace([np.inf, -np.inf], 0.0, inplace=True)
    df.fillna(0.0, inplace=True)

    clean_path = ART / "step1_clean.parquet"
    save_df(df, clean_path)
    log(f"전처리 완료 -> {clean_path.name}")

    if getattr(CFG, "force_user_group_key", False):
        CFG.user_col = "userID"
    else:
        CFG.user_col = _choose_group_key_for_sequences(
            df, min_len=2, user_key=CFG.user_col, dest_key="nameDest"
        )
    (ART / "group_key.txt").write_text(CFG.user_col, encoding="utf-8")
    log(f"[group-key] using '{CFG.user_col}' for all subsequent steps")
    return clean_path

def step1_load_or_prepare() -> Path:
    dst = ART / "step1_clean.parquet"
    if dst.exists():
        log("STEP1: 현재 런 캐시 사용.")
        gk = ART / "group_key.txt"
        if gk.exists():
            CFG.user_col = gk.read_text(encoding="utf-8").strip()
            log(f"[group-key] loaded from cache: '{CFG.user_col}'")
        return dst
    if RESUME_SRC is not None:
        src = RESUME_SRC / "step1_clean.parquet"
        if src.exists():
            log(f"STEP1: 재시작 소스에서 로드 -> 현재 런에 복사 저장 (src={RESUME_SRC.name})")
            df = load_df(src); save_df(df, dst)
            src_scaler = RESUME_SRC / "scaler_base.pkl"
            if src_scaler.exists():
                joblib.dump(joblib.load(src_scaler), ART / "scaler_base.pkl")
            gk_src = RESUME_SRC / "group_key.txt"
            if gk_src.exists():
                picked = gk_src.read_text(encoding="utf-8").strip()
                CFG.user_col = picked
                (ART / "group_key.txt").write_text(picked, encoding="utf-8")
                log(f"[group-key] loaded from resume source: '{CFG.user_col}'")
            return dst
    return step1_load_and_prepare()

def step2_group_split(clean_path: Path):
    log("STEP2: Group-aware *Stratified* Train/Test split...")
    df = load_df(clean_path)

    if CFG.temporal_holdout_enable:
        cfg_cut = CFG.temporal_cut_step
        q80 = float(df["step_raw"].quantile(0.8))
        cut = float(cfg_cut if cfg_cut is not None else q80)

        df_train_raw = df[df["step_raw"] <= cut].copy()
        df_test_raw  = df[df["step_raw"]  > cut].copy()

        inter = set(df_train_raw[CFG.user_col]) & set(df_test_raw[CFG.user_col])
        if inter:
            df_test_raw = df_test_raw[~df_test_raw[CFG.user_col].isin(inter)]

        if len(df_test_raw) == 0 or len(df_train_raw) == 0:
            log("[WARN] temporal split produced empty train/test → switching to group split.")
            CFG.temporal_holdout_enable = False
        else:
            num_cols = [c for c in CFG.numeric_cols if c in df.columns]
            df_train_scaled, df_test_scaled, scaler_base = fit_transform_scalers(df_train_raw, df_test_raw, num_cols)
            joblib.dump(scaler_base, ART / "scaler_base.pkl")
            detect_data_leakage(df_train_scaled, df_test_scaled, id_col=CFG.user_col, time_col="step_raw", label_col="Class")
            tr_stats = _split_stats(df_train_scaled, user_col=CFG.user_col, step_col="step_raw")
            te_stats = _split_stats(df_test_scaled,  user_col=CFG.user_col, step_col="step_raw")
            _emit_split_stats_artifacts(tr_stats, te_stats, ART / "run_stats")
            overlap = len(set(df_train_scaled[CFG.user_col]) & set(df_test_scaled[CFG.user_col]))
            log(f"[SPLIT] user-disjoint check: overlap={overlap}")
            save_df(df_train_scaled, ART / "train.parquet")
            save_df(df_test_scaled,  ART / "test.parquet")
            log("분할/스케일 완료 -> train/test 저장 (temporal holdout)")
            return

    user_has_fraud = df.groupby(CFG.user_col)["Class"].max()
    fraud_users  = user_has_fraud[user_has_fraud == 1].index.to_numpy()
    normal_users = user_has_fraud[user_has_fraud == 0].index.to_numpy()

    rng = np.random.RandomState(CFG.random_state)
    rng.shuffle(fraud_users); rng.shuffle(normal_users)

    n_f_test = max(1, int(len(fraud_users)  * CFG.test_size))
    n_n_test = max(1, int(len(normal_users) * CFG.test_size))

    test_users  = set(fraud_users[:n_f_test].tolist()  + normal_users[:n_n_test].tolist())
    train_users = set(fraud_users[n_f_test:].tolist() + normal_users[n_n_test:].tolist())

    if not any(user_has_fraud.loc[list(train_users)] == 1):
        if n_f_test > 0:
            u = fraud_users[0]; test_users.discard(u); train_users.add(u)
    if not any(user_has_fraud.loc[list(test_users)] == 1) and len(fraud_users) > 1:
        u = fraud_users[1]; train_users.discard(u); test_users.add(u)

    train_mask = df[CFG.user_col].isin(train_users)
    test_mask  = df[CFG.user_col].isin(test_users)

    df_train_raw = df[train_mask].copy()
    df_test_raw  = df[test_mask].copy()

    num_cols = [c for c in CFG.numeric_cols if c in df.columns]
    df_train_scaled, df_test_scaled, scaler_base = fit_transform_scalers(df_train_raw, df_test_raw, num_cols)
    joblib.dump(scaler_base, ART / "scaler_base.pkl")

    detect_data_leakage(df_train_scaled, df_test_scaled, id_col=CFG.user_col, time_col="step_raw", label_col="Class")
    overlap = len(set(df_train_scaled[CFG.user_col]) & set(df_test_scaled[CFG.user_col]))
    log(f"[SPLIT] user-disjoint check: overlap={overlap}")

    def _summ(d):
        uhf = d.groupby(CFG.user_col)["Class"].max()
        return dict(users=len(uhf), fraud_users=int((uhf==1).sum()), rows=len(d), fraud_rows=int(d["Class"].sum()))
    log(f"TRAIN summary: {_summ(df_train_scaled)}")
    log(f"TEST  summary: {_summ(df_test_scaled)}")
    tr_stats = _split_stats(df_train_scaled, user_col=CFG.user_col, step_col="step_raw")
    te_stats = _split_stats(df_test_scaled,  user_col=CFG.user_col, step_col="step_raw")
    _emit_split_stats_artifacts(tr_stats, te_stats, ART / "run_stats")
    save_df(df_train_scaled, ART / "train.parquet")
    save_df(df_test_scaled,  ART / "test.parquet")
    log("분할/스케일 완료 -> train/test 저장")

def step2_group_split_or_load(clean_path: Path):
    tr_dst, te_dst = ART / "train.parquet", ART / "test.parquet"
    if tr_dst.exists() and te_dst.exists():
        log("STEP2: 현재 런 캐시 사용.")
        return
    if RESUME_SRC is not None:
        tr_src, te_src = RESUME_SRC / "train.parquet", RESUME_SRC / "test.parquet"
        if tr_src.exists() and te_src.exists():
            log(f"STEP2: 재시작 소스에서 train/test 복사 (src={RESUME_SRC.name})")
            save_df(load_df(tr_src), tr_dst); save_df(load_df(te_src), te_dst)
            return
    step2_group_split(clean_path)

# ======================================================================
# STEP 3: 상태변수 엔지니어링
# ======================================================================
def engineer_state_variables(df_user: pd.DataFrame) -> pd.DataFrame:
    df_user = df_user.sort_values("step_raw").copy()
    denom = (df_user["oldbalanceOrg_raw"].abs() + 1e-6)
    df_user["x1_balance_exhaust"] = (df_user["amount_raw"] / denom).clip(-1e6, 1e6)
    df_user["x2_tx_freq"] = df_user["step_raw"].rolling(window=3, min_periods=1).apply(
        lambda x: len(x) / max((x.max() - x.min()), 1e-6), raw=False
    )
    df_user["x3_time_since_last"] = df_user["step_raw"].diff().fillna(0.0).clip(0, None)
    df_user["x4_amount_mean_w5"] = df_user["amount_raw"].rolling(window=5, min_periods=2).mean().ffill().fillna(0.0)
    df_user["x5_amount_std_w5"]  = df_user["amount_raw"].rolling(window=5, min_periods=2).std().fillna(0.0)
    return df_user.replace([np.inf, -np.inf], 0.0).fillna(0.0)

def build_states_by_group(df: pd.DataFrame, state_cols: List[str]) -> pd.DataFrame:
    return pd.concat(
        [g.pipe(engineer_state_variables) for _, g in tqdm(df.groupby(CFG.user_col), desc="Build State Vars")]
    )

def step3_engineer_or_load_states(train_df: pd.DataFrame, test_df: pd.DataFrame, state_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_dst = ART / "train_with_states.parquet"
    test_dst  = ART / "test_with_states.parquet"

    if _states_exist_and_valid(train_dst, state_cols) and _states_exist_and_valid(test_dst, state_cols):
        log("동적 상태 변수: 현재 런 캐시 사용.")
        return load_df(train_dst), load_df(test_dst)

    if RESUME_SRC is not None:
        train_src = RESUME_SRC / "train_with_states.parquet"
        test_src  = RESUME_SRC / "test_with_states.parquet"
        if _states_exist_and_valid(train_src, state_cols) and _states_exist_and_valid(test_src, state_cols):
            log(f"동적 상태 변수: 재시작 소스에서 로드 (src={RESUME_SRC.name})")
            tr, te = load_df(train_src), load_df(test_src)
            save_df(tr, train_dst); save_df(te, test_dst)
            return tr, te

    log("동적 상태 변수 생성 중...")
    tr = build_states_by_group(train_df, state_cols)
    te = build_states_by_group(test_df, state_cols)
    save_df(tr, train_dst); save_df(te, test_dst)
    log("동적 상태 변수 저장 완료 -> train/test_with_states.parquet")
    return tr, te

# ======================================================================
# STEP 4: SINDy 모델
# ======================================================================
from sklearn.metrics.pairwise import cosine_similarity

def _cos_sim_models(model_a, model_b, state_vars: List[str]) -> float:
    def _term_weights(model, state_vars) -> Dict[str, float]:
        try:
            names = model.get_feature_names(input_features=state_vars)
        except Exception:
            C = np.asarray(model.coefficients())
            names = [f"phi_{i}" for i in range(C.shape[0] if C.ndim >= 1 else 0)]
        C = np.asarray(model.coefficients())
        if C.ndim == 1: C = C.reshape(-1, 1)
        if C.shape[0] != len(names) and C.shape[1] == len(names):
            C = C.T
        n_terms = min(len(names), C.shape[0])
        names = names[:n_terms]; C = C[:n_terms, :]
        w = np.linalg.norm(C, axis=1)
        return {term: weight for term, weight in zip(names, w)}

    wa = _term_weights(model_a, state_vars)
    wb = _term_weights(model_b, state_vars)
    all_terms = sorted(set(wa) | set(wb))
    va = np.array([wa.get(t, 0.0) for t in all_terms], dtype=float)
    vb = np.array([wb.get(t, 0.0) for t in all_terms], dtype=float)
    if not np.any(va) and not np.any(vb): return 1.0
    if not np.any(va) or not np.any(vb):  return 0.0
    return float(cosine_similarity(va.reshape(1, -1), vb.reshape(1, -1))[0, 0])

# === SINDy Model Selection (유사도 + sparsity 제어) ===
def _fit_sindy_once(X_list, t_list, state_cols, degree, include_interaction, include_bias, threshold, diff):
    lib = ps.PolynomialLibrary(
        degree=degree,
        include_interaction=include_interaction,
        include_bias=include_bias
    )
    # SR3로 교체(희소성 강제)
    opt = _make_sr3(threshold)
    if opt is None:
        # SR3 시그니처가 맞지 않는 버전이면 STLSQ로 즉시 폴백
        opt = _make_stlsq(threshold)

    m = ps.SINDy(
        feature_library=lib,
        optimizer=opt,
        differentiation_method=diff,
    )

    m.fit(X_list, t=t_list, multiple_trajectories=True)
    C = np.asarray(m.coefficients())
    if C.ndim == 1:
        C = C.reshape(-1, 1)
    nnz = int(np.count_nonzero(np.abs(C) > 1e-9))
    return m, nnz

def _zero_small_coeffs(model, eps: float = 1e-3) -> int:
    """
    학습된 SINDy 모델 계수 중 |coef| < eps 값을 0으로 강제.
    가능한 다양한 내부 경로를 시도하고, 실제로 0으로 만든 개수를 반환.
    """
    zeroed = 0
    try:
        C = np.asarray(model.coefficients())
        if C.ndim == 1:
            C = C.reshape(-1, 1)
        mask = np.abs(C) < float(eps)
        zeroed = int(mask.sum())
        C2 = C.copy()
        C2[mask] = 0.0

        # 일반 경로 1: optimizer.coef_
        try:
            if hasattr(model, "optimizer") and hasattr(model.optimizer, "coef_"):
                if model.optimizer.coef_ is not None:
                    model.optimizer.coef_ = C2.T if C2.shape[0] != model.optimizer.coef_.shape[-1] else C2
                    return zeroed
        except Exception:
            pass

        # 일반 경로 2: private 속성들
        for attr in ["coef_", "_coef_list", "_coefficients", "_coefs"]:
            if hasattr(model, attr):
                try:
                    setattr(model, attr, C2)
                    return zeroed
                except Exception:
                    continue
    except Exception:
        pass
    return zeroed


def _nnz_of(model) -> int:
    C = np.asarray(model.coefficients())
    if C.ndim == 1:
        C = C.reshape(-1, 1)
    return int(np.count_nonzero(np.abs(C) > 1e-12))

def _pick_best_sindy(n_ts, n_t, f_ts, f_t, state_cols, diff_n, diff_f,
                     degree=2, include_interaction=False, include_bias=False,
                     thr_grid=(3000.0, 2000.0, 1500.0, 1000.0, 800.0, 600.0, 400.0, 300.0, 200.0, 150.0, 120.0, 100.0, 80.0, 60.0, 40.0, 30.0, 20.0, 10.0, 7.0, 5.0, 3.0, 2.0),
                     target_nnz=(3, 8), cos_tol=0.90):
    """
    - thr_grid 확장(큰 threshold 우선)으로 희소성 유도
    - 정상/사기 모두 '희소성 범위'를 만족하지 못하면 스킵
    - 최종 선택 후, 사후 소거(_zero_small_coeffs)로 추가 희소화
    """
    mid = 0.5 * (target_nnz[0] + target_nnz[1])

    # 1) 정상 모델: 희소성 범위 만족하는 후보만 허용
    best_n = None
    best_n_rank = (1e9, 1e9)  # (|nnz-mid|, nnz)
    for thr in thr_grid:
        mn, nnz = _fit_sindy_once(n_ts, n_t, state_cols, degree,
                                  include_interaction, include_bias, thr, diff_n)
        if not (target_nnz[0] <= nnz <= target_nnz[1]):
            continue
        gap = abs(nnz - mid)
        rank = (gap, nnz)
        if rank < best_n_rank:
            best_n, best_n_rank = mn, rank

    # 희소성 충족 후보가 하나도 없으면, nnz가 mid에 가장 가까운 후보 중 가장 큰 thr 결과를 선택
    if best_n is None:
        best_n = None
        best_n_rank = (1e9, 1e9, -1.0)  # (|nnz-mid|, nnz, thr_negative)
        for thr in thr_grid:
            mn, nnz = _fit_sindy_once(n_ts, n_t, state_cols, degree,
                                      include_interaction, include_bias, thr, diff_n)
            gap = abs(nnz - mid)
            rank = (gap, nnz, -thr)  # 큰 thr 선호
            if rank < best_n_rank:
                best_n, best_n_rank = mn, rank

    # 2) 사기 모델: 정상과의 유사도↓ + 희소성 범위
    best_f = None
    best_f_rank = (True, 2.0, 1e9, 1e9)  # (sim>cos_tol, sim, |nnz-mid|, nnz)
    for thr in thr_grid:
        mf, nnz = _fit_sindy_once(f_ts, f_t, state_cols, degree,
                                  include_interaction, include_bias, thr, diff_f)
        sim = _cos_sim_models(best_n, mf, state_cols)
        valid_sparse = (target_nnz[0] <= nnz <= target_nnz[1])
        if not valid_sparse:
            continue
        rank = (sim > cos_tol, sim, abs(nnz - mid), nnz)
        if rank < best_f_rank:
            best_f, best_f_rank = mf, rank

    # 사기 모델도 못 찾았으면, 유사도 최소 + nnz mid 근접 + 큰 thr 선호
    if best_f is None:
        best_f = None
        best_f_rank = (True, 2.0, 1e9, 1e9, -1.0)  # (sim>cos_tol, sim, |nnz-mid|, nnz, -thr)
        for thr in thr_grid:
            mf, nnz = _fit_sindy_once(f_ts, f_t, state_cols, degree,
                                      include_interaction, include_bias, thr, diff_f)
            sim = _cos_sim_models(best_n, mf, state_cols)
            rank = (sim > cos_tol, sim, abs(nnz - mid), nnz, -thr)
            if rank < best_f_rank:
                best_f, best_f_rank = mf, rank

    # 3) 사후 소거로 추가 희소화
    _zero_small_coeffs(best_n, eps=1e-3)
    _zero_small_coeffs(best_f, eps=1e-3)

    # 4) 최종 유사도 반환
    sim_final = _cos_sim_models(best_n, best_f, state_cols)
    return best_n, best_f, sim_final

def _save_sindy_equations_pretty(model, state_cols, out_path: Path, tag: str,
                                 coef_fmt="{:+.3e}", sort_by="abs", max_terms=None,
                                 style="plain"):
    # 계수/특징명 안전 추출
    try:
        names = model.get_feature_names(input_features=state_cols)
    except Exception:
        C = np.asarray(model.coefficients())
        n_terms = (C.shape[0] if C.ndim >= 1 else 0)
        names = [f"phi_{i}" for i in range(n_terms)]
    C = np.asarray(model.coefficients())
    if C.ndim == 1:
        C = C.reshape(-1, 1)
    if C.shape[0] != len(names) and C.shape[1] == len(names):
        C = C.T
    n_terms = min(len(names), C.shape[0])
    names = names[:n_terms]; C = C[:n_terms, :]

    def _order(col):
        vals = C[:, col]
        return np.argsort(-np.abs(vals)) if sort_by == "abs" else np.arange(len(vals))

    def _latex_escape(s: str) -> str:
        return (s.replace('\\', r'\textbackslash{}')
                 .replace('_', r'\_')
                 .replace('%', r'\%')
                 .replace('&', r'\&')
                 .replace('#', r'\#')
                 .replace('{', r'\{')
                 .replace('}', r'\}'))

    lines = []
    for j, xj in enumerate(state_cols):
        idx = _order(j)
        # term 수집
        terms = []
        for i in idx:
            coef = float(C[i, j])
            if abs(coef) <= 1e-12:
                continue
            phi = names[i]
            terms.append((abs(coef), f"{coef_fmt.format(coef)}*({phi})"))
        if max_terms is not None:
            terms = terms[:max_terms]

        if style == "latex":
            terms_ltx = []
            for i in idx:
                coef = float(C[i, j])
                if abs(coef) <= 1e-12:
                    continue
                phi = _latex_escape(names[i])
                terms_ltx.append(f"{coef_fmt.format(coef)}\\cdot({phi})")
                if max_terms is not None and len(terms_ltx) >= max_terms:
                    break
            rhs = " + ".join(terms_ltx) if terms_ltx else "0"
            line = rf"\frac{{d\,{_latex_escape(xj)}}}{{dt}} = {rhs}"
        else:
            rhs = " + ".join([t[1] for t in terms]) if terms else "0"
            line = f"d({xj})/dt = {rhs}"

        lines.append(line)

    out_path.write_text("\n".join(lines), encoding="utf-8")
    log(f"[SINDy {tag}] 방정식 저장 -> {out_path.name}")
    print("\n".join(lines[:2]), flush=True)

def _print_sindy_stats(model, tag):
    C = np.asarray(model.coefficients())
    if C.ndim == 1: C = C.reshape(-1, 1)
    nnz = int(np.count_nonzero(np.abs(C) > 1e-9))
    total = int(C.size)
    l2 = float(np.linalg.norm(C))
    log(f"{tag}: coef shape={C.shape}, nnz={nnz}/{total} ({nnz/total:.2%}), L2={l2:.3e}")

def step4_train_sindy_models(train_df_with_states: pd.DataFrame, state_cols: list):
    log("STEP4: SINDy 정상/사기 모델 학습…")

    # 상태변수 수치형 강제
    for c in state_cols:
        train_df_with_states[c] = pd.to_numeric(train_df_with_states[c], errors="coerce").fillna(0.0).astype(np.float64)

    # 라벨 인지형 표준화 + z-스케일
    ccs_stats = fit_ccs_stats(train_df_with_states, state_cols)
    train_df_ccs = apply_ccs(train_df_with_states, ccs_stats, label_col="Class")
    zscaler = StandardScaler().fit(train_df_ccs[state_cols].values)
    train_df_ccs[state_cols] = zscaler.transform(train_df_ccs[state_cols].values)
    joblib.dump(zscaler, ART / "sindy_zscaler.joblib")

    # 윈도우 궤적
    n_ts, n_t, f_ts, f_t, w_used, s_used = build_window_labeled_trajectories(
        train_df_ccs, state_cols, window=CFG.sindy_window, stride=CFG.sindy_stride, max_traj_per_class=CFG.max_traj_per_class
    )
    log(f"SINDy 궤적 수: normal={len(n_ts)}, fraud={len(f_ts)} (window={w_used}, stride={s_used})")
    if len(f_ts) == 0 or len(n_ts) == 0:
        raise RuntimeError("SINDy 궤적이 부족합니다. 파라미터를 완화하세요.")

    # 미분기 선택(신호 길이에 따라)
    def _make_diff_for_length(n: int):
        if n >= 7:
            return ps.SmoothedFiniteDifference(smoother_kws={"window_length":7, "polyorder":2})
        if n >= 5:
            return ps.SmoothedFiniteDifference(smoother_kws={"window_length":5, "polyorder":2})
        return ps.FiniteDifference()
    min_len_n = min(x.shape[0] for x in n_ts)
    min_len_f = min(x.shape[0] for x in f_ts)
    diff_n = _make_diff_for_length(min_len_n)
    diff_f = _make_diff_for_length(min_len_f)

    # 선택 루틴으로 정상/사기 동시 최적화(유사도 + 희소성)
    t0 = time.perf_counter()
    model_n, model_f, sim = _pick_best_sindy(
        n_ts, n_t, f_ts, f_t, state_cols, diff_n, diff_f,
        degree=CFG.sindy_degree,
        include_interaction=CFG.sindy_include_interaction,
        include_bias=CFG.sindy_include_bias,
        thr_grid=(3000.0, 2000.0, 1500.0, 1000.0, 800.0, 600.0, 400.0,
                    300.0, 200.0, 150.0, 120.0, 100.0, 80.0, 60.0, 40.0,
                    30.0, 20.0, 10.0, 7.0, 5.0, 3.0, 2.0),
        target_nnz=(5, 12),
        cos_tol=CFG.sindy_retry_cos_tol  # 권장: 0.90
    )
    log(f"SINDy 선택 완료 ({time.perf_counter()-t0:.1f}s), cos_sim={sim:.4f}")

    for _ in range(3):
        z = _zero_small_coeffs(model_n, eps=5e-2)
        z += _zero_small_coeffs(model_f, eps=5e-2)
        if z == 0: break
        # 필요시 같은 데이터로 다시 fit (multiple_trajectories=True)
        model_n.fit(n_ts, t=n_t, multiple_trajectories=True)
        model_f.fit(f_ts, t=f_t, multiple_trajectories=True)

    _print_sindy_stats(model_n, "Normal")
    _print_sindy_stats(model_f, "Fraud")
    _save_sindy_equations_pretty(model_n, state_cols, ART / "sindy_equations_normal.txt",
                                 tag="NORMAL", coef_fmt="{:+.3e}", sort_by="abs",
                                 max_terms=None, style="plain")
    _save_sindy_equations_pretty(model_f, state_cols, ART / "sindy_equations_fraud.txt",
                                 tag="FRAUD", coef_fmt="{:+.3e}", sort_by="abs",
                                 max_terms=None, style="plain")


    # 추론용 스케일러(정상 기준)
    inference_scaler = StandardScaler()
    inference_scaler.n_features_in_ = len(state_cols)
    inference_scaler.mean_  = ccs_stats.mean0.values.astype(float)
    inference_scaler.scale_ = ccs_stats.std0.values.astype(float)

    dump_safely(model_n, ART / "sindy_model_normal.dill")
    dump_safely(model_f, ART / "sindy_model_fraud.dill")
    # 스케일러는 pickle 가능하므로 그대로 joblib 사용 OK (원하시면 통일해도 무방)
    joblib.dump(inference_scaler, ART / "sindy_inference_scaler.joblib")
    return model_n, model_f, inference_scaler, state_cols

def step4_load_or_train_sindy_models(train_df_with_states: pd.DataFrame, state_cols: list):
    n_dst = ART / "sindy_model_normal.dill"
    f_dst = ART / "sindy_model_fraud.dill"
    sc_dst = ART / "sindy_inference_scaler.joblib"

    n_dst_gz = Path(str(n_dst) + ".gz")
    f_dst_gz = Path(str(f_dst) + ".gz")

    if (n_dst.exists() or n_dst_gz.exists()) and (f_dst.exists() or f_dst_gz.exists()) and sc_dst.exists():
        log("SINDy 모델: 현재 런 캐시 사용.")
        return load_safely(n_dst), load_safely(f_dst), joblib.load(sc_dst), state_cols

    if RESUME_SRC is not None:
        n_src = RESUME_SRC / "sindy_model_normal.dill"
        f_src = RESUME_SRC / "sindy_model_fraud.dill"
        sc_src = RESUME_SRC / "sindy_inference_scaler.joblib"
        n_src_gz = Path(str(n_src) + ".gz")
        f_src_gz = Path(str(f_src) + ".gz")
        if (n_src.exists() or n_src_gz.exists()) and (f_src.exists() or f_src_gz.exists()) and sc_src.exists():
            n, f, sc = load_safely(n_src), load_safely(f_src), joblib.load(sc_src)
            dump_safely(n, n_dst); dump_safely(f, f_dst); joblib.dump(sc, sc_dst)
            return n, f, sc, state_cols

    return step4_train_sindy_models(train_df_with_states, state_cols)

# ======================================================================
# STEP 5: SINDy 특징 계산 + 분류기 학습
# ======================================================================
def calculate_sindy_features(df: pd.DataFrame, model_n, model_f, scaler, state_cols) -> pd.DataFrame:
    log(f"SINDy 동적 특징 계산 중 ({len(df)} rows)...")
    sindy_features = pd.DataFrame(index=df.index)
    sindy_features[CFG.user_col] = df[CFG.user_col].values
    sindy_features["step_raw"] = df["step_raw"].values

    mse_n_full = np.zeros(len(df), dtype=float)
    mse_f_full = np.zeros(len(df), dtype=float)

    zscaler = None
    try:
        z_path = ART / "sindy_zscaler.joblib"
        if z_path.exists():
            zscaler = joblib.load(z_path)
    except Exception:
        pass

    # 추가: 전역 위치 인덱서 (df.index -> 0..N-1)
    pos_indexer = pd.Series(np.arange(len(df)), index=df.index)

    for _, group in tqdm(df.groupby(CFG.user_col), desc="SINDy MSE per user"):
        group = group.sort_values("step_raw")
        idx = group.index.values
        pos = pos_indexer.loc[idx].values  # ← 위치 인덱스로 변환
        if len(group) < 3:
            continue
        X = group[state_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
        t = make_strictly_increasing(group['step_raw'].values)
        X_scaled = _safe_scale(X, scaler)
        if zscaler is not None:
            try: X_scaled = zscaler.transform(X_scaled)
            except Exception: pass

        # calculate_sindy_features() 안, X_scaled 만든 직후
        X_scaled = np.clip(X_scaled, -3.0, 3.0)

        try:
            Xdot = _safe_fd(X_scaled, t)
        except Exception:
            continue

        def _rhs(m, X_):
            try: return m.predict(X_)
            except Exception:
                try: return m._rhs(X_)
                except Exception: return None

        Xdot_hat_n = _rhs(model_n, X_scaled)
        Xdot_hat_f = _rhs(model_f, X_scaled)
        if Xdot_hat_n is None or Xdot_hat_f is None:
            continue

        resid_n = Xdot - Xdot_hat_n
        resid_f = Xdot - Xdot_hat_f
        mse_n = np.einsum('ij,ij->i', resid_n, resid_n)
        mse_f = np.einsum('ij,ij->i', resid_f, resid_f)
        L = min(len(pos), len(mse_n), len(mse_f))
        mse_n_full[pos[:L]] = mse_n[:L]
        mse_f_full[pos[:L]] = mse_f[:L]

    sindy_features['sindy_mse_normal'] = mse_n_full
    sindy_features['sindy_mse_fraud']  = mse_f_full
    sindy_features['sindy_mse_ratio']  = np.log((sindy_features['sindy_mse_normal'] + 1e-8) /
                                                (sindy_features['sindy_mse_fraud'] + 1e-8))

    # 사용자 단위 과거 집계(누수 방지: shift(1))
    sindy_features = sindy_features.sort_values([CFG.user_col, "step_raw"])
    def _q90_exp(s): return s.expanding().quantile(0.9).shift(1)

    for col in ["sindy_mse_normal", "sindy_mse_fraud", "sindy_mse_ratio"]:
        sindy_features[f"{col}_user_mean"]   = (
            sindy_features.groupby(CFG.user_col, group_keys=False)[col].apply(lambda s: s.expanding().mean().shift(1)).fillna(0.0)
        )
        sindy_features[f"{col}_user_median"] = (
            sindy_features.groupby(CFG.user_col, group_keys=False)[col].apply(lambda s: s.expanding().median().shift(1)).fillna(0.0)
        )
        sindy_features[f"{col}_user_p90"] = (
            sindy_features.groupby(CFG.user_col, group_keys=False)[col].apply(_q90_exp).fillna(0.0)
        )
        sindy_features[f"{col}_diff1"] = sindy_features.groupby(CFG.user_col)[col].diff().fillna(0.0)
        sindy_features[f"{col}_z_roll20"] = (
            sindy_features.groupby(CFG.user_col)[col]
            .transform(lambda s: (s - s.rolling(20, min_periods=5).mean()) /
                                (s.rolling(20, min_periods=5).std() + 1e-9))
            .fillna(0.0)
        )
    # --- 여기부터 추가: 롤링 p90 (윈도우 기반) ---
    def _q90_roll(s, w=20):
        return s.rolling(window=w, min_periods=3).quantile(0.9).shift(1)

    for col in ["sindy_mse_normal","sindy_mse_fraud","sindy_mse_ratio"]:
        sindy_features[f"{col}_user_p90_roll20"] = (
            sindy_features.groupby(CFG.user_col, group_keys=False)[col].apply(_q90_roll).fillna(0.0)
        )
    # --- 추가 끝 ---

    sindy_features = sindy_features.sort_index()
    sindy_features['sindy_pred'] = (sindy_features['sindy_mse_fraud'] < sindy_features['sindy_mse_normal']).astype(int)
    return sindy_features

def _drop_constant_and_align(Xtr: pd.DataFrame, Xte: pd.DataFrame):
    nunique = Xtr.nunique(dropna=False)
    keep_cols = nunique[nunique > 1].index.tolist()
    if keep_cols:
        stds = Xtr[keep_cols].std(ddof=0)
        keep_cols = [c for c in keep_cols if stds[c] > 0]
    if not keep_cols:
        log("[WARN] 모든 컬럼이 상수로 판정되어 모델을 건너뜁니다.")
        return Xtr.iloc[:, :0], Xte.iloc[:, :0]
    dropped = [c for c in Xtr.columns if c not in keep_cols]
    if dropped:
        log(f"[WARN] 상수/무의미 컬럼 제거: {dropped[:8]}{'...' if len(dropped)>8 else ''}")
    return Xtr[keep_cols], Xte[keep_cols]

def step5_train_models_and_hybrid(sindy_n, sindy_f, scaler, state_cols):
    log("STEP5: SINDy 특징 결합 후 분류기 학습")
    dump_environment_snapshot()  # ← 재현성 아티팩트 저장(수치 영향 없음)
    train_df_base = load_df(ART / "train_with_states.parquet")
    test_df_base  = load_df(ART / "test_with_states.parquet")

    # 1) SINDy 특징 생성
    train_sindy = calculate_sindy_features(train_df_base, sindy_n, sindy_f, scaler, state_cols)
    test_sindy  = calculate_sindy_features(test_df_base,  sindy_n, sindy_f, scaler, state_cols)

    # --- VAL 슬라이스: train 안에서 step_raw 상위 10%만 검증으로 사용(임계값 선정용, 재학습 없음)
    assert "step_raw" in train_df_base.columns, "train_with_states에 step_raw가 필요합니다."
    cut = float(train_df_base["step_raw"].quantile(0.90))
    val_mask = (train_df_base["step_raw"] >= cut)
    train_mask = ~val_mask

    # 베이스라인/하이브리드용 입력 행렬 재구성(훈련·검증 나눔)
    def _make_xy(df_base, df_sindy, mask):
        dfb = df_base.loc[mask]
        dfs = df_sindy.loc[mask]
        Xb = dfb.drop(columns=['Class', CFG.user_col, 'step_raw'], errors="ignore")
        Xh = pd.concat([Xb.reset_index(drop=True), 
                        dfs.filter(regex=r'^sindy_').reset_index(drop=True)], axis=1)
        y  = dfb['Class'].astype(int).values
        # 숫자화/결측/무한대 보정 (기존 파이프라인과 동일 처리)
        Xb = (Xb.replace([np.inf,-np.inf], np.nan).apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(np.float64))
        Xh = (Xh.replace([np.inf,-np.inf], np.nan).apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(np.float64))
        return Xb, Xh, y

    X_train_base_val, X_train_hybrid_val, y_val = _make_xy(train_df_base, train_sindy, val_mask)

    # 2) SINDy-only 레퍼런스
    mu_p90 = float(train_sindy["sindy_mse_ratio_user_p90"].mean())
    sd_p90 = float(train_sindy["sindy_mse_ratio_user_p90"].std() + 1e-9)
    test_prob_sindy = expit((test_sindy["sindy_mse_ratio_user_p90"] - mu_p90) / sd_p90).astype(float)
    y_test = test_df_base["Class"].astype(int).values
    eval_from_proba(y_test, (test_sindy["sindy_mse_fraud"] < test_sindy["sindy_mse_normal"]).astype(float), "sindy_only_binary")
    sanity_check_shuffle(y_test, (test_sindy["sindy_mse_fraud"] < test_sindy["sindy_mse_normal"]).astype(float), "sindy_only_binary")
    eval_from_proba(y_test, test_prob_sindy.values, "sindy_only_score")
    sanity_check_shuffle(y_test, test_prob_sindy.values, "sindy_only_score")

    # 3) 하이브리드 테이블 (누수/금지 컬럼 제거 + 수치화)
    train_df_h = ensure_unique_columns(pd.concat([train_df_base, train_sindy.drop(columns=[CFG.user_col])], axis=1), where="train_hybrid")
    test_df_h  = ensure_unique_columns(pd.concat([test_df_base,  test_sindy.drop(columns=[CFG.user_col])], axis=1), where="test_hybrid")

    train_df_h = train_df_h.drop(columns=[c for c in FORBID_COLS if c in train_df_h.columns], errors="ignore")
    test_df_h  = test_df_h.drop(columns=[c for c in FORBID_COLS if c in test_df_h.columns], errors="ignore")

    train_df_h = to_numeric_df(train_df_h)
    test_df_h  = to_numeric_df(test_df_h)
    train_df_h = boost_sindy_columns(train_df_h, factor=4.0)
    test_df_h  = boost_sindy_columns(test_df_h,  factor=4.0)

    save_df(train_df_h, ART / "train_hybrid_features.parquet")
    save_df(test_df_h,  ART / "test_hybrid_features.parquet")

    # 4) 학습 데이터 구성(+SMOTE)
    feature_block = (
        train_df_h.drop(columns=['Class', CFG.user_col, 'step_raw'], errors="ignore")
                  .select_dtypes(include=[np.number])
                  .replace([np.inf, -np.inf], np.nan)
                  .fillna(0.0)
    )
    X_train_full = feature_block.copy()
    y_train_full = train_df_h['Class'].astype(int).values

    const_mask = X_train_full.std(ddof=0) > 1e-12
    X_train_full = X_train_full.loc[:, const_mask]

    pos = int((y_train_full == 1).sum())
    neg = int((y_train_full == 0).sum())
    min_class = min(pos, neg)
    use_smote = (pos > 0 and neg > 0 and min_class >= 2)

    if use_smote:
        k_neighbors = max(1, min(5, min_class - 1))
        r = float(CFG.smote_classifier_sampling)
        r = max(0.01, min(1.0, r))
        sm = SMOTE(random_state=CFG.random_state, k_neighbors=k_neighbors, sampling_strategy=r)
        X_train_sm, y_train_sm = sm.fit_resample(X_train_full, y_train_full)
        spw_effective = 1.0
    else:
        log("[WARN] 소수 클래스가 너무 작아 SMOTE 미적용. class_weight 기반으로 진행.")
        X_train_sm, y_train_sm = X_train_full.copy(), y_train_full.copy()
        spw_effective = neg / max(1, pos)

    if not isinstance(X_train_sm, pd.DataFrame):
        X_train_sm = pd.DataFrame(X_train_sm, columns=X_train_full.columns)
    X_train_sm = X_train_sm.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # 테스트 특징
    X_test_full = (
        test_df_h.drop(columns=['Class', CFG.user_col, 'step_raw'], errors="ignore")
                 .select_dtypes(include=[np.number])
                 .replace([np.inf, -np.inf], np.nan)
                 .fillna(0.0)
    )
    # 정렬/정합
    common_cols = sorted(list(set(X_train_sm.columns) & set(X_test_full.columns)))
    X_train_sm = X_train_sm[common_cols]
    X_test_full = X_test_full[common_cols]

    # 5) 모델들
    base_lgb = lgb.LGBMClassifier(**make_lgb_params(spw_effective, CFG.use_gpu))
    base_xgb = xgb.XGBClassifier(
        random_state=CFG.random_state,
        scale_pos_weight=spw_effective,
        n_estimators=1000,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        max_bin=256,
        eval_metric='aucpr',
        enable_categorical=False
    )
    models = {
        "logistic_regression": CalibratedClassifierCV(
            LogisticRegression(random_state=CFG.random_state, max_iter=2000, class_weight="balanced"),
            method="isotonic", cv=3
        ),
        "decision_tree": CalibratedClassifierCV(
            DecisionTreeClassifier(random_state=CFG.random_state, class_weight="balanced", max_depth=None, min_samples_leaf=5),
            method="isotonic", cv=3
        ),
        "lightgbm": CalibratedClassifierCV(base_lgb, method="isotonic", cv=3),
        "xgboost": CalibratedClassifierCV(base_xgb, method="isotonic", cv=3),
    }

    # 상수/불일치 제거(이중 안전)
    X_train_use, X_test_use = _drop_constant_and_align(X_train_sm, X_test_full)

    # 학습 & 평가 & 저장
    for name, clf in models.items():
        if X_train_use.shape[1] == 0:
            log(f"[WARN] {name}: 사용 가능한 특징이 없어 스킵.")
            continue
        clf.fit(X_train_use, y_train_sm)
        prob = clf.predict_proba(X_test_use)[:, 1]
        eval_from_proba(y_test, prob, name)
        # --- 추가 보고: τ_val (하이브리드용) ---
        hyb_cols_ref = list(X_train_sm.columns)  # SMOTE 후 최종 학습 칼럼
        X_val_hyb_aligned = _align_columns(X_train_hybrid_val, hyb_cols_ref)
        prob_val = clf.predict_proba(X_val_hyb_aligned)[:, 1]
        tau_val = _tau_max_f1(y_val, prob_val)
        _dump_tau(f"{name}_tau_val", tau_val, ART / f"{name}_tau_val.json")
        eval_from_proba(y_test, prob, f"{name}_tau_val", fixed_thr=tau_val)
        reliability_plot(y_test, prob, ART / f"reliability_{name}.png")
        sanity_check_shuffle(y_test, prob, name)
        save_feature_list(name, X_train_use.columns.tolist())
        joblib.dump(clf, ART / f"model_{name}.joblib")
        # Permutation Importance (트리 계열만)
        if name in ("lightgbm", "xgboost"):
            try:
                base_est = _extract_fitted_tree_for_shap(clf)  # ← 통일된 내부 추출
                permutation_importance_safely(base_est, X_test_use, y_test, ART / f"perm_importance_{name}.csv", n_repeats=5)
            except Exception as e:
                log(f"[WARN] PI 실패({name}): {e}")

    # === (베이스라인) SINDy 미사용 버전과 공정 비교 ===
    X_train_base = (
        train_df_base.drop(columns=['Class', CFG.user_col, 'step_raw'], errors="ignore")
                    .select_dtypes(include=[np.number]).replace([np.inf,-np.inf], np.nan).fillna(0.0)
    )
    X_test_base = (
        test_df_base.drop(columns=['Class', CFG.user_col, 'step_raw'], errors="ignore")
                    .select_dtypes(include=[np.number]).replace([np.inf,-np.inf], np.nan).fillna(0.0)
    )

    common0 = sorted(list(set(X_train_base.columns) & set(X_test_base.columns)))
    X_train_base = X_train_base[common0]
    X_test_base  = X_test_base[common0]

    # ★ 베이스라인은 원본 y를 사용 (SMOTE/리샘플 금지)
    y_train_base = train_df_base['Class'].astype(int).values

    # (선택) 안전 체크
    if len(X_train_base) != len(y_train_base):
        raise RuntimeError(f"[Baseline] X_train_base({len(X_train_base)}) vs y_train_base({len(y_train_base)}) 길이 불일치")
    
    _xgb_gpu0 = xgb_has_gpu() and CFG.use_gpu
    base_lgb0 = lgb.LGBMClassifier(**make_lgb_params(spw_effective, CFG.use_gpu))
    base_xgb0 = xgb.XGBClassifier(
        random_state=CFG.random_state, scale_pos_weight=spw_effective,
        n_estimators=1000, max_depth=5, subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", max_bin=256, eval_metric='aucpr', enable_categorical=False
    )

    # 확률 보정: XGBoost는 캘리브레이션 후 비교 (isotonic)
    cal_xgb0 = CalibratedClassifierCV(base_xgb0, method="isotonic", cv=3)

    for name, mdl in {
        "baseline_lightgbm": base_lgb0,
        "baseline_xgboost": cal_xgb0,  # ← 보정된 XGB로 교체
    }.items():
        mdl.fit(X_train_base, y_train_base)
        # 학습에 사용한 열 목록 저장
        base_cols_ref = list(X_train_base.columns)
        prob = mdl.predict_proba(X_test_base)[:, 1]
        eval_from_proba(y_test, prob, name)
        # --- 추가 보고: τ_val로 고정한 테스트 성능 ---
        X_val_base_aligned = _align_columns(X_train_base_val, base_cols_ref)
        prob_val = mdl.predict_proba(X_val_base_aligned)[:, 1]
        tau_val = _tau_max_f1(y_val, prob_val)
        _dump_tau(f"{name}_tau_val", tau_val, ART / f"{name}_tau_val.json")
        eval_from_proba(y_test, prob, f"{name}_tau_val", fixed_thr=tau_val)
        sanity_check_shuffle(y_test, prob, name)

        # === 저장 (SHAP용) ===
        joblib.dump(mdl, ART / f"model_{name}.joblib")
        save_feature_list(name, list(X_train_base.columns))

    # --- 추가: Baseline Logistic / DecisionTree (SINDy 미포함) ---
    baseline_more = {
        "baseline_logistic_regression": CalibratedClassifierCV(
            LogisticRegression(random_state=CFG.random_state,
                               max_iter=2000,
                               class_weight="balanced"),
            method="isotonic", cv=3
        ),
        "baseline_decision_tree": CalibratedClassifierCV(
            DecisionTreeClassifier(random_state=CFG.random_state,
                                   class_weight="balanced",
                                   max_depth=None,
                                   min_samples_leaf=5),
            method="isotonic", cv=3
        ),
    }
    for name, mdl in baseline_more.items():
        mdl.fit(X_train_base, y_train_base)
        # 학습에 사용한 열 목록 저장
        base_cols_ref = list(X_train_base.columns)
        prob = mdl.predict_proba(X_test_base)[:, 1]
        eval_from_proba(y_test, prob, name)
        # --- 추가 보고: τ_val로 고정한 테스트 성능 ---
        X_val_base_aligned = _align_columns(X_train_base_val, base_cols_ref)
        prob_val = mdl.predict_proba(X_val_base_aligned)[:, 1]
        tau_val = _tau_max_f1(y_val, prob_val)
        _dump_tau(f"{name}_tau_val", tau_val, ART / f"{name}_tau_val.json")
        eval_from_proba(y_test, prob, f"{name}_tau_val", fixed_thr=tau_val)
        sanity_check_shuffle(y_test, prob, name)
        joblib.dump(mdl, ART / f"model_{name}.joblib")
        save_feature_list(name, list(X_train_base.columns))

    # === 하이브리드가 베이스라인에 지면 SINDy 톱-K만 써서 재시도 (간이 오토튜닝) ===
    try:
        m_h_lgb = json.load(open(ART / "metrics_lightgbm.json"))["avg_precision"]
        m_b_lgb = json.load(open(ART / "metrics_baseline_lightgbm.json"))["avg_precision"]
        if m_h_lgb < m_b_lgb - 1e-4:
            log("[AUTO] Hybrid < Baseline(LGBM) → SINDy 상위 특징만 선택해 재학습")
            sindy_cols = [c for c in X_train_sm.columns if c.startswith("sindy_")]
            pri = [c for c in sindy_cols if ("ratio" in c and "p90" in c)] + \
                  [c for c in sindy_cols if ("ratio" in c and "median" in c)] + \
                  [c for c in sindy_cols if ("ratio" in c and "mean" in c)] + \
                  [c for c in sindy_cols if "mse_" in c]
            base_cols = [c for c in X_train_sm.columns if not c.startswith("sindy_")]
            keep = base_cols + pri[:8]  # 톱-8 SINDy만 사용
            keep = sorted(set(keep) & set(X_train_sm.columns) & set(X_test_full.columns))
            X_train_use2 = X_train_sm[keep]; X_test_use2 = X_test_full[keep]
            lgb2 = CalibratedClassifierCV(
                lgb.LGBMClassifier(**make_lgb_params(spw_effective, CFG.use_gpu)),
                method="isotonic", cv=3
            )
            lgb2.fit(X_train_use2, y_train_sm)
            prob2 = lgb2.predict_proba(X_test_use2)[:,1]
            eval_from_proba(y_test, prob2, "lightgbm_hybrid_topSINDy")
            reliability_plot(y_test, prob2, ART / "reliability_lightgbm_hybrid_topSINDy.png")
            sanity_check_shuffle(y_test, prob2, "lightgbm_hybrid_topSINDy")
            joblib.dump(lgb2, ART / "model_lightgbm_hybrid_topSINDy.joblib")
            save_feature_list("lightgbm_hybrid_topSINDy", list(X_train_use2.columns))
    except Exception as _e:
        log(f"[AUTO] fallback 재학습 스킵: {_e}")

# ======================================================================
# STEP 6: SHAP
# ======================================================================
def _extract_fitted_tree_for_shap(model):
    try:
        if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
            cc = model.calibrated_classifiers_[0]
            if hasattr(cc, "estimator") and cc.estimator is not None:
                return cc.estimator
            if hasattr(cc, "base_estimator") and cc.base_estimator is not None:
                return cc.base_estimator
    except Exception:
        pass
    return model

def step6_run_shap():
    log("STEP6: SHAP 분석 실행...")

    # === 대상 모델과 사용할 특징 테이블 *및 해당 라벨* 매핑 ===
    test_hybrid = load_df(ART / "test_hybrid_features.parquet")
    X_test_hybrid = (
        test_hybrid.drop(columns=['Class', CFG.user_col, 'step_raw'], errors="ignore")
                   .replace([np.inf, -np.inf], np.nan)
                   .apply(pd.to_numeric, errors="coerce")
                   .fillna(0.0)
                   .astype(np.float64)
    )
    y_hybrid = test_hybrid['Class'].astype(int)            # ← 라벨(Series, 인덱스 유지)

    test_base = load_df(ART / "test_with_states.parquet")
    X_test_base = (
        test_base.drop(columns=['Class', CFG.user_col, 'step_raw'], errors="ignore")
                 .replace([np.inf, -np.inf], np.nan)
                 .apply(pd.to_numeric, errors="coerce")
                 .fillna(0.0)
                 .astype(np.float64)
    )
    y_base = test_base['Class'].astype(int)                # ← 라벨(Series, 인덱스 유지)

    targets = [
        ("lightgbm",           X_test_hybrid, y_hybrid),
        ("baseline_lightgbm",  X_test_base,   y_base),
        ("lightgbm_hybrid_topSINDy", X_test_hybrid, y_hybrid),
    ]

    for name, X_pool, y_pool in targets:
        model_path = ART / f"model_{name}.joblib"
        cols_path  = ART / f"{name}_cols.json"
        if not model_path.exists() or not cols_path.exists():
            log(f"[SHAP] {name}: 모델/컬럼 파일 없음 → 스킵")
            continue

        model = joblib.load(model_path)
        feature_cols = json.load(open(cols_path, "r"))
        X_for_model = align_features(X_pool, feature_cols).astype(np.float64)

        # 인덱스 정합된 라벨 준비
        y_for_model = y_pool.loc[X_for_model.index].values

        # 샘플링(클래스 보존)
        k = min(CFG.shap_sample, len(X_for_model))
        if k < len(X_for_model):
            from sklearn.model_selection import StratifiedShuffleSplit
            sss = StratifiedShuffleSplit(n_splits=1, test_size=k, random_state=CFG.random_state)
            _, idx = next(sss.split(X_for_model, y_for_model))
            X_shap = X_for_model.iloc[idx]
        else:
            X_shap = X_for_model

        # Calibrated 래퍼 내부 트리 추출 (있으면)
        tree_for_shap = _extract_fitted_tree_for_shap(model)

        # SHAP 계산 & 저장  (이진분류는 양성(1) 클래스 기준으로 고정)
        try:
            explainer = shap.TreeExplainer(tree_for_shap)
            shap_values = explainer.shap_values(X_shap)
        except Exception:
            explainer = shap.TreeExplainer(tree_for_shap)
            shap_values = explainer.shap_values(X_shap.to_numpy(dtype=np.float64))

        # --- PATCH: shap_values 정규화 (list -> 양성 클래스, ndarray 보장) ---
        if isinstance(shap_values, list):
            if len(shap_values) == 2:
                # 일반적인 이진분류: [neg, pos]
                sv = np.asarray(shap_values[1], dtype=np.float64)
            else:
                # 드문 다중분류 대비: 평균 절대 SHAP가 가장 큰 클래스를 선택
                idx = int(np.argmax([np.mean(np.abs(np.asarray(sv_i))) for sv_i in shap_values]))
                sv = np.asarray(shap_values[idx], dtype=np.float64)
        else:
            sv = np.asarray(shap_values, dtype=np.float64)

        # 차원 정합 보정: (n_samples, n_features) 형태로 강제
        if sv.ndim == 3:
            # (n_class, n_samples, n_features) 같은 형태가 오는 경우
            if sv.shape[0] == 2:
                sv = sv[1]  # 양성 클래스
            else:
                sv = sv[0]
        elif sv.ndim == 1:
            sv = sv.reshape(-1, 1)
        # --------------------------------------------------------------------

        # 통계 저장
        pd.DataFrame({
            "feature": list(X_shap.columns),
            "mean_abs_shap": np.mean(np.abs(sv), axis=0)
        }).sort_values("mean_abs_shap", ascending=False) \
         .to_csv(ART / f"shap_stats_{name}.csv", index=False)

        # 요약 플롯 저장
        plt.figure(figsize=PLOT_FIGSIZE_WIDE, dpi=PLOT_DPI)
        shap.summary_plot(sv, X_shap, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(ART / f"shap_{name}.png", dpi=PLOT_DPI, bbox_inches="tight")
        plt.close('all')
        log(f"[SHAP] {name}: shap_{name}.png / shap_stats_{name}.csv 저장 완료")

# ======================================================================
# STEP 6B: PDP / ICE
# ======================================================================
def step6b_run_pdp():
    log("STEP6B: PDP/ICE 생성...")

    # 하이브리드(증강) 테스트셋과 베이스라인 테스트셋 로드
    test_hybrid = load_df(ART / "test_hybrid_features.parquet")
    X_test_hybrid = (
        test_hybrid.drop(columns=['Class', CFG.user_col, 'step_raw'], errors="ignore")
                   .replace([np.inf, -np.inf], np.nan)
                   .apply(pd.to_numeric, errors="coerce")
                   .fillna(0.0)
                   .astype(np.float64)
    )
    test_base = load_df(ART / "test_with_states.parquet")
    X_test_base = (
        test_base.drop(columns=['Class', CFG.user_col, 'step_raw'], errors="ignore")
                 .replace([np.inf, -np.inf], np.nan)
                 .apply(pd.to_numeric, errors="coerce")
                 .fillna(0.0)
                 .astype(np.float64)
    )

    targets = [
        ("lightgbm",           X_test_hybrid),
        ("baseline_lightgbm",  X_test_base),
        ("xgboost",            X_test_hybrid),
        ("baseline_xgboost",   X_test_base),
    ]

    for name, X_pool in targets:
        model_path = ART / f"model_{name}.joblib"
        cols_path  = ART / f"{name}_cols.json"
        if not model_path.exists() or not cols_path.exists():
            log(f"[PDP] {name}: 모델/컬럼 파일 없음 → 스킵")
            continue

        model = joblib.load(model_path)
        feature_cols = json.load(open(cols_path, "r"))
        X = align_features(X_pool, feature_cols).astype(np.float64)

        # 캘리브레이션 래퍼일 수 있으므로 내부 트리 추출
        base_for_pdp = _extract_fitted_tree_for_shap(model)

        # 중요도 기반 상위 특징 고르기(있으면 perm_importance, 없으면 분산)
        imp_csv = ART / f"perm_importance_{name}.csv"
        if imp_csv.exists():
            imp_df = pd.read_csv(imp_csv).sort_values("importance_mean", ascending=False)
            top_feats = imp_df["feature"].tolist()[:CFG.pdp_features_topk]
            # 논문 본문/부록과 동일한 PDP를 보장하기 위한 필수 목록
            must_have = [
                "amount_raw", "oldbalanceOrg_raw",
                "x1_balance_exhaust", "x2_tx_freq", "x3_time_since_last"
            ]
            # 중복 제거 + 우선 포함
            top_feats = list(dict.fromkeys(must_have + top_feats))[:CFG.pdp_features_topk]
        else:
            stds = X.std().sort_values(ascending=False)
            top_feats = stds.index.tolist()[:CFG.pdp_features_topk]

        for f in top_feats:
            # 상수/준상수 피처 스킵
            if f not in X.columns or float(X[f].std(ddof=0)) < 1e-8:
                log(f"[PDP] skip nearly-constant feature: {name}/{f}")
                continue

            try:
                fig, ax = plt.subplots(figsize=PLOT_FIGSIZE_WIDE, dpi=PLOT_DPI)
                PartialDependenceDisplay.from_estimator(
                    base_for_pdp, X, [f],
                    kind="both",
                    grid_resolution=CFG.pdp_grid_resolution,
                    percentiles=(0.005, 0.995),   # ← 퍼센타일 범위 확대
                    ax=ax,
                )
                plt.tight_layout()
                out = ART / f"pdp_{name}_{f}.png"
                fig.savefig(out, dpi=PLOT_DPI, bbox_inches='tight')
                plt.close(fig)        # ← 해당 fig 즉시 닫기
                plt.close('all')      # ← 혹시 남은 핸들러도 정리
                log(f"[PDP] saved -> {out.name}")
            except Exception as e:
                log(f"[WARN][PDP] {name}/{f}: {e}")

# ======================================================================
# STEP 6C: LIME (국소 설명)
# ======================================================================
def step6c_run_lime():
    if LimeTabularExplainer is None:
        log("[LIME] lime 패키지가 설치되어 있지 않습니다 (pip install lime). 스킵.")
        return
    log("STEP6C: LIME 국소 설명 생성...")

    # 훈련/테스트 테이블 로드
    train_h = load_df(ART / "train_hybrid_features.parquet")
    test_h  = load_df(ART / "test_hybrid_features.parquet")

    # 대상: 하이브리드 라이트GBM과 XGBoost (캘리브레이션 래퍼 그대로 사용 가능)
    targets = [
        ("lightgbm",          test_h),
        ("xgboost",           test_h),
        ("baseline_lightgbm", load_df(ART / 'test_with_states.parquet')),
        ("baseline_xgboost",  load_df(ART / 'test_with_states.parquet')),
        ("baseline_logistic_regression", load_df(ART / 'test_with_states.parquet')),
        ("baseline_decision_tree",      load_df(ART / 'test_with_states.parquet')),
    ]

    for name, test_tbl in targets:
        model_path = ART / f"model_{name}.joblib"
        cols_path  = ART / f"{name}_cols.json"
        if not model_path.exists() or not cols_path.exists():
            log(f"[LIME] {name}: 모델/컬럼 파일 없음 → 스킵")
            continue

        model = joblib.load(model_path)
        feature_cols = json.load(open(cols_path, "r"))

        # 훈련/테스트에서 특징 정합
        X_train_full = (
            train_h.drop(columns=['Class', CFG.user_col, 'step_raw'], errors="ignore")
                   .replace([np.inf, -np.inf], np.nan)
                   .apply(pd.to_numeric, errors="coerce")
                   .fillna(0.0)
                   .astype(np.float64)
        )
        X_train = align_features(X_train_full, feature_cols).astype(np.float64)

        X_test_full = (
            test_tbl.drop(columns=['Class', CFG.user_col, 'step_raw'], errors="ignore")
                    .replace([np.inf, -np.inf], np.nan)
                    .apply(pd.to_numeric, errors="coerce")
                    .fillna(0.0)
                    .astype(np.float64)
        )
        X_test = align_features(X_test_full, feature_cols).astype(np.float64)

        # Explainer 학습(연속형 디스크리타이즈: 신용카드 데이터에서 안정적)
        explainer = LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=feature_cols,
            class_names=['normal','fraud'],
            discretize_continuous=True,
            mode='classification',
            random_state=CFG.random_state,
        )

        # 샘플링: 예측 확률 상위 K개 (fraud=1 쪽)
        probs = np.asarray(model.predict_proba(X_test)[:, 1])
        order = np.argsort(-probs)  # 내림차순
        idx = order[:min(CFG.lime_samples, len(order))]

        for j, i in enumerate(idx):
            try:
                exp = explainer.explain_instance(
                    X_test.iloc[i].values, 
                    model.predict_proba,  # CalibratedClassifierCV 포함 호환
                    num_features=10
                )
                fig = exp.as_pyplot_figure()
                plt.tight_layout()
                out = ART / f"lime_{name}_{i}.png"
                fig.savefig(out, dpi=PLOT_DPI, bbox_inches='tight')
                plt.close('all')

                # 가독성을 위해 CSV로도 저장
                pairs = exp.as_list()  # [(feature, weight), ...]
                pd.DataFrame(pairs, columns=["feature_condition", "weight"]) \
                  .to_csv(ART / f"lime_{name}_{i}.csv", index=False)
                log(f"[LIME] saved -> {out.name}")
            except Exception as e:
                log(f"[WARN][LIME] {name} idx={i}: {e}")

# ======================================================================
# STEP 8: 전통 모델 간단 시각화(트리, 로지스틱)
# ======================================================================
def step8_visualize_classic_models():
    log("STEP8: 결정 트리/로지스틱 간단 시각화...")
    models = {}
    for p, key in [
        (ART / "model_decision_tree.joblib", "decision_tree"),
        (ART / "model_logistic_regression.joblib", "logistic_regression"),
        (ART / "model_baseline_decision_tree.joblib", "baseline_decision_tree"),
        (ART / "model_baseline_logistic_regression.joblib", "baseline_logistic_regression"),
    ]:
        if p.exists(): models[key] = joblib.load(p)
    if not models:
        log("[STEP8] 시각화 대상 없음.")
        return

    feature_names = None
    for cp in [ART / "decision_tree_cols.json", ART / "logistic_regression_cols.json"]:
        if cp.exists():
            feature_names = json.load(open(cp, "r"))
            break
    if feature_names is None:
        te = load_df(ART / "test_hybrid_features.parquet")
        feature_names = te.select_dtypes(include=[np.number]).columns.difference(['Class', CFG.user_col, 'step_raw']).tolist()

    if "decision_tree" in models:
        est = models["decision_tree"]
        try:
            if hasattr(est, "calibrated_classifiers_") and est.calibrated_classifiers_:
                est = est.calibrated_classifiers_[0].estimator
        except Exception:
            pass
        if hasattr(est, "tree_"):
            fig = plt.figure(figsize=PLOT_FIGSIZE_TALL, dpi=PLOT_DPI)  # 또는 PLOT_FIGSIZE_WIDE
            from sklearn import tree as _tree
            _tree.plot_tree(est, feature_names=feature_names, filled=True, max_depth=4)
            fig.tight_layout()
            fig.savefig(ART / "tree_preview.png", dpi=PLOT_DPI, bbox_inches="tight")
            plt.close(fig)
            log("결정 트리 시각화 저장 -> tree_preview.png")

    if "logistic_regression" in models:
        est = models["logistic_regression"]
        try:
            if hasattr(est, "calibrated_classifiers_") and est.calibrated_classifiers_:
                est = est.calibrated_classifiers_[0].estimator
        except Exception:
            pass
        if hasattr(est, "coef_") and est.coef_ is not None:
            coefs = est.coef_.ravel()
            idx = np.argsort(np.abs(coefs))[::-1][:30]
            plt.figure(figsize=PLOT_FIGSIZE_WIDE, dpi=PLOT_DPI)
            plt.bar(range(len(idx)), coefs[idx])
            plt.xticks(range(len(idx)), [feature_names[i] for i in idx], rotation=75, ha="right")
            plt.tight_layout()
            plt.savefig(ART / "logreg_coeffs.png", dpi=PLOT_DPI, bbox_inches="tight")
            plt.close()
            log("로지스틱 회귀 시각화 저장 -> logreg_coeffs.png")

# ======================================================================
# Main
# ======================================================================
def main():
    save_json(asdict(CFG), ART / "config.json")
    # 1. 데이터 준비
    clean_path = step1_load_or_prepare()
    # 2. 그룹 분할
    step2_group_split_or_load(clean_path)
    # 3. 동적 상태 변수
    log("엔지니어링: 동적 상태 변수 생성/로드...")
    train_df = load_df(ART / "train.parquet")
    test_df  = load_df(ART / "test.parquet")
    state_cols = ["x1_balance_exhaust","x2_tx_freq","x3_time_since_last","x4_amount_mean_w5","x5_amount_std_w5"]
    train_with_states, test_with_states = step3_engineer_or_load_states(train_df, test_df, state_cols)
    # 4. SINDy 학습/로드
    sindy_n, sindy_f, scaler, state_cols = step4_load_or_train_sindy_models(train_with_states, state_cols)
    # 5. 분류기 학습(로지스틱/DT/LGB/XGB) + SINDy-only 레퍼런스
    step5_train_models_and_hybrid(sindy_n, sindy_f, scaler, state_cols)
    # 6. SHAP (트리 계열)
    if CFG.run_shap:
        step6_run_shap()

    # === [ADD] PDP / LIME ===
    if CFG.run_pdp:
        step6b_run_pdp()
    if CFG.run_lime:
        step6c_run_lime()
    # 8. 간단 시각화
    step8_visualize_classic_models()
    log("=== 모든 단계 완료 ===")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"[FATAL] {e}")
        import traceback
        traceback.print_exc()
