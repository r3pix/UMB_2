from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

EPS = 1e-12

FEATURE_NAMES = [
    "packets_per_sec",
    "avg_packet_size_bytes",
    "port_entropy",
    "syn_ratio",
    "unique_dst_ips",
    "connection_duration_sec",
    "repeated_connections",
]

def zscore_fit(X_train: np.ndarray, eps: float = EPS):
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0) + eps
    return mu, sigma

def zscore_apply(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray):
    return (X - mu) / sigma

def metrics_binary(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision_pos": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "Recall_pos": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "F1_pos": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "AUC": float(roc_auc_score(y_true, y_proba)),
    }

def youden_opt_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y_true, y_proba)
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])

def cost_opt_threshold(y_true: np.ndarray, y_proba: np.ndarray, fn_cost: float = 100.0, fp_cost: float = 1.0):
    taus = np.arange(0.01, 1.00, 0.01)
    best = (float("inf"), 0.5)
    costs = []
    for tau in taus:
        yhat = (y_proba >= tau).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, yhat).ravel()
        cost = fn_cost * fn + fp_cost * fp
        costs.append(cost)
        if cost < best[0]:
            best = (cost, float(tau))
    return best[1], np.array(taus), np.array(costs)

def shannon_entropy(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    uniq, cnt = np.unique(values, return_counts=True)
    p = cnt / cnt.sum()
    return float(-(p * np.log2(p + 1e-10)).sum())

def safe_datetime(col: pd.Series) -> pd.Series:
    # CICIDS often uses string timestamps; try best-effort parsing
    s = pd.to_datetime(col, errors="coerce", utc=False)
    if s.isna().all():
        # sometimes timestamp is numeric (epoch)
        s = pd.to_datetime(pd.to_numeric(col, errors="coerce"), unit="s", errors="coerce")
    return s
