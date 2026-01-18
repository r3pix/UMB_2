from __future__ import annotations

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from src.utils import FEATURE_NAMES, zscore_fit, zscore_apply, metrics_binary, shannon_entropy, safe_datetime

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

EPS = 1e-10


def clean_common(D: pd.DataFrame) -> pd.DataFrame:
    # Algorytm 5: inf -> NaN, NaN -> mediana, usuń prawie zerową wariancję
    D = D.replace([np.inf, -np.inf], np.nan)
    D = D.fillna(D.median(numeric_only=True))
    num_cols = D.select_dtypes(include=[np.number]).columns
    low_var = num_cols[D[num_cols].var() < 1e-10]
    return D.drop(columns=list(low_var), errors="ignore")


def cicids_build_features(df: pd.DataFrame) -> pd.DataFrame:
    # Upewnij się, że nazwy kolumn są bez spacji
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Minimalny zestaw 7 cech dostępnych w MachineLearningCSV
    feature_cols = [
        "Destination Port",
        "Flow Packets/s",
        "Average Packet Size",
        "Flow Duration",
        "SYN Flag Count",
        "Total Fwd Packets",
        "Total Backward Packets",
    ]

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Brak wymaganych kolumn CICIDS2017 (ML CSV): {missing}")

    X = df[feature_cols].copy()

    # Konwersja do float + czyszczenie inf/nan
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    return X


def unsw_build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Bazujemy na kolumnach, które masz w CSV:
    required = ["dur", "spkts", "dpkts", "sbytes", "dbytes"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Brak wymaganych kolumn UNSW: {missing}")

    # Konwersja do liczbowych
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    dur = df["dur"]
    total_pkts = df["spkts"] + df["dpkts"]
    total_bytes = df["sbytes"] + df["dbytes"]

    # 7 cech (analogicznie do idei z Z1/Z2)
    X = pd.DataFrame({
        "packets_per_sec": total_pkts / dur.replace(0, np.nan),
        "avg_packet_size": total_bytes / total_pkts.replace(0, np.nan),
        "dur": dur,
        "rate": pd.to_numeric(df["rate"], errors="coerce") if "rate" in df.columns else np.nan,
        "sttl": pd.to_numeric(df["sttl"], errors="coerce") if "sttl" in df.columns else np.nan,
        "total_fwd_pkts": df["spkts"],
        "total_bwd_pkts": df["dpkts"],
    })

    # czyszczenie inf/nan
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    return X


def load_cicids(monday_csv: str, ddos_csv: str, portscan_csv: str) -> tuple[pd.DataFrame, np.ndarray]:
    D1 = pd.read_csv(monday_csv, low_memory=False)
    D2 = pd.read_csv(ddos_csv, low_memory=False)
    D3 = pd.read_csv(portscan_csv, low_memory=False)

    for D in (D1, D2, D3):
        D.columns = D.columns.str.strip()

    D = pd.concat([D1, D2, D3], ignore_index=True)

    # etykieta
    if "Label" not in D.columns:
        raise ValueError("CICIDS2017: brak kolumny Label")

    # strip wartości label (czasem bywa " BENIGN")
    labels = D["Label"].astype(str).str.strip().str.upper()
    y = (labels != "BENIGN").astype(int).to_numpy()

    X = cicids_build_features(D)
    return X, y


def load_unsw(train_csv: str, test_csv: str, label_col: str = "label") -> tuple[pd.DataFrame, np.ndarray]:
    Dtr = pd.read_csv(train_csv, low_memory=False)
    Dte = pd.read_csv(test_csv, low_memory=False)

    # strip kolumn (dla spójności z CICIDS)
    Dtr.columns = Dtr.columns.str.strip()
    Dte.columns = Dte.columns.str.strip()

    D = pd.concat([Dtr, Dte], ignore_index=True)

    if label_col not in D.columns:
        raise ValueError(f"UNSW: brak kolumny {label_col}")

    y = pd.to_numeric(D[label_col], errors="coerce").fillna(0).astype(int).to_numpy()
    X = unsw_build_features(D)
    return X, y


def split_60_20_20(X: np.ndarray, y: np.ndarray):
    idx = np.arange(len(y))
    I_tr, I_tmp = train_test_split(idx, train_size=0.6, random_state=42, stratify=y)
    I_val, I_te = train_test_split(I_tmp, train_size=0.5, random_state=42, stratify=y[I_tmp])
    return I_tr, I_val, I_te


def train_eval(Xdf: pd.DataFrame, y: np.ndarray, out_dir: str):
    X = Xdf.to_numpy(dtype=float)
    I_tr, I_val, I_te = split_60_20_20(X, y)
    X_tr, y_tr = X[I_tr], y[I_tr]
    X_te, y_te = X[I_te], y[I_te]

    mu, sigma = zscore_fit(X_tr)
    X_trz = zscore_apply(X_tr, mu, sigma)
    X_tez = zscore_apply(X_te, mu, sigma)

    M = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=42)
    M.fit(X_trz, y_tr)

    proba = M.predict_proba(X_tez)[:, 1]
    yhat = (proba >= 0.5).astype(int)
    cm = confusion_matrix(y_te, yhat)
    met = metrics_binary(y_te, yhat, proba)

    # zapisz
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame([met]).to_csv(os.path.join(out_dir, "task3_metrics.csv"), index=False)
    pd.DataFrame(cm, index=["True0", "True1"], columns=["Pred0", "Pred1"]).to_csv(
        os.path.join(out_dir, "task3_confusion.csv")
    )
    pd.DataFrame({"feature": Xdf.columns, "beta": M.coef_[0]}).to_csv(
        os.path.join(out_dir, "task3_betas.csv"), index=False
    )

    return met, cm


def train_eval_models(Xdf: pd.DataFrame, y: np.ndarray, out_dir: str):
    """
    Uruchamia zestaw modeli na tym samym podziale 60/20/20 (używamy test),
    z tym samym skalowaniem Z-score fitowanym na train.
    Zapisuje:
      - models_metrics.csv (wszystkie modele)
      - per model: confusion + coef/importances (jeśli dostępne)
      - per model: predictions.csv (KROK 0: do ROC/PR/CM bez retrenowania)
    """
    X = Xdf.to_numpy(dtype=float)
    I_tr, I_val, I_te = split_60_20_20(X, y)
    X_tr, y_tr = X[I_tr], y[I_tr]
    X_te, y_te = X[I_te], y[I_te]

    # skalowanie (SVM potrzebuje; drzewa/XGB też przeżyją)
    mu, sigma = zscore_fit(X_tr)
    X_trz = zscore_apply(X_tr, mu, sigma)
    X_tez = zscore_apply(X_te, mu, sigma)

    # imbalance ratio do XGB
    pos = max(int(y_tr.sum()), 1)
    neg = max(int((y_tr == 0).sum()), 1)
    scale_pos_weight = neg / pos

    models: list[tuple[str, object]] = []

    # 0) Logistic Regression (baseline)
    models.append((
        "LogisticRegression",
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        )
    ))

    # 1) Decision Tree
    models.append((
        "DecisionTree",
        DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced",
            max_depth=None
        )
    ))

    # 2) Random Forest
    models.append((
        "RandomForest",
        RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        )
    ))

    # 3) SVM RBF (Kernel / KVM)
    # (zostawiam probability=True, bo u Ciebie działa; jeśli będzie wolno -> probability=False)
    models.append((
        "SVM_RBF",
        SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=42
        )
    ))

    # 4) XGBoost
    if XGBClassifier is not None:
        models.append((
            "XGBoost",
            XGBClassifier(
                n_estimators=400,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight,
            )
        ))

    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for name, M in models:
        print(f"[Task3] Training {name} ...")

        # === SUBSAMPLING TYLKO DLA SVM ===
        if name == "SVM_RBF":
            MAX_SVM_SAMPLES = 20000  # możesz zmienić na 10000 / 30000
            if len(X_trz) > MAX_SVM_SAMPLES:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(X_trz), size=MAX_SVM_SAMPLES, replace=False)
                X_fit = X_trz[idx]
                y_fit = y_tr[idx]
                print(f"[Task3]   SVM trained on subsample: {len(X_fit)} samples")
            else:
                X_fit = X_trz
                y_fit = y_tr
            M.fit(X_fit, y_fit)
        else:
            M.fit(X_trz, y_tr)

        # proba / score
        if hasattr(M, "predict_proba"):
            proba = M.predict_proba(X_tez)[:, 1]
        else:
            scores = M.decision_function(X_tez)
            mn, mx = float(np.min(scores)), float(np.max(scores))
            proba = (scores - mn) / (mx - mn + EPS)

        yhat = (proba >= 0.5).astype(int)
        cm = confusion_matrix(y_te, yhat)
        met = metrics_binary(y_te, yhat, proba)
        met["model"] = name
        rows.append(met)

        # ===== KROK 0: zapisz predykcje do późniejszych wykresów =====
        pd.DataFrame({
            "y_true": y_te,
            "y_pred": yhat,
            "y_score": proba,
        }).to_csv(
            os.path.join(out_dir, f"{name}_predictions.csv"),
            index=False
        )

        # zapis per model (CM + ważności)
        pd.DataFrame(cm, index=["True0", "True1"], columns=["Pred0", "Pred1"]).to_csv(
            os.path.join(out_dir, f"{name}_confusion.csv")
        )

        if hasattr(M, "coef_"):
            pd.DataFrame({"feature": Xdf.columns, "coef": M.coef_[0]}).to_csv(
                os.path.join(out_dir, f"{name}_coef.csv"), index=False
            )
        if hasattr(M, "feature_importances_"):
            pd.DataFrame({"feature": Xdf.columns, "importance": M.feature_importances_}).to_csv(
                os.path.join(out_dir, f"{name}_feature_importances.csv"), index=False
            )

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "models_metrics.csv"), index=False)
    print(f"[Task3] Saved metrics to {os.path.join(out_dir, 'models_metrics.csv')}")


if __name__ == "__main__":
    print("Użycie:")
    print("  CICIDS2017: uzupełnij ścieżki do Monday-WorkingHours.csv, Friday-...DDoS.csv, Friday-...PortScan.csv")
    print("  UNSW-NB15:  uzupełnij ścieżki do UNSW_NB15_training-set.csv i UNSW_NB15_testing-set.csv")
