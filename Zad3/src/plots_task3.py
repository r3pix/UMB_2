import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

try:
    import seaborn as sns
except Exception:
    sns = None

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None


# ---------- helpers ----------

def _list_prediction_files(pred_dir: str):
    return sorted(
        [f for f in os.listdir(pred_dir) if f.endswith("_predictions.csv")]
    )

def _read_preds(pred_dir: str, fname: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(pred_dir, fname))
    # sanity
    for c in ["y_true", "y_pred", "y_score"]:
        if c not in df.columns:
            raise ValueError(f"{fname}: brak kolumny {c}")
    return df

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ---------- 1) ROC + PR (many models) ----------

def plot_roc_pr(pred_dir: str, out_dir: str, title_suffix: str = ""):
    _ensure_dir(out_dir)
    files = _list_prediction_files(pred_dir)
    if not files:
        raise ValueError(f"Brak *_predictions.csv w {pred_dir}")

    # ROC
    plt.figure(figsize=(7, 6))
    for f in files:
        name = f.replace("_predictions.csv", "")
        df = _read_preds(pred_dir, f)
        fpr, tpr, _ = roc_curve(df["y_true"], df["y_score"])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC – all models {title_suffix}".strip())
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "roc_all_models.png"), dpi=200)
    plt.close()

    # PR
    plt.figure(figsize=(7, 6))
    for f in files:
        name = f.replace("_predictions.csv", "")
        df = _read_preds(pred_dir, f)
        p, r, _ = precision_recall_curve(df["y_true"], df["y_score"])
        plt.plot(r, p, label=name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision–Recall – all models {title_suffix}".strip())
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pr_all_models.png"), dpi=200)
    plt.close()


# ---------- 2) Confusion matrices (per model) ----------

def plot_confusion_matrices(pred_dir: str, out_dir: str):
    _ensure_dir(out_dir)
    files = _list_prediction_files(pred_dir)
    if not files:
        raise ValueError(f"Brak *_predictions.csv w {pred_dir}")

    for f in files:
        name = f.replace("_predictions.csv", "")
        df = _read_preds(pred_dir, f)
        cm = confusion_matrix(df["y_true"], df["y_pred"])

        plt.figure(figsize=(4.2, 4.0))
        if sns is not None:
            sns.heatmap(
                cm, annot=True, fmt="d",
                xticklabels=["Pred 0", "Pred 1"],
                yticklabels=["True 0", "True 1"],
                cbar=False
            )
        else:
            plt.imshow(cm)
            for (i, j), v in np.ndenumerate(cm):
                plt.text(j, i, str(v), ha="center", va="center")
            plt.xticks([0, 1], ["Pred 0", "Pred 1"])
            plt.yticks([0, 1], ["True 0", "True 1"])

        plt.title(f"Confusion Matrix – {name}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{name}_cm.png"), dpi=200)
        plt.close()


# ---------- 3) Decision boundaries on PCA(2) ----------

def _make_models_for_pca(scale_pos_weight: float = 1.0):
    models = [
        ("LogisticRegression", LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42)),
        ("DecisionTree", DecisionTreeClassifier(random_state=42, class_weight="balanced")),
        ("RandomForest", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1)),
        ("SVM_RBF", SVC(kernel="rbf", probability=False, class_weight="balanced", random_state=42)),
    ]
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
    return models

def plot_decision_boundaries_pca(
    Xdf: pd.DataFrame,
    y: np.ndarray,
    out_dir: str,
    dataset_name: str = "",
    max_points: int = 20000
):
    """
    Rysuje granice decyzyjne w przestrzeni PCA(2).
    Uwaga: modele są trenowane na PCA2 (to jest wizualizacja, nie “główne wyniki”).
    """
    _ensure_dir(out_dir)

    # subsample do rysowania (żeby było szybko i czytelnie)
    n = len(y)
    if n > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=max_points, replace=False)
        X = Xdf.to_numpy(dtype=float)[idx]
        yy = y[idx]
    else:
        X = Xdf.to_numpy(dtype=float)
        yy = y

    # scale_pos_weight dla XGB (na podstawie yy)
    pos = max(int(yy.sum()), 1)
    neg = max(int((yy == 0).sum()), 1)
    spw = neg / pos

    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X)

    # siatka
    x_min, x_max = X2[:, 0].min(), X2[:, 0].max()
    y_min, y_max = X2[:, 1].min(), X2[:, 1].max()
    xx, yy_grid = np.meshgrid(
        np.linspace(x_min, x_max, 350),
        np.linspace(y_min, y_max, 350),
    )
    grid = np.c_[xx.ravel(), yy_grid.ravel()]

    for name, model in _make_models_for_pca(scale_pos_weight=spw):
        print(f"[PCA2] boundary for {dataset_name} – {name}")
        model.fit(X2, yy)
        Z = model.predict(grid).reshape(xx.shape)

        plt.figure(figsize=(6.2, 5.2))
        plt.contourf(xx, yy_grid, Z, alpha=0.25)
        plt.scatter(X2[:, 0], X2[:, 1], c=yy, s=6)
        title = f"Decision boundary (PCA2) – {name}"
        if dataset_name:
            title += f" ({dataset_name})"
        plt.title(title)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{name}_decision_boundary_pca2.png"), dpi=200)
        plt.close()


# ---------- 4) KDE / hist features ----------

def plot_feature_distributions(
    Xdf: pd.DataFrame,
    y: np.ndarray,
    out_dir: str,
    dataset_name: str = ""
):
    """
    Dla każdej cechy:
      - histogram (normal vs attack)
      - KDE (normal vs attack) – jeśli dane pozwalają (ciągłe)
    """
    _ensure_dir(out_dir)

    for col in Xdf.columns:
        safe_col = (
            col.replace(" ", "_")
                .replace("/", "_per_")
                .replace("(", "")
                .replace(")", "")
        )   

        x0 = pd.to_numeric(Xdf.loc[y == 0, col], errors="coerce").dropna()
        x1 = pd.to_numeric(Xdf.loc[y == 1, col], errors="coerce").dropna()

        # histogram
        plt.figure(figsize=(6.5, 4.5))
        plt.hist(x0, bins=50, alpha=0.5, density=True, label="Normal")
        plt.hist(x1, bins=50, alpha=0.5, density=True, label="Attack")
        title = f"Histogram – {col}"
        if dataset_name:
            title += f" ({dataset_name})"
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"hist_{safe_col}.png"), dpi=200)
        plt.close()

        # KDE (czasem nie ma sensu dla portów, ale zostawiamy)
        # KDE (może się wysypać dla danych dyskretnych / małej liczby unikatów)
        plt.figure(figsize=(6.5, 4.5))
        try:
            # warunki sensowności KDE
            ok0 = (len(x0) >= 50) and (x0.nunique() >= 10)
            ok1 = (len(x1) >= 50) and (x1.nunique() >= 10)

            if ok0:
                x0.plot(kind="kde", label="Normal")
            if ok1:
                x1.plot(kind="kde", label="Attack")

            if ok0 or ok1:
                title = f"KDE – {col}"
                if dataset_name:
                    title += f" ({dataset_name})"
                plt.title(title)
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f"kde_{safe_col}.png"), dpi=200)
            # jeśli KDE nie ma sensu (za mało/za dyskretne) -> nie zapisujemy pliku
        except Exception as e:
            # jeśli KDE padnie (np. singular covariance), po prostu pomijamy KDE
            pass
        finally:
            plt.close()

