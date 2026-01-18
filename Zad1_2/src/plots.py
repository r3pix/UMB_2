from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.stats import gaussian_kde

def save_confusion_heatmap(cm, title, path):
    fig, ax = plt.subplots(figsize=(5,4))
    ax.imshow(cm, aspect="auto")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Pred 0","Pred 1"])
    ax.set_yticklabels(["True 0","True 1"])
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

def density_two(x0, x1, xlabel, title, path, lab0="normal", lab1="attack"):
    fig, ax = plt.subplots(figsize=(6,4))
    xs = np.linspace(min(x0.min(), x1.min()), max(x0.max(), x1.max()), 400)
    ax.plot(xs, gaussian_kde(x0)(xs), label=lab0)
    ax.plot(xs, gaussian_kde(x1)(xs), label=lab1)
    ax.set_xlabel(xlabel); ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

def roc_plot(y_true, y_proba, title, path, mark_tau=None):
    fpr, tpr, thr = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(fpr, tpr, label=f"ROC (AUC={auc:.4f})")
    ax.plot([0,1],[0,1], linestyle="--", label="chance")
    if mark_tau is not None:
        # mark closest threshold
        idx = int(np.argmin(np.abs(thr - mark_tau)))
        ax.scatter([fpr[idx]],[tpr[idx]], label=f"τ={thr[idx]:.2f}")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(title); ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return float(auc)

def proba_hist(y_true, y_proba, tau, title, path):
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(y_proba[y_true==0], bins=30, alpha=0.6, label="normal")
    ax.hist(y_proba[y_true==1], bins=30, alpha=0.6, label="attack")
    ax.axvline(tau, linestyle="--", label=f"τ={tau:.2f}")
    ax.set_xlabel("P(y=1|x)"); ax.set_ylabel("Count")
    ax.set_title(title); ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

def betas_barh(beta, names, title, path):
    order = np.argsort(np.abs(beta))[::-1]
    beta_s = beta[order]
    names_s = [names[i] for i in order]
    fig, ax = plt.subplots(figsize=(7,4.5))
    y = np.arange(len(beta_s))
    ax.barh(y, beta_s)
    ax.set_yticks(y); ax.set_yticklabels(names_s)
    ax.invert_yaxis()
    ax.set_xlabel("β (z-scored)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

def contrib_bar(beta, x_z, names, title, path):
    contrib = beta * x_z
    fig, ax = plt.subplots(figsize=(7,4))
    y = np.arange(len(beta))
    ax.barh(y, contrib)
    ax.set_yticks(y); ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("β_i · x_i")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

def corr_heatmap(X, names, title, path):
    corr = np.corrcoef(X, rowvar=False)
    fig, ax = plt.subplots(figsize=(6.5,5.5))
    im = ax.imshow(corr, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return corr
