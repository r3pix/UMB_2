from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from src.utils import FEATURE_NAMES, zscore_fit, zscore_apply, metrics_binary, cost_opt_threshold
from src.plots import (
    density_two, betas_barh, save_confusion_heatmap, roc_plot, proba_hist,
    contrib_bar, corr_heatmap
)

RNG = np.random.default_rng(42)

def ensure_dirs(base: str):
    for d in ["results/figures", "results/metrics"]:
        os.makedirs(os.path.join(base, d), exist_ok=True)

def task1(base: str):
    # Dane idealne (Algorytm 1)
    n_norm, n_attack = 800, 200
    X = np.zeros((n_norm+n_attack, 7))
    y = np.zeros(n_norm+n_attack, dtype=int)

    X[:n_norm,0] = RNG.normal(50, 15, size=n_norm)
    X[:n_norm,1] = RNG.normal(800, 200, size=n_norm)
    X[:n_norm,2] = RNG.normal(2.5, 0.5, size=n_norm)
    X[:n_norm,3] = np.clip(RNG.normal(0.2, 0.05, size=n_norm), 0, 1)
    X[:n_norm,4] = RNG.poisson(5, size=n_norm) + 1
    X[:n_norm,5] = RNG.exponential(30, size=n_norm)         # Exp(lambda=1/30) => scale=30
    X[:n_norm,6] = RNG.poisson(2, size=n_norm)

    X[n_norm:,0] = RNG.normal(250, 30, size=n_attack)
    X[n_norm:,1] = RNG.normal(300, 100, size=n_attack)
    X[n_norm:,2] = RNG.normal(4.0, 0.3, size=n_attack)
    X[n_norm:,3] = np.clip(RNG.normal(0.8, 0.05, size=n_attack), 0, 1)
    X[n_norm:,4] = RNG.poisson(50, size=n_attack) + 1
    X[n_norm:,5] = RNG.exponential(2, size=n_attack)         # Exp(lambda=1/2) => scale=2
    X[n_norm:,6] = RNG.poisson(20, size=n_attack)
    y[n_norm:] = 1

    X, y = shuffle(X, y, random_state=42)

    idx = np.arange(len(y))
    I_tr, I_te = train_test_split(idx, train_size=0.7, random_state=42, stratify=y)
    X_tr, y_tr = X[I_tr], y[I_tr]
    X_te, y_te = X[I_te], y[I_te]

    mu, sigma = zscore_fit(X_tr)
    X_trz = zscore_apply(X_tr, mu, sigma)
    X_tez = zscore_apply(X_te, mu, sigma)

    M = LogisticRegression(C=1.0, penalty="l2", max_iter=1000, random_state=42)
    M.fit(X_trz, y_tr)

    proba = M.predict_proba(X_tez)[:,1]
    yhat = (proba >= 0.5).astype(int)
    cm = confusion_matrix(y_te, yhat)
    met = metrics_binary(y_te, yhat, proba)
    beta = M.coef_[0]
    beta0 = float(M.intercept_[0])

    figdir = os.path.join(base, "results/figures")
    density_two(X[y==0,0], X[y==1,0], "packets/s", "Z1: packets_per_sec (raw)", os.path.join(figdir,"z1_density_packets_per_sec.png"))
    density_two(X[y==0,1], X[y==1,1], "bytes", "Z1: avg_packet_size (raw)", os.path.join(figdir,"z1_density_avg_packet_size.png"))
    density_two(X[y==0,2], X[y==1,2], "bits", "Z1: port_entropy (raw)", os.path.join(figdir,"z1_density_port_entropy.png"))
    density_two(X[y==0,3], X[y==1,3], "ratio", "Z1: syn_ratio (raw)", os.path.join(figdir,"z1_density_syn_ratio.png"))
    betas_barh(beta, FEATURE_NAMES, "Z1: β coefficients", os.path.join(figdir,"z1_betas.png"))
    save_confusion_heatmap(cm, "Z1: confusion matrix (τ=0.5)", os.path.join(figdir,"z1_confusion.png"))
    roc_plot(y_te, proba, "Z1: ROC", os.path.join(figdir,"z1_roc.png"))
    proba_hist(y_te, proba, 0.5, "Z1: P(y=1|x) (τ=0.5)", os.path.join(figdir,"z1_proba_hist.png"))

    # 3 przykłady wkładów β_i·x_i
    for k in [0,1,2]:
        contrib_bar(beta, X_tez[k], FEATURE_NAMES, f"Z1: contributions sample {k}", os.path.join(figdir,f"z1_contrib_{k}.png"))

    corr_heatmap(X, FEATURE_NAMES, "Z1: Pearson corr (raw)", os.path.join(figdir,"z1_corr.png"))

    return met, beta, beta0

def task2(base: str):
    # Dane realistyczne (Algorytm 2)
    nnorm, nobv, nmed, nsub = 950, 20, 15, 15
    N = nnorm + nobv + nmed + nsub
    X = np.zeros((N,7))
    y = np.zeros(N, dtype=int)
    atype = np.array(["normal"]*N, dtype=object)

    mu = np.array([50, 800, 2.5, 0.2, 5, 30, 2], dtype=float)
    sig = np.array([15, 200, 0.5, 0.05], dtype=float)

    # normal
    X[:nnorm,0] = RNG.normal(mu[0], sig[0], size=nnorm)
    X[:nnorm,1] = RNG.normal(mu[1], sig[1], size=nnorm)
    X[:nnorm,2] = RNG.normal(mu[2], sig[2], size=nnorm)
    X[:nnorm,3] = np.clip(RNG.normal(mu[3], sig[3], size=nnorm), 0, 1)
    X[:nnorm,4] = RNG.poisson(mu[4], size=nnorm) + 1
    X[:nnorm,5] = RNG.exponential(mu[5], size=nnorm)            # Exp(1/mu[5]) => scale=mu[5]
    X[:nnorm,6] = RNG.poisson(mu[6], size=nnorm)
    y[:nnorm] = 0

    def gen_block(start, n, k, label):
        # rosnące: [0,2,3,4,6], malejące: [1,5]
        X[start:start+n,0] = RNG.normal(mu[0] + k*sig[0], sig[0], size=n)
        X[start:start+n,1] = RNG.normal(mu[1] - k*sig[1], sig[1], size=n)
        X[start:start+n,2] = RNG.normal(mu[2] + k*sig[2], sig[2], size=n)
        X[start:start+n,3] = np.clip(RNG.normal(mu[3] + k*sig[3], sig[3], size=n), 0, 1)
        X[start:start+n,4] = RNG.poisson(mu[4] + k*10, size=n) + 1
        X[start:start+n,5] = RNG.exponential(mu[5]/k, size=n)    # Exp(1/(mu/k)) => scale=mu/k
        X[start:start+n,6] = RNG.poisson(mu[6] + k*5, size=n)
        y[start:start+n] = 1
        atype[start:start+n] = label

    gen_block(nnorm, nobv, 4, "obvious")
    gen_block(nnorm+nobv, nmed, 2, "medium")
    gen_block(nnorm+nobv+nmed, nsub, 1, "subtle")

    X, y, atype = shuffle(X, y, atype, random_state=42)

    idx = np.arange(N)
    I_tr, I_te = train_test_split(idx, train_size=0.7, random_state=42, stratify=y)
    X_tr, y_tr = X[I_tr], y[I_tr]
    X_te, y_te = X[I_te], y[I_te]
    at_te = atype[I_te]

    muhat, sighat = zscore_fit(X_tr)
    X_trz = zscore_apply(X_tr, muhat, sighat)
    X_tez = zscore_apply(X_te, muhat, sighat)

    # model 1 std
    M_std = LogisticRegression(C=1.0, penalty="l2", max_iter=1000, random_state=42)
    M_std.fit(X_trz, y_tr)
    P_std = M_std.predict_proba(X_tez)[:,1]
    yhat_std = (P_std >= 0.5).astype(int)
    cm_std = confusion_matrix(y_te, yhat_std)
    met_std = metrics_binary(y_te, yhat_std, P_std)

    # model 2 balanced
    M_bal = LogisticRegression(C=1.0, penalty="l2", max_iter=1000, random_state=42, class_weight="balanced")
    M_bal.fit(X_trz, y_tr)
    P_bal = M_bal.predict_proba(X_tez)[:,1]
    yhat_bal = (P_bal >= 0.5).astype(int)
    cm_bal = confusion_matrix(y_te, yhat_bal)
    met_bal = metrics_binary(y_te, yhat_bal, P_bal)

    # model 3 threshold opt (na std)
    tau_opt, taus, costs = cost_opt_threshold(y_te, P_std, fn_cost=100.0, fp_cost=1.0)
    yhat_thr = (P_std >= tau_opt).astype(int)
    cm_thr = confusion_matrix(y_te, yhat_thr)
    met_thr = metrics_binary(y_te, yhat_thr, P_std)

    # oszczędność kosztu
    C_05 = float(costs[np.where(np.isclose(taus, 0.5))[0][0]])
    C_opt = float(costs.min())
    deltaC = C_05 - C_opt

    # wykresy
    figdir = os.path.join(base, "results/figures")
    import matplotlib.pyplot as plt

    # 3 macierze pomyłek obok siebie
    fig, axes = plt.subplots(1,3, figsize=(12,4))
    for ax, cm, title in zip(axes, [cm_std, cm_bal, cm_thr], ["std τ=0.5","balanced τ=0.5",f"std τopt={tau_opt:.2f}"]):
        ax.imshow(cm, aspect="auto")
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["Pred0","Pred1"]); ax.set_yticklabels(["True0","True1"])
        ax.set_title(title)
        for i in range(2):
            for j in range(2):
                ax.text(j,i,str(cm[i,j]),ha="center",va="center")
        ax.text(0,1,f"FN={cm[1,0]}",ha="center",va="bottom")
        ax.text(1,0,f"FP={cm[0,1]}",ha="center",va="bottom")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir,"z2_confusions_3models.png"), dpi=300)
    plt.close(fig)

    # ROC: std vs balanced
    from sklearn.metrics import roc_curve, roc_auc_score
    fpr_s, tpr_s, _ = roc_curve(y_te, P_std)
    fpr_b, tpr_b, _ = roc_curve(y_te, P_bal)
    auc_s = roc_auc_score(y_te, P_std)
    auc_b = roc_auc_score(y_te, P_bal)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(fpr_s, tpr_s, label=f"std (AUC={auc_s:.4f})")
    ax.plot(fpr_b, tpr_b, label=f"balanced (AUC={auc_b:.4f})")
    ax.plot([0,1],[0,1], linestyle="--", label="chance")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("Z2: ROC std vs balanced"); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(figdir,"z2_roc_std_vs_balanced.png"), dpi=300)
    plt.close(fig)

    proba_hist(y_te, P_bal, 0.5, "Z2: P(y=1|x) balanced (τ=0.5)", os.path.join(figdir,"z2_proba_hist_balanced.png"))

    # β porównanie
    beta_std = M_std.coef_[0]
    beta_bal = M_bal.coef_[0]
    order = np.argsort(np.abs(beta_std))[::-1]
    fig, ax = plt.subplots(figsize=(8,4.5))
    x = np.arange(len(FEATURE_NAMES))
    ax.bar(x-0.2, beta_std[order], width=0.4, label="β std")
    ax.bar(x+0.2, beta_bal[order], width=0.4, label="β balanced")
    ax.set_xticks(x); ax.set_xticklabels([FEATURE_NAMES[i] for i in order], rotation=30, ha="right")
    ax.set_ylabel("β (z-scored)")
    ax.set_title("Z2: β comparison"); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(figdir,"z2_betas_std_vs_balanced.png"), dpi=300)
    plt.close(fig)

    # 3 gęstości: przykładowe cechy
    from scipy.stats import gaussian_kde
    labels = ["normal","obvious","medium","subtle"]
    for j in [0,1,3]:
        fig, ax = plt.subplots(figsize=(6,4))
        xs = np.linspace(X[:,j].min(), X[:,j].max(), 400)
        for lab in labels:
            vals = X[(atype==lab), j]
            ax.plot(xs, gaussian_kde(vals)(xs), label=lab)
        ax.set_xlabel(FEATURE_NAMES[j]); ax.set_ylabel("Density")
        ax.set_title(f"Z2: density overlay {FEATURE_NAMES[j]}"); ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(figdir,f"z2_density_overlay_{FEATURE_NAMES[j]}.png"), dpi=300)
        plt.close(fig)

    # Precision/Recall/F1 vs tau + cost vs tau
    from sklearn.metrics import precision_score, recall_score, f1_score
    prec, rec, f1s = [], [], []
    for tau in taus:
        yh = (P_std >= tau).astype(int)
        prec.append(precision_score(y_te, yh, zero_division=0))
        rec.append(recall_score(y_te, yh, zero_division=0))
        f1s.append(f1_score(y_te, yh, zero_division=0))

    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(taus, prec, label="Precision(τ)")
    ax.plot(taus, rec, label="Recall(τ)")
    ax.plot(taus, f1s, label="F1(τ)")
    ax.axvline(tau_opt, linestyle="--", label=f"τopt={tau_opt:.2f}")
    ax.set_xlabel("τ"); ax.set_ylabel("metric"); ax.set_title("Z2: metrics vs τ"); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(figdir,"z2_metrics_vs_tau.png"), dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(taus, costs, label="C(τ)=100·FN+FP")
    ax.scatter([tau_opt], [C_opt], label=f"min at τopt={tau_opt:.2f}")
    ax.set_xlabel("τ"); ax.set_ylabel("Cost"); ax.set_title("Z2: cost vs τ"); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(figdir,"z2_cost_vs_tau.png"), dpi=300)
    plt.close(fig)

    # ile subtelnych wykryto
    subtle_mask = (at_te=="subtle") & (y_te==1)
    det_sub_std = int(((yhat_std==1) & subtle_mask).sum())
    det_sub_bal = int(((yhat_bal==1) & subtle_mask).sum())
    det_sub_thr = int(((yhat_thr==1) & subtle_mask).sum())
    n_sub = int(subtle_mask.sum())

    return (met_std, met_bal, met_thr, beta_std, beta_bal, float(M_std.intercept_[0]), float(M_bal.intercept_[0]), tau_opt, float(deltaC),
            det_sub_std, det_sub_bal, det_sub_thr, n_sub)

def main():
    base = os.path.dirname(os.path.dirname(__file__))
    ensure_dirs(base)

    m1, b1, b01 = task1(base)
    (m2s, m2b, m2t, b2s, b2b, b02s, b02b, tau_opt, deltaC, ds, db, dt, nsub) = task2(base)

    # zapisz metryki i bety
    metrics = pd.DataFrame([
        {"Experiment":"Z1", **m1, "tau_used":0.5, "beta0":b01},
        {"Experiment":"Z2 std", **m2s, "tau_used":0.5, "beta0":b02s},
        {"Experiment":"Z2 balanced", **m2b, "tau_used":0.5, "beta0":b02b},
        {"Experiment":"Z2 thr-opt", **m2t, "tau_used":tau_opt, "beta0":b02s},
    ])
    metrics.to_csv(os.path.join(base,"results/metrics/metrics_summary.csv"), index=False)

    betas = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "beta_task1": b1,
        "beta_task2_std": b2s,
        "beta_task2_balanced": b2b,
    })
    betas.to_csv(os.path.join(base,"results/metrics/betas.csv"), index=False)

    with open(os.path.join(base,"results/metrics/task2_subtle_detection.txt"),"w",encoding="utf-8") as f:
        f.write(f"subtle detected (test): std={ds}/{nsub}, balanced={db}/{nsub}, thr-opt={dt}/{nsub}\n")
        f.write(f"tau_opt={tau_opt:.2f}, deltaC={deltaC:.0f}\n")

if __name__ == "__main__":
    main()
