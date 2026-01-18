from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from src.utils import FEATURE_NAMES, zscore_fit, zscore_apply, metrics_binary, shannon_entropy, safe_datetime

EPS = 1e-10

def clean_common(D: pd.DataFrame) -> pd.DataFrame:
    # Algorytm 5: inf -> NaN, NaN -> mediana, usuń prawie zerową wariancję
    D = D.replace([np.inf, -np.inf], np.nan)
    D = D.fillna(D.median(numeric_only=True))
    num_cols = D.select_dtypes(include=[np.number]).columns
    low_var = num_cols[D[num_cols].var() < 1e-10]
    return D.drop(columns=list(low_var), errors="ignore")

def cicids_build_features(D: pd.DataFrame) -> pd.DataFrame:
    """
    Zadanie 3 – CICIDS2017 (zgodnie z opisem w PDF):
    - packets_per_sec: Flow Packets/s
    - avg_packet_size_bytes: Average Packet Size
    - connection_duration_sec: Flow Duration / 1e6
    - syn_ratio: SYN Flag Count / (Total Fwd Packets + Total Backward Packets + eps), clip[0,1]
    - port_entropy: okna 300s po Source IP, entropia Destination Port
    - unique_dst_ips: okna 3600s po Source IP, liczba unikalnych Destination IP
    - repeated_connections: okna 900s, max liczby powtórzeń pary (Source IP, Destination IP)
    """
    # mapowanie nazw (w razie różnic w CSV)
    col = {c.lower(): c for c in D.columns}
    def pick(*names):
        for n in names:
            if n.lower() in col:
                return col[n.lower()]
        return None

    ts_c = pick("Timestamp")
    srcip_c = pick("Source IP", "Src IP", "srcip")
    dstip_c = pick("Destination IP", "Dst IP", "dstip")
    dport_c = pick("Destination Port", "Dst Port", "dport")
    # cechy bezpośrednie
    pps_c = pick("Flow Packets/s")
    aps_c = pick("Average Packet Size")
    dur_c = pick("Flow Duration")
    syn_c = pick("SYN Flag Count")
    fwd_c = pick("Total Fwd Packets")
    bwd_c = pick("Total Backward Packets")

    missing = [n for n,v in {
        "Timestamp": ts_c, "Source IP": srcip_c, "Destination IP": dstip_c, "Destination Port": dport_c,
        "Flow Packets/s": pps_c, "Average Packet Size": aps_c, "Flow Duration": dur_c,
        "SYN Flag Count": syn_c, "Total Fwd Packets": fwd_c, "Total Backward Packets": bwd_c,
    }.items() if v is None]
    if missing:
        raise ValueError(f"Brak wymaganych kolumn CICIDS2017: {missing}")

    ts = safe_datetime(D[ts_c])
    if ts.isna().any():
        # jeśli część nieparsowalna – usuń (bez timestampu nie policzymy agregacji)
        keep = ~ts.isna()
        D = D.loc[keep].copy()
        ts = ts.loc[keep]

    # time bins
    t_sec = (ts.view("int64") // 10**9).astype("int64")
    bin300 = (t_sec // 300).astype("int64")
    bin900 = (t_sec // 900).astype("int64")
    bin3600 = (t_sec // 3600).astype("int64")

    out = pd.DataFrame(index=D.index)
    out["packets_per_sec"] = pd.to_numeric(D[pps_c], errors="coerce")
    out["avg_packet_size_bytes"] = pd.to_numeric(D[aps_c], errors="coerce")
    out["connection_duration_sec"] = pd.to_numeric(D[dur_c], errors="coerce") / 1e6

    total_pkts = pd.to_numeric(D[fwd_c], errors="coerce") + pd.to_numeric(D[bwd_c], errors="coerce") + EPS
    syn = pd.to_numeric(D[syn_c], errors="coerce") / total_pkts
    out["syn_ratio"] = syn.clip(0, 1)

    # port entropy: group by (srcip, bin300)
    grp_pe = pd.DataFrame({
        "src": D[srcip_c].astype(str).values,
        "bin": bin300.values,
        "dport": pd.to_numeric(D[dport_c], errors="coerce").fillna(-1).astype(int).values
    }, index=D.index)
    ent = grp_pe.groupby(["src","bin"])["dport"].apply(lambda s: shannon_entropy(s.values))
    out["port_entropy"] = grp_pe.set_index(["src","bin"]).index.map(ent)

    # unique dst ips: group by (srcip, bin3600)
    grp_ud = pd.DataFrame({"src": D[srcip_c].astype(str).values, "bin": bin3600.values, "dst": D[dstip_c].astype(str).values}, index=D.index)
    nun = grp_ud.groupby(["src","bin"])["dst"].nunique()
    out["unique_dst_ips"] = grp_ud.set_index(["src","bin"]).index.map(nun).astype(float)

    # repeated connections: in each (src, bin900) take max count of pair (src,dst)
    grp_rc = pd.DataFrame({"src": D[srcip_c].astype(str).values, "bin": bin900.values, "dst": D[dstip_c].astype(str).values}, index=D.index)
    # count pairs per window
    pair_counts = grp_rc.groupby(["src","bin","dst"]).size()
    max_per_window = pair_counts.groupby(["src","bin"]).max()
    out["repeated_connections"] = grp_rc.set_index(["src","bin"]).index.map(max_per_window).astype(float)

    out = clean_common(out)
    return out

def unsw_build_features(D: pd.DataFrame) -> pd.DataFrame:
    """
    Zadanie 3 – UNSW-NB15 (wg opisu):
    - connection_duration_sec: dur
    - packets_per_sec: spkts/(dur+eps)
    - avg_packet_size_bytes: (smeansz + dmeansz)/2
    - repeated_connections: ct_srv_src (przybliżenie)
    - port_entropy: analogicznie używając sport/dsport (jeśli jest czas: stime)
    - syn_ratio i unique_dst_ips: trudne -> ustawiane jako NaN (później median fill)
    """
    col = {c.lower(): c for c in D.columns}
    def pick(*names):
        for n in names:
            if n.lower() in col:
                return col[n.lower()]
        return None

    dur = pick("dur")
    spkts = pick("spkts")
    smeansz = pick("smeansz")
    dmeansz = pick("dmeansz")
    ctsrv = pick("ct_srv_src")
    sport = pick("sport")
    dsport = pick("dsport")
    stime = pick("stime")  # epoch seconds typically
    srcip = pick("srcip")
    if dur is None or spkts is None or smeansz is None or dmeansz is None:
        raise ValueError("Brak wymaganych kolumn UNSW dla packets_per_sec/avg_packet_size/dur")

    out = pd.DataFrame(index=D.index)
    durv = pd.to_numeric(D[dur], errors="coerce")
    out["connection_duration_sec"] = durv
    out["packets_per_sec"] = pd.to_numeric(D[spkts], errors="coerce") / (durv + EPS)
    out["avg_packet_size_bytes"] = (pd.to_numeric(D[smeansz], errors="coerce") + pd.to_numeric(D[dmeansz], errors="coerce")) / 2.0
    out["repeated_connections"] = pd.to_numeric(D[ctsrv], errors="coerce") if ctsrv is not None else np.nan

    # port entropy: jeśli mamy stime i srcip i dsport
    if stime is not None and srcip is not None and dsport is not None:
        t = pd.to_numeric(D[stime], errors="coerce")
        bin300 = (t.fillna(0).astype("int64") // 300).astype("int64")
        grp = pd.DataFrame({"src": D[srcip].astype(str).values, "bin": bin300.values,
                            "dport": pd.to_numeric(D[dsport], errors="coerce").fillna(-1).astype(int).values}, index=D.index)
        ent = grp.groupby(["src","bin"])["dport"].apply(lambda s: shannon_entropy(s.values))
        out["port_entropy"] = grp.set_index(["src","bin"]).index.map(ent)
    elif sport is not None:
        # fallback: global entropy of sport (less faithful)
        out["port_entropy"] = shannon_entropy(pd.to_numeric(D[sport], errors="coerce").fillna(-1).astype(int).values)
    else:
        out["port_entropy"] = np.nan

    out["syn_ratio"] = np.nan
    out["unique_dst_ips"] = np.nan

    out = clean_common(out)
    return out

def load_cicids(monday_csv: str, ddos_csv: str, portscan_csv: str) -> tuple[pd.DataFrame, np.ndarray]:
    D1 = pd.read_csv(monday_csv, low_memory=False)
    D2 = pd.read_csv(ddos_csv, low_memory=False)
    D3 = pd.read_csv(portscan_csv, low_memory=False)
    D = pd.concat([D1, D2, D3], ignore_index=True)

    # etykieta
    if "Label" not in D.columns:
        raise ValueError("CICIDS2017: brak kolumny Label")
    y = (D["Label"].astype(str).str.upper() != "BENIGN").astype(int).to_numpy()
    X = cicids_build_features(D)
    return X, y

def load_unsw(train_csv: str, test_csv: str, label_col: str = "label") -> tuple[pd.DataFrame, np.ndarray]:
    Dtr = pd.read_csv(train_csv, low_memory=False)
    Dte = pd.read_csv(test_csv, low_memory=False)
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

    M = LogisticRegression(C=1.0, penalty="l2", class_weight="balanced", max_iter=2000, random_state=42)
    M.fit(X_trz, y_tr)

    proba = M.predict_proba(X_tez)[:,1]
    yhat = (proba >= 0.5).astype(int)
    cm = confusion_matrix(y_te, yhat)
    met = metrics_binary(y_te, yhat, proba)

    # zapisz
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame([met]).to_csv(os.path.join(out_dir,"task3_metrics.csv"), index=False)
    pd.DataFrame(cm, index=["True0","True1"], columns=["Pred0","Pred1"]).to_csv(os.path.join(out_dir,"task3_confusion.csv"))
    pd.DataFrame({"feature": Xdf.columns, "beta": M.coef_[0]}).to_csv(os.path.join(out_dir,"task3_betas.csv"), index=False)

    return met, cm

if __name__ == "__main__":
    print("Użycie:")
    print("  CICIDS2017: uzupełnij ścieżki do Monday-WorkingHours.csv, Friday-...DDoS.csv, Friday-...PortScan.csv")
    print("  UNSW-NB15:  uzupełnij ścieżki do UNSW_NB15_training-set.csv i UNSW_NB15_testing-set.csv")
