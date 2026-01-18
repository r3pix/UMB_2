from src.task3_real_data import load_cicids, load_unsw, train_eval_models

if __name__ == "__main__":
    # === CICIDS2017 ===
    cicids_dir = "data/cicids2017"
    monday = f"{cicids_dir}/Monday-WorkingHours.csv"
    ddos   = f"{cicids_dir}/Friday-WorkingHours-Afternoon-DDoS.csv"
    port   = f"{cicids_dir}/Friday-WorkingHours-Afternoon-PortScan.csv"

    print("Running Task 3 – CICIDS2017...")
    Xc, yc = load_cicids(monday, ddos, port)
    train_eval_models(Xc, yc, out_dir="results/task3_cicids_models")

    # === UNSW-NB15 ===
    unsw_dir = "data/unsw_nb15"
    train_csv = f"{unsw_dir}/UNSW_NB15_training-set.csv"
    test_csv  = f"{unsw_dir}/UNSW_NB15_testing-set.csv"

    print("Running Task 3 – UNSW-NB15...")
    Xu, yu = load_unsw(train_csv, test_csv, label_col="label")
    train_eval_models(Xu, yu, out_dir="results/task3_unsw_models")

    print("Task 3 completed for both datasets.")
