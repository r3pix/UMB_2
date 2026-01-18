from src.task3_real_data import load_cicids, load_unsw
from src.plots_task3 import (
    plot_roc_pr,
    plot_confusion_matrices,
    plot_decision_boundaries_pca,
    plot_feature_distributions
)

def main():
    # --- CICIDS ---
    cicids_dir = "data/cicids2017"
    monday = f"{cicids_dir}/Monday-WorkingHours.csv"
    ddos   = f"{cicids_dir}/Friday-WorkingHours-Afternoon-DDoS.csv"
    port   = f"{cicids_dir}/Friday-WorkingHours-Afternoon-PortScan.csv"

    Xc, yc = load_cicids(monday, ddos, port)
    pred_dir_c = "results/task3_cicids_models"
    out_c = f"{pred_dir_c}/plots"

    plot_roc_pr(pred_dir_c, out_c, title_suffix="(CICIDS2017)")
    plot_confusion_matrices(pred_dir_c, out_c)
    plot_decision_boundaries_pca(Xc, yc, out_dir=f"{out_c}/pca2_boundaries", dataset_name="CICIDS2017")
    plot_feature_distributions(Xc, yc, out_dir=f"{out_c}/feature_distributions", dataset_name="CICIDS2017")

    # --- UNSW ---
    unsw_dir = "data/unsw_nb15"
    train_csv = f"{unsw_dir}/UNSW_NB15_training-set.csv"
    test_csv  = f"{unsw_dir}/UNSW_NB15_testing-set.csv"

    Xu, yu = load_unsw(train_csv, test_csv, label_col="label")
    pred_dir_u = "results/task3_unsw_models"
    out_u = f"{pred_dir_u}/plots"

    plot_roc_pr(pred_dir_u, out_u, title_suffix="(UNSW-NB15)")
    plot_confusion_matrices(pred_dir_u, out_u)
    plot_decision_boundaries_pca(Xu, yu, out_dir=f"{out_u}/pca2_boundaries", dataset_name="UNSW-NB15")
    plot_feature_distributions(Xu, yu, out_dir=f"{out_u}/feature_distributions", dataset_name="UNSW-NB15")

    print("All plots generated.")

if __name__ == "__main__":
    main()
