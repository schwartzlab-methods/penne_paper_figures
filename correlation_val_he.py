'''
Validate the correlation between predicted gene expression and ground truth data from H&E images
This is computed at a per gene level
'''
import os
import numpy as np
import pandas as pd
import argparse
from scipy.stats import spearmanr, pearsonr
from correlation_validation import compute_stats_gt
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
## plotting settings
if True:  # In order to bypass isort when saving
    from altairThemes import altairThemes
alt.themes.register("publishTheme", altairThemes.publishTheme)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_npy', type=str, required=True, help='Path to the predicted npy file')
    parser.add_argument('--gene_names', type=str, required=True, help='Path to the gene symbol list npy')
    parser.add_argument('--ground_truth', type=str, required=True, help='Path to the ground truth directory')
    parser.add_argument('--output', type=str, required=True, help='Output file path for correlation results')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    if args.gene_names.endswith("tsv.gz"):
            gene_symbols_list = pd.read_csv(args.gene_names, sep="\t", header = None)[1].values
    else:
        with open(args.gene_names, 'r') as f:
            gene_symbols_list = [line.strip() for line in f.readlines() if "Unnamed: 0" not in line] # list of gene symbols
    pred = np.load(args.pred_npy)  # shape (cells, genes), in the same order as gene_symbols_list
    gt_data_df = pd.DataFrame(np.load(args.ground_truth), columns=gene_symbols_list)  # shape (cells, genes), in the same order as gene_symbols_list

    print("Shape of predicted data:", pred.shape)
    print("Shape of ground truth data:", gt_data_df.shape)
    
    print("Generating per-gene stats about prediction and ground truth...")
    compute_stats_gt(gt_data_df, pred, args.output)
    metrics = []
    for j in range(gt_data_df.shape[1]):
        mae = np.mean(np.abs(gt_data_df.iloc[:, j] - pred[:, j]))
        rmse = np.sqrt(np.mean((gt_data_df.iloc[:, j] - pred[:, j])**2))
        if np.sum(gt_data_df.iloc[:, j]) > 1e-8 and np.sum(pred[:, j]) > 1e-8: #more than only zero
            spearman = spearmanr(gt_data_df.iloc[:, j], pred[:, j]).correlation
            var_ratio = np.var(pred[:, j]) / (np.var(gt_data_df.iloc[:, j]) + 1e-8)
            nrmse = rmse / (np.std(gt_data_df.iloc[:, j]) + 1e-8)
            ev = 1 - np.var(gt_data_df.iloc[:, j] - pred[:, j]) / (np.var(gt_data_df.iloc[:, j]) + 1e-8)
        elif np.sum(pred[:, j]) > 1e-8 and np.sum(gt_data_df.iloc[:, j]) < 1e-8: # all zero in gt but not in pred
            spearman = 0.0
            var_ratio = np.nan
            ev = 0.0
            nrmse = np.nan
        elif np.sum(pred[:, j]) < 1e-8 and np.sum(gt_data_df.iloc[:, j]) > 1e-8: # all zero in pred but not in gt
            spearman = 0.0
            var_ratio = 0.0
            ev = 0.0
            nrmse = np.nan
        else: # all zero in both
            continue
        metrics.append([gene_symbols_list[j], mae, rmse, nrmse, ev, spearman, var_ratio])

    df_metrics = pd.DataFrame(metrics, columns=["Gene", "MAE", "RMSE", "NRMSE", "Explained Variance", "Spearman", "VarRatio"])
    df_metrics.to_csv(os.path.join(args.output, "gene_correlation_metrics_nonezero.csv"), index=False)
    # plot each sns.violinplot on its on subfigure and save a one big figure
    plt.figure(figsize=(12, 6))
    for i, col in enumerate(["MAE", "RMSE", "NRMSE", "Explained Variance", "Spearman", "VarRatio"]):
        plt.subplot(2, 3, i + 1)
        sns.violinplot(data=df_metrics[col])
        plt.title(col)
    plt.savefig(os.path.join(args.output, "gene_correlation_metrics_nonezero.png"))
    plt.close()

    # write that filter by Explain variance >= 0.1 and Spearman >= 0.3
    filtered_genes = df_metrics[(df_metrics["Explained Variance"] >= 0.1) & (df_metrics["Spearman"] >= 0.3)]["Gene"].tolist()
    with open(os.path.join(args.output, "gene_correlation_metrics_filtered_ev0.1_spearman0.3.txt"), "w") as f:
        f.write("\n".join(filtered_genes))

    #* plot explained variance vs spearman
    df_metrics = df_metrics.dropna(subset=["Explained Variance", "Spearman"])
    # drop very negative explained variance
    df_metrics = df_metrics[df_metrics["Explained Variance"] >= -1]
    plt.figure(figsize=(8, 8))
    plt.scatter(df_metrics["Explained Variance"], df_metrics["Spearman"], alpha=0.5)
    plt.xlabel("Explained Variance")
    plt.ylabel("Spearman Correlation")
    plt.title("Explained Variance vs Spearman Correlation")
    plt.axvline(x=0.1, color='r', linestyle='--')
    plt.axhline(y=0.3, color='r', linestyle='--')
    plt.savefig(os.path.join(args.output, "gene_correlation_ev_vs_spearman.png"))
    plt.close()
    # plot with altair
    chart = alt.Chart(df_metrics).mark_circle(opacity=0.5).encode(
        x=alt.X("Explained Variance:Q", title="Explained Variance"),
        y=alt.Y("Spearman:Q", title="Spearman Correlation"),
        tooltip=["Gene:N", "Explained Variance:Q", "Spearman:Q"]
    ).interactive()
    v_line = alt.Chart(df_metrics).mark_rule(color='red', strokeDash=[5,5]).encode(
       x=alt.datum(0.1)
    )
    h_line = alt.Chart(df_metrics).mark_rule(color='red', strokeDash=[5,5]).encode(
        y=alt.datum(0.3)
    )   
    chart = chart + v_line + h_line
    chart.save(os.path.join(args.output, "gene_correlation_ev_vs_spearman.html"))

    # plot explained variance vs variance of GT
    genes = df_metrics["Gene"].values
    pred_mean_per_gene = pred[:, np.isin(gene_symbols_list, genes)].mean(axis=0) # shape (num_genes,)
    var_explained_all = df_metrics["Explained Variance"].values
    plt.figure(figsize=(8, 8))
    plt.scatter(pred_mean_per_gene, var_explained_all, alpha=0.5)
    plt.xlabel("Mean of Predictions per Gene")
    plt.ylabel("Explained Variance")
    plt.title("Mean of Predictions vs Explained Variance")
    plt.savefig(os.path.join(args.output, "gene_correlation_pred_mean_vs_ev.png"))
    plt.close()

    #! All genes
    print("======================================== ALL GENES ========================================")
    print("Calculating statistics on GENE correlation with ALL genes...")
    corr_genes = []
    corr_val_genes = []
    non_zero_pred_genes = []
    non_zero_corr = []
    for i, gene in enumerate(gene_symbols_list):
        # Compute correlation
        pred_values = pred[:, i].flatten()
        exp_values = gt_data_df[gene].values.flatten()
        if np.sum(pred_values) < 1e-8 and np.sum(exp_values) < 1e-8: # both are all zero
            correlation = 1.0
        elif np.sum(exp_values) < 1e-8 or np.sum(pred_values) < 1e-8: # one array is constant but the other is not
            correlation = 0.0
        else: # at least one is non-zero
            correlation = spearmanr(exp_values, pred_values).correlation
            non_zero_pred_genes.append(gene)
            non_zero_corr.append(correlation)
        corr_genes.append(str(gene))
        corr_val_genes.append(correlation)
    df_non_zero_pred = pd.DataFrame({"Gene": non_zero_pred_genes, "Correlation": non_zero_corr})
    df_non_zero_pred.to_csv(os.path.join(args.output, "all_gene_correlation_non_zero.csv"), index=False)
    print("Mean of non-zero correlations:", df_non_zero_pred["Correlation"].mean())

    print("Calculating statistics on SAMPLE correlation using ALL GENES...")
    samples = []
    corr_val_samples = []
    for i, sample in enumerate(gt_data_df.index):
        pred_values = pred[i, :].flatten()
        exp_values = gt_data_df.iloc[i, :].values.flatten()
        correlation = spearmanr(exp_values, pred_values).correlation
        corr_val_samples.append(correlation)
        samples.append(sample)
    print("Mean sample correlation all genes:", np.mean(corr_val_samples))
    print("Std sample correlation all genes:", np.std(corr_val_samples))
    corr_val_samples_non_zero = []
    for i, sample in enumerate(gt_data_df.index):
        pred_values = pred[i, np.isin(gene_symbols_list, non_zero_pred_genes)].flatten()
        exp_values = gt_data_df.loc[sample, non_zero_pred_genes].values.flatten()
        correlation = spearmanr(exp_values, pred_values).correlation
        corr_val_samples_non_zero.append(correlation)
    print("Mean sample correlation non-zero genes:", np.mean(corr_val_samples_non_zero))
    print("Std sample correlation non-zero genes:", np.std(corr_val_samples_non_zero))
    # plot correlation violin plots for both
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=[corr_val_genes, corr_val_samples, corr_val_samples_non_zero])
    plt.xticks([0, 1, 2], ['Gene Correlations', 'Sample All Gene Correlations', 'Sample All Non-Zero Gene Correlations'])
    plt.title('Violin Plot of Correlation Coefficients')
    plt.ylabel('Correlation Coefficient')
    plt.savefig(os.path.join(args.output, "correlation_violin_plot_all_genes.png"))
    plt.close()
    # plot with altair box plot for gene correlations, sample correlations, sample non-zero gene correlations
    df_melt = pd.DataFrame({
        "Type": ["All Gene Correlations"] * len(corr_val_genes) + ["Sample All Gene Correlations"] * len(corr_val_samples) + ["Sample All Non-Zero Gene Correlations"] * len(corr_val_samples_non_zero),
        "Correlation": corr_val_genes + corr_val_samples + corr_val_samples_non_zero
    })
    chart = alt.Chart(df_melt).mark_boxplot().encode(
        x="Type:N",
        y="Correlation:Q"
    ).interactive()
    chart.save(os.path.join(args.output, "correlation_boxplot_all_genes.html"))


    #! High correlation genes
    print("======================================== HIGH CORRELATION GENES ========================================")
    print("Calculating statistics on GENE correlation using HIGH correlation genes...")
    corr_genes = []
    corr_val_genes = []
    non_zero_pred_genes = []
    non_zero_corr = []
    for i, gene in enumerate(gene_symbols_list):
        if gene in filtered_genes:
            # Compute correlation
            pred_values = pred[:, i].flatten()
            exp_values = gt_data_df[gene].values.flatten()
            if np.sum(pred_values) < 1e-8 and np.sum(exp_values) < 1e-8: # both are all zero
                correlation = 1.0
            elif np.sum(exp_values) < 1e-8 or np.sum(pred_values) < 1e-8: # one array is constant but the other is not
                correlation = 0.0
            else: # at least one is non-zero
                correlation = spearmanr(exp_values, pred_values).correlation
                non_zero_pred_genes.append(gene)
                non_zero_corr.append(correlation)
            corr_genes.append(str(gene))
            corr_val_genes.append(correlation)
    df_non_zero_pred = pd.DataFrame({"Gene": non_zero_pred_genes, "Correlation": non_zero_corr})
    df_non_zero_pred.to_csv(os.path.join(args.output, "filtered_gene_correlation_non_zero.csv"), index=False)
    print("Mean of non-zero correlations:", df_non_zero_pred["Correlation"].mean())

    print("Calculating statistics on SAMPLE using HIGH correlation genes...")
    samples = []
    corr_val_samples = []
    # filter such that both pred and exp only contains corr_genes
    pred_filtered = pred[:, np.isin(gene_symbols_list, corr_genes)]
    gt_data_df_filtered = gt_data_df.loc[:, corr_genes].reindex(columns=corr_genes)
    for i, sample in enumerate(gt_data_df_filtered.index):
        pred_values = pred_filtered[i, :].flatten()
        exp_values = gt_data_df_filtered.iloc[i, :].values.flatten()
        correlation = spearmanr(exp_values, pred_values).correlation
        corr_val_samples.append(correlation)
        samples.append(sample)

    corr_val_samples_non_zero = []
    pred_filtered_non_zero = pred[:, np.isin(gene_symbols_list, non_zero_pred_genes)]
    gt_data_df_filtered_non_zero = gt_data_df.loc[:, non_zero_pred_genes].reindex(columns=non_zero_pred_genes)
    for i, sample in enumerate(gt_data_df_filtered_non_zero.index):
        pred_values = pred_filtered_non_zero[i, :].flatten()
        exp_values = gt_data_df_filtered_non_zero.iloc[i, :].values.flatten()
        correlation = spearmanr(exp_values, pred_values).correlation
        corr_val_samples_non_zero.append(correlation)
    
    print("Mean sample correlation high confident genes:", np.mean(corr_val_samples))
    print("Std sample correlation high confident genes:", np.std(corr_val_samples))
    print("Mean sample correlation high confident non-zero genes:", np.mean(corr_val_samples_non_zero))
    print("Std sample correlation high confident non-zero genes:", np.std(corr_val_samples_non_zero))

    # plot correlation violin plots for both
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=[corr_val_genes, corr_val_samples, corr_val_samples_non_zero])
    plt.xticks([0, 1, 2], ['Gene Correlations', 'Sample Filtered Gene Correlations', 'Sample Filtered Non-Zero Gene Correlations'])
    plt.title('Violin Plot of Correlation Coefficients')
    plt.ylabel('Correlation Coefficient')
    plt.savefig(os.path.join(args.output, "correlation_violin_plot_filtered_genes.png"))
    plt.close()
    # plot with altair box plot for gene correlations, sample correlations, sample non-zero gene correlations
    df_melted = pd.DataFrame({
        "Type": ["Filtered Gene Correlations"] * len(corr_val_genes) + ["Sample Filtered Gene Correlations"] * len(corr_val_samples) + ["Sample Filtered Non-Zero Gene Correlations"] * len(corr_val_samples_non_zero),
        "Correlation": corr_val_genes + corr_val_samples + corr_val_samples_non_zero
    })
    chart = alt.Chart(df_melted).mark_boxplot().encode(
        x="Type:N",
        y="Correlation:Q"
    ).interactive()
    chart.save(os.path.join(args.output, "correlation_boxplot_filtered_genes.html"))

    # Save correlation results
    df_gene_cor = pd.DataFrame({"Gene": corr_genes, "Correlation": corr_val_genes})
    df_gene_cor.to_csv(os.path.join(args.output, "filtered_gene_correlation_results.csv"), index=False)

    # write gene correlation > 0.3 to a txt
    with open(os.path.join(args.output, "gene_correlation_greater_0.3.txt"), "w") as f:
        for gene, corr in zip(corr_genes, corr_val_genes):
            if corr > 0.3:
                f.write(f"{gene}\n")
    
    # write the top 200 genes to a txt
    df_gene_cor = df_gene_cor.sort_values(by="Correlation", ascending=False).head(200)
    with open(os.path.join(args.output, "gene_correlation_top_200.txt"), "w") as f:
        for gene in df_gene_cor["Gene"]:
            f.write(f"{gene}\n")

    df_sample_cor = pd.DataFrame({"Sample": samples, 
                                 "Correlation all genes": corr_val_samples,
                                 "Correlation non-zero genes": corr_val_samples_non_zero})
    df_sample_cor.to_csv(os.path.join(args.output, "sample_correlation_results.csv"), index=False)

    
if __name__ == "__main__":
    main()