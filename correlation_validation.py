'''
Validate the gene expression correlation between the microscopy images and the gene expression data.
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import QuantileTransformer
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
import argparse
import altair as alt
## plotting settings
if True:  # In order to bypass isort when saving
    from altairThemes import altairThemes
alt.themes.register("publishTheme", altairThemes.publishTheme)

def parse_gt_files(files:list[str]) -> pd.DataFrame:
    '''Parse ground truth files and return a DataFrame.

    Args:
        files (list[str]): List of paths to the ground truth files.

    Returns:
        pd.DataFrame: DataFrame containing the ground truth data.
    '''
    gt_data = []
    for f in files:
        exp_name = str(os.path.basename(f)).replace('_gene_counts_gene_symbols.txt', '')
        df = pd.read_csv(f, sep='\t', header=0, names=[exp_name,"gene_symbol"])
        # make it such that one row is one sample, and one column is one gene, and the value is the expression level
        df = df.set_index("gene_symbol").T
        # normalize to 1e6 then log2 + 1
        df = np.log2((df / df.sum(axis=1).values[0]) * 1e6 + 1)
        gt_data.append(df)
    return pd.concat(gt_data)

def compute_stats_gt(gt_df: pd.DataFrame, pred_df: np.array, save: str) -> None:
    '''Compute statistics for the ground truth DataFrame.

    Args:
        df (pd.DataFrame): Ground truth DataFrame.
        save (str): Path to save the statistics.
    '''
    # plot violin plots of the number of features expressed per sample for both gt and pred
    num_features_per_sample_gt = (gt_df > 0).sum(axis=1)
    num_features_per_sample_pred = (pred_df > 0).sum(axis=1)
    df = pd.DataFrame({
        "Ground Truth": num_features_per_sample_gt,
        "Predicted": num_features_per_sample_pred
    })
    df.to_csv(os.path.join(save, "gt_num_features_per_sample.csv"), index=False)
    # plot side by size violin
    plt.figure(figsize=(12, 12))
    sns.violinplot(data=df, orient="v")
    plt.title("Number of Features Expressed per Sample")
    plt.ylabel("Number of Features")
    plt.savefig(os.path.join(save, "gt_num_features_violin.png"))
    plt.close()
    # altair boxplot
    altair_df = pd.DataFrame({
        "Type": ["Ground Truth"] * len(num_features_per_sample_gt) + ["Predicted"] * len(num_features_per_sample_pred),
        "Number of Features": num_features_per_sample_gt.tolist() + num_features_per_sample_pred.tolist()
    })
    chart = alt.Chart(altair_df).mark_boxplot().encode(
        x="Type:N",
        y="Number of Features:Q"
    ).interactive()
    chart.save(os.path.join(save, "gt_num_features_boxplot.html"))

    # plot the distribution of mean expression of genes per sample for both gt and pred
    mean_expression_per_sample_gt = gt_df.mean(axis=1)
    mean_expression_per_sample_pred = pred_df.mean(axis=1)
    df = pd.DataFrame({
        "Ground Truth": mean_expression_per_sample_gt,
        "Predicted": mean_expression_per_sample_pred
    })
    plt.figure(figsize=(12, 12))
    sns.violinplot(data=df, orient="v")
    plt.title("Mean Gene Expression per Sample")
    plt.ylabel("Mean Expression")
    plt.savefig(os.path.join(save, "gt_mean_expression_violin.png"))
    plt.close()
    # plot boxplot with altair
    altair_df = pd.DataFrame({
        "Type": ["Ground Truth"] * len(mean_expression_per_sample_gt) + ["Predicted"] * len(mean_expression_per_sample_pred),
        "Mean Expression": mean_expression_per_sample_gt.tolist() + mean_expression_per_sample_pred.tolist()
    })
    chart = alt.Chart(altair_df).mark_boxplot().encode(
        x="Type:N",
        y="Mean Expression:Q"
    ).properties(
        title="Mean Gene Expression per Sample"
    ).interactive()
    chart.save(os.path.join(save, "gt_mean_expression_boxplot.html"))

    # plot the distribution of gene with highest std in gt
    std_across_genes = gt_df.std(axis=0)
    idx_to_plot = std_across_genes.argsort()[-10:]
    df = pd.DataFrame({
        "truth": gt_df.to_numpy().flatten()[idx_to_plot],
        "pred": pred_df.flatten()[idx_to_plot],
        "feature": gt_df.columns.values[idx_to_plot]
    })
    df.to_csv(os.path.join(save, "gt_gene_distribution.csv"), index=False)

    plt.figure(figsize=(12,12))
    melted = pd.melt(df, id_vars="feature", value_vars=["truth", "pred"], var_name="type")
    sns.violinplot(x="feature", y="value", hue="type", data=melted)
    plt.title("Distribution per gene: predictions vs truth of top genes with highest std")
    plt.xlabel("Gene")
    plt.ylabel("Value")
    plt.savefig(os.path.join(save, "gt_gene_distribution_violin.png"))
    plt.close()
    # altair
    altair_df = pd.DataFrame({
        "Feature": df["feature"].tolist() * 2,
        "Value": df["truth"].tolist() + df["pred"].tolist(),
        "Type": ["Ground Truth"] * len(df) + ["Predicted"] * len(df)
    })
    chart = alt.Chart(altair_df).mark_boxplot().encode(
        x="Feature:N",
        y="Value:Q",
        color="Type:N"
    ).interactive()
    chart.save(os.path.join(save, "gt_gene_distribution_boxplot.html"))

    highest_pred_genes = pd.DataFrame(pred_df).std(axis=0).nlargest(10).index
    df = pd.DataFrame({
        "truth": gt_df.to_numpy().flatten()[highest_pred_genes],
        "pred": pred_df.flatten()[highest_pred_genes],
        "feature": gt_df.columns.values[highest_pred_genes]
    })
    plt.figure(figsize=(12,12))
    sns.violinplot(x="feature", y="value", hue="type",
                data=pd.melt(df, id_vars="feature", value_vars=["truth", "pred"], var_name="type"))
    plt.title("Distribution per gene: predictions vs truth of top genes with highest std")
    plt.xlabel("Gene")
    plt.ylabel("Value")
    plt.savefig(os.path.join(save, "gt_gene_distribution_violin_largest_in_pred.png"))
    plt.close()
    # altair
    altair_df = pd.DataFrame({
        "Feature": df["feature"].tolist() * 2,
        "Value": df["truth"].tolist() + df["pred"].tolist(),
        "Type": ["Ground Truth"] * len(df) + ["Predicted"] * len(df)
    })
    chart = alt.Chart(altair_df).mark_boxplot().encode(
        x="Feature:N",
        y="Value:Q",
        color="Type:N"
    ).interactive()
    chart.save(os.path.join(save, "gt_gene_distribution_boxplot_largest_in_pred.html"))

    # plot variance per sample vs per gene
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=pd.DataFrame({
        "truth": gt_df.var(axis=1),
        "pred": pred_df.var(axis=1)
    }))
    plt.title("Variance per Sample: Predictions vs Truth")
    plt.ylabel("Variance")
    plt.savefig(os.path.join(save, "gt_variance_per_sample.png"))
    plt.close()
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=pd.DataFrame({
        "truth": gt_df.var(axis=0),
        "pred": pred_df.var(axis=0)
    }))
    plt.title("Variance per Gene: Predictions vs Truth")
    plt.ylabel("Variance")
    plt.savefig(os.path.join(save, "gt_variance_per_gene.png"))
    plt.close()
    # altair variance per sample
    altair_df = pd.DataFrame({
        "Type": ["Ground Truth"] * gt_df.shape[0] + ["Predicted"] * pred_df.shape[0],
        "Variance": gt_df.var(axis=1).tolist() + np.var(pred_df, axis=1).tolist()
    })
    chart = alt.Chart(altair_df).mark_boxplot().encode(
        x="Type:N",
        y="Variance:Q"
    ).properties(
        title="Variance per Sample: Predictions vs Truth"
    ).interactive()
    chart.save(os.path.join(save, "gt_variance_per_sample_boxplot.html"))

def main():
    print("Starting correlation validation...")
    parser = argparse.ArgumentParser(description="Validate gene expression correlation")
    parser.add_argument('--pred_npy', type=str, required=True, help='Path to the predicted npy file')
    parser.add_argument('--exp_label', type=str, required=True, help='Path to the expression label npy file')
    parser.add_argument('--gene_list', type=str, required=True, help='Path to the gene symbol list npy')
    parser.add_argument('--ground_truth', type=str, required=True, help='Path to the ground truth directory')
    parser.add_argument('--output', type=str, required=True, help='Output file path for correlation results')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    gt_files = [os.path.join(args.ground_truth, f) 
                for f in os.listdir(args.ground_truth) if f.endswith('_gene_symbols.txt')]
    
    gene_symbols_list = np.load(args.gene_list, allow_pickle=True).reshape(-1)
    pred = np.load(args.pred_npy) # shape (samples, num_genes), order of genes is in the gene_symbols_list
    exp_label = np.load(args.exp_label, allow_pickle=True).reshape(-1) # shape (samples,)
    # average over the pred with the same experiment label
    pred = pd.DataFrame(pred, columns=gene_symbols_list, index=exp_label).groupby(exp_label).mean()
    labels = pred.index.values
    pred = pred.values
    gt_data_df = parse_gt_files(gt_files) # row: samples, columns: genes

    # reorder such that the experiment orders are the same on rows as exp_label
    gt_data_df = gt_data_df.reindex(labels)

    common_genes = gt_data_df.columns.intersection(gene_symbols_list)
    print("Number of common genes: ", common_genes.shape[0])

    # get common genes only
    gt_data_df = gt_data_df.loc[:, common_genes]
    pred = pred[:, np.isin(gene_symbols_list, common_genes)]

    # reorder such that the columns are in the same order
    common_genes_ordered = [gene for gene in gene_symbols_list if gene in common_genes]
    gt_data_df = gt_data_df.reindex(columns=common_genes_ordered)
    print(gt_data_df.head(10))

    # dimension reduction on shane's data
    print("Performing PCA on ground truth data...")
    pca = PCA()
    gt_data_reduced = pca.fit_transform(gt_data_df.fillna(0).to_numpy())
    # plot, where label is the sample (row name)
    # get colors for each unique label
    unique_labels = np.unique(gt_data_df.index)
    label_to_color = {label: color for label, color in zip(unique_labels, sns.color_palette("hsv", len(unique_labels)))}
    colors = [label_to_color[label] for label in gt_data_df.index]

    plt.figure(figsize=(8, 8))
    plt.scatter(gt_data_reduced[:, 0], gt_data_reduced[:, 1], c=colors, s=50)
    plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', label=label,
                          markerfacecolor=color, markersize=10) for label, color in label_to_color.items()])
    plt.title("PCA of Ground Truth Gene Expression Data")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)")
    plt.savefig(os.path.join(args.output, "gt_pca.png"))
    plt.close()

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
        metrics.append([common_genes_ordered[j], mae, rmse, nrmse, ev, spearman, var_ratio])

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

    # write that filter by Explain variance >= 0.5 and Spearman >= 0.3
    filtered_genes = df_metrics[(df_metrics["Explained Variance"] >= 0.5) & (df_metrics["Spearman"] >= 0.3)]["Gene"].tolist()
    with open(os.path.join(args.output, "gene_correlation_metrics_filtered_ev0.5_spearman0.3.txt"), "w") as f:
        f.write("\n".join(filtered_genes))

    # run EnrichR to see if the filtered genes are enriched in any cellular processes, and save the results
    from gseapy import enrichr
    enr_res = enrichr(
        gene_list=filtered_genes,
        gene_sets="GO_Biological_Process_2025",
        outdir=args.output,
        cutoff=0.25,  # Only consider terms with adj p-value < 0.25
        verbose=True
    )

    # Sort by combined score and select top 10 UP enriched
    top_terms_sorted = enr_res.results.sort_values("Combined Score", ascending=False)
    top_terms_sorted = top_terms_sorted[top_terms_sorted["Adjusted P-value"] < 0.25]
    top_terms_sorted = top_terms_sorted[top_terms_sorted["Combined Score"] > 0]

    # plot with Altair
    altair_df = pd.DataFrame({
        "Term": top_terms_sorted["Term"],
        "Combined Score": top_terms_sorted["Combined Score"],
    })
    chart = alt.Chart(altair_df).mark_bar().encode(
        x=alt.X('Combined Score:Q', title='Combined Score'),
        y=alt.Y('Term:N', sort='-x', title='Gene Set'),
        color=alt.value('indianred')  # Color for bars
    ).properties(
        title="Top Positively Enriched Gene Sets from GO Biological Process (Enrichr)",
    ).interactive()
    chart.save(os.path.join(args.output, 'enrichr_GOprocess_filter_results_top_altair.html'))

    #* plot explained variance vs spearman
    df_metrics = df_metrics.dropna(subset=["Explained Variance", "Spearman"])
    # drop very negative explained variance
    df_metrics = df_metrics[df_metrics["Explained Variance"] >= -1]
    plt.figure(figsize=(8, 8))
    plt.scatter(df_metrics["Explained Variance"], df_metrics["Spearman"], alpha=0.5)
    plt.xlabel("Explained Variance")
    plt.ylabel("Spearman Correlation")
    plt.title("Explained Variance vs Spearman Correlation")
    plt.axvline(x=0.5, color='r', linestyle='--')
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
       x=alt.datum(0.5)
    )
    h_line = alt.Chart(df_metrics).mark_rule(color='red', strokeDash=[5,5]).encode(
        y=alt.datum(0.3)
    )   
    chart = chart + v_line + h_line
    chart.save(os.path.join(args.output, "gene_correlation_ev_vs_spearman.html"))

    # plot explained variance vs variance of GT
    genes = df_metrics["Gene"].values
    pred_mean_per_gene = pred[:, np.isin(common_genes_ordered, genes)].mean(axis=0) # shape (num_genes,)
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
    for i, gene in enumerate(common_genes_ordered):
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
        pred_values = pred[i, np.isin(common_genes_ordered, non_zero_pred_genes)].flatten()
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
    for i, gene in enumerate(common_genes_ordered):
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
    pred_filtered = pred[:, np.isin(common_genes_ordered, corr_genes)]
    gt_data_df_filtered = gt_data_df.loc[:, corr_genes].reindex(columns=corr_genes)
    for i, sample in enumerate(gt_data_df_filtered.index):
        pred_values = pred_filtered[i, :].flatten()
        exp_values = gt_data_df_filtered.iloc[i, :].values.flatten()
        correlation = spearmanr(exp_values, pred_values).correlation
        corr_val_samples.append(correlation)
        samples.append(sample)

    corr_val_samples_non_zero = []
    pred_filtered_non_zero = pred[:, np.isin(common_genes_ordered, non_zero_pred_genes)]
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

    # print("Calculating statistics on gene correlation with ALL genes...")

    # # Quantile normalization between pred and actual
    # qt = QuantileTransformer(random_state=0)
    # pred = qt.fit_transform(pred.T).T
    # gt_data_df = pd.DataFrame(qt.fit_transform(gt_data_df.T).T, columns=gt_data_df.columns, index=gt_data_df.index)
    # gt_data_df = (gt_data_df - np.mean(gt_data_df, axis=0)) / np.std(gt_data_df, axis=0) if np.std(gt_data_df, axis=0).all() > 0 else gt_data_df
    # Validate the gene expression correlation
    # corr_genes = []
    # corr_val_genes = []
    # non_zero_pred_genes = []
    # non_zero_corr = []
    # for gene in gene_symbols_list:
    #     if (gene in gt_data_df.columns) and (gene in common_genes):
    #         # Compute correlation
    #         pred_values = pred[:, gene_symbols_list == gene].flatten()
    #         exp_values = gt_data_df[gene].values.flatten()
    #         if np.sum(pred_values) < 1e-8 and np.sum(exp_values) < 1e-8: # both are all zero
    #             correlation = 1.0
    #         elif np.sum(exp_values) < 1e-8 or np.sum(pred_values) < 1e-8: # one array is constant but the other is not
    #             correlation = 0.0
    #         else: # at least one is non-zero
    #             correlation = spearmanr(exp_values, pred_values).correlation
    #             non_zero_pred_genes.append(gene)
    #             non_zero_corr.append(correlation)
    #         corr_genes.append(str(gene))
    #         corr_val_genes.append(correlation)
    # df_non_zero_pred = pd.DataFrame({"Gene": non_zero_pred_genes, "Correlation": non_zero_corr})
    # df_non_zero_pred.to_csv(os.path.join(args.output, "gene_correlation_non_zero.csv"), index=False)
    # print("Mean of non-zero correlations:", df_non_zero_pred["Correlation"].mean())
    
    # # Validate sample correlation
    # print("Calculating statistics on sample using high correlation genes...")
    # samples = []
    # corr_val_samples = []
    # # filter such that both pred and exp only contains corr_genes
    # pred_filtered = pred[:, np.isin(gene_symbols_list, corr_genes)]
    # gt_data_df_filtered = gt_data_df.loc[:, corr_genes].reindex(columns=corr_genes)
    # for i, sample in enumerate(gt_data_df_filtered.index):
    #     pred_values = pred_filtered[i, :].flatten()
    #     exp_values = gt_data_df_filtered.iloc[i, :].values.flatten()
    #     correlation = spearmanr(exp_values, pred_values).correlation
    #     corr_val_samples.append(correlation)
    #     samples.append(sample)

    # corr_val_samples_non_zero = []
    # pred_filtered_non_zero = pred[:, np.isin(gene_symbols_list, non_zero_pred_genes)]
    # gt_data_df_filtered_non_zero = gt_data_df.loc[:, non_zero_pred_genes].reindex(columns=non_zero_pred_genes)
    # for i, sample in enumerate(gt_data_df_filtered_non_zero.index):
    #     pred_values = pred_filtered_non_zero[i, :].flatten()
    #     exp_values = gt_data_df_filtered_non_zero.iloc[i, :].values.flatten()
    #     correlation = spearmanr(exp_values, pred_values).correlation
    #     corr_val_samples_non_zero.append(correlation)
    
    # print("Mean sample correlation all genes:", np.mean(corr_val_samples))
    # print("Std sample correlation all genes:", np.std(corr_val_samples))
    # print("Mean sample correlation non-zero genes:", np.mean(corr_val_samples_non_zero))
    # print("Std sample correlation non-zero genes:", np.std(corr_val_samples_non_zero))

if __name__ == "__main__":
    main()