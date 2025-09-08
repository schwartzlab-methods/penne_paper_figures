'''
Validate the gene expression correlation between the microscopy images and the gene expression data.
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import QuantileTransformer
import argparse

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
        # make it such that gene symbols are row
        df = df.set_index("gene_symbol").T
        # normalize to 1e6 then log2 + 1
        df = np.log2((df / df.sum(axis=1).values[0]) * 1e6 + 1)
        gt_data.append(df)
    return pd.concat(gt_data)

def compute_stats_gt(gt_df: pd.DataFrame, pred_df: pd.DataFrame, save: str) -> None:
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
    # plot side by size violin
    plt.figure(figsize=(12, 12))
    sns.violinplot(data=df, orient="v")
    plt.title("Number of Features Expressed per Sample")
    plt.ylabel("Number of Features")
    plt.savefig(os.path.join(save, "gt_num_features_violin.png"))
    plt.close()

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

def main():
    parser = argparse.ArgumentParser(description="Validate gene expression correlation")
    parser.add_argument('--pred_npy', type=str, required=True, help='Path to the predicted npy file')
    parser.add_argument('--exp_label', type=str, required=True, help='Path to the expression label npy file')
    parser.add_argument('--gene_list', type=str, required=True, help='Path to the gene symbol list npy')
    parser.add_argument('--ground_truth', type=str, required=True, help='Path to the ground truth directory')
    parser.add_argument('--output', type=str, required=True, help='Output file path for correlation results')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    rng = np.random.default_rng()

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

    compute_stats_gt(gt_data_df, pred, args.output)

    # Quantil normalization between pred and actual
    qt = QuantileTransformer(random_state=0)
    pred = qt.fit_transform(pred)
    gt_data_df = qt.fit_transform(gt_data_df)
    # gt_data_df = (gt_data_df - np.mean(gt_data_df, axis=0)) / np.std(gt_data_df, axis=0) if np.std(gt_data_df, axis=0).all() > 0 else gt_data_df
    # Validate the gene expression correlation
    corr_genes = []
    corr_val_genes = []
    non_zero_pred_genes = []
    non_zero_corr = []
    for gene in gene_symbols_list:
        if gene in gt_data_df.columns:
            # Compute correlation
            # due to small sample sizes, we will add a random small number to each to avoid nan
            pred_values = pred[:, gene_symbols_list == gene].flatten()# + np.abs(rng.normal(0, 1e-6, pred[:, gene_symbols_list == gene].flatten().shape))
            exp_values = gt_data_df[gene].values# + np.abs(rng.normal(0, 1e-6, gt_data_df[gene].values.shape))
            # Z-score normalization
            if np.sum(pred_values) == 0 and np.sum(exp_values) == 0:
                correlation = 1.0
            elif np.sum(pred_values) == 0 or np.sum(exp_values) == 0:
                correlation = 0.0
            else:
                correlation = np.corrcoef(pred_values, exp_values)[0, 1]
                non_zero_pred_genes.append(gene)
                non_zero_corr.append(correlation)

            # if correlation is nan, print exp_val and pre_val
            corr_genes.append(str(gene))
            corr_val_genes.append(correlation)
    df_non_zero_pred = pd.DataFrame({"Gene": non_zero_pred_genes, "Correlation": non_zero_corr})
    df_non_zero_pred.to_csv(os.path.join(args.output, "gene_correlation_non_zero.csv"), index=False)
    print("Mean of non-zero correlations:", df_non_zero_pred["Correlation"].mean())
    
    # Validate sample correlation
    samples = []
    corr_val_samples = []
    # filter such that both pred and exp only contains corr_genes
    pred_filtered = pred[:, np.isin(gene_symbols_list, corr_genes)]
    gt_data_df_filtered = gt_data_df.loc[:, corr_genes].reindex(columns=corr_genes)
    for i, sample in enumerate(gt_data_df_filtered.index):
        pred_values = pred_filtered[i, :].flatten()
        exp_values = gt_data_df_filtered.iloc[i, :].values.flatten()
        correlation = np.corrcoef(pred_values, exp_values)[0, 1]
        corr_val_samples.append(correlation)
        samples.append(sample)

    corr_val_samples_non_zero = []
    pred_filtered_non_zero = pred[:, np.isin(gene_symbols_list, non_zero_pred_genes)]
    gt_data_df_filtered_non_zero = gt_data_df.loc[:, non_zero_pred_genes].reindex(columns=non_zero_pred_genes)
    for i, sample in enumerate(gt_data_df_filtered_non_zero.index):
        pred_values = pred_filtered_non_zero[i, :].flatten()
        exp_values = gt_data_df_filtered_non_zero.iloc[i, :].values.flatten()
        correlation = np.corrcoef(pred_values, exp_values)[0, 1]
        corr_val_samples_non_zero.append(correlation)

    # plot correlation violin plots for both
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=[corr_val_genes, corr_val_samples, corr_val_samples_non_zero], inner="quartile")
    plt.xticks([0, 1, 2], ['Gene Correlations', 'Sample All Gene Correlations', 'Sample Non-Zero Gene Correlations'])
    plt.title('Violin Plot of Correlation Coefficients')
    plt.ylabel('Correlation Coefficient')
    plt.savefig(os.path.join(args.output, "correlation_violin_plot.png"))
    plt.close()

    # Save correlation results
    df_gene_cor = pd.DataFrame({"Gene": corr_genes, "Correlation": corr_val_genes})
    df_gene_cor.to_csv(os.path.join(args.output, "gene_correlation_results.csv"), index=False)

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