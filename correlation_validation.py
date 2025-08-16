'''
Validate the gene expression correlation between the microscopy images and the gene expression data.
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
        exp_name = os.path.basename(f).replace('_gene_counts__gene_symbols.txt', '')
        df = pd.read_csv(f, sep='\t', header=1, names=[exp_name,"gene_symbol"])
        # make it such that gene symbols are row
        df = df.set_index("gene_symbol").T
        # normalize to 1e6 then log2 + 1
        df = np.log2((df / df.sum(axis=1).values[0]) * 1e6 + 1)
        gt_data.append(df)
    return pd.concat(gt_data)

def main():
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
    
    pred = np.load(args.pred_npy) # shape (samples, num_genes), order of genes is in the gene_symbols_list
    exp_label = np.load(args.exp_label, allow_pickle=True).reshape(-1) # shape (samples,)
    gene_symbols_list = np.load(args.gene_list, allow_pickle=True).reshape(-1)

    gt_data_df = parse_gt_files(gt_files) # row: samples, columns: genes

    # reorder such that the experiment orders are the same on rows as exp_label
    gt_data_df = gt_data_df.reindex(exp_label)

    # Validate the gene expression correlation
    corr_genes = []
    corr_val_genes = []
    for gene in gene_symbols_list:
        if gene in gt_data_df.columns:
            # Compute correlation
            pred_values = pred[:, gene_symbols_list == gene].flatten()
            exp_values = gt_data_df[gene].values
            correlation = np.corrcoef(pred_values, exp_values)[0, 1]
            corr_genes.append(gene)
            corr_val_genes.append(correlation)

    # Validate sample correlation
    samples = []
    corr_val_samples = []
    # filter such that both pred and exp only contains corr_genes
    pred_filtered = pred[:, np.isin(gene_symbols_list, corr_genes)]
    gt_data_df_filtered = gt_data_df[:, corr_genes].reindex(columns=corr_genes)
    for i, sample in enumerate(gt_data_df.columns):
        pred_values = pred_filtered[i, :].flatten()
        exp_values = gt_data_df_filtered[i, :].values.flatten()
        correlation = np.corrcoef(pred_values, exp_values)[0, 1]
        corr_val_samples.append(correlation)
        samples.append(sample)
    
    # plot correlation violin plots for both
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=[corr_val_genes, corr_val_samples], inner="quartile")
    plt.xticks([0, 1], ['Gene Correlations', 'Sample Correlations'])
    plt.title('Violin Plot of Correlation Coefficients')
    plt.ylabel('Correlation Coefficient')
    plt.savefig(args.output, "correlation_violin_plot")
    plt.close()

if __name__ == "__main__":
    main()