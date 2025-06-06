'''
Perform differential gene expression analysis on the predictions
'''

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from tqdm import tqdm

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Differential Gene Expression Analysis")
    parser.add_argument('--counts', type=str, required=True, help='Path to the counts matrix npy file')
    parser.add_argument('--labels', type=str, required=True, help='Path to the labels npy file')
    parser.add_argument('--genes', type=str, required=True, help='Path to the gene names npy file')
    parser.add_argument('--cell_types', type=str, required=True, nargs="+", help='The two cell types to compare')
    parser.add_argument('--output', type=str, required=True, help='Output directory for results')
    args = parser.parse_args()
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    args.cell_types.sort() #sort as the alphabetical first is the reference in linear modeling
    # Load count data (rows: cells, columns: gemes)
    counts = np.load(args.counts).astype(float)
    cell_types = np.load(args.labels)
    gene_names = np.load(args.genes)
    assert counts.shape[0] == cell_types.shape[0], "Counts and labels must have the same number of cells"
    assert counts.shape[1] == gene_names.shape[0], "Counts and gene names must match in number of genes"
    assert len(args.cell_types) == 2, "Please provide exactly two cell types for comparison"
    for each in args.cell_types:
        assert each in cell_types, f"Cell type {each} not found in labels"

    # normalize to counts per million
    counts = counts / np.sum(counts, axis=1, keepdims=True) * 1e6
    # log2 transform
    counts = np.log2(counts + 1)
    
    # get only the cell types of interest from counts
    cell_type_mask = np.isin(cell_types, args.cell_types)
    counts = counts[cell_type_mask]
    cell_types = cell_types[cell_type_mask].astype(str)

    # Create a DataFrame for the counts
    counts_df = pd.DataFrame(counts, columns=gene_names)

    # do differential expression analysis
    results = []
    new_gene_names = []
    for gene in tqdm(gene_names):
        # Create a DataFrame for the current gene
        gene_df = counts_df[[gene]].copy()
        if (0 == gene_df.to_numpy()).all():
            continue
        new_gene_names.append(gene)
        gene_df['cell_type'] = cell_types

        # Create the design matrix
        X = sm.add_constant(pd.get_dummies(gene_df['cell_type'], drop_first=True)).astype(float)
        y = gene_df[gene].astype(float)
        # combine all the genes in y if there are multiple with the same name
        if len(y.shape) != 1:
            y = y.mean(axis=1)
        # Fit a linear model
        model = sm.OLS(y, X, missing="drop")
        results.append(model.fit())

    # Extract p-values and coefficients
    p_values = [result.pvalues.iloc[1] for result in results]
    coefficients = [result.params.iloc[1] for result in results]
    print("Final number of genes analyzed: ", len(new_gene_names))
    # Create a DataFrame for the results
    results_df = pd.DataFrame({
        'gene': new_gene_names,
        'p_value': p_values,
        'log_fc': coefficients
    })
    # Adjust p-values for multiple testing
    results_df['adj_p_value'] = sm.stats.multipletests(results_df['p_value'], method='fdr_bh')[1]
    results_df['Adj p < 0.05 and |Log2FC| > 1'] = results_df.apply(lambda x: (x["adj_p_value"] < 0.05) and (np.absolute(x["log_fc"]) > 1), axis=1).astype(str)
    # results_df["adj_p_value < 0.05"] = (results_df['adj_p_value'] < 0.05).astype(str)
    # Save results
    results_df.to_csv(os.path.join(args.output, f'deg_ref_{args.cell_types[0]}vs{args.cell_types[1]}.csv'), index=False)
    # Plot results
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=results_df, x='log_fc', y=-np.log10(results_df['adj_p_value']), hue='Adj p < 0.05 and Log2FC > 1')
    plt.axhline(y=-np.log10(0.05), color='r', linestyle='--')
    plt.axvline(x=1, color='r', linestyle='--')
    plt.axvline(x=-1, color='r', linestyle='--')
    plt.title(f'Differential Gene Expression Analysis: {args.cell_types[0]}(Reference) vs {args.cell_types[1]}')
    plt.xlabel('Log2 FC')
    plt.ylabel('-log10(FDR Adjusted p-value)')
    plt.savefig(os.path.join(args.output, f'deg_plot_ref_{args.cell_types[0]}vs{args.cell_types[1]}.png'))
    plt.close()
    # get a list of differentially expressed genes
    deg_genes_up = results_df[results_df['adj_p_value'] < 0.05]
    deg_genes_up = deg_genes_up[deg_genes_up['log_fc'] > 1]['gene'].tolist()
    deg_genes_down = results_df[results_df['adj_p_value'] < 0.05]
    deg_genes_down = deg_genes_down[deg_genes_down['log_fc'] < -1]['gene'].tolist()

    # save the list of differentially expressed genes
    with open(os.path.join(args.output, f'deg_genes_ref_{args.cell_types[0]}vs{args.cell_types[1]}_up.txt'), 'w') as f:
        for gene in deg_genes_up:
            f.write(f"{gene}\n")
    with open(os.path.join(args.output, f'deg_genes_ref_{args.cell_types[0]}vs{args.cell_types[1]}_down.txt'), 'w') as f:
        for gene in deg_genes_down:
            f.write(f"{gene}\n")
    print(f"DEG analysis completed. Results saved to {args.output}")

if __name__ == "__main__":
    main()
