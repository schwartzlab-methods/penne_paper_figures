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

    # Load count data (rows: cells, columns: gemes)
    counts = np.load(args.counts)
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
    cell_types = cell_types[cell_type_mask]

    # Create a DataFrame for the counts
    counts_df = pd.DataFrame(counts, columns=gene_names)

    # do differential expression analysis
    results = []
    for gene in gene_names:
        # Create a DataFrame for the current gene
        gene_df = counts_df[[gene]].copy()
        gene_df['cell_type'] = cell_types

        # Fit a linear model
        model = sm.OLS(gene_df[gene], sm.add_constant(pd.get_dummies(gene_df['cell_type'], drop_first=True)))
        results.append(model.fit())
    
    # Extract p-values and coefficients
    p_values = [result.pvalues[1] for result in results]
    coefficients = [result.params[1] for result in results]
    # Create a DataFrame for the results
    results_df = pd.DataFrame({
        'gene': gene_names,
        'p_value': p_values,
        'coefficient': coefficients
    })
    # Adjust p-values for multiple testing
    results_df['adj_p_value'] = sm.stats.multipletests(results_df['p_value'], method='fdr_bh')[1]
    # Save results
    results_df.to_csv(os.path.join(args.output, f'deg_{args.cell_types[0]}vs{args.cell_types[1]}.csv'), index=False)
    # Plot results
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=results_df, x='coefficient', y=-np.log10(results_df['adj_p_value']), hue='adj_p_value < 0.05')
    plt.axhline(y=-np.log10(0.05), color='r', linestyle='--')
    plt.title(f'Differential Gene Expression Analysis: {args.cell_types[0]} vs {args.cell_types[1]}')
    plt.xlabel('Coefficient')
    plt.ylabel('-log10(Adjusted p-value)')
    plt.savefig(os.path.join(args.output, f'deg_plot_{args.cell_types[0]}vs{args.cell_types[1]}.png'))
    plt.close()
    # get a list of differentially expressed genes
    deg_genes = results_df[results_df['adj_p_value'] < 0.05]['gene'].tolist()
    # save the list of differentially expressed genes
    with open(os.path.join(args.output, f'deg_genes_{args.cell_types[0]}vs{args.cell_types[1]}.txt'), 'w') as f:
        for gene in deg_genes:
            f.write(f"{gene}\n")
    print(f"DEG analysis completed. Results saved to {args.output}")

if __name__ == "__main__":
    main()
