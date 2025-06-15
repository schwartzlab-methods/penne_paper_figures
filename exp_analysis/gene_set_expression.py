import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from tqdm import tqdm

def read_tsv(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip().split('\t') for line in f]
    df = pd.DataFrame(lines).T #number of genes x number of cells
    df.columns = [cell.lower() for cell in df.iloc[0].tolist()]  # set the first row as header
    df = df[2:]  # remove the first row
    return df

def validate_enrichment(expression_matrix, cell_labels, gene_names, enriched_gene_sets, out):
    """
    For each cell type, tests whether its enriched genes are significantly
    more expressed in cells of that type than in others.

    Parameters:
    - expression_matrix: numpy array of shape (cells x genes)
    - cell_labels: numpy array of shape (cells,)
    - gene_names: list of gene names corresponding to columns in expression_matrix
    - enriched_gene_sets: dict mapping cell type -> list of gene names
    - out: path to save all figures and csvs
    """
    gene_df = pd.DataFrame(expression_matrix, columns=gene_names)
    label_series = pd.Series(cell_labels, name="cell_type")
    
    results = []

    for cell_type, gene_set in tqdm(enriched_gene_sets.items()):
        # Filter for genes in the set that exist in gene_names
        valid_genes = [g for g in gene_set if g in gene_names]
        if not valid_genes:
            print("no valid genes found for cell type:", cell_type)
            continue
        # Compute per-cell average expression of the gene set
        module_score = gene_df[valid_genes].mean(axis=1)

        # Compare scores between matching and non-matching cell types
        is_target = label_series == cell_type
        target_scores = module_score[is_target]
        background_scores = module_score[~is_target]

        # compare the mean exp of gene set vs everything else
        # get module score for the complement of the gene set
        complement_genes = [g for g in gene_names if g not in valid_genes]
        complement_scores = gene_df[complement_genes].mean(axis=1)[is_target]

        # Mann-Whitney U test (non-parametric)
        _, pval_cross_celltype = mannwhitneyu(target_scores, background_scores)
        _, pval_within_celltype = mannwhitneyu(target_scores, complement_scores)

        # Record result
        results.append({
            "cell_type": cell_type,
            "num_genes_in_set": len(valid_genes),
            "mean_score_in_type": target_scores.mean(),
            "mean_score_in_type_nonmarker": complement_scores.mean(),
            "mean_score_elsewhere": background_scores.mean(),
            "p_value_cross_celltype": pval_cross_celltype,
            "p_value_within_celltype": pval_within_celltype
        })

        results_df = pd.DataFrame(results)
        results_df["FDR_cross"] = results_df["p_value_cross_celltype"] * len(results_df)  # Bonferroni correction
        results_df["FDR_within"] = results_df["p_value_within_celltype"] * len(results_df)  # Bonferroni correction
        
        # plot violin plot
        plt.figure(figsize=(4, 4))
        sns.violinplot(
            x=label_series.isin([cell_type]).map({True: cell_type, False: "Other"}),
            y=module_score,
            palette="muted"
        )
        plt.title(f"Mean Expression for {cell_type} Marker Genes")
        plt.ylabel("Mean Expression of Gene Set")
        plt.xlabel("Cell Type")
        plt.tight_layout()
        plt.savefig(os.path.join(out, f"{cell_type}_gene_set_mean_exp_across.png"))
        plt.close()

        plt.figure(figsize=(4, 4))
        sns.violinplot(
            x=["Marker"]*len(target_scores.tolist()) + ["Other"]*len(complement_scores.tolist()),
            y=target_scores.tolist()+complement_scores.tolist(),
            palette="muted"
        )
        plt.title(f"Mean Expression for {cell_type} Marker Genes")
        plt.ylabel("Mean Expression of Gene Set")
        plt.xlabel("Gene Type")
        plt.tight_layout()
        plt.savefig(os.path.join(out, f"{cell_type}_gene_set_mean_exp_within.png"))
        plt.close()


    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(out, "gene_set_enrichment.csv"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expression_npy', type=str, help='Path to expression numpy matrix')
    parser.add_argument('--up_gene_sets', type=str, help='Path to the .gmt file containg the up-regulated gene sets')
    parser.add_argument('--gene_names', type=str, help='Path to the numpy file containg the names of the genes')
    parser.add_argument('--cell_names', type=str, help='Path to the numpy file containg the names of the cells')
    parser.add_argument('--gene_template', type=str, default=None, help='Optional, path to the features.tsv file for converting gene names to gene symbols')
    parser.add_argument('--output', type=str, help='Path to the output directory')
    parser.add_argument('--cell_types', type=str, nargs="+", help="Cell types that you want to compare")
    args = parser.parse_args()
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    expression_matrix = np.load(args.expression_npy)
    # normalize to counts per million
    expression_matrix = expression_matrix / np.sum(expression_matrix, axis=1, keepdims=True) * 1e6
    # # log2 transform
    # expression_matrix = np.log2(cexpression_matrixounts + 1)
    gene_names = np.load(args.gene_names)
    cell_labels = [cell.lower() for cell in np.load(args.cell_names).tolist()]
    celltype_of_interest = [cell.lower() for cell in args.cell_types]
    signature = read_tsv(args.up_gene_sets)

    if args.gene_template:
        # convert the signature from gene names to gene symbols
        template = pd.read_csv(args.gene_template, sep='\t', header=None)
        name_to_symbol = {row[1]: row[0] for _, row in template.iterrows()}  # map from gene name to gene symbol
        # convert the signature to gene symbols by mapping the dataframe
        signature = signature.map(lambda x: name_to_symbol.get(x, x))  # map gene names to symbols

    # select only the cell type of interest
    enriched_gene_sets = {
        celltype: signature[celltype].dropna().tolist() for celltype in celltype_of_interest if celltype in signature.columns
    }
    for key, value in enriched_gene_sets.items():
        print(f"Number of marker genes of {key}:", len(value))
    
    # Validate enrichment
    results_df = validate_enrichment(expression_matrix, cell_labels, gene_names, enriched_gene_sets, args.output)

if __name__ == "__main__":
    main()