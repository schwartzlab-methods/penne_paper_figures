import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from itertools import chain
# from functools import reduce
from tqdm import tqdm
import altair as alt
## plotting settings
if True:  # In order to bypass isort when saving
    from altairThemes import altairThemes
alt.themes.register("publishTheme", altairThemes.publishTheme)
alt.themes.enable("publishTheme")

def permutation_test(group1, group2, num_permutations=100000):
    """
    Perform a two-sided permutation test to compare the means of two groups.

    Parameters:
    - group1: array-like, first group of data
    - group2: array-like, second group of data
    - num_permutations: int, number of permutations to perform

    Returns:
    - p_value: float, the p-value from the permutation test
    """
    np.random.seed(42)

    observed_diff = abs(np.mean(group1) - np.mean(group2))
    print("Observed difference in means:", observed_diff)
    combined = np.concatenate([group1, group2])
    count = 0

    for _ in range(num_permutations):
        np.random.shuffle(combined)
        new_group1 = combined[:len(group1)]
        new_group2 = combined[len(group1):]
        new_diff = abs(np.mean(new_group1) - np.mean(new_group2))
        if new_diff >= observed_diff:
            count += 1

    p_value = count / num_permutations
    return p_value

def read_tsv(file_paths):
    '''
    Read multile .gmt files in a tsv format and combine them into a single dataframe
    Shape: number of genes x number of enrichment sets
    '''
    print("Reading gene set files...")
    df_all = pd.DataFrame()
    for each in file_paths:
        with open(each, 'r', encoding='utf-8') as f:
            lines = [line.strip().split('\t') for line in f]
        df = pd.DataFrame(lines).T #number of genes x number of enrichment sets
        df.columns = [cell.lower().split("-")[0] for cell in df.iloc[0].tolist()]  # set the first row as header
        df = df[2:]  # remove the first row and description
        df_all = pd.concat([df_all, df], axis=1, join="outer")
    # combine the columns with the same name by taking the union of their values
    # df_all = df_all.groupby(df_all.columns, axis=1).agg(lambda x: pd.Series(x).explode().unique().tolist())
    df_all = (
        df_all.groupby(df_all.columns, axis=1)
            .apply(lambda df:
                    pd.Series(df.values.ravel())
                    .explode()
                    .dropna()
                    .unique()
                    .tolist())
    ).to_frame().T

    print("Number of enrichment sets:", df_all.shape[1])
    return df_all

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

    # track summary stats for all the values tested
    marker_in_cell = []
    marker_across_cell = []

    auc_df_final = pd.DataFrame(columns=["cell_type", "tpr", "fpr"])
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

        # plot AUC curve for how well the module score can separate the target cell type from the rest
        from sklearn.metrics import roc_auc_score
        from sklearn.metrics import roc_curve
        auc_score = roc_auc_score(is_target.astype(int), module_score)
        print(f"AUC for {cell_type} marker genes: {auc_score:.4f}")
        roc_output = roc_curve(is_target.astype(int), module_score)
        roc_data = pd.DataFrame({"fpr": roc_output[0], "tpr": roc_output[1]})
        auc_df_final = pd.concat([auc_df_final, pd.DataFrame({"cell_type": [cell_type]*len(roc_data), "fpr": roc_data["fpr"], "tpr": roc_data["tpr"]})], ignore_index=True)
        plt.figure(figsize=(6, 6))
        sns.lineplot(data=roc_data, x="fpr", y="tpr")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title(f"ROC Curve for {cell_type} Marker Genes (AUC={auc_score:.4f})")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.tight_layout()
        plt.savefig(os.path.join(out, f"{cell_type}_marker_genes_roc_curve.png"))
        plt.close()

        # repeat with altair plot
        roc_plot = alt.Chart(roc_data).mark_line().encode(
            x='fpr:Q',
            y='tpr:Q'
        ).properties(
            title=f"ROC Curve for {cell_type} Marker Genes (AUC={auc_score:.4f})",
        ).interactive()
        roc_plot.save(os.path.join(out, f"{cell_type}_marker_genes_roc_curve_altair.html"))

        # compare the mean exp of gene set vs everything else
        # get module score for the complement of the gene set
        complement_genes = [g for g in gene_names if g not in valid_genes]

        # sample the complement genes to have the same number of genes as the target genes
        if len(complement_genes) > len(valid_genes):
            complement_genes = np.random.choice(complement_genes, size=len(valid_genes), replace=False)
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

        # normalization for combining
        current_mean_all = np.concatenate([target_scores, background_scores]).mean()
        current_std_all = np.concatenate([target_scores, background_scores]).std()
        z_norm_in_type = (target_scores - current_mean_all) / current_std_all
        z_norm_elsewhere = (background_scores - current_mean_all) / current_std_all
        marker_in_cell.append(z_norm_in_type)
        marker_across_cell.append(z_norm_elsewhere)

        results_df = pd.DataFrame(results)
        results_df["FDR_cross"] = results_df["p_value_cross_celltype"] * len(results_df)  # Bonferroni correction
        results_df["FDR_within"] = results_df["p_value_within_celltype"] * len(results_df)  # Bonferroni correction
        
        # plot violin plot across cell types (ie: marker of this cell type in this cell type vs in other cell types)
        plt.figure(figsize=(8, 8))
        sns.violinplot(
            x=label_series.isin([cell_type]).map({True: cell_type, False: "Other"}),
            y=module_score,
            palette="muted",
            order=[cell_type, "Other"]
        )
        plt.title(f"Mean Log2-Normalized Expression for {cell_type} Marker Genes")
        plt.ylabel("Mean Log2-Normalized Expression of Gene Set")
        plt.xlabel("Cell Type")
        plt.tight_layout()
        plt.savefig(os.path.join(out, f"{cell_type}_gene_set_mean_exp_across.png"))
        plt.close()
        # plot altair box plot
        plot_df = pd.DataFrame({
            "Mean Log2-Normalized Expression of Gene Set": module_score.tolist(),
            "Cell Type": label_series.isin([cell_type]).map({True: cell_type, False: "Other"})
        })
        box_plot = alt.Chart(plot_df).mark_boxplot().encode(
            y='Mean Log2-Normalized Expression of Gene Set:Q',
            x=alt.X('Cell Type:N', sort=[cell_type, "Other"], axis=alt.Axis(labelAngle=-45)),
            color='Cell Type:N'
        ).properties(
            title=f"Mean Log2-Normalized Expression for {cell_type} Marker Genes",
        ).interactive()
        box_plot.save(os.path.join(out, f"{cell_type}_gene_set_mean_exp_across_altair.html"))

        # plot violin plot within cell type
        plt.figure(figsize=(8, 8))
        sns.violinplot(
            x=["Marker"]*len(target_scores.tolist()) + ["Other"]*len(complement_scores.tolist()),
            y=target_scores.tolist()+complement_scores.tolist(),
            palette="muted",
            order=["Marker", "Other"]
        )
        plt.title(f"Mean Log2-Normalized Expression for {cell_type} Marker Genes")
        plt.ylabel("Mean Log2-Normalized Expression of Gene Set")
        plt.xlabel("Gene Type")
        plt.tight_layout()
        plt.savefig(os.path.join(out, f"{cell_type}_gene_set_mean_exp_within.png"))
        plt.close()

        # plot altair box plot
        plot_df = pd.DataFrame({
            "Mean Log2-Normalized Expression of Gene Set": target_scores.tolist()+complement_scores.tolist(),
            "Type": ["Marker"]*len(target_scores.tolist()) + ["Other"]*len(complement_scores.tolist())
        })
        box_plot = alt.Chart(plot_df).mark_boxplot().encode(
            y='Mean Log2-Normalized Expression of Gene Set:Q',
            x=alt.X('Type:N', sort=["Marker", "Other"], axis=alt.Axis(labelAngle=-45)),
            color='Type:N'
        ).properties(
            title=f"Mean Log2-Normalized Expression for {cell_type} Marker Genes",
        ).interactive()
        box_plot.save(os.path.join(out, f"{cell_type}_gene_set_mean_exp_within_altair.html"))
    # add diagnal line for the AUC plot
    auc_df_final = pd.concat([auc_df_final, pd.DataFrame({"cell_type": ["Diagonal"]*2, "fpr": [0, 1], "tpr": [0, 1]})], ignore_index=True)
    # plot final AUC with altair
    auc_plot = alt.Chart(auc_df_final).mark_line().encode(
        x='fpr:Q',
        y='tpr:Q',
        color='cell_type:N'
    ).properties(
        title=f"ROC Curve for Marker Genes Across Cell Types",
    ).interactive()
    auc_plot.save(os.path.join(out, f"all_cell_types_marker_genes_roc_curve_altair.html"))

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(out, "gene_set_enrichment.csv"))

    # plot summary boxplot of marker expression with altair
    if len(enriched_gene_sets.keys()) < 2:
        print("Not enough cell types for summary plot.")
    else:
        marker_in_cell_concat = np.concatenate(marker_in_cell).ravel()
        marker_across_cell_concat = np.concatenate(marker_across_cell).ravel()    
        summary_melted = pd.DataFrame({
            "mean_z_normalized_expression": np.concatenate([marker_in_cell_concat, marker_across_cell_concat]),
            "context": ["within_cell_type"]*len(marker_in_cell_concat) + ["across_cell_type"]*len(marker_across_cell_concat)
        })
        box_plot = alt.Chart(summary_melted).mark_boxplot().encode(
            y='mean_z_normalized_expression:Q',
            x=alt.X('context:N', sort=["within_cell_type", "across_cell_type"], axis=alt.Axis(labelAngle=-45)),
            color='context:N'
        ).properties(
            title="Mean Z-Normalized Expression of Marker Genes",
        ).interactive()
        box_plot.save(os.path.join(out, f"summary_marker_expression_boxplot.html"))
    
        # stats test for summary plot
        p_value_summary_permutation = permutation_test(marker_in_cell_concat, marker_across_cell_concat)
        stats, p_value_summary_mannwhitney = mannwhitneyu(marker_in_cell_concat, marker_across_cell_concat)
        print("Permutation test p-value for summary marker expression:", p_value_summary_permutation)
        print("Mann-Whitney U test p-value for summary marker expression:", p_value_summary_mannwhitney)
        with open(os.path.join(out, "summary_marker_expression_stats.txt"), "w") as f:
            f.write("Summary statistics for marker gene expression comparison:\n")
            f.write("Mean for in cell type: {:.4f}\n".format(marker_in_cell_concat.mean()))
            f.write("Mean for across cell type: {:.4f}\n".format(marker_across_cell_concat.mean()))
            f.write(f"Permutation test p-value: {p_value_summary_permutation:.4e}\n")
            f.write(f"Mann-Whitney U test p-value: {p_value_summary_mannwhitney:.4e}\n")
            f.write(f"Mann-Whitney U test statistic: {stats:.4f}\n")

    return results_df

def main():
    np.random.seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--expression_npy', type=str, nargs="+", help='Path(s) to expression numpy matrix')
    parser.add_argument('--up_gene_sets', type=str, nargs="+", help='Path(s) to the .gmt file containg the up-regulated gene sets')
    parser.add_argument('--gene_names', type=str, help='Path to the numpy file containg the names of the genes')
    parser.add_argument('--cell_names', type=str, nargs="+", help='Path to the numpy file containg the names of the cells')
    parser.add_argument('--gene_template', type=str, default=None, help='Optional, path to the features.tsv file for converting gene names to gene symbols')
    parser.add_argument('--output', type=str, help='Path to the output directory')
    parser.add_argument('--cell_types', type=str, nargs="+", help="Cell types that you want to compare")
    parser.add_argument('--ignore_cell_types', type=str, nargs="+", default=None, help="Cell types that you want to ignore")
    args = parser.parse_args()
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    exp_L = []
    for each in args.expression_npy:
        exp_L.append(np.load(each))
    expression_matrix = np.concatenate(exp_L)
    gene_names = np.load(args.gene_names)
    cell_labels = []
    for each in args.cell_names:
        cell_labels.extend([cell.lower() for cell in np.load(each).tolist()])
    celltype_of_interest = [cell.lower() for cell in args.cell_types]
    signature = read_tsv(args.up_gene_sets)

    # remove those that are ignored
    if args.ignore_cell_types:
        for cell_type in args.ignore_cell_types:
            # find the location in cell_labels
            loc = [i for i, label in enumerate(cell_labels) if label == cell_type]
            # drop the corresponding rows in the expression matrix
            expression_matrix = np.delete(expression_matrix, loc, axis=0)
            # drop from cell_labels
            cell_labels = [label for i, label in enumerate(cell_labels) if i not in loc]

    if args.gene_template:
        # convert the signature from gene names to gene symbols
        template = pd.read_csv(args.gene_template, sep='\t', header=None)
        name_to_symbol = {row[1]: row[0] for _, row in template.iterrows()}  # map from gene name to gene symbol
        # convert the signature to gene symbols by mapping the dataframe
        signature = signature.map(lambda x: name_to_symbol.get(x, x))  # map gene names to symbols

    # select only the cell type of interest
    enriched_gene_sets = {
        celltype: sorted(set(chain.from_iterable(
                    signature[celltype].dropna().tolist()
                )))
                    for celltype in celltype_of_interest if celltype in signature.columns
    }
    # enriched_gene_sets = {
    #     celltype: list(reduce(set.intersection, (set(col) for _, col in signature.loc[:, celltype].dropna().items()))) 
    #                 for celltype in celltype_of_interest if celltype in signature.columns
    # }
    for key, value in enriched_gene_sets.items():
        print(f"Number of marker genes of {key}:", len(value))
    
    # Validate enrichment
    results_df = validate_enrichment(expression_matrix, cell_labels, gene_names, enriched_gene_sets, args.output)

if __name__ == "__main__":
    main()