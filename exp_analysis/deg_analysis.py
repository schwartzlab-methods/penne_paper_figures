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
from cmapPy.pandasGEXpress.parse_gct import parse

import altair as alt
## plotting settings
if True:  # In order to bypass isort when saving
    from altairThemes import altairThemes
alt.themes.register("publishTheme", altairThemes.publishTheme)
alt.themes.enable("publishTheme")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Differential Gene Expression Analysis")
    parser.add_argument('--counts', type=str, required=True, nargs="+", help='Path to the counts matrix npy file')
    parser.add_argument('--labels', type=str, required=True, nargs="+", help='Path to the labels npy file')
    parser.add_argument('--genes', type=str, required=True, help='Path to the gene names npy file')
    parser.add_argument('--cell_types', type=str, required=True, nargs="+", 
                        help='The two cell types to compare. Use to select from the dataset.')
    parser.add_argument('--comparison', type=str, nargs="+", default=None,
                        help='The comparison type (e.g. "control" or "treatment"). If none, will use cell_types')
    parser.add_argument('--output', type=str, required=True, help='Output directory for results')
    parser.add_argument('--gt', type=str, default=None, help='Path to the ground truth RNA seq .gct file (optional)')
    parser.add_argument('--gene_template', type=str, default=None, help='Path to the tsv file for mapping gene symbols tp gene names (optinal)')
    args = parser.parse_args()
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    if args.comparison:
        # Use the comparison argument to filter the dataset
        # sort alphabetically by comparison, but ensure that the relative order does not change
        dict_cell_type_to_comp_group = {args.cell_types[0]: args.comparison[0], args.cell_types[1]: args.comparison[1]}
        comparison = sorted(args.comparison)
    else:
        comparison = sorted(args.cell_types)
    # Load count data (rows: cells, columns: gemes)
    count_L = []
    for each in args.counts:
        count_L.append(np.load(each).astype(float))
    counts = np.concatenate(count_L)
    # counts = np.load(args.counts).astype(float)
    cell_L = []
    for each in args.labels:
        cell_L.append(np.load(each))
    cell_types = np.concatenate(cell_L).ravel()
    # cell_types = np.load(args.labels)
    gene_names = np.load(args.genes)
    assert counts.shape[0] == cell_types.shape[0], f"Counts and labels must have the same number of cells. Got {counts.shape[0]} and {cell_types.shape[0]}"
    assert counts.shape[1] == gene_names.shape[0], "Counts and gene names must match in number of genes"
    assert len(args.cell_types) == 2, "Please provide exactly two cell types for comparison"
    for each in args.cell_types:
        assert each in cell_types, f"Cell type {each} not found in labels"
    
    if args.gt is not None:
        # compute the ground truth gene expression
        gt_data = parse(args.gt).data_df
        gt_data.index = [x.split(".")[0] for x in gt_data.index]  # remove the version number from gene names
        gt_data.columns = [x.split("_")[0].lower() for x in gt_data.columns]  # get cell names

        if args.gene_template:
            # convert the signature from gene names to gene symbols
            template = pd.read_csv(args.gene_template, sep='\t', header=None)
            name_to_symbol = {row[0]: row[1] for _, row in template.iterrows()}  # map from gene symbols to gene names
            # convert the signature to gene symbols by mapping the dataframe
            gt_data.index = [name_to_symbol[name] if name in name_to_symbol.keys() else name for name in gt_data.index]  # map gene names to symbols

        # check if the ground truth data has the two cell types
        if not all(cell.lower() in gt_data.columns for cell in comparison):
            print(f"Ground truth data does not contain the specified comparison: {comparison}.")
            print("Will proceed without ground truth data.")
        else:
            print(f"Process GT data for {comparison}.")
            # process the ground truth data
            # filter to only have the two cell types
            gt_data = gt_data.loc[:, [cell.lower() for cell in comparison]]
            # select only the genes that are in the intersection of gt and gene_names
            common_genes = list(set(gt_data.index).intersection(set(gene_names)))
            # filter the gene names
            # find duplicated genes
            duplicated_genes = gt_data.index[gt_data.index.duplicated()].tolist()
            gene_names = [gene if ((gene in common_genes) and (gene not in duplicated_genes)) else None for gene in gene_names]
            order = [gene for gene in gene_names if gene in common_genes]
            gt_data = gt_data.loc[order, :]  # filter to only have the common genes
            # sort the ground truth data by gene names
            gt_data = gt_data.reindex(order)
            # normalize to counts per million and log2 transform
            gt_data = gt_data / np.sum(gt_data, axis=0) * 1e6
            gt_data = np.log2(gt_data + 1)
            l2fc_gt = (gt_data[args.cell_types[1].lower()] - gt_data[args.cell_types[0].lower()]).to_numpy()
            print("GT processed, running DGE on data")
    
    # get only the cell types of interest from counts
    cell_type_mask = np.isin(cell_types, args.cell_types)
    counts = counts[cell_type_mask]
    cell_types = cell_types[cell_type_mask].astype(str)
    print(f"Analyzing {cell_types.shape[0]} cells")

    # Create a DataFrame for the counts
    counts_df = pd.DataFrame(counts, columns=gene_names)

    # do differential expression analysis
    results = []
    new_gene_names = []
    l2fc_gt_L = []
    i = 0
    for gene in tqdm(gene_names):
        if gene is None:
            continue
        # Create a DataFrame for the current gene
        gene_df = counts_df[[gene]].copy()
        if (0 == gene_df.to_numpy()).all():
            i += 1
            continue
        new_gene_names.append(gene)
        if args.gt:
            l2fc_gt_L.append(l2fc_gt[i])
        i += 1
        if args.comparison:
            gene_df['cell_type'] = [dict_cell_type_to_comp_group[cell_type] for cell_type in cell_types]
        else:
            gene_df['cell_type'] = cell_types

        # Create the design matrix
        X = sm.add_constant(pd.get_dummies(gene_df['cell_type'], drop_first=True)).astype(float)
        y = gene_df[gene].astype(float)
        # combine all the genes in y if there are multiple with the same name
        if len(y.shape) != 1:
            y = y.mean(axis=1)
        # Fit a linear model
        model = sm.OLS(y, X, missing="drop")
        results.append(model.fit(cov_type='HC3'))

    # Extract p-values and coefficients
    p_values = [result.pvalues.iloc[1] for result in results]
    coefficients = [result.params.iloc[1] for result in results]
    t_stats = [result.tvalues.iloc[1] for result in results]
    print("Final number of genes analyzed: ", len(new_gene_names))
    if len(new_gene_names) == 0:
        print("No genes with sufficient expressions found. Program ends.")
        return 0
    # Create a DataFrame for the results
    if args.gt is not None:
        # add the ground truth log2 fold change to the results
        results_df = pd.DataFrame({
            'gene': new_gene_names,
            'p_value': p_values,
            'log_fc': coefficients,
            'log_fc_gt': l2fc_gt_L,
            't_stat': t_stats,
        })
    else:
        results_df = pd.DataFrame({
            'gene': new_gene_names,
            'p_value': p_values,
            'log_fc': coefficients,
            't_stat': t_stats
        })
    # Adjust p-values for multiple testing
    results_df['adj_p_value'] = sm.stats.multipletests(results_df['p_value'], method='fdr_bh')[1]
    results_df["neg_log10_adj_p_value"] = -np.log10(results_df['adj_p_value'])
    results_df['adj_p_005_log2fc_1'] = results_df.apply(lambda x: (x["adj_p_value"] < 0.05) and (np.absolute(x["log_fc"]) > 1), axis=1).astype(str)
    if args.gt is not None:
        results_df['correct_dir'] = results_df.apply(lambda x: "Same_significant" if (np.sign(x['log_fc']) == np.sign(x['log_fc_gt'])) and x["adj_p_value"] < 0.05
                                                     else "Different_significant" if (np.sign(x['log_fc']) != np.sign(x['log_fc_gt'])) and x["adj_p_value"] < 0.05 
                                                     else "Same_nonsignificant" if (np.sign(x['log_fc']) == np.sign(x['log_fc_gt']))
                                                     else "Different_nonsignificant", axis=1)

    # Save results
    results_df.to_csv(os.path.join(args.output, f'deg_ref_{comparison[0]}vs{comparison[1]}.csv'), index=False)
    # Plot results
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=results_df, x='log_fc', y='neg_log10_adj_p_value', 
                    hue='adj_p_005_log2fc_1', hue_order=["True", "False"])
    plt.axhline(y=-np.log10(0.05), color='r', linestyle='--')
    plt.axvline(x=1, color='r', linestyle='--')
    plt.axvline(x=-1, color='r', linestyle='--')
    plt.title(f'Differential Gene Expression Analysis: {comparison[0]}(Reference) vs {comparison[1]}')
    plt.xlabel('Log2 FC')
    plt.ylabel('-log10(FDR Adjusted p-value)')
    plt.savefig(os.path.join(args.output, f'deg_plot_ref_{comparison[0]}vs{comparison[1]}.png'))
    plt.close()

    # plot with Altair
    deg_chart = alt.Chart(results_df).mark_circle().encode(
        x=alt.X('log_fc:Q', title='Log2 FC'),
        y=alt.Y('neg_log10_adj_p_value:Q', title='-log10(FDR Adjusted p-value)'),
        color=alt.Color('adj_p_005_log2fc_1', scale=alt.Scale(domain=["True", "False"], range=['red', 'blue']), 
                        legend=alt.Legend(title='FDR<0.05 & |Log2FC|>1')),
        tooltip=['gene', 'log_fc', 'adj_p_value']
    ).properties(
        title=f'Differential Gene Expression Analysis: {comparison[0]}(Reference) vs {comparison[1]}'
    ).add_selection(
        alt.selection_single(fields=['gene'], on='click')
    ).interactive()
    # add two lines for p-value and log2fc thresholds
    deg_chart += alt.Chart(pd.DataFrame({'y': [-np.log10(0.05)]})).mark_rule(color='black', strokeDash=[5,5]).encode(y='y:Q')
    deg_chart += alt.Chart(pd.DataFrame({'x': [1]})).mark_rule(color='black', strokeDash=[5,5]).encode(x='x:Q')
    deg_chart += alt.Chart(pd.DataFrame({'x': [-1]})).mark_rule(color='black', strokeDash=[5,5]).encode(x='x:Q')
    deg_chart.save(os.path.join(args.output, f'deg_plot_ref_{comparison[0]}vs{comparison[1]}.html'))
    # get a list of differentially expressed genes
    deg_genes_up = results_df[results_df['adj_p_value'] < 0.05]
    deg_genes_up = deg_genes_up[deg_genes_up['log_fc'] > 1]['gene'].tolist()
    deg_genes_down = results_df[results_df['adj_p_value'] < 0.05]
    deg_genes_down = deg_genes_down[deg_genes_down['log_fc'] < -1]['gene'].tolist()

    # save the list of differentially expressed genes
    with open(os.path.join(args.output, f'deg_genes_ref_{comparison[0]}vs{comparison[1]}_up.txt'), 'w') as f:
        for gene in deg_genes_up:
            f.write(f"{gene}\n")
    with open(os.path.join(args.output, f'deg_genes_ref_{comparison[0]}vs{comparison[1]}_down.txt'), 'w') as f:
        for gene in deg_genes_down:
            f.write(f"{gene}\n")
    print(f"DEG analysis completed. Results saved to {args.output}")
    if args.gt is not None:
        # create a bar plot of the correct direction of the log2 fold change
        plt.figure(figsize=(10, 6))
        sns.countplot(data=results_df, x='correct_dir', order=["Same_significant", "Same_nonsignificant", "Different_significant", "Different_nonsignificant"])
        plt.title(f'Correct Direction of Log2 FC: {comparison[0]} vs {comparison[1]}')
        plt.xlabel('Correct Direction')
        plt.ylabel('Count')
        plt.savefig(os.path.join(args.output, f'correct_direction_ref_{comparison[0]}vs{comparison[1]}.png'))
        plt.close()

if __name__ == "__main__":
    main()
