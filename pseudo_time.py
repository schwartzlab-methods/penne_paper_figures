'''
Pseudo-time analysis for confluency data using py-Monocle
'''

from py_monocle import pseudotime
import numpy as np
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
import altair as alt
## plotting settings
if True:  # In order to bypass isort when saving
    from altairThemes import altairThemes
alt.themes.register("publishTheme", altairThemes.publishTheme)
alt.themes.enable("publishTheme")

def wilcoxon_rank_sum_test(group1, group2):
    '''
    Perform Wilcoxon rank-sum test between two groups.
    '''
    from scipy.stats import ranksums
    stat, p_value = ranksums(group1, group2)
    return stat, p_value

def pseudotime_analysis_and_plot(array, labels, output_dir):
    '''
    Run pseudotime analysis and plot the results.
    '''
    # prep for py-Monocle
    adata = ad.AnnData(array)
    # Louvain clustering
    sc.pp.neighbors(adata, n_neighbors=15, use_rep='X')
    sc.tl.louvain(adata, key_added='louvain')
    # UMAP
    sc.tl.umap(adata, min_dist=0.5)

    # pseudotime analysis using py-Monocle
    print("Running pseudotime analysis...")
    cells = adata.obsm['X_umap']
    clusters = adata.obs['louvain'].to_numpy().astype(int)
    day_0_label = np.where(labels == "day0")[0][0]

    # pseudotime computation, output is a 1D array of pseudotime values for each cell
    pseudotime_array = pseudotime(matrix=cells, root_cells=day_0_label, clusters=clusters)

    # compute the mean pseudotime for each label
    unique_labels = np.unique(labels)
    mean_pseudotime = {label: np.mean(pseudotime_array[labels == label]) for label in unique_labels}
    df = pd.DataFrame(list(mean_pseudotime.items()), columns=['Label', 'Mean_Pseudotime'])
    df.to_csv(os.path.join(output_dir, "mean_pseudotime.csv"), index=False)

    # order by pseudotime, plot a line plot of the cells coloured by the labels
    order = np.argsort(pseudotime_array)
    ordered_labels = labels[order]
    ordered_pseudotime = pseudotime_array[order]

    # get colour map
    cmap = plt.get_cmap('viridis', len(unique_labels))
    label_to_color = {label: cmap(i) for i, label in enumerate(unique_labels)}
    ordered_colors = [label_to_color[label] for label in ordered_labels]
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(range(len(ordered_pseudotime)), ordered_pseudotime, c=ordered_colors, s=5)
    # create a legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=label_to_color[label], markersize=5) for label in unique_labels]
    plt.legend(handles, unique_labels, title="Experiment Types", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel('Cells ordered by pseudotime')
    plt.ylabel('Pseudotime')
    plt.title('Pseudotime Analysis of Confluency Data')
    plt.savefig(os.path.join(output_dir, "pseudotime_confluency.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # plot an altair tick plot, where only along x is pseudotime, y is just a constant
    df_altair = pd.DataFrame({'Cell_Index': range(len(ordered_pseudotime)),
                              'Pseudotime': ordered_pseudotime,
                              'Label': ordered_labels})
    chart = alt.Chart(df_altair).mark_tick(size=10).encode(
        x='Pseudotime',
        y=alt.value(1),
        color=alt.Color('Label', scale=alt.Scale(scheme='viridis'))
    ).properties(
        title='Pseudotime',
    ).interactive()
    chart.save(os.path.join(output_dir, 'pseudotime_confluency_altair.html'))

    chart = alt.Chart(df_altair).mark_boxplot().encode(
        x=alt.X('Label', sort=np.unique(labels).tolist(), axis=alt.Axis(labelAngle=-45)),
        y='Pseudotime',
        color='Label'
    ).properties(
        title='Pseudotime Distribution by Label'
    ).interactive()
    chart.save(os.path.join(output_dir, 'pseudotime_boxplot.html'))

    # perform Wilcoxon rank-sum test between each pair of labels for pseudotime
    with open(os.path.join(output_dir, 'pseudotime_wilcoxon.txt'), 'w') as f:
        f.write("Wilcoxon Rank-Sum Test Results for Pseudotime:\n")
        for i in range(len(unique_labels)):
            for j in range(i + 1, len(unique_labels)):
                group1 = pseudotime_array[labels == unique_labels[i]]
                group2 = pseudotime_array[labels == unique_labels[j]]
                stat, p_val = wilcoxon_rank_sum_test(group1, group2)
                f.write(f"{unique_labels[i]} vs {unique_labels[j]}: statistic={stat:.4f}, p-value={p_val:.4e}\n")

    print("Pseudotime analysis completed and plot saved.")

    return pseudotime_array

def compute_marker_gene_scores(array, labels, pseudotime_array, gene_set, gene_names, output_dir, gene_set_name):
    '''
    Compute marker gene scores and plot against pseudotime.
    '''
    gene_indices = [i for i, gene in enumerate(gene_names) if gene in gene_set]
    if not gene_indices:
        print("No genes from the gene set found in the data.")
        return

    marker_scores = array[:, gene_indices].mean(axis=1)

    # colour by label
    colours = plt.get_cmap('viridis', len(np.unique(labels)))
    label_to_colour = {lab: colours(i) for i, lab in enumerate(np.unique(labels))}
    marker_colours = [label_to_colour[lab] for lab in labels]

    # create a linear regression and report R^2 and p-value
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(pseudotime_array, marker_scores)
    if p_value < 2.2e-16:
        p_value = 0.0
    print(f"Linear regression results for {gene_set_name} marker scores vs pseudotime:")
    print(f"R-squared: {r_value**2:.4f}, p-value: {p_value:.4e}")

    # plot marker scores against pseudotime
    plt.figure(figsize=(10, 6))
    plt.scatter(pseudotime_array, marker_scores, c=marker_colours, s=5)
    # plot regression line
    x_vals = np.array(plt.gca().get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, color='red', linestyle='--', label='Linear Regression')
    # annotate R^2 and p-value
    plt.text(0.05, 0.95, f'R² = {r_value**2:.4f}\np = {p_value:.4e}', transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    plt.xlabel('Pseudotime')
    plt.ylabel(f'{gene_set_name} Marker Gene Score')
    plt.title(f'{gene_set_name} Marker Gene Score vs Pseudotime')
    # create a legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=label_to_colour[label], markersize=5) for label in np.unique(labels)]
    plt.legend(handles, np.unique(labels), title="Day", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(os.path.join(output_dir, f"marker_score_vs_pseudotime_{gene_set_name}.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # save marker scores
    pd.DataFrame({'Pseudotime': pseudotime_array, 'Marker_Score': marker_scores}).to_csv(
        os.path.join(output_dir, f"marker_scores_{gene_set_name}.csv"), index=False)
    
    # plot a altair scatter plot
    df_altair = pd.DataFrame({'Pseudotime': pseudotime_array,
                              'Marker_Score': marker_scores,
                              'Label': labels})
    chart = alt.Chart(df_altair).mark_circle().encode(
        x='Pseudotime',
        y='Marker_Score',
        color=alt.Color('Label', scale=alt.Scale(scheme='viridis'))
    ).properties(
        title=f'{gene_set_name} Marker Gene Score vs Pseudotime',
    ).interactive()
    # fit a regression line
    regression = chart.transform_regression('Pseudotime', 'Marker_Score').mark_line(color='red').interactive()
    chart = chart + regression
    chart = chart.interactive()
    chart.save(os.path.join(output_dir, f'marker_score_vs_pseudotime_{gene_set_name}_altair.html'))

    # plot a box plot of average marker score per label and avg pseudotime per label
    df_box = pd.DataFrame({'Label': labels,
                           'Marker_Score': marker_scores,
                           'Pseudotime': pseudotime_array})
    chart = alt.Chart(df_box).mark_boxplot().encode(
        x=alt.X('Label', sort=np.unique(labels).tolist(), axis=alt.Axis(labelAngle=-45)),
        y='Marker_Score',
        color='Label'
    ).properties(
        title=f'{gene_set_name} Marker Gene Score Distribution by Label'
    ).interactive()
    chart.save(os.path.join(output_dir, f'marker_score_boxplot_{gene_set_name}.html'))

    # perform Wilcoxon rank-sum test between each pair of labels for marker scores
    unique_labels = np.unique(labels)
    with open(os.path.join(output_dir, f'marker_score_wilcoxon_{gene_set_name}.txt'), 'w') as f:
        f.write("Wilcoxon Rank-Sum Test Results for Marker Gene Scores:\n")
        for i in range(len(unique_labels)):
            for j in range(i + 1, len(unique_labels)):
                group1 = marker_scores[labels == unique_labels[i]]
                group2 = marker_scores[labels == unique_labels[j]]
                stat, p_val = wilcoxon_rank_sum_test(group1, group2)
                f.write(f"{unique_labels[i]} vs {unique_labels[j]}: statistic={stat:.4f}, p-value={p_val:.4e}\n")


    
    print("Marker gene scores computed and plot saved.")

def main():
    parser = argparse.ArgumentParser(description="Pseudo-time analysis for confluency data")
    parser.add_argument('--input', type=str, help='Path to input numpy array file of the exp matrix')
    parser.add_argument('--labels', type=str, help='Path to input numpy array file of the confluency labels')
    parser.add_argument('--genes', type=str, default=None, help='Path to input numpy array file of the gene names')
    parser.add_argument('--gene_set', type=str, default=None, help='Path to input gmt file of gene set of interest')
    parser.add_argument('--output', type=str, help='Path to output directory for results')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("Loading and processing data...")
    # load data
    array = np.load(args.input)
    labels = np.load(args.labels)

    # pseudotime analysis
    pseudotime_array = pseudotime_analysis_and_plot(array, labels, args.output)
    
    # compute marker gene score if gene set is provided
    if args.gene_set and args.genes:
        with open(args.gene_set, 'r') as f:
            file_L = f.readline().strip().split('\t')
        gene_set = set(file_L[2:])  # skip the first two entry which is the gene set name and description
        gene_set_name = file_L[0]  # get the gene set name
        print(f"Loaded gene set: {gene_set_name} with {len(gene_set)} genes.")
        gene_names = np.load(args.genes)
        compute_marker_gene_scores(array, labels, pseudotime_array, gene_set, gene_names, args.output, gene_set_name)


if __name__ == "__main__":
    main()