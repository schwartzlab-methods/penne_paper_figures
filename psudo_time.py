'''
Pseudo-time analysis for confluency data using py-Monocle
'''

from py_monocle import pseudotime
import numpy as np
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Pseudo-time analysis for confluency data")
    parser.add_argument('--input', type=str, help='Path to input numpy array file of the exp matrix')
    parser.add_argument('--labels', type=str, help='Path to input numpy array file of the confluency labels')
    parser.add_argument('--output', type=str, help='Path to output directory for results')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("Loading and processing data...")
    # load data
    array = np.load(args.input)
    labels = np.load(args.labels)

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

    pseudotime_array = pseudotime(matrix=cells, root_cells=day_0_label, clusters=clusters)

    # order by pseudotime, plot a line plot of the cells coloured by the labels
    order = np.argsort(pseudotime_array)
    ordered_labels = labels[order]
    ordered_pseudotime = pseudotime_array[order]
    # get colour map
    unique_labels = np.unique(labels)
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
    plt.savefig(os.path.join(args.output, "pseudotime_confluency.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # show enrichment of apical markers along pseudotime

if __name__ == "__main__":
    main()