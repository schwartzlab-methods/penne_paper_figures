'''
Visualize the gating vector from gated MLP for a set of genes
'''

from dataset import ShaneSeqCellTypeDataset
from model import GeneExpPredVisiumHD
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel
from _feature_extractors import init_spaghetti, pre_processing_phikon
import os
import numpy as np
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list, fcluster
import altair as alt

## plotting settings
if True:  # In order to bypass isort when saving
    from altairThemes import altairThemes
alt.themes.register("publishTheme", altairThemes.publishTheme)
alt.themes.enable("publishTheme")

def main():
    parser = argparse.ArgumentParser(description="Validate GFP levels")
    parser.add_argument("--input_file", type=str, required=True, 
                        help="Path to the input directory with GFP and PCM images")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model")
    parser.add_argument("--spaghetti_model", type=str, required=True,
                        help="Path to the spaghetti model")
    parser.add_argument("--gene_names", type=str, required=True,
                        help="Path to the gene names file")
    parser.add_argument("--genes_to_use", type=str, nargs="+", default=None,
                        help="Path to the file with genes to use, or the name of the gene. If no supplied, use all genes")  
    parser.add_argument("--plot_per_gene", action="store_true",
                        help="Whether to plot per-gene gate correlation heatmaps. If set, will plot gene x gate correlation matrix.")
    parser.add_argument("--max_num_clusters", type=int, default=6, help="Maximum number of clusters for gating vectors")
    parser.add_argument("--scramble", action="store_true", help="Whether to scramble the input images for testing as a baseline.")               
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # prep data
    dataset = ShaneSeqCellTypeDataset(args.input_file)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    extractor = AutoModel.from_pretrained("owkin/phikon-v2").eval()
    image_processor = pre_processing_phikon()
    feature_extractor = (image_processor, extractor)
    converter = init_spaghetti(args.spaghetti_model)

    if args.gene_names.endswith(".tsv.gz"):
        genes = np.loadtxt(args.gene_names, dtype=str, delimiter='\t')
        gene_names = genes[:,1].reshape(-1)
        gene_symbols = genes[:,0].reshape(-1)
        np.save(os.path.join(args.output_dir, f"gene_symbols.npy"), gene_symbols)
    else: # assume txt file
        with open(args.gene_names, 'r') as f:
            gene_names = [line.strip() for line in f.readlines() if "Unnamed: 0" not in line]
        gene_names = np.array(gene_names).reshape(-1)
        np.save(os.path.join(args.output_dir, f"gene_names.npy"), gene_names)
    print("Gene symbols saved to ", args.output_dir)
    num_genes = gene_names.shape[0]

    # load genes to use
    if args.genes_to_use:
        if args.genes_to_use[0].endswith(".npy"):
            genes_to_use = np.load(args.genes_to_use[0], allow_pickle=True)
        elif args.genes_to_use[0].endswith(".txt") or args.genes_to_use[0].endswith(".tsv"):
            with open(args.genes_to_use[0], 'r') as f:
                genes_to_use = [line.strip() for line in f.readlines() if "Unnamed: 0" not in line]
            genes_to_use = np.array(genes_to_use).reshape(-1)
        else: # assume it's a gene name or a list of gene names
            genes_to_use = np.array(args.genes_to_use)
        print(f"Using {len(genes_to_use)} genes from {args.genes_to_use}")
        gene_indices = []
        for gene in genes_to_use:
            if gene in gene_names:
                gene_indices.append(np.where(gene_names==gene)[0][0])
            else:
                print(f"Gene {gene} not found in gene list. Skipping.")
        gene_indices = np.array(gene_indices)
        print(f"Using {len(gene_indices)} genes after filtering")
    else:
        gene_indices = np.arange(num_genes)
        genes_to_use = gene_names
        print("Using all genes: ", num_genes)

    # prep model
    model = GeneExpPredVisiumHD.load_from_checkpoint(args.model_path, num_genes = num_genes, 
                                converter = converter, feature_extractor = feature_extractor)
    model.freeze()
    model.eval()

    # inference
    pred_L = []
    pred_mean_L = []
    gates = []
    with torch.no_grad():
        for img, label in tqdm(loader):
            img = img.to(model.device)
            img = img.squeeze(0) #remove the default batch dimension
            pred = model(img, if_convert=True, scramble=args.scramble) #shape: num_patch, num_genes
            # get only the genes we want
            pred = pred[:, gene_indices] # shape: num_patch, num_genes_used
            pred_L.append(pred.cpu().numpy())
            mean_exp = torch.mean(pred, dim=1) # shape: num_patch
            pred_mean_L.append(mean_exp.cpu().numpy())
            # compute gate value
            gate = model.compute_gate(img, if_convert=True) # shape: (n_layers=3, num_patches, dim_in=960)
            gates.append(gate.permute(1,0,2).cpu().numpy()) # shape: num_patches, n_layers=3, dim_in=960
    pred_L = np.concatenate(pred_L, axis=0) # shape: num_samples, num_genes_used
    pred_mean_L = np.concatenate(pred_mean_L, axis=0) # shape: num_samples
    gates =  np.concatenate(gates, axis=0) # shape: num_samples, n_layers=3, dim_in=960

    # generate a heatmap of the gate values, where x-axis is the layer index, y-axis is the gate dimension, 
    # also cluster using hierarchical clustering
    # color is the correlation between the gate value and the predicted expression
    num_layers = gates.shape[1]
    gate_dim = gates.shape[2]
    num_genes_used = len(gene_indices)
    if not args.plot_per_gene:
        print("Generating gate correlation heatmap using avg predicted expression...")
        heatmap = np.zeros((num_layers, gate_dim))
        flatten_pred_L = pred_mean_L.reshape(-1) # shape: num_samples
        for layer_idx in tqdm(range(num_layers)):
            for dim_idx in range(gate_dim):
                gate_values = gates[:, layer_idx, dim_idx].reshape(-1) # shape: num_samples
                if np.std(gate_values) == 0 or np.std(flatten_pred_L) == 0: # constant value, correlation undefined
                    corr = 0
                else:
                    corr, _ = spearmanr(gate_values, flatten_pred_L)
                heatmap[layer_idx, dim_idx] = corr
        
        plt.figure(figsize=(10,6))
        sns.clustermap(heatmap, cmap='bwr', center=0)
        plt.xlabel("Gate Dimension")
        plt.ylabel("Layer Index")
        plt.title("Spearman Correlation between Gate Values and Predicted Expression")
        plt.savefig(os.path.join(args.output_dir, "gate_expression_correlation_heatmap.png"))
        plt.close()
        print("Gate correlation heatmap saved to ", args.output_dir)
    else:
        # generate a gene x gene matrix of correlation values, then cluster genes based on gate correlation profiles
        print("Generating gene x gene correlation matrix...")
        gene_gate_corr = np.zeros((num_genes_used, gate_dim))
        for gene_idx in tqdm(range(num_genes_used)):
            gene_exp = pred_L[:, gene_idx].reshape(-1) # shape: num_samples
            for dim_idx in range(gate_dim):
                gate_values = gates[:, -1, dim_idx].reshape(-1) # use last layer's gate values, shape: num_samples
                if np.std(gate_values) == 0 or np.std(gene_exp) == 0:
                    corr = 0
                else:
                    corr, _ = spearmanr(gate_values, gene_exp)
                gene_gate_corr[gene_idx, dim_idx] = corr
        
        all_cols = np.array([i for i in range(gate_dim)])
        # Filter out nans
        valid_rows = ~np.isnan(gene_gate_corr).any(axis=1)
        gene_gate_corr = gene_gate_corr[valid_rows, :]
        gene_names = genes_to_use[valid_rows]
        valid_cols = ~np.isnan(gene_gate_corr).any(axis=0)
        gene_gate_corr = gene_gate_corr[:, valid_cols]
        all_cols = all_cols[valid_cols]


        # cluster
        dendro = linkage(gene_gate_corr, method='ward')
        # sort the datafram by the order of the dendrogram, where the rows (genes) are reordered according to the hierarchical clustering
        leaf_indices = leaves_list(dendro)
        row_sorted_data = gene_gate_corr[leaf_indices, :]
        row_sorted_genes = gene_names[leaf_indices]
        
        dendro_col = linkage(gene_gate_corr.T, method='ward')
        col_leaf_indices = leaves_list(dendro_col)
        col_sorted_data = row_sorted_data[:, col_leaf_indices]
        col_sorted_cols = all_cols[col_leaf_indices]

        # generate heatmap
        heatmap_df_nonmelt = pd.DataFrame(col_sorted_data, index=row_sorted_genes, columns=col_sorted_cols)
        heatmap_df_nonmelt.to_csv(os.path.join(args.output_dir, "gene_gate_correlation_matrix.csv"))
        heatmap_df = heatmap_df_nonmelt.reset_index().melt(id_vars='index', var_name='Gate Dimension', value_name='Correlation')
        heatmap_df.rename(columns={"index": "Gene"}, inplace=True)

        # plot with alair
        heatmap_chat = alt.Chart(heatmap_df).mark_rect().encode(
            x=alt.X('Gate Dimension:O', title='Gate Dimension', sort=col_sorted_cols.tolist()),
            y=alt.Y('Gene:N', title='Gene', sort=row_sorted_genes.tolist()),
            color=alt.Color('Correlation:Q', title='Correlation', scale=alt.Scale(scheme='redblue', domainMid=0))
        )
        heatmap_chat.save(os.path.join(args.output_dir, "gene_gate_correlation_heatmap.html"))

        # plot dendrogram
        plt.figure(figsize=(60, 60))
        dendrogram(dendro, labels=gene_names, orientation='right', truncate_mode='lastp', p=3)
        # remove figure axes and save
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "gene_clustering_dendrogram.svg"))
        plt.close()
    
        # cluster genes based on gate correlation profiles
        plt.figure(figsize=(50,35))
        # make text font size smaller for better visualization
        sns.set(font_scale=0.5)
        cg = sns.clustermap(gene_gate_corr, cmap='bwr', center=0, yticklabels=gene_names)
        plt.xlabel("Gate Dimension")
        plt.ylabel("Genes")
        plt.title("Gene x Gate Correlation Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "gene_gate_correlation_heatmap_last_layer.svg"))
        plt.savefig(os.path.join(args.output_dir, "gene_gate_correlation_heatmap_last_layer.png"))
        plt.close()
        print("Gene x Gate correlation heatmap saved to ", args.output_dir)
        # save the linkage matrix for genes
        linkage_cg = cg.dendrogram_row.linkage
        np.save(os.path.join(args.output_dir, "gene_gate_correlation_linkage.npy"), linkage_cg)
        print("Gene linkage matrix saved to ", os.path.join(args.output_dir, "gene_gate_correlation_linkage.npy"))
        # get the clusters based on the linkage distance
        clusters = fcluster(linkage_cg, args.max_num_clusters, criterion='maxclust')
        gene_cluster_df = pd.DataFrame({"Gene": gene_names, "Cluster": clusters})
        gene_cluster_df.to_csv(os.path.join(args.output_dir, "gene_clusters.csv"), index=False)
        print("Gene clusters saved to ", os.path.join(args.output_dir, "gene_clusters.csv"))
        # for each cluster, get the top 50 features with highest average correlation
        top_features = []
        for cluster_id in np.unique(clusters):
            cluster_gene_indices = np.where(clusters==cluster_id)[0]
            cluster_gene_corrs = gene_gate_corr[cluster_gene_indices, :] # shape: num_genes_in_cluster, gate_dim
            avg_corrs = np.mean(cluster_gene_corrs, axis=0) # shape: gate_dim
            top_gate_indices = np.argsort(-np.abs(avg_corrs))[:50] # top 50 features with highest absolute average correlation
            for gate_idx in top_gate_indices:
                top_features.append({
                    "Cluster": cluster_id,
                    "Gate Dimension": gate_idx,
                    "Average Correlation": avg_corrs[gate_idx]
                })
        top_features_df = pd.DataFrame(top_features)
        top_features_df.to_csv(os.path.join(args.output_dir, "top_gate_features_per_cluster.csv"), index=False)
        print("Top gate features per cluster saved to ", os.path.join(args.output_dir, "top_gate_features_per_cluster.csv"))
        # perform enrichment analysis with enrichr for each cluster
        print("Performing enrichment analysis for each gene cluster...")
        import gseapy as gp
        clusters = np.unique(clusters)
        final_enrichment_results = {}
        for cluster_id in tqdm(clusters):
            cluster_genes = gene_cluster_df[gene_cluster_df["Cluster"]==cluster_id]["Gene"].values.tolist()
            if len(cluster_genes) < 2:
                print(f"Cluster {cluster_id} has less than 2 genes. Skipping enrichment analysis.")
                continue
            enr = gp.enrichr(gene_list=cluster_genes,
                             gene_sets='GO_Cellular_Component_2025',
                             outdir=None, # do not save to disk
                             background=genes_to_use.tolist(),
                            )
            final_enrichment_results[cluster_id] = enr.results
            # save the enrichment results to csv
            enr.results.to_csv(os.path.join(args.output_dir, f"gene_cluster_{cluster_id}_enrichment.csv"), index=False)

        # plot top enriched terms for each cluster in one horizontal bar plot figure with Altair
        # y axis: top terms grouped by cluster id, x axis: combined score
        all_terms = []
        for cluster_id, df in final_enrichment_results.items():
            df = df[df["Adjusted P-value"] < 0.25] # filter by adjusted p-value
            # only keep infinity if there are more than two items in Gene
            df = df[df["Genes"].str.split(";").apply(len) >= 2]
            # replace inf with a fixed large number (1.1 times the max finite combined score)
            top_terms = df.sort_values(by="Combined Score", ascending=False)# .head(2)
            top_terms["Cluster"] = cluster_id
            all_terms.append(top_terms)
        all_terms_df = pd.concat(all_terms, axis=0)
        max_finite_score = all_terms_df.loc[np.isfinite(all_terms_df["Combined Score"]), "Combined Score"].max()
        all_terms_df["Combined Score"] = all_terms_df["Combined Score"].replace(np.inf, 1.1 * max_finite_score)
        chart = alt.Chart(all_terms_df).mark_bar().encode(
            x=alt.X('Combined Score:Q', title='Combined Score'),
            y=alt.Y('Term:N', title='', sort=all_terms_df["Cluster"].tolist()), # by cluster
            color=alt.Color('Cluster:N', title='Cluster ID')
        ).interactive()
        # set labellimit to 0 to show full term names
        chart = chart.configure_axisY(labelLimit=0)
        chart.save(os.path.join(args.output_dir, "gene_cluster_enrichment.html"))
        print("Gene cluster enrichment plot saved to ", os.path.join(args.output_dir, "gene_cluster_enrichment.html"))
        

if __name__ == "__main__":
    main()