'''
Visualize the gating vector from gated MLP for a set of genes
'''

from dataset import ShaneSeqCellTypeDataset
import pytorch_lightning as pl
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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
from scipy.stats import spearmanr, fisher_exact
# from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
import umap
import torchvision.utils

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
            pred = model(img, if_convert=True) #shape: num_patch, num_genes
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
        # generate a gene x gate matrix of correlation values, then cluster genes based on gate correlation profiles
        print("Generating gene x gate correlation matrix...")
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

        # cluster genes based on gate correlation profiles
        plt.figure(figsize=(12,10))
        cg = sns.clustermap(gene_gate_corr, cmap='bwr', center=0, yticklabels=genes_to_use)
        plt.xlabel("Gate Dimension")
        plt.ylabel("Genes")
        plt.title("Gene x Gate Correlation Matrix")
        plt.savefig(os.path.join(args.output_dir, "gene_gate_correlation_heatmap_last_layer.svg"))
        plt.close()
        print("Gene x Gate correlation heatmap saved to ", args.output_dir)
        # save the linkage matrix
        linkage = cg.dendrogram_row.linkage
        np.save(os.path.join(args.output_dir, "gene_gate_correlation_linkage.npy"), linkage)
        print("Gene linkage matrix saved to ", os.path.join(args.output_dir, "gene_gate_correlation_linkage.npy"))
        # get the clusters based on the linkage distance
        from scipy.cluster.hierarchy import fcluster
        clusters = fcluster(linkage, args.max_num_clusters, criterion='maxclust')
        gene_cluster_df = pd.DataFrame({"Gene": genes_to_use, "Cluster": clusters})
        gene_cluster_df.to_csv(os.path.join(args.output_dir, "gene_clusters.csv"), index=False)
        print("Gene clusters saved to ", os.path.join(args.output_dir, "gene_clusters.csv"))
        

if __name__ == "__main__":
    main()