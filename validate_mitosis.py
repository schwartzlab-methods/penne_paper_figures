'''
Code to validate the mitosis levels in the dataset. It:
1. infers the gene expression
2. get G2-M gene set expression (ILF3;CDKN1B;RAD21;CUL3;HNRNPD;SRSF2;HNRNPU;KIF22;SMC1A)
3. correlate with mitosis levels
'''
# from dataset import MitoticDataset
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
from scipy.stats import pearsonr
import umap
from toomanycells import TooManyCells as tmc
import anndata as ad
from itertools import chain

import altair as alt
## plotting settings
if True:  # In order to bypass isort when saving
    from altairThemes import altairThemes
alt.themes.register("publishTheme", altairThemes.publishTheme)
alt.themes.enable("publishTheme")

def compute_correlation(mean_exp, mit_levels):
    '''
    Compute Pearson, Spearman correlation between mean expression and mitosis levels
    '''
    pearson_corr, pearson_p = pearsonr(mean_exp.reshape(-1), mit_levels.reshape(-1))
    spearman_corr, spearman_p = spearmanr(mean_exp.reshape(-1), mit_levels.reshape(-1))
    # return linear regression
    model = LinearRegression().fit(mean_exp.reshape(-1, 1), mit_levels.reshape(-1, 1))
    r2 = model.score(mean_exp.reshape(-1, 1), mit_levels.reshape(-1, 1))
    return pearson_corr, pearson_p, spearman_corr, spearman_p, r2

def main():
    parser = argparse.ArgumentParser(description="Validate mitotic levels")
    parser.add_argument("--input_file", type=str, required=True, nargs='+', 
                        help="Path to the input directory with GFP and PCM images")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model")
    parser.add_argument("--spaghetti_model", type=str, default=None,
                        help="Path to the spaghetti model. If None, will not use spaghetti conversion")
    parser.add_argument("--no_spaghetti", action="store_true", help='Whether to not use the Spaghetti model')
    parser.add_argument("--gene_names", type=str, required=True,
                        help="Path to the gene names file")
    parser.add_argument("--gmt_file", type=str, default=None,
                        help="Gene sets to see how many genes are the same with marker genes")
    parser.add_argument("--genes_to_use", type=str, default=None,
                        help="Path to the file with genes to use. If no supplied, use all genes")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # prep data
    # dataset = MitoticDataset(args.input_file[0])
    dataset = ShaneSeqCellTypeDataset(args.input_file, load_mitotitc=True)
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
    try: # load existing inference results if available
        pred_L = np.load(os.path.join(args.output_dir, "mitosis_pred.npy"))
        mitosis_L = np.load(os.path.join(args.output_dir, "mitosis_val.npy"))
        img_name_L = np.load(os.path.join(args.output_dir, "mitosis_img_names.npy"))
    except FileNotFoundError:
        # prep model
        model = GeneExpPredVisiumHD.load_from_checkpoint(args.model_path, num_genes = num_genes, 
                                    converter = converter, feature_extractor = feature_extractor,
                                    bio_feature_size = 960, domain_feature_size = 64)
        model.freeze()
        model.eval()

        # inference
        pred_L = []
        mitosis_L = []
        img_name_L = []
        with torch.no_grad():
            for img, label in tqdm(loader):
                img = img.to(model.device)
                img = img.squeeze(0) #remove the default batch dimension
                pred = model(img, if_convert=not args.no_spaghetti) #shape: num_patches, num_genes
                pred_L.append(pred.cpu().numpy())
                mitosis_L.append(np.array(label[0]))
                img_name_L.append(np.array(label[1], dtype=str))
        pred_L = np.concatenate(pred_L, axis=0) # shape: num_samples, num_genes
        mitosis_L = np.concatenate(mitosis_L, axis=0).reshape(-1) # shape: num_samples
        img_name_L = np.concatenate(img_name_L, axis=0).reshape(-1) # shape: num_samples
        # save
        np.save(os.path.join(args.output_dir, "mitosis_pred.npy"), pred_L)
        np.save(os.path.join(args.output_dir, "mitosis_val.npy"), mitosis_L)
        np.save(os.path.join(args.output_dir, "mitosis_img_names.npy"), img_name_L)

    # load marker gene file
    with open(args.gmt_file, 'r') as f:
        file_L = f.readline().strip().split('\t')
    gene_set = set(file_L[2:])  # skip the first two entry which is the gene set name and description
    gene_set_name = file_L[0]  # get the gene set name
    print(f"Loaded gene set: {gene_set_name} with {len(gene_set)} genes.")
    if args.genes_to_use:
        genes_to_use = np.loadtxt(args.genes_to_use, dtype=str)
        mask = np.isin(gene_names, genes_to_use)
        pred_L = pred_L[:, mask]
        gene_names = gene_names[mask]
    
    # get subsect of genes that are in the gene set
    gene_set_indices = [i for i, g in enumerate(gene_names) if g in gene_set]
    pred_L = pred_L[:, gene_set_indices]
    gene_names = gene_names[gene_set_indices]
    print(f"Using {len(gene_names)} genes from the gene set for analysis.")
    
    # correlation analysis

    pearson_corr, pearson_p, spearman_corr, spearman_p, r2 = compute_correlation(
        mean_exp = np.mean(pred_L, axis=1),
        mit_levels = mitosis_L
    )

    # plot scatter plot of mean expression vs mitosis levels
    mean_exp = np.mean(pred_L, axis=1)
    plt.figure(figsize=(8, 6))
    plt.scatter(mean_exp, mitosis_L, alpha=0.5)
    plt.xlabel("Mean Gene Expression")
    plt.ylabel("Mitosis Levels")
    plt.title(f"Mean Gene Expression vs Mitosis Levels\nPearson r={pearson_corr:.2f} (p={pearson_p:.2e}), Spearman r={spearman_corr:.2f} (p={spearman_p:.2e}), R2={r2:.2f}")
    plt.savefig(os.path.join(args.output_dir, "mean_exp_vs_mitosis.png"))
    plt.close()

    df = pd.DataFrame({
        "Mean_Gene_Expression": mean_exp,
        "Mitosis_Levels": mitosis_L
    })
    chart = alt.Chart(df).mark_circle(size=60).encode(
        x='Mean_Gene_Expression',
        y='Mitosis_Levels'
    ).interactive()

    # add a regression line
    regression = alt.Chart(df).transform_regression(
        'Mean_Gene_Expression', 'Mitosis_Levels'
    ).mark_line(color='red').encode(
        x='Mean_Gene_Expression',
        y='Mitosis_Levels'
    )
    chart = chart + regression
    chart.save(os.path.join(args.output_dir, "mean_exp_vs_mitosis_altair.html"))
    plt.close()

    # compute PCA according to gene expression profiles
    print("Running PCA")
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(pred_L)
    plt.figure(figsize=(12, 6))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=mitosis_L, cmap="viridis")
    plt.colorbar(label="Mitosis Level")
    plt.title("PCA of Gene Expression Profiles")
    plt.xlabel(f"PC 1 (Variance Explained: {pca.explained_variance_ratio_[0]:.2f})")
    plt.ylabel(f"PC 2 (Variance Explained: {pca.explained_variance_ratio_[1]:.2f})")
    plt.savefig(os.path.join(args.output_dir, "mitosis_pca.png"))
    plt.close()

    # running UMAP
    print("Running UMAP")
    umap_fit = umap.UMAP(n_components=2, random_state=42, n_neighbors=50, min_dist=1, metric='cosine')
    umap_result = umap_fit.fit_transform(pred_L)
    plt.figure(figsize=(12, 6))
    plt.scatter(umap_result[:, 0], umap_result[:, 1], c=mitosis_L, cmap="viridis")
    plt.colorbar(label="Mitosis Level")
    plt.title("UMAP of Gene Expression Profiles")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.savefig(os.path.join(args.output_dir, "mitosis_umap.png"))
    plt.close()

    # Run TMC
    print("Running TMC")
    adata = ad.AnnData(pred_L)
    adata.obs["node_id"] = [str(i) for i in range(pred_L.shape[0])]
    tmc_obj = tmc(adata, os.path.join(args.output_dir, "tmc_output"))
    tmc_obj.run_spectral_clustering(modularity_threshold=1e-9)
    tmc_obj.store_outputs(
        cell_ann_col="node_id",
    )
    # save the labels of each cell to a csv with node_ids, values
    # get the node ids of each cell
    node_ids = tmc_obj.A.obs["sp_cluster"].values
    cell_info = pd.DataFrame({
        "node_id": node_ids,
        "mitosis_level": mitosis_L
    })
    # collapse to get the mean mitosis level of each node
    cell_info = cell_info.groupby("node_id").mean().reset_index()
    cell_info.to_csv(os.path.join(args.output_dir, "tmc_output", "cell_info.csv"), index=False, header=True)

if __name__ == "__main__":
    main()
