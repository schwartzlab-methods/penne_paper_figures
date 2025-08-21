'''
Code to validate the GFP levels in the dataset. It:
1. infers the gene expression
2. compute the correlation between the inferred gene expression and the GFP levels
3. Plot a PCA using inferred gene expression showing the value of GFP levels through colouring
4. look at the highest contributing genes (PC loading) to the GFP levels
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
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression

def feature_label_analysis(X, X_label, y):
    """
    Compute Pearson, Spearman correlations and Mutual Information
    between each sample's gene expression and a single label.
    
    Args:
        X: np.ndarray of shape (B, F)
        X_label: np.ndarray of shape (B, F)
        y: np.ndarray of shape (B,) or (B, 1)
    
    Returns:
        pd.DataFrame with results for each feature
    """
    y = y.squeeze()
    B, _ = X.shape

    results = {"Feature": [], "Pearson": [], "Spearman": [], "MutualInfo": []}
    for i in range(B):
        results["Feature"].append(X_label[i])
        results["Pearson"].append(np.corrcoef(X[:, i].reshape(-1), y)[0, 1])
        results["Spearman"].append(spearmanr(X[:, i].reshape(-1), y).correlation)
    results["MutualInfo"]= list((mutual_info_regression(X, y, discrete_features='auto')))

    return pd.DataFrame(results)

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
    # prep model
    model = GeneExpPredVisiumHD.load_from_checkpoint(args.model_path, num_genes = num_genes, 
                                converter = converter, feature_extractor = feature_extractor, orthogonal_loss_weight=0.1)
    model.freeze()
    model.eval()

    # inference
    pred_L = []
    gfp_L = []
    img_name_L = []
    with torch.no_grad():
        for img, label in tqdm(loader):
            img = img.to(model.device)
            img = img.squeeze(0) #remove the default batch dimension
            pred = model(img, if_convert=True) #shape: 1, num_genes
            pred_L.append(pred.cpu().numpy())
            gfp_L.append(np.array(label[0]))
            img_name_L.append(np.array(label[1], dtype=str))
    pred_L = np.concatenate(pred_L, axis=0) # shape: num_samples, num_genes
    gfp_L = np.concatenate(gfp_L, axis=0).reshape(-1) # shape: num_samples
    img_name_L = np.concatenate(img_name_L, axis=0).reshape(-1) # shape: num_samples
    # save
    np.save(os.path.join(args.output_dir, f"shane_cell_type_pred.npy"), pred_L)
    np.save(os.path.join(args.output_dir, f"shane_cell_type_gfp.npy"), gfp_L)
    np.save(os.path.join(args.output_dir, f"shane_cell_type_img_names.npy"), img_name_L)

    corr_df = feature_label_analysis(pred_L, gene_names, gfp_L)
    corr_df.to_csv(os.path.join(args.output_dir, "shane_cell_type_corr.csv"), index=False)
    # plot violin plots for each stats
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=[corr_df["Pearson"], corr_df["Spearman"], corr_df["MutualInfo"]], 
                      palette="muted", inner="quartile")
    plt.xticks(ticks=[0, 1, 2], labels=["Pearson", "Spearman", "MutualInfo"])
    plt.title("Distribution of Correlation Statistics")
    plt.ylabel("Correlation Statistics")
    plt.savefig(os.path.join(args.output_dir, "shane_cell_type_corr_violin.png"))
    plt.close()

    # compute PCA according to gene expression profiles
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(pred_L)
    plt.figure(figsize=(12, 6))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=gfp_L, cmap="viridis")
    plt.colorbar(label="GFP Level")
    plt.title("PCA of Gene Expression Profiles")
    plt.xlabel(f"PCA 1 ({pca.explained_variance_ratio_[0]:.2f})")
    plt.ylabel(f"PCA 2 ({pca.explained_variance_ratio_[1]:.2f})")
    plt.savefig(os.path.join(args.output_dir, "shane_cell_type_pca.png"))
    plt.close()

    # save the loading vector for each gene
    loading_df = pd.DataFrame(pca.components_.T, index=gene_names, columns=[f"PC{i+1}" for i in range(pca.n_components_)])
    loading_df.to_csv(os.path.join(args.output_dir, "shane_cell_type_pca_loadings.csv"))

if __name__ == "__main__":
    main()
