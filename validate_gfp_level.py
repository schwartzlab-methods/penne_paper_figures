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
# from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import umap

def read_tsv(file_path):
    df_all = pd.DataFrame()
    for each in file_path:
        with open(each, 'r', encoding='utf-8') as f:
            lines = [line.strip().split('\t') for line in f]
        df = pd.DataFrame(lines).T #number of genes x number of cells
        df.columns = [cell.lower().split("-")[0] for cell in df.iloc[0].tolist()]  # set the first row as header
        df = df[2:]  # remove the first row and description
        df_all = pd.concat([df_all, df], axis=1, join="outer")
    return df_all

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
    _, F = X.shape

    results = {"Feature": [], "Pearson": [], "Spearman": []}#, "MutualInfo": []}
    for i in range(F):
        results["Feature"].append(X_label[i])
        results["Pearson"].append(np.corrcoef(X[:, i].reshape(-1), y)[0, 1])
        results["Spearman"].append(spearmanr(X[:, i].reshape(-1), y).correlation)
    # results["MutualInfo"]= list((mutual_info_regression(X, y, discrete_features='auto')))

    return pd.DataFrame(results).fillna(0)

def linear_regression(X, X_label, y, out):
    '''
    Perform linear regression on the given data.
    '''
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)

    reg = LinearRegression()
    reg.fit(X, y)
    y_pred = reg.predict(X)
    r2 = r2_score(y, y_pred)

    # plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y, y_pred, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Linear Regression (R2 = {r2:.4f})")
    plt.savefig(os.path.join(out, "shane_cell_type_linear_regression.png"))
    plt.close()

    coef_df = pd.DataFrame({
        "Feature": X_label,
        "Coefficient": reg.coef_.flatten()
    })

    pred_df = pd.DataFrame({
        "Actual": y.flatten(),
        "Prediction": y_pred.flatten()
    })

    print(f"Linear Regression R2 score: {r2:.4f}")
    return pred_df, coef_df, r2

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
    parser.add_argument("--gmt_file", type=str, default=None, nargs="+",
                        help="Gene sets to see how many genes are the same with marker genes")
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
    try:
        pred_L = np.load(os.path.join(args.output_dir, f"shane_cell_type_pred.npy"))
        gfp_L = np.load(os.path.join(args.output_dir, f"shane_cell_type_gfp.npy"))
        img_name_L = np.load(os.path.join(args.output_dir, f"shane_cell_type_img_names.npy"))
    except FileNotFoundError:
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

    # load marker gene file
    if args.gmt_file:
        # compute the ground truth gene expression
        signature = read_tsv(args.gmt_file)
        marker_mcf10a = list(set(signature["mcf10a"].dropna().values.ravel().tolist()))
        marker_hct116 = list(set(signature["hct116"].dropna().values.ravel().tolist()))
        # # get only those genes
        # mask = np.isin(gene_names, marker_mcf10a)
        # pred_L = pred_L[:,mask]
        # gene_names = gene_names[mask]
    # remove features (genes) that only have 0
    non_zero_genes = np.any(pred_L != 0, axis=0)
    pred_L = pred_L[:,non_zero_genes]
    gene_names = gene_names[non_zero_genes]
    assert pred_L.shape[1] == gene_names.shape[0]
    print("Number of genes after filtering:", gene_names.shape[0])

    # zscore normalize pred
    pred_L = (pred_L - np.mean(pred_L, axis=0)) / np.std(pred_L, axis=0) if np.std(pred_L, axis=0).all() > 0 else pred_L

    # linear regression
    print("Running Linear Regression")
    pred_df, coef_df, r2 = linear_regression(pred_L, gene_names, gfp_L, args.output_dir)
    pred_df.to_csv(os.path.join(args.output_dir, "shane_cell_type_linear_regression_predictions.csv"), index=False)
    coef_df.to_csv(os.path.join(args.output_dir, "shane_cell_type_linear_regression_coefficients.csv"), index=False)
    # see how many top features are MCF10A genes and how many bottom features are HCT116 genes
    if args.gmt_file:
        # pos_features = coef_df[coef_df['Coefficient'] > 0]["Feature"].values.reshape(-1)
        # neg_features = coef_df[coef_df['Coefficient'] < 0]["Feature"].values.reshape(-1)
        pos_features = coef_df.nlargest(200, 'Coefficient')['Feature'].values.reshape(-1)
        neg_features = coef_df.nsmallest(200, 'Coefficient')['Feature'].values.reshape(-1)
        # calculate intersection
        num_top_mcf10a = sum([1 if f in marker_mcf10a else 0 for f in pos_features])
        num_bottom_hct116 = sum([1 if f in marker_hct116 else 0 for f in neg_features])
        print(f"Number of 200 positive features that are MCF10A genes: {num_top_mcf10a} out of {len(marker_mcf10a)} marker genes")
        print(f"Number of 200 negative features that are HCT116 genes: {num_bottom_hct116} out of {len(marker_hct116)} marker genes")

    # compute correlation statistics
    print("Correlating Features")
    corr_df = feature_label_analysis(pred_L, gene_names, gfp_L)
    corr_df.to_csv(os.path.join(args.output_dir, "shane_cell_type_corr.csv"), index=False)
    # plot violin plots for each stats
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=[corr_df["Pearson"], corr_df["Spearman"]],# corr_df["MutualInfo"]], 
                      palette="muted", inner="quartile")
    plt.xticks(ticks=[0, 1],#, 2], 
                labels=["Pearson", "Spearman"])#, "MutualInfo"])
    plt.title("Distribution of Correlation Statistics")
    plt.ylabel("Correlation Statistics")
    plt.savefig(os.path.join(args.output_dir, "shane_cell_type_corr_violin.png"))
    plt.close()

    # compute PCA according to gene expression profiles
    print("Running PCA")
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(pred_L)
    plt.figure(figsize=(12, 6))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=gfp_L, cmap="viridis")
    plt.colorbar(label="GFP Level")
    plt.title("PCA of Gene Expression Profiles")
    plt.xlabel(f"PC 1 (Variance Explained: {pca.explained_variance_ratio_[0]:.2f})")
    plt.ylabel(f"PC 2 (Variance Explained: {pca.explained_variance_ratio_[1]:.2f})")
    plt.savefig(os.path.join(args.output_dir, "shane_cell_type_pca.png"))
    plt.close()

    # save the loading vector for each gene
    loading_df = pd.DataFrame(pca.components_.T, index=gene_names, columns=[f"PC{i+1}" for i in range(pca.n_components_)])
    loading_df.to_csv(os.path.join(args.output_dir, "shane_cell_type_pca_loadings.csv"))

    # running UMAP
    print("Running UMAP")
    umap_fit = umap.UMAP(n_components=2, random_state=42)
    umap_result = umap_fit.fit_transform(pred_L)
    plt.figure(figsize=(12, 6))
    plt.scatter(umap_result[:, 0], umap_result[:, 1], c=gfp_L, cmap="viridis")
    plt.colorbar(label="GFP Level")
    plt.title("UMAP of Gene Expression Profiles")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.savefig(os.path.join(args.output_dir, "shane_cell_type_umap.png"))
    plt.close()

if __name__ == "__main__":
    main()
