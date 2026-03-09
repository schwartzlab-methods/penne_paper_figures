'''
Code to validate the GFP levels in the dataset. It:
1. infers the gene expression
2. compute the correlation between the inferred gene expression and the GFP levels
3. Plot a PCA using inferred gene expression showing the value of GFP levels through colouring
4. look at the highest contributing genes (PC loading) to the GFP levels
'''
from dataset import ShaneSeqCellTypeDataset
from correlation_validation import parse_gt_files
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
from scipy.stats import spearmanr, fisher_exact, pearsonr
# from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score
import umap
from toomanycells import TooManyCells as tmc
import anndata as ad
from exp_analysis.gene_set_expression import read_tsv
from itertools import chain
import gseapy as gp
import altair as alt
## plotting settings
if True:  # In order to bypass isort when saving
    from altairThemes import altairThemes
alt.themes.register("publishTheme", altairThemes.publishTheme)
alt.themes.enable("publishTheme")

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

    results = {"Gene": [], "Pearson": [], "Spearman": []}#, "MutualInfo": []}
    for i in range(F):
        results["Gene"].append(X_label[i])
        results["Pearson"].append(np.corrcoef(X[:, i].reshape(-1), y)[0, 1])
        results["Spearman"].append(spearmanr(X[:, i].reshape(-1), y).correlation)
    # results["MutualInfo"]= list((mutual_info_regression(X, y, discrete_features='auto')))

    return pd.DataFrame(results).fillna(0)

def compute_number_of_correct_features(coef_df, marker_mcf10a, marker_hct116, output, name="", up_thresh=0, down_thresh=0):
    pos_features = coef_df[coef_df['Coefficient'] > up_thresh]["Gene"].values.reshape(-1)
    neg_features = coef_df[coef_df['Coefficient'] < down_thresh]["Gene"].values.reshape(-1)
    # pos_features = coef_df.nlargest(200, 'Coefficient')['gene'].values.reshape(-1)
    # neg_features = coef_df.nsmallest(200, 'Coefficient')['gene'].values.reshape(-1)
    # calculate intersection
    num_top_mcf10a = sum([1 if f in marker_mcf10a else 0 for f in pos_features])
    num_bottom_mcf10a = sum([1 if f in marker_mcf10a else 0 for f in neg_features])
    num_bottom_hct116 = sum([1 if f in marker_hct116 else 0 for f in neg_features])
    num_top_hct116 = sum([1 if f in marker_hct116 else 0 for f in pos_features])
    print(f"Number of positive features that are MCF10A genes: {num_top_mcf10a} out of {len(marker_mcf10a)} marker genes")
    print(f"Number of negative features that are MCF10A genes: {num_bottom_mcf10a} out of {len(marker_mcf10a)} marker genes")
    print(f"Number of negative features that are HCT116 genes: {num_bottom_hct116} out of {len(marker_hct116)} marker genes")
    print(f"Number of positive features that are HCT116 genes: {num_top_hct116} out of {len(marker_hct116)} marker genes")
    # perform Fisher's exact test
    contingency_table_mcf10a = np.array([[num_top_mcf10a, len(pos_features) - num_top_mcf10a],
                                            [num_bottom_mcf10a, len(neg_features) - num_bottom_mcf10a]])
    odds_ratio_mcf10a, p_value_mcf10a = fisher_exact(contingency_table_mcf10a, alternative='greater')
    print(f"Fisher's exact test p-value for MCF10A: {p_value_mcf10a}")

    contingency_table_hct116 = np.array([[num_bottom_hct116, len(neg_features) - num_bottom_hct116],
                                            [num_top_hct116, len(pos_features) - num_top_hct116]])
    odds_ratio_hct116, p_value_hct116 = fisher_exact(contingency_table_hct116, alternative='greater')
    print(f"Fisher's exact test p-value for HCT116: {p_value_hct116}")

    with open(os.path.join(output, f"shane_feature_linear_fisher_exact_results_{name}.txt"), "w") as f:
        f.write("G: The feature with correct sign; E: the gene set\n")
        f.write(f"Using thresholds: up_thresh={up_thresh}, down_thresh={down_thresh}\n")
        f.write("---------------------------------------------------\n")
        f.write(f"MCF10A,odds ratio: {odds_ratio_mcf10a}, p-value: {p_value_mcf10a}\n")
        f.write("contingency ([(G+, E+),(G+, E-)],[(G-, E+), (G-, E-)])\n")
        f.write(f"{num_top_mcf10a}, {len(pos_features) - num_top_mcf10a}\n")
        f.write(f"{num_bottom_mcf10a}, {len(neg_features) - num_bottom_mcf10a}\n")
        f.write("---------------------------------------------------\n")
        f.write(f"HCT116,odds ratio: {odds_ratio_hct116}, p-value: {p_value_hct116}\n")
        f.write("contingency ([(G+, E+),(G+, E-)],[(G-, E+), (G-, E-)])\n")
        f.write(f"{num_top_hct116}, {len(pos_features) - num_top_hct116}\n")
        f.write(f"{num_bottom_hct116}, {len(neg_features) - num_bottom_hct116}\n")

def enrichr_analysis(coef_df, database, output, name="", up_thresh=0, down_thresh=0):
    '''
    run enrichr analysis on the top and bottom gene from database (str or dict)
    '''
    pos_features = coef_df[coef_df['Coefficient'] > up_thresh]["Gene"].values.reshape(-1)
    neg_features = coef_df[coef_df['Coefficient'] < down_thresh]["Gene"].values.reshape(-1)
    # run enrichr
    enr_pos = gp.enrichr(gene_list=pos_features.tolist(),
                        gene_sets=database,
                        outdir=None,
                        verbose=True)
    enr_neg = gp.enrichr(gene_list=neg_features.tolist(),
                        gene_sets=database,
                        outdir=None,
                        verbose=True)
    pos_sorted = enr_pos.results.sort_values(by='Combined Score', ascending=False)
    pos_final = pos_sorted[pos_sorted['Adjusted P-value'] < 0.05]
    pos_final = pos_final[pos_final["Combined Score"] > 0]
    neg_sorted = enr_neg.results.sort_values(by='Combined Score', ascending=False)
    neg_final = neg_sorted[neg_sorted['Adjusted P-value'] < 0.05]
    neg_final = neg_final[neg_final["Combined Score"] > 0]
    # save results
    pos_final.to_csv(os.path.join(output, f"shane_feature_linear_enrichr_pos_{name}.csv"), index=False)
    neg_final.to_csv(os.path.join(output, f"shane_feature_linear_enrichr_neg_{name}.csv"), index=False)
    # plot top 10 terms for both
    top_pos = pos_final.head(10)
    top_neg = neg_final.head(10)
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    sns.barplot(x='Combined Score', y='Term', data=top_pos, color='blue')
    plt.title('Top 10 Enrichr Terms for Positive Features')
    plt.subplot(1, 2, 2)
    sns.barplot(x='Combined Score', y='Term', data=top_neg, color='red')
    plt.title('Top 10 Enrichr Terms for Negative Features')
    plt.tight_layout()
    plt.savefig(os.path.join(output, f"shane_feature_linear_enrichr_{name}.png"))
    plt.close()

def prerank_gsea(coef_df, database, output, name=""):
    '''
    run prerank gsea analysis on the top and bottom gene from gmt file
    '''
    # create preranked file
    prerank_df = coef_df[['Gene', 'Coefficient']]
    prerank_df_sorted = prerank_df.sort_values(by='Coefficient', ascending=False)
    # reindex
    prerank_df_sorted = prerank_df_sorted.loc[:, ['Gene', 'Coefficient']]
    # run prerank
    pre_res = gp.prerank(rnk=prerank_df_sorted,
                         gene_sets=database,
                         outdir=None,
                         threads=4,
                         min_size=1,
                         max_size=10000,
                         permutation_num=1000,
                         seed=42,
                         verbose=True)
    pre_sorted = pre_res.res2d.sort_values(by='NES', ascending=False)
    
    up_terms = pre_sorted[pre_sorted['NES'] > 0]
    down_terms = pre_sorted[pre_sorted['NES'] < 0]
    # save results
    pre_sorted.to_csv(os.path.join(output, f"shane_feature_linear_prerank_gsea_{name}.csv"), index=False)
    # plot top 10 terms for both
    top_terms = up_terms.head(10)
    plt.figure(figsize=(16, 6))
    sns.barplot(x='NES', y='Term', data=top_terms, color='red')
    plt.title('Top 10 Prerank GSEA Terms')
    plt.tight_layout()
    plt.savefig(os.path.join(output, f"shane_feature_linear_prerank_gsea_{name}_up.png"))
    plt.close()
    
    top_terms_down = down_terms.head(10)
    plt.figure(figsize=(16, 6))
    sns.barplot(x='NES', y='Term', data=top_terms_down, color='blue')
    plt.title('Top 10 Prerank GSEA Terms (Down)')
    plt.tight_layout()
    plt.savefig(os.path.join(output, f"shane_feature_linear_prerank_gsea_{name}_down.png"))
    plt.close()

def linear_regression(X, X_label, y, out , gt_level=None, gt_level_neg=None):
    '''
    Perform regression on the given data.
    '''
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)

    # reg = LinearRegression()
    reg = Ridge(alpha=1.0)
    # reg = Lasso(alpha=0.1)
    reg.fit(X, y)
    y_pred = reg.predict(X)
    # p-values and r2
    pearson_corr, p_value = pearsonr(y.flatten(), y_pred.flatten())
    r2 = pearson_corr ** 2
    # r2 = r2_score(y, y_pred)

    # generate regression line values
    reg_line = LinearRegression()
    reg_line.fit(y, y_pred)
    y_pred_reg = reg_line.predict(y)

    # if gt_level is provided, compute the cosine similarity between the inferred gene expression and the gt_level
    if gt_level is not None and gt_level_neg is not None:
        similarity = cosine_similarity(X, gt_level) - cosine_similarity(X, gt_level_neg)
        # get colour palette based on gt_level
        norm = plt.Normalize(similarity.min(), similarity.max())
        cmap = plt.cm.viridis
        colors = cmap(norm(similarity))
    else:
        colors = 'blue'
    # plot
    plt.figure(figsize=(15, 6))
    plt.scatter(y, y_pred, color=colors, alpha=0.7)
    # add a legend for the colour based on gt_level if provided
    if gt_level is not None and gt_level_neg is not None:
        plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), 
        label='Cosine Similarity to GT Level', cax=plt.gca().inset_axes([1.05, 0.1, 0.02, 0.8]))
    # plot the regression line genrated
    plt.plot(y, y_pred_reg, color='red', linewidth=2)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Ridge Regression (R2 = {r2:.4f}, p-value = {p_value:.4e})")
    plt.savefig(os.path.join(out, "shane_cell_type_ridge_regression.png"))
    plt.close()

    print(f"Ridge Regression R2 score: {r2:.4f}")
    print(f"Ridge Regression p-value: {p_value:.4e}")

    if X_label is not None:
        coef_df = pd.DataFrame({
            "Gene": X_label,
            "Coefficient": reg.coef_.flatten()
        })
    else:
        coef_df = pd.DataFrame({
            "Feature_Index": np.arange(X.shape[1]),
            "Coefficient": reg.coef_.flatten()
        })

    pred_df = pd.DataFrame({
        "Actual": y.flatten(),
        "Prediction": y_pred.flatten()
    })

    # altair plot
    scatter = alt.Chart(pred_df).mark_circle(opacity=0.5).encode(
        x='Actual:Q',
        y='Prediction:Q',
        tooltip=['Actual', 'Prediction']
    ).interactive()
    chart = scatter + scatter.transform_regression('Actual', 'Prediction').mark_line(color='red')
    chart.save(os.path.join(out, "shane_cell_type_ridge_regression_altair.html"))

    return pred_df, coef_df, r2, p_value

def cosine_similarity(exp, gt_level):
    '''
    Compute cosine similarity between the inferred gene expression and the ground truth expression
    Return the similarity score for each sample
    '''
    exp_norm = exp / np.linalg.norm(exp, axis=1, keepdims=True)
    gt_norm = gt_level / np.linalg.norm(gt_level)
    similarity = np.dot(exp_norm, gt_norm)
    return similarity

def main():
    parser = argparse.ArgumentParser(description="Validate GFP levels")
    parser.add_argument("--input_file", type=str, required=True, 
                        help="Path to the input directory with GFP and PCM images")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model")
    parser.add_argument("--spaghetti_model", type=str, default=None,
                        help="Path to the spaghetti model. If None, will not use spaghetti conversion")
    parser.add_argument("--no_spaghetti", action="store_true", help='Whether to not use the Spaghetti model')
    parser.add_argument("--gene_names", type=str, required=True,
                        help="Path to the gene names file")
    parser.add_argument("--gmt_file", type=str, default=None, nargs="+",
                        help="Gene sets to see how many genes are the same with marker genes")
    parser.add_argument("--only_markers", action="store_true",
                        help="Whether to only use MCF10A marker genes")
    parser.add_argument("--genes_to_use", type=str, default=None,
                        help="Path to the file with genes to use. If no supplied, use all genes")
    parser.add_argument("--threshold", type=float, default=10.0,
                        help="Threshold for filtering genes based on correlation. Range: [0,100]")
    parser.add_argument("--scramble", action="store_true", help="Whether to scramble/permute the input images for baseline evaluation")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory")
    parser.add_argument("--load", action="store_true", help="Whether to load existing predictions")
    parser.add_argument("--gt", type=str, default=None, help="Path to the ground truth gene expression of MCF10A for correlation analysis")
    parser.add_argument("--gt_marker", type=str, default=None, help="Path to the ground truth marker gene expression of MCF10A for correlation analysis")
    parser.add_argument("--test_mcf_hct", action="store_true", help="Whether to test MCF10A and HCT116 marker gene enrichment")
    parser.add_argument("--show_example", action="store_true", help="Whether to show example images with high and low GFP levels")
    parser.add_argument("--run_dim_reduction", action="store_true", help="Whether to run dimensionality reduction (PCA, UMAP, TMC)")
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
        if args.load:
            pred_L = np.load(os.path.join(args.output_dir, f"shane_cell_type_pred.npy"))
            gfp_L = np.load(os.path.join(args.output_dir, f"shane_cell_type_gfp.npy"))
            img_name_L = np.load(os.path.join(args.output_dir, f"shane_cell_type_img_names.npy"))
        else:
            raise FileNotFoundError
    except FileNotFoundError:
        # prep model
        model = GeneExpPredVisiumHD.load_from_checkpoint(args.model_path, num_genes = num_genes, 
                                    converter = converter, feature_extractor = feature_extractor,
                                    bio_feature_size = 960, domain_feature_size = 64)
        model.freeze()
        model.eval()

        # inference
        pred_L = []
        gfp_L = []
        img_name_L = []
        img_L = []
        pcm_img_L = []
        with torch.no_grad():
            for img, label in tqdm(loader):
                if "00d" not in label[1][0]:
                     print(f"Skipping sample {label[1][0]} as it is too confluent to measure GFP levels reliably")
                     continue
                img = img.to(model.device)
                img = img.squeeze(0) #remove the default batch dimension
                pred = model(img, if_convert=not args.no_spaghetti, scramble=args.scramble) #shape: num_patches, num_genes
                pred_L.append(pred.cpu().numpy())
                gfp_L.append(np.array(label[0]))
                img_name_L.append(np.array(label[1], dtype=str))
                img_L.append(label[2][0].cpu().numpy())
                pcm_img_L.append(img.cpu().numpy())
        pred_L = np.concatenate(pred_L, axis=0) # shape: num_samples, num_genes
        gfp_L = np.concatenate(gfp_L, axis=0).reshape(-1) # shape: num_samples
        img_name_L = np.concatenate(img_name_L, axis=0).reshape(-1) # shape: num_samples
        img_L = np.concatenate(img_L, axis=0) # shape: num_samples, channels, height, width
        pcm_img_L = np.concatenate(pcm_img_L, axis=0) # shape: num_samples, channels, height, width
        # save
        np.save(os.path.join(args.output_dir, f"shane_cell_type_pred.npy"), pred_L)
        np.save(os.path.join(args.output_dir, f"shane_cell_type_gfp.npy"), gfp_L)
        np.save(os.path.join(args.output_dir, f"shane_cell_type_img_names.npy"), img_name_L)
    
    # filter out negative GFP levels which are unreliable measurements
    valid_indices = gfp_L >= 0
    pred_L_z = (pred_L - pred_L.mean(axis=1, keepdims=True)) / (pred_L.std(axis=1, keepdims=True) + 1e-8)  # z-score normalization
    pred_L = pred_L[valid_indices]
    pred_L_z = pred_L_z[valid_indices]
    gfp_L = gfp_L[valid_indices]
    
    # parse gt if provided
    if args.gt:
        gt_files = [os.path.join(args.gt, f) 
                for f in os.listdir(args.gt) if f.endswith('_gene_symbols.txt') and "NIR" in f]
        gt_df = parse_gt_files(gt_files) # shape: num_samples, num_genes
        # # z-score normalize
        # gt_df = gt_df.apply(lambda x: (x - x.mean()) / (x.std() + 1e-8), axis=0)
        # take the mean across all rows (samples) to get the average expression level for each gene
        gt_neg_level = gt_df[gt_df.index.str.contains("HCT")].mean(axis=0).values.reshape(-1)
        gt_level = gt_df[gt_df.index.str.contains("GFP")].mean(axis=0).values.reshape(-1) # shape: num_genes
        # order according to gene_names
        gt_level_ordered = []
        gt_neg_level_ordered = []
        for gene in gene_names:
            if gene in gt_df.columns:
                gt_level_ordered.append(gt_level[gt_df.columns.get_loc(gene)])
                gt_neg_level_ordered.append(gt_neg_level[gt_df.columns.get_loc(gene)])
            else:
                gt_level_ordered.append(0.0) # if gene not found in gt, assign 0
                gt_neg_level_ordered.append(0.0)
        # get sample with name GFP and HCT
        gt_level_ordered = np.array(gt_level_ordered).reshape(-1) # shape: num_genes
        gt_neg_level_ordered = np.array(gt_neg_level_ordered).reshape(-1) # shape: num_genes
        if args.gt_marker:
            gt_marker_df = pd.read_csv(args.gt_marker, sep=',', header=0)
            # order according to log2FC
            gt_marker_df.sort_values(by="log2FC", ascending=False, inplace=True)
            # get top and bottom genes as marker genes
            # gt_marker_top = gt_marker_df.head(200)["gene_symbol"].values
            # gt_marker_bottom = gt_marker_df.tail(200)["gene_symbol"].values
            gt_marker_top = gt_marker_df[gt_marker_df["log2FC"] > 6]["gene_symbol"].values
            gt_marker_bottom = gt_marker_df[gt_marker_df["log2FC"] < -6]["gene_symbol"].values
            all_markers = list(set(gt_marker_top).union(set(gt_marker_bottom)))
            # filter the gene_names to only include marker genes
            masks = np.isin(gene_names, all_markers)
            gene_names = gene_names[masks]
            pred_L = pred_L[:,masks]
            pred_L_z = pred_L_z[:,masks]
            gt_level_ordered = gt_level_ordered[masks]
            gt_neg_level_ordered = gt_neg_level_ordered[masks]

    # plot green fluorescence distribution
    plt.figure(figsize=(8,6))
    sns.histplot(gfp_L, bins=50, kde=True)
    plt.title("Distribution of GFP Levels")
    plt.xlabel("GFP Level")
    plt.ylabel("Count")
    plt.savefig(os.path.join(args.output_dir, "shane_cell_type_gfp_distribution.png"))
    plt.close()

    if args.show_example and not args.load:
        # show example images with high and low GFP levels
        img_L = img_L[valid_indices]
        pcm_img_L = pcm_img_L[valid_indices]
        sorted_indices = np.argsort(gfp_L)
        low_indices = sorted_indices[:5]
        high_indices = sorted_indices[-5:]
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        for i, idx in enumerate(low_indices):
            img = img_L[idx]
            img_np = np.transpose(img, (1, 2, 0))
            axes[0, i].imshow(img_np)
            axes[0, i].set_title(f"Low GFP: {gfp_L[idx]:.4f}")
            axes[0, i].axis('off')
        for i, idx in enumerate(high_indices):
            img = img_L[idx]
            img_np = np.transpose(img, (1, 2, 0))
            axes[1, i].imshow(img_np)
            axes[1, i].set_title(f"High GFP: {gfp_L[idx]:.4f}")
            axes[1, i].axis('off')
        plt.suptitle("Example Images with Low and High GFP Levels")
        plt.savefig(os.path.join(args.output_dir, "shane_cell_type_example_images.png"))
        plt.close()
        # show example PCM images with high and low GFP levels
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        for i, idx in enumerate(low_indices):
            pcm_img = pcm_img_L[idx]
            pcm_img_np = np.transpose(pcm_img, (1, 2, 0))
            axes[0, i].imshow(pcm_img_np)
            axes[0, i].set_title(f"Low GFP: {gfp_L[idx]:.4f}")
            axes[0, i].axis('off')
        for i, idx in enumerate(high_indices):
            pcm_img = pcm_img_L[idx]
            pcm_img_np = np.transpose(pcm_img, (1, 2, 0))
            axes[1, i].imshow(pcm_img_np)
            axes[1, i].set_title(f"High GFP: {gfp_L[idx]:.4f}")
            axes[1, i].axis('off')
        plt.suptitle("Example PCM Images with Low and High GFP Levels")
        plt.savefig(os.path.join(args.output_dir, "shane_cell_type_example_pcm_images.png"))
        plt.close()
        #! display all gfp of 0
        zero_gfp_indices = np.where(gfp_L == 0)[0]
        # save a pair of image and pcm with gfp of 0, each as individual file, in a folder
        os.makedirs(os.path.join(args.output_dir, "zero_gfp_examples"), exist_ok=True)
        for i, idx in enumerate(zero_gfp_indices):
            img = img_L[idx]
            pcm_img = pcm_img_L[idx]
            # save image
            img_np = np.transpose(img, (1, 2, 0))
            plt.imsave(os.path.join(args.output_dir, "zero_gfp_examples", f"zero_gfp_image_{i}.png"), img_np)
            # save pcm image
            pcm_img_np = np.transpose(pcm_img, (1, 2, 0))
            plt.imsave(os.path.join(args.output_dir, "zero_gfp_examples", f"zero_gfp_pcm_image_{i}.png"), pcm_img_np)
    # load marker gene file
    if args.gmt_file and args.test_mcf_hct: # compute correlation with marker genes
        signature = read_tsv(args.gmt_file)

        marker_mcf10a = sorted(set(chain.from_iterable(
                            signature["mcf10a"].dropna().tolist()
                        )))
        marker_hct116 = sorted(set(chain.from_iterable(
                            signature["hct116"].dropna().tolist()
                        )))

        # remove the intersection between the two
        intersection = set(marker_mcf10a).intersection(set(marker_hct116))
        marker_mcf10a = list(set(marker_mcf10a) - intersection)
        marker_hct116 = list(set(marker_hct116) - intersection)
    if args.genes_to_use:
        genes_to_use = np.loadtxt(args.genes_to_use, dtype=str)
        mask = np.isin(gene_names, genes_to_use)
        pred_L = pred_L[:,mask]
        gene_names = gene_names[mask]
        gt_level_ordered = gt_level_ordered[mask] if args.gt else None
        gt_neg_level_ordered = gt_neg_level_ordered[mask] if args.gt else None

    if args.test_mcf_hct:
        # get the mcf10a marker expressions
        mask_mcf = np.isin(gene_names, marker_mcf10a)
        mask_hct = np.isin(gene_names, marker_hct116)
        mask_either = mask_mcf | mask_hct
        pred_L_hct = np.mean(pred_L[:,mask_hct], axis=1, keepdims=True)
        pred_L_mcf = pred_L[:,mask_mcf] # shape: num_samples, num_marker_genes
        pred_L_mcf = np.mean(pred_L_mcf, axis=1, keepdims=True) # shape: num_samples, marker_gene_exp
        pred_L_neither = pred_L[:,~mask_either]
        pred_L_neither = np.mean(pred_L_neither, axis=1, keepdims=True)
        # compute both spearman and pearson correlation
        spearman_mcf, p_value_mcf = spearmanr(pred_L_mcf.flatten(), gfp_L.flatten())
        print(f"Spearman correlation between mean MCF10A marker gene expression and GFP levels: {spearman_mcf:.4f} (p-value: {p_value_mcf:.4e})")
        spearman_hct, p_value_hct = spearmanr(pred_L_hct.flatten(), gfp_L.flatten())
        print(f"Spearman correlation between mean HCT116 marker gene expression and GFP levels: {spearman_hct:.4f} (p-value: {p_value_hct:.4e})")
        spearman_neither, p_value_neither = spearmanr(pred_L_neither.flatten(), gfp_L.flatten())
        print(f"Spearman correlation between mean non-marker gene expression and GFP levels: {spearman_neither:.4f} (p-value: {p_value_neither:.4e})")
        pearson_mcf, p_value_mcf_pearson = pearsonr(pred_L_mcf.flatten(), gfp_L.flatten())
        print(f"Pearson correlation between mean MCF10A marker gene expression and GFP levels: {pearson_mcf:.4f} (p-value: {p_value_mcf_pearson:.4e})")
        pearson_hct, p_value_hct_pearson = pearsonr(pred_L_hct.flatten(), gfp_L.flatten())
        print(f"Pearson correlation between mean HCT116 marker gene expression and GFP levels: {pearson_hct:.4f} (p-value: {p_value_hct_pearson:.4e})")
        pearson_neither, p_value_neither_pearson = pearsonr(pred_L_neither.flatten(), gfp_L.flatten())
        print(f"Pearson correlation between mean non-marker gene expression and GFP levels: {pearson_neither:.4f} (p-value: {p_value_neither_pearson:.4e})")
        with open(os.path.join(args.output_dir, "shane_cell_type_marker_gene_correlations.txt"), "w") as f:
            f.write("Spearman Correlations:\n")
            f.write(f"MCF10A: {spearman_mcf:.4f} (p-value: {p_value_mcf:.4e})\n")
            f.write(f"HCT116: {spearman_hct:.4f} (p-value: {p_value_hct:.4e})\n")
            f.write(f"Non-marker genes: {spearman_neither:.4f} (p-value: {p_value_neither:.4e})\n")
            f.write("\nPearson Correlations:\n")
            f.write(f"MCF10A: {pearson_mcf:.4f} (p-value: {p_value_mcf_pearson:.4e})\n")
            f.write(f"HCT116: {pearson_hct:.4f} (p-value: {p_value_hct_pearson:.4e})\n")
            f.write(f"Non-marker genes: {pearson_neither:.4f} (p-value: {p_value_neither_pearson:.4e})\n")

    # remove features (genes) that only have 0
    non_zero_genes = np.any(pred_L != 0, axis=0)
    pred_L = pred_L[:,non_zero_genes]
    pred_L_z = pred_L_z[:,non_zero_genes]
    gene_names = gene_names[non_zero_genes]
    gt_level_ordered = gt_level_ordered[non_zero_genes] if args.gt else None
    gt_neg_level_ordered = gt_neg_level_ordered[non_zero_genes] if args.gt else None
    assert pred_L.shape[1] == gene_names.shape[0], "Number of genes after filtering does not match the number of columns in pred_L"

    print("Number of genes after filtering zero genes:", gene_names.shape[0])
    if args.test_mcf_hct:
        num_genes_in_mcf10a = sum([1 if g in marker_mcf10a else 0 for g in gene_names])
        num_genes_in_hct116 = sum([1 if g in marker_hct116 else 0 for g in gene_names])
        print(f"Number of genes after filtering in MCF10A: {num_genes_in_mcf10a}")
        print(f"Number of genes after filtering in HCT116: {num_genes_in_hct116}")
    
    if args.gt:
        # treat this as a binary classifination
        # take top 33% gfp as high, mid 33% as mid, bottom 33% as low
        # then plot a box plot of cosine similarity between the inferred gene expression and the gt_level for each group
        print("Computing cosine similarity between inferred gene expression and GT level across GFP levels")
        print("Total samples: ", len(pred_L))
        final_high_marker_genes = set(gt_marker_top).intersection(set(gene_names))
        final_low_marker_genes = set(gt_marker_bottom).intersection(set(gene_names))
        df_final_marker_genes = pd.DataFrame({
            "Gene": gene_names,
            "Is_MCF10A_Marker": [1 if gene in final_high_marker_genes else 0 for gene in gene_names],
            "Is_HCT116_Marker": [1 if gene in final_low_marker_genes else 0 for gene in gene_names]
        })
        df_final_marker_genes.to_csv(os.path.join(args.output_dir, "shane_cell_type_final_marker_genes.csv"), index=False)
        num_cut = 2
        print("Cutting GFP levels into ", num_cut, " bins")
        gfp_bins = pd.cut(gfp_L, bins=num_cut, labels=[f"bin_{i}" for i in range(num_cut)])
        # gfp_bins = pd.qcut(gfp_L, q=num_cut, labels=[f"bin_{i}" for i in range(num_cut)])
        high_idx = gfp_bins == f"bin_{num_cut-1}"
        low_idx = gfp_bins == f"bin_0"
        # similarity_pos = cosine_similarity(pred_L, gt_level_ordered)
        # similarity_neg = cosine_similarity(pred_L, gt_neg_level_ordered)
        # save a DEG with gene names and diff between high and low
        deg_df = pd.DataFrame({
            "Gene": gene_names,
            "exp_in_high": np.mean(pred_L[high_idx, :], axis=0),
            "exp_in_low": np.mean(pred_L[low_idx, :], axis=0),
            "log2fc": np.mean(pred_L[high_idx, :], axis=0) - np.mean(pred_L[low_idx, :], axis=0)
        })
        deg_df.to_csv(os.path.join(args.output_dir, "shane_cell_type_deg_high33_vs_low33_gfp.csv"), index=False)
        #* compute a z-score difference between high marker gene and low marker gene
        pos_gene_marker = np.isin(gene_names, gt_marker_top)
        neg_gene_marker = np.isin(gene_names, gt_marker_bottom)
        similarity_pos = np.mean(pred_L_z[:,pos_gene_marker], axis=1)
        similarity_neg = np.mean(pred_L_z[:,neg_gene_marker], axis=1)
        sim_df = pd.DataFrame({
            "gfp_values": gfp_L,
            "GFP_Level": gfp_bins,
            "Cosine_Similarity": similarity_pos - similarity_neg
        })
        # run a statistical test to see if there is a significant difference between the three groups
        # use manwhiney u test since the data is not normally distributed
        from scipy.stats import mannwhitneyu
        low_high = sim_df[sim_df["GFP_Level"].isin([f"bin_0", f"bin_{num_cut-1}"])]
        u_stat_low_high, p_value_low_high = mannwhitneyu(low_high[low_high["GFP_Level"]==f"bin_0"]["Cosine_Similarity"], low_high[low_high["GFP_Level"]==f"bin_{num_cut-1}"]["Cosine_Similarity"])
        print(f"Mann-Whitney U test between Low and High GFP levels: U={u_stat_low_high}, p-value={p_value_low_high:.4e}")
        print(sim_df.groupby("GFP_Level")["Cosine_Similarity"].describe())
        plt.figure(figsize=(15,6))
        sns.boxplot(x="GFP_Level", y="Cosine_Similarity", data=sim_df, palette="muted")
        plt.xlabel("GFP Level")
        plt.ylabel("Log2FC\nMCF10A and HCT116 Marker Gene Expression")
        plt.title("Log2FC between MCF10A and HCT116 Marker Gene Expression across GFP Levels")
        plt.savefig(os.path.join(args.output_dir, "shane_cell_type_log2fc_boxplot.png"))
        plt.close()
        # plot a scatter plot between GFP levels and cosine similarity
        plt.figure(figsize=(15,6))
        sns.scatterplot(x=gfp_L, y=similarity_pos - similarity_neg)
        # fit a regression line
        sns.regplot(x=gfp_L, y=similarity_pos - similarity_neg, scatter=False, color='red')
        # show the correlation coefficient in the plot
        corr_coef, p_value_corr = pearsonr(gfp_L, similarity_pos - similarity_neg)
        plt.text(0.05, 0.95, f"Pearson r = {corr_coef:.4f}\n(p-value = {p_value_corr:.4e})", transform=plt.gca().transAxes, 
                 fontsize=12, verticalalignment='top')
        plt.title("Log2FC between MCF10A and HCT116 Marker Gene Expression across GFP Levels")
        plt.xlabel("GFP Level")
        plt.ylabel("Log2FC (MCF10A - HCT116)")
        plt.savefig(os.path.join(args.output_dir, "shane_cell_type_log2fc_scatter.png"))
        plt.close()
        #! make GFP into more bins and plot mean for each bin
        num_bins = 10
        sim_df["bin"]=pd.cut(sim_df["gfp_values"], bins=num_bins, labels=[f"Bin_{i}" for i in range(num_bins)])
        pseudo_bulk = sim_df.groupby('bin')[['gfp_values', 'Cosine_Similarity']].mean().dropna()
        r, p = pearsonr(pseudo_bulk['gfp_values'], pseudo_bulk['Cosine_Similarity'])
        plt.figure(figsize=(15,6))
        plt.errorbar(pseudo_bulk['gfp_values'], pseudo_bulk['Cosine_Similarity'], 
                 yerr=sim_df.groupby('bin')['Cosine_Similarity'].sem(), 
                 fmt='o', color='black', capsize=5, label='Bin Mean +- SEM')
    
        sns.regplot(x='gfp_values', y='Cosine_Similarity', data=pseudo_bulk, scatter=False, color='red')
        
        plt.title(f"Pseudo-Bulk Validation (n={num_bins} bins)\nPearson r: {r:.4f}\n(p-value: {p:.4e})")
        plt.xlabel("Mean GFP Level (MCF10A Abundance)")
        plt.ylabel("Log2FC\nMCF10A and HCT116 Marker Gene Expression")
        plt.legend()
        plt.savefig(os.path.join(args.output_dir, "shane_cell_type_log2fc_binned_scatter.png"))
        plt.close()

    # ridge regression
    print("Running Ridge Regression")
    pred_df, coef_df, r2, p_value = linear_regression(pred_L, None if args.only_markers else gene_names, gfp_L, args.output_dir, 
                                                    gt_level_ordered if args.gt else None, gt_neg_level_ordered if args.gt else None)
    pred_df.to_csv(os.path.join(args.output_dir, "shane_cell_type_ridge_regression_predictions.csv"), index=False)
    coef_df.to_csv(os.path.join(args.output_dir, "shane_cell_type_ridge_regression_coefficients.csv"), index=False)
    # see how many top features are MCF10A genes and how many bottom features are HCT116 genes
    if args.gmt_file and not args.only_markers:
        top_ten_percent_coef = np.percentile(coef_df["Coefficient"], 100 - args.threshold)
        bottom_ten_percent_coef = np.percentile(coef_df["Coefficient"], args.threshold)
        if args.test_mcf_hct:
            compute_number_of_correct_features(coef_df, marker_mcf10a, marker_hct116, args.output_dir, name="ridge",
                                                up_thresh=top_ten_percent_coef, down_thresh=bottom_ten_percent_coef)
            # plot coefficients and label MCF10A or HCT116 or neither with three colours using altair tick plot
            coef_df["Colour"] = ["MCF10A Marker" if g in marker_mcf10a 
                                else "HCT116 Marker" if g in marker_hct116 
                                else "Neither" 
                                for g in coef_df["Gene"]]
            coef_df = coef_df.sort_values(by="Coefficient", ascending=False)
            ticks = alt.Chart(coef_df).mark_tick(opacity=0.5, thickness=1).encode(
                color=alt.Color('Colour:N', scale=alt.Scale(domain=["MCF10A Marker", "HCT116 Marker", "Neither"], range=["red", "blue", "grey"])),
                x=alt.X('Coefficient:Q', title='Ridge Coefficient')
            ).interactive()
            ticks.save(os.path.join(args.output_dir, "shane_cell_type_ridge_coefficients.html"))
        # run enrichr analysis
        enrichr_analysis(coef_df, database=args.gmt_file, output=args.output_dir, name="ridge",
                        up_thresh=top_ten_percent_coef, down_thresh=bottom_ten_percent_coef)
        # run prerank gsea
        prerank_gsea(coef_df, database=args.gmt_file, output=args.output_dir, name="ridge")

    # compute correlation statistics
    if not args.only_markers:
        print("Correlating Features using Pearson and Spearman from expression to GFP levels")
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
        if args.test_mcf_hct:
            # plot scatter plot between Pearson and Spearman, allow tool tip labels to select genes
            # colour if gene is in MCF10A (red) or HCT116 (blue) or none (grey)
            # plot red and blue points on top of grey points
            corr_df["Colour"] = ["MCF10A Marker" if g in marker_mcf10a 
                                else "HCT116 Marker" if g in marker_hct116 
                                else "Neither" 
                                for g in corr_df["Gene"]]
            corr_df = corr_df.sort_values(by="Colour", ascending=False)
            scatter = alt.Chart(corr_df).mark_circle(opacity=0.5).encode(
                x='Pearson:Q',
                y='Spearman:Q',
                color=alt.Color('Colour:N', scale=alt.Scale(domain=["MCF10A Marker", "HCT116 Marker", "Neither"], range=["red", "blue", "grey"])),
                tooltip=['Gene', 'Pearson', 'Spearman']
            ).interactive()
            scatter.save(os.path.join(args.output_dir, "shane_cell_type_corr_scatter.html"))
        # rename columns for fisher exact test
        corr_df["Coefficient"] = corr_df["Pearson"] + corr_df["Spearman"] # simple sum
        top_ten_percent_coef = np.percentile(corr_df["Coefficient"], 100 - args.threshold)
        bottom_ten_percent_coef = np.percentile(corr_df["Coefficient"], args.threshold)
        if args.test_mcf_hct:
            compute_number_of_correct_features(corr_df, marker_mcf10a, marker_hct116, args.output_dir, name="correlation",
                                                up_thresh=top_ten_percent_coef, down_thresh=bottom_ten_percent_coef)
        # run enrichr analysis
        enrichr_analysis(corr_df, database=args.gmt_file, output=args.output_dir, name="correlation",
                        up_thresh=top_ten_percent_coef, down_thresh=bottom_ten_percent_coef)
        prerank_gsea(corr_df, database=args.gmt_file, output=args.output_dir, name="correlation")

    if args.run_dim_reduction:
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
        umap_fit = umap.UMAP(n_components=2, random_state=42, n_neighbors=50, min_dist=1, metric='cosine')
        umap_result = umap_fit.fit_transform(pred_L)
        plt.figure(figsize=(12, 6))
        plt.scatter(umap_result[:, 0], umap_result[:, 1], c=gfp_L, cmap="viridis")
        plt.colorbar(label="GFP Level")
        plt.title("UMAP of Gene Expression Profiles")
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.savefig(os.path.join(args.output_dir, "shane_cell_type_umap.png"))
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
        # save the GFP labels of each cell to a csv with node_ids, values
        # get the node ids of each cell
        node_ids = tmc_obj.A.obs["sp_cluster"].values
        cell_info = pd.DataFrame({
            "node_id": node_ids,
            "GFP_level": gfp_L
        })
        # collapse to get the mean GFP level of each node
        cell_info = cell_info.groupby("node_id").mean().reset_index()
        cell_info.to_csv(os.path.join(args.output_dir, "tmc_output", "cell_info.csv"), index=False, header=True)

if __name__ == "__main__":
    main()
