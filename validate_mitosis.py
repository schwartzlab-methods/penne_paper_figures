'''
Code to validate the mitosis levels in the dataset. It:
1. infers the gene expression
2. get gene set expression
3. correlate with mitosis levels
'''
# from dataset import MitoticDataset
from tqdm import tqdm
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
from scipy.stats import pearsonr, zscore
from scipy import signal
from scipy.cluster.hierarchy import linkage, dendrogram
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

def compute_granger_causality(mean_exp, mit_levels, max_lag=5, save=".", name="all_genes"):
    '''
    Compute Granger causality between mean expression and mitosis levels
    '''
    from statsmodels.tsa.stattools import grangercausalitytests
    df_forward = pd.DataFrame({
        "mit_levels": mit_levels.reshape(-1),
        "mean_exp": mean_exp.reshape(-1)
    })
    results = grangercausalitytests(df_forward, maxlag=max_lag, verbose=True)
    # save the results dict along with the direction of correlation to a text file
    with open(os.path.join(save, f"granger_causality_results_gene_cause_green_{name}.txt"), 'w') as f:
        for lag in results.keys():
            corr, p_value = pearsonr(df_forward["mit_levels"].shift(int(lag)).dropna(), df_forward["mean_exp"][:-int(lag)])
            f.write(f"Lag {lag}:\n")
            f.write(str(results[lag][0]))
            f.write(f"Pearson correlation at lag {lag}: r={corr}, p={p_value}\n")
            f.write("\n\n")
    # also compute reverse direction
    df_reverse = pd.DataFrame({
        "mean_exp": mean_exp.reshape(-1),
        "mit_levels": mit_levels.reshape(-1)
    })
    results_reverse = grangercausalitytests(df_reverse, maxlag=max_lag, verbose=True)
    with open(os.path.join(save, f"granger_causality_results_green_cause_gene_{name}.txt"), 'w') as f:
        for lag in results_reverse.keys():
            corr, p_value = pearsonr(df_reverse["mean_exp"].shift(int(lag)).dropna(), df_reverse["mit_levels"][:-int(lag)])
            f.write(f"Lag {lag}:\n")
            f.write(str(results_reverse[lag][0]))
            f.write(f"Pearson correlation at lag {lag}: r={corr}, p={p_value}\n")
            f.write("\n\n")

def compute_difference_test(mean_exp, mit_levels, out, name="all_genes"):
    '''
    Compute first difference sign test between high mitosis and low mitosis groups
    '''
    # do a moving average
    window_size = 10
    mean_exp = np.convolve(mean_exp, np.ones(window_size)/window_size, mode='valid')
    mit_levels = np.convolve(mit_levels, np.ones(window_size)/window_size, mode='valid')
    # compute first difference
    mean_exp_diff = np.diff(mean_exp)
    mit_levels_diff = np.diff(mit_levels)
    # get the sign of the differences
    mean_exp_sign = np.sign(mean_exp_diff)
    mit_levels_sign = np.sign(mit_levels_diff)
    valid_indices = np.where((mean_exp_sign != 0) & (mit_levels_sign != 0))[0]
    matches = (mean_exp_sign[valid_indices] == mit_levels_sign[valid_indices])
    # compute bionomial test
    from scipy.stats import binomtest
    num_matches = np.sum(matches)
    num_trials = len(matches)
    p_value = binomtest(num_matches, num_trials, p=0.5).pvalue
    print(f"Total Moving Avg Time Steps Evaluated: {num_trials}")
    print(f"Number of Matching Directions: {num_matches}")
    print(f"Directional Similarity: {(num_matches/num_trials)*100:.2f}%")
    print("---------------------------------------------------------------------")
    print(f"P-Value: {p_value:.5f}")
    with open(os.path.join(out, f"difference_test_results_{name}.txt"), 'w') as f:
        f.write(f"Total Time Steps Evaluated: {num_trials}\n")
        f.write(f"Number of Matching Directions: {num_matches}\n")
        f.write(f"Directional Similarity: {(num_matches/num_trials)*100:.2f}%\n")
        f.write("---------------------------------------------------------------------\n")
        f.write(f"P-Value: {p_value:.5f}\n")

def compute_correlation(mean_exp: np.ndarray, mit_levels: np.ndarray):
    '''
    Compute Pearson, Spearman correlation between mean expression and mitosis levels
    '''
    print("Shapes of inputs - mean_exp: ", mean_exp.shape, "mit_levels: ", mit_levels.shape)
    pearson_corr, pearson_p = pearsonr(mean_exp.reshape(-1), mit_levels.reshape(-1))
    spearman_corr, spearman_p = spearmanr(mean_exp.reshape(-1), mit_levels.reshape(-1))
    # return linear regression
    model = LinearRegression().fit(mean_exp.reshape(-1, 1), mit_levels.reshape(-1, 1))
    r2 = model.score(mean_exp.reshape(-1, 1), mit_levels.reshape(-1, 1))
    print(f"Pearson correlation: r={pearson_corr}, p={pearson_p}")
    print(f"Spearman correlation: r={spearman_corr}, p={spearman_p}")
    print(f"R2 score: {r2}")
    return pearson_corr, pearson_p, spearman_corr, spearman_p, r2

def validate_biological_relevance(pred_genes, ground_truth_geminin, save, name="all_genes"):
    """
    Args:
        pred_genes: List/Array of predicted gene expression (Blue Line)
        ground_truth_geminin: List/Array of Geminin levels (Orange Line)
    """
    
    # 1. Z-Score Normalization (Fixes the Amplitude Issue)
    z_genes = zscore(pred_genes)
    z_geminin = zscore(ground_truth_geminin)
    
    # 2. Pearson Correlation (Global Trend)
    r_val = np.corrcoef(z_genes, z_geminin)[0, 1]
    print(f"Global Pearson Correlation (r): {r_val:.4f}")
    
    # 3. Cross-Correlation (Finding the Phase Shift)
    # This checks lags from -10 to +10 time points
    lags = np.arange(-10, 11)
    cross_corr = [np.corrcoef(np.roll(z_genes, lag), z_geminin)[0, 1] for lag in lags]
    
    # Find the best lag
    best_lag_idx = np.argmax(np.abs(cross_corr))
    best_lag = lags[best_lag_idx]
    best_r = cross_corr[best_lag_idx]
    
    print(f"Max Correlation of {best_r:.4f} found at Lag {best_lag}")

    # --- PLOTTING ---
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Z-Scored Comparison (The "Truth" about trends)
    # plt.subplot(1, 2, 1)
    plt.plot(z_genes, label=f'Pred: {name} Genes (Z-score)', color='tab:blue')
    plt.plot(z_geminin, label='True: Geminin (Z-score)', color='tab:orange', alpha=0.7)
    plt.title(f"Normalized Trend Comparison\n(Correlation: {r_val:.2f})")
    plt.legend()
    plt.xlabel("Time Points (in 30 Min Intervals)")
    plt.ylabel("Z-Score Normalized Values")
    plt.tight_layout()
    plt.savefig(os.path.join(save, f"{name}_trend_comparison.png"))
    
    # Plot 2: Cross-Correlation
    # plt.subplot(1, 2, 2)
    plt.figure(figsize=(10, 5))
    plt.stem(lags, cross_corr)
    plt.axvline(best_lag, color='r', linestyle='--', alpha=0.5)
    plt.xlabel("Lag (Time Points)")
    plt.ylabel("Correlation Coefficient")
    plt.title("Time-Lagged Cross-Correlation")
    plt.tight_layout()
    plt.savefig(os.path.join(save, f"{name}_cross_correlation.png"))
    plt.close()

    # new plot - expression minus geminin
    plt.figure(figsize=(10, 5))
    plt.plot(z_genes - z_geminin, color='tab:green')
    plt.title(f"Difference between Predicted {name} Gene Expression and Geminin (Z-score)")
    plt.xlabel("Time Points (in 30 Min Intervals)")
    plt.ylabel("Difference in Z-Score Normalized Values")
    plt.tight_layout()
    plt.savefig(os.path.join(save, f"{name}_expression_minus_geminin.png"))
    plt.close()


def per_gene_lag_correlation_heatmap(pred_genes, ground_truth_geminin, gene_names, save, name="all_genes"):
    """
    Compute the lagged correlation between each predicted gene and geminin, and plot a heatmap of the correlation values for each gene at different lags.
    Args:
        pred_genes: Array of shape (num_time_points, num_genes) of predicted gene expression
        ground_truth_geminin: Array of shape (num_time_points,) of Geminin levels
        gene_names: List of gene names corresponding to the columns in pred_genes
    """
    # lags = np.arange(-10, 11)
    # heatmap_data = np.zeros((len(gene_names), len(lags)))
    print("Computing lagged correlation for each gene...")
    lags = signal.correlation_lags(pred_genes.shape[0], ground_truth_geminin.shape[0], mode='full')
    heatmap_data = [(signal.correlate(zscore(pred_genes[:, i].reshape(-1)), zscore(ground_truth_geminin.reshape(-1)), 
                    mode='full') )/ground_truth_geminin.shape[0]
                    for i in range(gene_names.shape[0])]
    heatmap_data = np.array(heatmap_data)
    np.save(os.path.join(save, f"{name}_lagged_correlation_data.npy"), heatmap_data)
    print("Lagged correlation computed for all genes.")

    # remove rows with NaN values
    valid_rows = ~np.isnan(heatmap_data).any(axis=1)
    heatmap_data = heatmap_data[valid_rows, :]
    gene_names = gene_names[valid_rows]
    
    # sort by max correlation value across lags
    peak_lag_indices = np.argmax(heatmap_data, axis=1)
    sort_order = np.argsort(peak_lag_indices)
    sorted_data = heatmap_data[sort_order]
    sorted_genes = gene_names[sort_order]
    # for i in range(pred_genes.shape[1]):
    #     # for j, lag in enumerate(lags):
    #     z_genes = zscore(pred_genes[:, i])
    #     z_geminin = zscore(ground_truth_geminin)
    #     heatmap_data[i, j] = signal.correlate(z_genes, z_geminin, mode='full')[j]
    
    # Plot heatmap and cluster with dendrogram
    plt.figure(figsize=(20, 20))
    sns.heatmap(sorted_data, xticklabels=lags, yticklabels=sorted_genes, cmap='coolwarm', center=0)
    plt.title(f"Lagged Correlation between Predicted {name} Genes and Geminin")
    plt.xlabel("Lag (Time Points)")
    plt.ylabel("Genes")
    plt.tight_layout()
    plt.savefig(os.path.join(save, f"{name}_lagged_correlation_heatmap.png"))
    plt.close()
    # Plot dendrogram
    dendro = linkage(sorted_data, method='ward')
    sns.set(font_scale=0.4)
    plt.figure(figsize=(60, 60))
    # overlay the heatmap with the dendrogram
    cg = sns.clustermap(sorted_data, row_linkage=dendro, col_cluster=False, 
                        yticklabels=sorted_genes, xticklabels=lags, cmap='coolwarm', center=0)
    # dendrogram(dendro, labels=gene_names, orientation='right')
    plt.title("Hierarchical Clustering of Genes based on Lagged Correlation")
    plt.xlabel("Correlation")
    plt.ylabel("Genes")
    plt.tight_layout()
    plt.savefig(os.path.join(save, f"{name}_gene_clustering_dendrogram.png"))
    # save the list of cluster of genes
    from scipy.cluster.hierarchy import fcluster
    cluster_labels = fcluster(dendro, t=2, criterion='maxclust')
    gene_clusters = pd.DataFrame({
        "gene": gene_names,
        "cluster": cluster_labels
    })
    gene_clusters.to_csv(os.path.join(save, f"{name}_gene_clusters.csv"), index=False)
    plt.close()
    

def main():
    parser = argparse.ArgumentParser(description="Validate mitotic levels")
    parser.add_argument("--input_file", type=str, required=True, nargs='+', 
                        help="Path to the input directory with GFP and PCM images")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model")
    parser.add_argument("--spaghetti_model", type=str, default=None,
                        help="Path to the spaghetti model. If None, will not use spaghetti conversion")
    parser.add_argument("--no_spaghetti", action="store_true", help='Whether to not use the Spaghetti model')
    parser.add_argument("--name", type=str, default="all_genes", help="Name to identify the gene set used for validation in the output")
    parser.add_argument("--gene_names", type=str, required=True,
                        help="Path to the gene names file")
    parser.add_argument("--gene_set", type=str, default=None,
                        help="Gene set to see how many genes are the same with marker genes")
    parser.add_argument("--genes_to_use", type=str, default=None,
                        help="Path to the file with genes to use. If no supplied, use all genes")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory")
    parser.add_argument("--plot_line", action="store_true",
                        help="If used, a line plot showing the variation of gene expression of each frame will be generated.")
    parser.add_argument("--test_mode", action="store_true", help="Whether to run in test mode to visually examine green value")
    parser.add_argument("--load_saved_npy", action="store_true", help="Whether to load saved numpy files instead of running inference again")
    parser.add_argument("--scramble", action="store_true", help="Whether to scramble the input images for testing as a baseline.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # prep data
    # dataset = MitoticDataset(args.input_file[0])
    dataset = ShaneSeqCellTypeDataset(args.input_file, load_mitotic=True)
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

    # load marker gene file to be used for a particular pathway / gene
    if args.gene_set:
        if os.path.exists(args.gene_set):
            with open(args.gene_set, 'r') as f:
                file_L = f.readline().strip().split('\t')
            gene_set = set(file_L[2:])  # skip the first two entry which is the gene set name and description
            gene_set_name = file_L[0]  # get the gene set name
        else: # treat as comma separated gene names string entered directly
            gene_set = set(args.gene_set.strip().split(','))
            gene_set_name = args.gene_set
        print(f"Loaded gene set: {gene_set_name} with {len(gene_set)} genes.")
    else:
        gene_set = set(gene_names) # if no specific gene set provided, use all genes
        gene_set_name = "all_genes"
    
    # print("Genes in the gene set: ", list(gene_set))

    try: # load existing inference results if available
        if args.load_saved_npy:
            if args.test_mode:
                print("Running in test mode, will redo inference")
                raise FileNotFoundError
            mitosis_L = np.load(os.path.join(args.output_dir, "mitosis_val.npy"))
            pred_L = np.load(os.path.join(args.output_dir, "mitosis_pred.npy"))
            img_name_L = np.load(os.path.join(args.output_dir, "mitosis_img_names.npy"))
            mitosis_L_unconcat = np.load(os.path.join(args.output_dir, "mitosis_val_unconcat.npy"), allow_pickle=True)
            pred_L_unconcat = np.load(os.path.join(args.output_dir, "mitosis_pred_unconcat.npy"), allow_pickle=True)
            print("Files loaded")
            print("prediction shape: ", pred_L_unconcat.shape)
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
        pred_L_unconcat = []
        mitosis_L_unconcat = []
        img_name_L = []
        if args.test_mode: # store phase images and gfp images
            img_L_unconcat = []
            gfp_L_unconcat = [] 
        with torch.no_grad():
            for img, label in tqdm(loader):
                if "00d" not in label[1][0]: # only process day 0 images
                    print("Sample ", label[1][0], " is not day 0, skipping...")
                    continue
                img = img.to(model.device)
                img = img.squeeze(0) #remove the default batch dimension
                pred = model(img, if_convert=not args.no_spaghetti, scramble=args.scramble) #shape: num_patches, num_genes
                pred_L_unconcat.append(pred.cpu().numpy())
                img_name_L.append(np.array(label[1], dtype=str))
                mitosis_L_unconcat.append(np.array(label[0][0]))
                if args.test_mode:
                    img_L_unconcat.append(img.cpu().numpy())
                    gfp_L_unconcat.append(label[2].numpy().squeeze(0))  # shape: num_patches, 3, H, W
        mitosis_L = np.concatenate(mitosis_L_unconcat, axis=0).reshape(-1) # shape: num_samples
        np.save(os.path.join(args.output_dir, "mitosis_val.npy"), mitosis_L)
        pred_L = np.concatenate(pred_L_unconcat, axis=0) # shape: num_samples, num_genes
        img_name_L = np.concatenate(img_name_L, axis=0).reshape(-1) # shape: num_samples
        # save
        np.save(os.path.join(args.output_dir, "mitosis_pred.npy"), pred_L)
        np.save(os.path.join(args.output_dir, "mitosis_img_names.npy"), img_name_L)
        np.save(os.path.join(args.output_dir, "mitosis_val_unconcat.npy"), mitosis_L_unconcat)
        np.save(os.path.join(args.output_dir, "mitosis_pred_unconcat.npy"), pred_L_unconcat)
        if args.test_mode:
            # sample high and low mitosis images
            img_L = np.concatenate(img_L_unconcat, axis=0)  # shape: num_samples x num_patches, 1, H, W
            gfp_L = np.concatenate(gfp_L_unconcat, axis=0)  # shape: num_samples x num_patches, 3, H, W

    # get subsect of genes that are in the gene set
    gene_set_indices = [i for i, g in enumerate(gene_names) if g in gene_set]
    num_final_genes = len(gene_set_indices)
    if num_final_genes == 0:
        raise ValueError("No genes from the gene set found in the predicted gene list.")
    pred_L = pred_L[:, gene_set_indices].reshape(pred_L.shape[0], num_final_genes)
    pred_L_unconcat = [each[:, gene_set_indices].reshape(each.shape[0], num_final_genes) for each in pred_L_unconcat]
    gene_names = gene_names[gene_set_indices]
    print(f"Using {len(gene_names)} genes from the gene set for analysis.")
    
    # filter based on the high confidence genes only
    genes_to_use = np.loadtxt(args.genes_to_use, dtype=str)
    mask = np.isin(gene_names, genes_to_use)
    print(f"Using {np.sum(mask)} genes from the provided gene list for analysis.")
    pred_L = pred_L[:, mask]
    gene_names = gene_names[mask]
    pred_L_unconcat = [each[:, mask] for each in pred_L_unconcat]

    if not args.gene_set: # if no specific gene set is provided, do correlation analysis for all genes and plot the distribution of correlation values
        # compute the correlation of each gene with the mitosis levels
        gene_corrs = {
            "gene": [],
            "spearman_corr": [],
            "spearman_p": []
        }
        for i in range(pred_L.shape[1]):
            corr, p_value = spearmanr(pred_L[:, i], mitosis_L)
            gene_corrs["gene"].append(gene_names[i])
            gene_corrs["spearman_corr"].append(corr)
            gene_corrs["spearman_p"].append(p_value)
        gene_corrs_df = pd.DataFrame(gene_corrs)
        gene_corrs_df.to_csv(os.path.join(args.output_dir, "gene_correlation_metrics.csv"), index=False)
        print("Correlation metrics for each gene saved to ", os.path.join(args.output_dir, "gene_correlation_metrics.csv"))
        # plot the distribution of correlation values
        plt.figure(figsize=(8, 5))
        sns.histplot(gene_corrs_df["spearman_corr"], bins=50, kde=True)
        plt.title("Distribution of Spearman Correlation between Predicted Gene Expression and Mitosis Levels")
        plt.xlabel("Spearman Correlation")
        plt.ylabel("Count")
        plt.savefig(os.path.join(args.output_dir, "gene_correlation_distribution_all_genes.png"))
        plt.close()
        # print("Analysis of all high confidence genes completed. If you want to focus on a subset of genes, please provide a gene list using the --genes_set argument. Exiting for now.")
        # return 0
        
    print("Final number of genes used for correlation analysis: ", len(gene_names))
    print("Final shape of predicted expression: ", np.array(pred_L_unconcat).shape)
    print("Final mean expression values: ", np.mean(pred_L))

    # get only day 0 and day 1 samples
    num_patches_per_file = len(mitosis_L) // len(img_name_L)
    mean_exp_unconcat = [np.mean(each, axis=1) for each in pred_L_unconcat] # len: num_batches * num_time, each shape: num_patches, 1
    mean_exp_per_frame = {}
    mitosis_per_frame = {}
    exp_per_frame = {}
    for batch_num in np.unique([name.split('_')[2] for name in img_name_L]):
        # get all the indices for this batch of all time points
        batch_indices = [i for i, name in enumerate(img_name_L) if int(batch_num) == int(name.split('_')[2])]
        mitosis_batch = [mitosis_L_unconcat[i].reshape(-1) for i in batch_indices] # each shape: num_patches
        pred_batch = [mean_exp_unconcat[i] for i in batch_indices] # each shape: num_patches, 1
        pred_bath_unaverage = [pred_L_unconcat[i] for i in batch_indices] # each shape: num_patches, num_genes
        img_batch = [img_name_L[i] for i in batch_indices]
        # get all the time point names for this batch
        time_points = sorted(list(set([img_name_L[i].split('_')[3] for i in batch_indices])))
        #! remove anything over 00d
        time_points = [t for t in time_points if (('00d' in t))]# or ('01d' in t))]# or ('02d' in t))]
        # compute the mean expression and mitosis level for each time point for each patch in this batch
        for patches in range(num_patches_per_file):
            frame_mean_exp = []
            frame_mitosis = []
            frame_exp = []
            for t in time_points:
                # get the index of this time point in the batch_indices
                time_point_index = [i for i in range(len(img_batch)) if img_batch[i].split('_')[3] == t][0]
                frame_mean_exp.append(pred_batch[time_point_index][patches])
                frame_mitosis.append(mitosis_batch[time_point_index][patches])
                frame_exp.append(pred_bath_unaverage[time_point_index][patches])
            mean_exp_per_frame[f"batch_{batch_num}_patch_{patches}"] = frame_mean_exp
            mitosis_per_frame[f"batch_{batch_num}_patch_{patches}"] = frame_mitosis
            exp_per_frame[f"batch_{batch_num}_patch_{patches}"] = frame_exp # shape: num_time_points, num_genes
    all_keys = list(set(mean_exp_per_frame.keys()))

    if args.test_mode:
        # filter image names such that only day 0 images are kept
        new_mitosis_L = np.concatenate([mitosis_L_unconcat[i] for i, name in enumerate(img_name_L) if '00d' in name], axis=0).reshape(-1)
        new_img_L = np.concatenate([img_L_unconcat[i] for i, name in enumerate(img_name_L) if '00d' in name], axis=0) # shape: num_samples x num_patches, 1, H, W
        new_gfp_L = np.concatenate([gfp_L_unconcat[i] for i, name in enumerate(img_name_L) if '00d' in name], axis=0) # shape: num_samples x num_patches, 3, H, W
        print("Generating test mode mitosis images...")
        os.makedirs(os.path.join(args.output_dir, "test"), exist_ok=True)
        high_mitosis_indices = np.argsort(new_mitosis_L)[-10:]
        low_mitosis_indices = np.argsort(new_mitosis_L)[:10]
        sampled_indices = list(chain(high_mitosis_indices, low_mitosis_indices))
        sampled_imgs = [new_img_L[i] for i in sampled_indices]
        sampled_mitosis = [new_mitosis_L[i] for i in sampled_indices]
        sampled_gfp = [new_gfp_L[i] for i in sampled_indices]
        for i, idx in tqdm(enumerate(sampled_indices)):
            img = sampled_imgs[i] # shape: 3, H, W
            gfp = sampled_gfp[i] # shape: 3, H, W
            # change to H W 3 for plotting
            img = np.transpose(img, (1, 2, 0))
            gfp = np.transpose(gfp, (1, 2, 0))
            mitosis_level = sampled_mitosis[i]
            fig, axs = plt.subplots(1, 2, figsize=(8, 4))
            axs[0].imshow(img, cmap='gray')
            axs[0].set_title(f"PCM Image\nMitosis Level: {mitosis_level:.2f}")
            axs[0].axis('off')
            # show the entire gfp image with colour
            axs[1].imshow(gfp, cmap='viridis')
            axs[1].set_title("GFP Image")
            axs[1].axis('off')
            plt.savefig(os.path.join(args.output_dir, "test", f"mitosis_test_img_{i}_idx_{idx}.png"))
            plt.close()

    # correlation analysis
    pearson_corr, pearson_p, spearman_corr, spearman_p, r2 = compute_correlation(
        mean_exp = np.array([mean_exp_per_frame[key] for key in all_keys for _ in range(len(mean_exp_per_frame[key]))]),
        mit_levels = np.array([mitosis_per_frame[key] for key in all_keys for _ in range(len(mitosis_per_frame[key]))]),
        # mean_exp = np.mean(pred_L, axis=1),
        # mit_levels = mitosis_L
    )

    # plot scatter plot of mean expression vs mitosis levels
    mean_exp = np.mean(pred_L, axis=1)
    plt.figure(figsize=(8, 6))
    plt.scatter(mean_exp, mitosis_L, alpha=0.5)
    plt.xlabel("Mean Gene Expression")
    plt.ylabel("Image Geminin Green Intensity")
    plt.title(f"Mean Gene Expression vs Geminin Green Intensity\nPearson r={pearson_corr:.2f} (p={pearson_p:.2e}), Spearman r={spearman_corr:.2f} (p={spearman_p:.2e}), R2={r2:.2f}")
    plt.savefig(os.path.join(args.output_dir, f"mean_exp_vs_mitosis_{gene_set_name}.png"))
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
    chart.save(os.path.join(args.output_dir, f"mean_exp_vs_mitosis_altair_{gene_set_name}.html"))

    # plot a line plot of two lines showing the variation of gene expression and mitosis levels of each frame
    if args.plot_line:
        # plot line plot for each patch in each batch
        for key in tqdm(mean_exp_per_frame.keys()):
            time_points = list(range(len(mean_exp_per_frame[key])))
            plt.figure(figsize=(10, 6))
            plt.plot(time_points, mean_exp_per_frame[key], label="Mean Gene Expression")
            plt.plot(time_points, np.log(np.array(mitosis_per_frame[key]) + 1e-9), label="Log Geminin Green Levels")
            # plt.plot(time_points, mitosis_per_frame[key], label="Mitosis Levels")
            plt.xlabel("Number of 30 Minute Time Points")
            plt.ylabel("Value")
            plt.title(f"Variation of Gene Expression of {gene_set_name} and Geminin Green Levels over Time\n{key}")
            plt.legend()
            plt.savefig(os.path.join(args.output_dir, f"line_plot_{key}_{gene_set_name}.png"))
            plt.close()
            # plot with altair
            df_line = pd.DataFrame({
                "Time_Point": time_points,
                "Mean_Gene_Expression": mean_exp_per_frame[key],
                "Mitosis_Levels": mitosis_per_frame[key],
            })
            df_line_melted = df_line.melt(id_vars=["Time_Point"], value_vars=["Mean_Gene_Expression", "Mitosis_Levels"],
                                          var_name="Metric", value_name="Value")
            chart_line = alt.Chart(df_line_melted).mark_line(point=True).encode(
                x=alt.X('Time_Point:N', title='Number of 30 Minute Time Points'),
                y=alt.Y('Value:Q', title='Value'),
                color='Metric:N'
            ).interactive()
            chart_line.save(os.path.join(args.output_dir, f"line_plot_{key}_altair_{gene_set_name}.html"))
        # average across ALL patches for all the time points
        time_points = list(range(len(time_points)))
        keys = list(mean_exp_per_frame.keys())
        mean_exp_all = [mean_exp_per_frame[key] for key in keys]
        mitosis_all = [mitosis_per_frame[key] for key in keys]
        mean_exp_all = np.mean(np.array(mean_exp_all), axis=0)
        mitosis_all = np.mean(np.array(mitosis_all), axis=0)
        exp_all = np.mean(np.array([exp_per_frame[key] for key in keys]), axis=0) # shape: num_time_points, num_genes
        # run stats test
        compute_difference_test(mean_exp_all, mitosis_all, args.output_dir, name=gene_set_name)
        compute_granger_causality(mean_exp_all, mitosis_all, max_lag=10, save=args.output_dir, name=gene_set_name)
        if len(gene_names) > 1:
            per_gene_lag_correlation_heatmap(exp_all, mitosis_all, gene_names, save=args.output_dir, name=gene_set_name)

        # plot at different scale to see trend
        plt.figure(figsize=(10, 6))
        plt.plot(time_points, mean_exp_all, label="Mean Gene Expression")
        plt.plot(time_points, np.log(mitosis_all + 1e-9), label="Log Geminin Green Levels")
        # plt.plot(time_points, mitosis_all, label="Mitosis Levels")
        plt.xlabel("Number of 30 Minute Time Points")
        plt.ylabel("Value")
        plt.title(f"Variation of Gene Expression of {gene_set_name} and Geminin Green Levels over Time\n(Average across all patches)")
        plt.legend()
        plt.savefig(os.path.join(args.output_dir, f"line_plot_all_patches_avg_{gene_set_name}.png"))
        plt.close()
        # altair
        df_line = pd.DataFrame({
            "Time_Point": time_points,
            "Mean_Gene_Expression": mean_exp_all,
            "Mitosis_Levels": mitosis_all,
        })
        df_line_melted = df_line.melt(id_vars=["Time_Point"], value_vars=["Mean_Gene_Expression", "Mitosis_Levels"],
                                      var_name="Metric", value_name="Value")
        chart_line = alt.Chart(df_line_melted).mark_line(point=True).encode(
            x=alt.X('Time_Point:N', title='Number of 30 Minute Time Points'),
            y=alt.Y('Value:Q', title='Value'),
            color='Metric:N'
        ).interactive()
        chart_line.save(os.path.join(args.output_dir, f"line_plot_all_patches_avg_altair_{gene_set_name}.html"))

        # plot for all patches in ONE batch together by taking mean across patches
        # first get the unique batch numbers
        # get unique batch numbers
        unique_batches = set([key.split('_')[1] for key in mean_exp_per_frame.keys()])
        for batch_num in unique_batches:
            time_points = list(range(len(time_points)))
            batch_mean_exp = []
            batch_mitosis = []
            for key in mean_exp_per_frame.keys():
                if key.startswith(f"batch_{batch_num}_"):
                    batch_mean_exp.append(mean_exp_per_frame[key])
                    batch_mitosis.append(mitosis_per_frame[key])
            batch_mean_exp = np.mean(np.array(batch_mean_exp), axis=0)
            batch_mitosis = np.mean(np.array(batch_mitosis), axis=0)
            plt.figure(figsize=(10, 6))
            plt.plot(time_points, batch_mean_exp, label="Mean Gene Expression")
            plt.plot(time_points, np.log(batch_mitosis + 1e-9), label="Log Geminin Green Levels")
            # plt.plot(time_points, batch_mitosis, label="Mitosis Levels")
            plt.xlabel("Number of 30 Minute Time Points")
            plt.ylabel("Value")
            plt.title(f"Variation of Gene Expression of {gene_set_name} and Geminin Green Intensity over Time\nBatch {batch_num} (Mean across patches)")
            plt.legend()
            plt.savefig(os.path.join(args.output_dir, f"line_plot_batch_{batch_num}_mean_patches_{gene_set_name}.png"))
            plt.close()
            # altair
            df_line = pd.DataFrame({
                "Time_Point": time_points,
                "Mean_Gene_Expression": batch_mean_exp,
                "Mitosis_Levels": batch_mitosis,
            })
            df_line_melted = df_line.melt(id_vars=["Time_Point"], value_vars=["Mean_Gene_Expression", "Mitosis_Levels"],
                                          var_name="Metric", value_name="Value")
            chart_line = alt.Chart(df_line_melted).mark_line(point=True).encode(
                x=alt.X('Time_Point:N', title='Number of 30 Minute Time Points'),
                y=alt.Y('Value:Q', title='Value'),
                color='Metric:N'
            ).interactive()
            chart_line.save(os.path.join(args.output_dir, f"line_plot_batch_{batch_num}_mean_patches_altair_{gene_set_name}.html"))
        
        # biological relevance validation
        validate_biological_relevance(
            pred_genes = mean_exp_all,
            ground_truth_geminin = mitosis_all,
            save = args.output_dir,
            name = f"All_Patches_Avg_{gene_set_name}"
        )


    # compute PCA according to gene expression profiles
    if len(gene_names) > 1 and False: #! for now, disable PCA and UMAP since with the small number of genes in the gene set, the results are not very informative. We can enable it later if we want to visually inspect the clustering of samples based on gene expression profiles.
        print("Running PCA")
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(pred_L)
        plt.figure(figsize=(12, 6))
        plt.scatter(pca_result[:, 0], pca_result[:, 1], c=mitosis_L, cmap="viridis")
        plt.colorbar(label="Mitosis Level")
        plt.title("PCA of Gene Expression Profiles")
        plt.xlabel(f"PC 1 (Variance Explained: {pca.explained_variance_ratio_[0]:.2f})")
        plt.ylabel(f"PC 2 (Variance Explained: {pca.explained_variance_ratio_[1]:.2f})")
        plt.savefig(os.path.join(args.output_dir, f"mitosis_pca_{gene_set_name}.png"))
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
        plt.savefig(os.path.join(args.output_dir, f"mitosis_umap_{gene_set_name}.png"))
        plt.close()

        # Run TMC
        print("Running TMC")
        adata = ad.AnnData(pred_L)
        adata.obs["node_id"] = [str(i) for i in range(pred_L.shape[0])]
        tmc_obj = tmc(adata, os.path.join(args.output_dir, f"tmc_output_{gene_set_name}"))
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
        cell_info.to_csv(os.path.join(args.output_dir, f"tmc_output_{gene_set_name}", "cell_info.csv"), index=False, header=True)

if __name__ == "__main__":
    main()
