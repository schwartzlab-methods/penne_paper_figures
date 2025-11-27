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
    print(f"Pearson correlation: r={pearson_corr}, p={pearson_p}")
    print(f"Spearman correlation: r={spearman_corr}, p={spearman_p}")
    print(f"R2 score: {r2}")
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
    parser.add_argument("--plot_line", action="store_true",
                        help="If used, a line plot showing the variation of gene expression of each frame will be generated.")
    parser.add_argument("--test_mode", action="store_true", help="Whether to run in test mode to visually examine green value")
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
        mitosis_L = np.load(os.path.join(args.output_dir, "mitosis_val.npy"))
        if args.test_mode:
            img_L = np.load(os.path.join(args.output_dir, "mitosis_test_imgs.npy"))
            gfp_L = np.load(os.path.join(args.output_dir, "mitosis_test_gfp.npy"))
            print("Loaded existing test mode mitosis images and GFP images.")
        else:
            pred_L = np.load(os.path.join(args.output_dir, "mitosis_pred.npy"))
            img_name_L = np.load(os.path.join(args.output_dir, "mitosis_img_names.npy"))
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
            img_L = []
            gfp_L = [] 
        with torch.no_grad():
            for img, label in tqdm(loader):
                img = img.to(model.device)
                img = img.squeeze(0) #remove the default batch dimension
                pred = model(img, if_convert=not args.no_spaghetti) #shape: num_patches, num_genes
                pred_L_unconcat.append(pred.cpu().numpy())
                img_name_L.append(np.array(label[1], dtype=str))
                mitosis_L_unconcat.append(np.array(label[0][0]))
                if args.test_mode:
                    img_L.append(img.cpu().numpy())
                    gfp_L.append(label[2].numpy().squeeze(0))  # shape: num_patches, 3, H, W
        mitosis_L = np.concatenate(mitosis_L_unconcat, axis=0).reshape(-1) # shape: num_samples
        np.save(os.path.join(args.output_dir, "mitosis_val.npy"), mitosis_L)
        if not args.test_mode:
            pred_L = np.concatenate(pred_L_unconcat, axis=0) # shape: num_samples, num_genes
            img_name_L = np.concatenate(img_name_L, axis=0).reshape(-1) # shape: num_samples
            # save
            np.save(os.path.join(args.output_dir, "mitosis_pred.npy"), pred_L)
            np.save(os.path.join(args.output_dir, "mitosis_img_names.npy"), img_name_L)
        if args.test_mode:
            # sample high and low mitosis images and save
            img_L = np.concatenate(img_L, axis=0)  # shape: num_samples, num_patches, 1, H, W
            gfp_L = np.concatenate(gfp_L, axis=0)  # shape: num_samples, num_patches, 1, H, W
    if args.test_mode:
        print("Generating test mode mitosis images...")
        os.makedirs(os.path.join(args.output_dir, "test"), exist_ok=True)
        high_mitosis_indices = np.argsort(mitosis_L)[-10:]
        low_mitosis_indices = np.argsort(mitosis_L)[:10]
        sampled_indices = list(chain(high_mitosis_indices, low_mitosis_indices))
        sampled_imgs = [img_L[i] for i in sampled_indices]
        sampled_mitosis = [mitosis_L[i] for i in sampled_indices]
        sampled_gfp = [gfp_L[i] for i in sampled_indices]
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
            plt.savefig(os.path.join(args.output_dir, f"mitosis_test_img_{i}_idx_{idx}.png"))
            plt.close()

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
        pred_L_unconcat = [each[:, mask] for each in pred_L_unconcat]
    
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

    # plot a line plot of two lines showing the variation of gene expression and mitosis levels of each frame
    if args.plot_line:
        num_patches_per_file = len(mitosis_L) // len(img_name_L)
        mean_exp_unconcat = [np.mean(each, axis=1) for each in pred_L_unconcat] # len: num_batches * num_time, each shape: num_patches, 1
        mean_exp_per_frame = {}
        mitosis_per_frame = {}
        for batch_num in np.unique([name.split('_')[2] for name in img_name_L]):
            # get all the indices for this batch of all time points
            batch_indices = [i for i, name in enumerate(img_name_L) if int(batch_num) == int(name.split('_')[2])]
            mitosis_batch = [mitosis_L_unconcat[i].reshape(-1) for i in batch_indices] # each shape: num_patches
            pred_batch = [mean_exp_unconcat[i] for i in batch_indices] # each shape: num_patches, 1
            img_batch = [img_name_L[i] for i in batch_indices]
            print("Pred", len(pred_batch), "Mitosis", len(mitosis_batch), "Img", len(img_batch))
            print("For each:", pred_batch[0].shape, mitosis_batch[0].shape)
            # get all the time point names for this batch
            time_points = sorted(list(set([img_name_L[i].split('_')[3] for i in batch_indices])))
            # compute the mean expression and mitosis level for each time point for each patch in this batch
            for patches in range(num_patches_per_file):
                frame_mean_exp = []
                frame_mitosis = []
                for t in time_points:
                    # get the index of this time point in the batch_indices
                    time_point_index = [i for i in range(len(img_batch)) if img_batch[i].split('_')[3] == t][0]
                    frame_mean_exp.append(pred_batch[time_point_index][patches])
                    frame_mitosis.append(mitosis_batch[time_point_index][patches])
                mean_exp_per_frame[f"batch_{batch_num}_patch_{patches}"] = frame_mean_exp
                mitosis_per_frame[f"batch_{batch_num}_patch_{patches}"] = frame_mitosis
        # plot line plot for each patch in each batch
        for key in mean_exp_per_frame.keys():
            time_points = list(range(len(mean_exp_per_frame[key])))
            plt.figure(figsize=(10, 6))
            plt.plot(time_points, mean_exp_per_frame[key], label="Mean Gene Expression")
            plt.plot(time_points, mitosis_per_frame[key], label="Mitosis Levels")
            plt.xlabel("Number of 30 Minute Time Points")
            plt.ylabel("Value")
            plt.title(f"Variation of Gene Expression and Mitosis Levels over Time\n{key}")
            plt.legend()
            plt.savefig(os.path.join(args.output_dir, f"line_plot_{key}.png"))
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
            chart_line.save(os.path.join(args.output_dir, f"line_plot_{key}_altair.html"))
        # plot for all patches in one batch together by taking mean across patches
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
            plt.plot(time_points, batch_mitosis, label="Mitosis Levels")
            plt.xlabel("Number of 30 Minute Time Points")
            plt.ylabel("Value")
            plt.title(f"Variation of Gene Expression and Mitosis Levels over Time\nBatch {batch_num} (Mean across patches)")
            plt.legend()
            plt.savefig(os.path.join(args.output_dir, f"line_plot_batch_{batch_num}_mean_patches.png"))
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
            chart_line.save(os.path.join(args.output_dir, f"line_plot_batch_{batch_num}_mean_patches_altair.html"))

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
