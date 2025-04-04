'''
Validate the gene expression predictor
'''
import torch
import os
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from scipy.cluster import hierarchy
import altair as alt
from torch.utils.data import random_split, DataLoader
import pytorch_lightning as pl
from modules import SpaghettiGenerator
from model import GeneExpPredVisiumHD
from dataset import VisiumHD_Livecell_Dataset
from transformers import AutoImageProcessor, AutoModel
import argparse
from _feature_extractors import owkin_features, spaghetti_convertion

def init_spaghetti(model_path: str) -> torch.nn.Module:
    '''
    Initialize the SPAGHETTI model for image translation
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = SpaghettiGenerator(3, 9)
    generator.to(device)
    ckpt = torch.load(model_path, map_location=device)["state_dict"]
    # get only G_AB weights
    ckpt = {k[5:]: v for k, v in ckpt.items() if ("G_AB" in k)}
    generator.load_state_dict(ckpt)
    return generator

def plot_heatmap(save_dr, expression_gt, matched_spot_expression_pred, top_k=50,exp="heatmap.png"):
    #plot heatmap of top k genes ranked by mean
    #take mean of expression
    mean = np.mean(expression_gt, axis=1)
    #take ind of top k 
    ind = np.argpartition(mean, -top_k)[-top_k:]

    # Compute the correlation matrix
    corr_matrix = np.corrcoef(expression_gt[ind,:],matched_spot_expression_pred[ind,:])
    dendrogram = hierarchy.dendrogram(hierarchy.linkage(corr_matrix, method='ward'), no_plot=True)
    cluster_idx = dendrogram['leaves']

    # corr_matrix = np.corrcoef(matched_spot_expression_pred[ind,:])
    corr_matrix = corr_matrix[cluster_idx, :]
    corr_matrix = corr_matrix[:, cluster_idx]

    # Reorder the correlation matrix and plot the heatmap
    plt.figure(dpi=300, figsize=(5,5))
    sns.heatmap(corr_matrix, cmap='viridis', xticklabels=False, yticklabels=False, cbar= True, vmin=-1, vmax=1)
    plt.title(f"Top {top_k} genes ranked by mean expression")
    plt.xlabel("Genes")
    plt.ylabel("Genes")
    plt.savefig(os.path.join(save_dr,exp), dpi=300, bbox_inches='tight')
    print("Finished plotting heatmap")

def compute_corr(save_dir, name, expression_gt, matched_spot_expression_pred, top_k=50, qc_idx=None):
    #cells are in columns, genes are in rows
    if qc_idx is not None:
        expression_gt = expression_gt[:,qc_idx]
        matched_spot_expression_pred = matched_spot_expression_pred[:,qc_idx]
    mean = np.mean(expression_gt, axis=1)
    ind = np.argpartition(mean, -top_k)[-top_k:]
    corr = np.zeros(top_k)
    for i in range(top_k):
        corr[i] = np.corrcoef(expression_gt[ind[i],:], matched_spot_expression_pred[ind[i],:])[0,1]
    plt.figure(dpi=300, figsize=(5,5))
    plt.hist(corr, bins=20)
    plt.title(f"Correlation of top {top_k} genes ranked by mean expression")
    plt.xlabel("Correlation")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(save_dir,f"corr_hist_overall_{name}.png"), dpi=300, bbox_inches='tight')
    return np.mean(corr)

# def convert_pred(pred: np.ndarray, true: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
#     # convert the prediction to the same format as the ground truth
#     # works by selecting the same genes as the ground truth
#     features_true = pd.read_csv(CFG.visium_features, sep="\t", header = None)[0].values
#     features_pred = pd.read_csv(os.path.join(CFG.data_dir, 'dataset/features/GSM7697870_C73C1_features.tsv'), sep="\t", header = None)[0].values
#     # get the common genes
#     common_genes = np.intersect1d(features_true, features_pred)
#     # get the indices of the common genes
#     common_genes_idx = [np.where(features_pred == gene)[0][0] for gene in common_genes]
#     # select the common genes from the prediction
#     pred = pred[:, common_genes_idx]
#     features_pred = features_pred[common_genes_idx]
#     # change the order of the genes to match the ground truth, by sorting the genes
#     true = true[:, np.argsort(features_true)]
#     pred = pred[:, np.argsort(features_pred)]
#     return pred, true

def main():
     # seeds for reproducibility
    torch.manual_seed(42)
    pl.seed_everything(42, workers=True)
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--visiumhd_dir', type=str, help='Directory containing the VisiumHD patches')
    parser.add_argument('--livecell_dir', type=str, help='Directory containing the LIVECell patches')
    parser.add_argument('--mtx_dir', type=str, help='Directory containing the mtx files')
    parser.add_argument('--model_dir', type=str, help='Directory containing the model checkpoints')
    parser.add_argument('--gene_names', type=str, help='Path to the gene names feature tsv file')
    parser.add_argument('--spaghetti_model', type=str, help='Path to the Spaghetti model')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--name', type=str, default="gene_predictor", help='Name of the model for logging')
    args = parser.parse_args()
    # create dataset
    dataset = VisiumHD_Livecell_Dataset(args.visiumhd_dir, args.mtx_dir, args.livecell_dir)
    # split dataset into train and val
    _, val_dataset = random_split(dataset, [0.8, 0.2])
    # create dataloaders
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    # create feature extractor
    feature_extractor = AutoModel.from_pretrained("/fs01/home/richarddong/.cache/huggingface/hub/phikon-v2")
    image_processor = AutoImageProcessor.from_pretrained("owkin/phikon-v2")
    model = GeneExpPredVisiumHD.load_from_checkpoint(args.model_dir, num_genes = dataset.num_genes, 
                                converter = lambda device, x: spaghetti_convertion(init_spaghetti(args.spaghetti_model), device, x), 
                                feature_extractor = lambda device, x: owkin_features(feature_extractor, device, image_processor, x))
    model.freeze()
    # inference
    pred_L = []
    gt_L = []
    features_he_L = []
    features_pcm_L = []
    features_pcm_non_translate_L = []
    features_he_non_translated_L = []
    with torch.no_grad():
        for batch in tqdm(val_loader, total=len(val_loader)):
            he_image, mtx, pcm = batch
            pred_exp = model.forward(he_image, if_convert=False)
            # compute some features
            features_he = model.compute_feature(he_image, if_convert=False, if_translate=True)
            features_he_non_translated = model.compute_feature(he_image, if_convert=False, if_translate=False)
            features_pcm = model.compute_feature(pcm, if_convert=True, if_translate=True)
            features_pcm_non_translate = model.compute_feature(pcm, if_convert=True, if_translate=False)
            # remove all negative values
            pred_exp[pred_exp < 0] = 0
            pred_L.append(pred_exp.cpu().numpy())
            gt_L.append(mtx.view.cpu().numpy())
            features_he_L.append(features_he.cpu().numpy())
            features_pcm_L.append(features_pcm.cpu().numpy())
            features_pcm_non_translate_L.append(features_pcm_non_translate.cpu().numpy())
            features_he_non_translated_L.append(features_he_non_translated.cpu().numpy())
    pred = np.concatenate(pred_L, axis=0)
    true = np.concatenate(gt_L, axis=0)
    he_features_translated = np.concatenate(features_he_L, axis=0)
    he_features_non_translated = np.concatenate(features_he_non_translated_L, axis=0)
    pcm_features_translated = np.concatenate(features_pcm_L, axis=0)
    pcm_features_non_translated = np.concatenate(features_pcm_non_translate_L, axis=0)
    print(f"Final prediction shape: {pred.shape}") # spots x features
    print("Finished generating predictions")

    # save numpys
    np.save(os.path.join(args.output_dir, "pred.npy"), pred)
    np.save(os.path.join(args.output_dir, "true.npy"), true)
    np.save(os.path.join(args.output_dir, "he_features_translated.npy"), he_features_translated)
    np.save(os.path.join(args.output_dir, "he_features_non_translated.npy"), he_features_non_translated)
    np.save(os.path.join(args.output_dir, "pcm_features_translated.npy"), pcm_features_translated)
    np.save(os.path.join(args.output_dir, "pcm_features_non_translated.npy"), pcm_features_non_translated)

    #! across spots correlation
    corr = np.zeros(pred.shape[0])
    for i in range(pred.shape[0]):
        corr[i] = np.corrcoef(pred[i,:], true[i,:],)[0,1] #corrcoef returns a matrix
    #remove nan
    corr_spots = corr[~np.isnan(corr)]
    # plot histogram
    chart = alt.Chart(pd.DataFrame(corr_spots, columns=["correlation"])).mark_bar().encode(
        alt.X("correlation", bin=alt.Bin(maxbins=20)),
        y='count()',
    ).properties(
        title="Correlation of predicted vs ground truth expression across spots",
        width=300,
        height=150
    ).interactive()
    chart.save(os.path.join(args.output_dir, f"correlation_spots_hist_{args.name}.html"))

    #! across genes correlation
    corr = np.zeros(pred.shape[1])
    for i in range(pred.shape[1]):
        corr[i] = np.corrcoef(pred[:,i], true[:,i],)[0,1] #corrcoef returns a matrix
    #remove nan
    corr_genes = corr[~np.isnan(corr)]
    # plot histogram
    chart = alt.Chart(pd.DataFrame(corr_genes, columns=["correlation"])).mark_bar().encode(
        alt.X("correlation", bin=alt.Bin(maxbins=20)),
        y='count()',
    ).properties(
        title="Correlation of predicted vs ground truth expression across genes",
        width=300,
        height=150
    ).interactive()
    chart.save(os.path.join(args.output_dir, f"correlation_hist_genes_{args.name}.html"))

    # gene analysis
    gene_names = pd.read_csv(args.gene_names, sep="\t", header = None)[0].values
    adata_raw = sc.AnnData(true)
    adata_raw.var_names = pd.read_csv(gene_names, sep="\t", header = None)[0].values
    print(f"Shape of raw adata: {adata_raw.shape}")

    adata_pred = sc.AnnData(pred)
    adata_pred.var_names = pd.read_csv(gene_names, sep="\t", header = None)[0].values
    print(f"Shape of predicted adata: {adata_pred.shape}")

    # compute gene lists
    # normalize counts matrix so that each 'cell' (barcode) has counts summing to 1
    adata_pred.X_norm = sc.pp.normalize_total(adata_pred, target_sum=1, inplace=False)['X']
    adata_raw.X_norm = sc.pp.normalize_total(adata_raw, target_sum=1, inplace=False)['X']

    # create new adata.var column contaning mean of each column of adata.X_norm above
    # this is total normalized counts per gene a.k.a. 'mean_total_expression'
    adata_pred.var['mean_expression'] = np.ravel(adata_pred.X_norm.mean(0))
    adata_raw.var['mean_expression'] = np.ravel(adata_raw.X_norm.mean(0))

    # compute highly expressed genes
    # return pd.DataFrame of n top-ranked genes by mean expression
    n = 500
    most_expressed_pred = pd.DataFrame(adata_pred.var.nlargest(n, 'mean_expression')['mean_expression'])
    most_expressed_raw = pd.DataFrame(adata_raw.var.nlargest(n, 'mean_expression')['mean_expression'])
    total_most_exp = pd.concat([most_expressed_pred, most_expressed_raw], axis=1, join='outer')
    total_most_exp.to_csv(os.path.join(args.output_dir, f'most_expressed_genes_{args.name}.csv'), header=['predicted', 'raw'])

    sc.pp.normalize_total(adata_raw)
    sc.pp.normalize_total(adata_pred)
    sc.pp.log1p(adata_raw)
    sc.pp.log1p(adata_pred)

    sc.pp.highly_variable_genes(adata_pred, n_top_genes=n)
    sc.pp.highly_variable_genes(adata_raw, n_top_genes=n)

    highly_variable_pred = adata_pred[:,adata_pred.var['highly_variable']==True].to_df()
    highly_variable_raw = adata_raw[:,adata_raw.var['highly_variable']==True].to_df()
    total_highly_variable = pd.concat([highly_variable_pred, highly_variable_raw], axis=1, join='outer')
    total_highly_variable.to_csv(os.path.join(args.output_dir, f'highly_variable_genes_{args.name}.csv'))

    # plot correlation heatmap of top 50 highly variable genes for each
    # get expression matrix for top 50 highly variable genes
    hv_genes = adata_pred.var_names[adata_pred.var['highly_variable']==True]
    hv_genes = hv_genes[:50]
    hv_genes_idx = [adata_pred.var_names.get_loc(gene) for gene in hv_genes]
    plot_heatmap(args.output_dir, adata_pred[:,hv_genes].X.T, adata_pred[:,hv_genes].X.T, top_k=50, exp=f"hv_genes_pred_heatmap_{args.name}.png")

    hv_genes = adata_raw.var_names[adata_raw.var['highly_variable']==True]
    hv_genes = hv_genes[:50]
    hv_genes_idx = [adata_raw.var_names.get_loc(gene) for gene in hv_genes]
    plot_heatmap(args.output_dir, adata_raw[:,hv_genes].X.T, adata_raw[:,hv_genes].X.T, top_k=50, exp=f"hv_genes_gt_heatmap_{args.name}.png")

    hv_pred = adata_pred[:,adata_pred.var['highly_variable']==True]
    hv_raw = adata_raw[:,adata_raw.var['highly_variable']==True]
    compute_corr(args.output_dir, args.name, hv_raw.X.T, hv_pred.X.T, top_k=50)
        

if __name__ == "__main__":
    main()
    