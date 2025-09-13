'''
Visualize the patch for a particular gene. Label the image with the predicted gene exp
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
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    # make high/mid/low directories
    os.makedirs(os.path.join(args.output_dir, "high_exp"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "mid_exp"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "low_exp"), exist_ok=True)

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
        print("Using all genes: ", num_genes)

    # prep model
    model = GeneExpPredVisiumHD.load_from_checkpoint(args.model_path, num_genes = num_genes, 
                                converter = converter, feature_extractor = feature_extractor, orthogonal_loss_weight=0.1)
    model.freeze()
    model.eval()

    # inference
    pred_L = []
    image_L = []
    with torch.no_grad():
        for img, label in tqdm(loader):
            img = img.to(model.device)
            img = img.squeeze(0) #remove the default batch dimension
            pred = model(img, if_convert=True) #shape: num_patch, num_genes
            # get only the genes we want
            pred = pred[:, gene_indices]
            mean_exp = torch.mean(pred, dim=1) # shape: num_patch
            pred_L.append(mean_exp.cpu().numpy())
            image_L.append(img.cpu().numpy())
    pred_L = np.concatenate(pred_L, axis=0) # shape: num_samples, 1
    image_L = np.concatenate(image_L, axis=0) # shape: num_samples, 3, 256, 256

    # get the stats of mean exp, organized into three groups (low, medium, high)
    perc_33 = np.percentile(pred_L, 33, axis=0) # shape: 1
    perc_66 = np.percentile(pred_L, 66, axis=0)
    print("33 percentile: ", perc_33)
    print("66 percentile: ", perc_66)
    # save the images according to the expression levels
    image_L = torch.tensor(image_L)
    for i in range(pred_L.shape[0]):
        exp = pred_L[i].item()
        if exp <= perc_33:
            save_path = os.path.join(args.output_dir, "low_exp", f"img_{i}_exp_{exp:.4f}.png")
            torchvision.utils.save_image(image_L[i], save_path)
        elif exp <= perc_66:
            save_path = os.path.join(args.output_dir, "mid_exp", f"img_{i}_exp_{exp:.4f}.png")
            torchvision.utils.save_image(image_L[i], save_path)
        else:
            save_path = os.path.join(args.output_dir, "high_exp", f"img_{i}_exp_{exp:.4f}.png")
            torchvision.utils.save_image(image_L[i], save_path)
    print("Images saved to ", args.output_dir)
if __name__ == "__main__":
    main()