'''
Perform inference of the model on PCM data
#todo: figure out gsea on the predictions
'''

import os
import torch
from train_model import init_spaghetti
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from model import GeneExpPredVisiumHD
from dataset import LiveCellDataset
from transformers import AutoImageProcessor, AutoModel
import argparse
from _feature_extractors import owkin_features, spaghetti_convertion
import numpy as np
from tqdm import tqdm

def main():
    # seeds for reproducibility
    torch.manual_seed(42)
    pl.seed_everything(42, workers=True)
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--livecell_dir', type=str, nargs="+", help='Directory containing the LIVECell images')
    parser.add_argument('--model_dir', type=str, help='Directory containing the model checkpoints')
    parser.add_argument('--gene_names', type=str, help='Path to the gene names feature tsv file')
    parser.add_argument('--spaghetti_model', type=str, help='Path to the Spaghetti model')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--name', type=str, default="gene_predictor", help='Name of the model for logging')
    args = parser.parse_args()
    # check if the output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # create dataset
    dataset = LiveCellDataset(args.livecell_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    # create feature extractor
    feature_extractor = AutoModel.from_pretrained("/fs01/home/richarddong/.cache/huggingface/hub/phikon-v2")
    image_processor = AutoImageProcessor.from_pretrained("owkin/phikon-v2")
    # save the gene names
    gene_names = np.loadtxt(args.gene_names, dtype=str, delimiter='\t')
    gene_symbols = gene_names[:,1].reshape(-1)
    np.save(os.path.join(args.output_dir, f"{args.name}_gene_symbols.npy"), gene_symbols)
    print("Gene symbols saved to ", args.output_dir)
    num_genes = gene_names.shape[0]
    # prep model
    model = GeneExpPredVisiumHD.load_from_checkpoint(args.model_dir, num_genes = num_genes, 
                                converter = lambda device, x: spaghetti_convertion(init_spaghetti(args.spaghetti_model), device, x), 
                                feature_extractor = lambda device, x: owkin_features(feature_extractor, device, image_processor, x))
    model.freeze()
    # inference
    pred_L = []
    cell_type_L = []
    cell_type_indices = [] # useful later for cell type classification (if needed)
    for img, labels in tqdm(loader):
        img = img.to(model.device)
        pred = model(img, if_convert=True)
        pred_L.append(pred.cpu().numpy())
        cell_type_indices.append(labels[0].cpu().numpy())
        cell_type_L.append(np.array(labels[2], dtype=str))
    pred_L = np.concatenate(pred_L, axis=0)
    cell_type_L = np.concatenate(cell_type_L, axis=0)
    cell_type_indices = np.concatenate(cell_type_indices, axis=0)
    # save the predictions
    np.save(os.path.join(args.output_dir, f"{args.name}_predictions.npy"), pred_L)
    np.save(os.path.join(args.output_dir, f"{args.name}_cell_types.npy"), cell_type_L)
    np.save(os.path.join(args.output_dir, f"{args.name}_cell_type_indices.npy"), cell_type_indices)
    print("Inference finished. Predictions saved to ", args.output_dir)

if __name__ == "__main__":
    main()



