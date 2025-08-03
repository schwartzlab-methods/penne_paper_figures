'''
Perform inference of the model on PCM data
'''

import os
import torch
from train_model import init_spaghetti
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from model import GeneExpPredVisiumHD
from dataset import LiveCellDataset, U373Dataset, TrizinaCaco2Dataset, ShaneMCF10ADataset
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
    parser.add_argument('--img_dir', type=str, nargs="+", help='Directory containing the LIVECell images (or other PCM images)')
    parser.add_argument('--model_dir', type=str, help='Directory containing the model checkpoints')
    parser.add_argument('--gene_names', type=str, help='Path to the gene names feature tsv or txt file')
    parser.add_argument('--spaghetti_model', type=str, help='Path to the Spaghetti model')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument("--u373_dataset", action="store_true", help="use the u373 dataset instead of the livecell data")
    parser.add_argument("--caco2_dataset", action="store_true", help="use the cao2 dataset instead of the livecell data")
    parser.add_argument("--shane_mcf10a_dataset", action="store_true", help="use the MCF10A dataset from Shane instead of the livecell data")
    parser.add_argument('--name', type=str, default="gene_predictor", help='Name of the model for logging')
    parser.add_argument('--cell_type', type=str, default=None, help='Cell type of the image. Supply this if all cells are the same type.')
    args = parser.parse_args()
    # check if the output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # create dataset
    if args.u373_dataset:
        dataset = U373Dataset(args.img_dir[0])
    elif args.caco2_dataset:
        dataset = TrizinaCaco2Dataset(args.img_dir[0])
    elif args.shane_mcf10a_dataset:
        dataset = ShaneMCF10ADataset(args.img_dir[0])
    else:
        dataset = LiveCellDataset(args.img_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    # create feature extractor
    feature_extractor = AutoModel.from_pretrained("owkin/phikon-v2")
    image_processor = AutoImageProcessor.from_pretrained("owkin/phikon-v2")
    # save the gene names
    if args.gene_names.endswith(".tsv.gz"):
        genes = np.loadtxt(args.gene_names, dtype=str, delimiter='\t')
        gene_names = genes[:,1].reshape(-1)
        gene_symbols = genes[:,0].reshape(-1)
        np.save(os.path.join(args.output_dir, f"{args.name}_gene_symbols.npy"), gene_symbols)
    else: # assume txt file
        with open(args.gene_names, 'r') as f:
            gene_names = [line.strip() for line in f.readlines() if "Unnamed: 0" not in line]
        gene_names = np.array(gene_names).reshape(-1)
    np.save(os.path.join(args.output_dir, f"{args.name}_gene_names.npy"), gene_names)
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
    img_name = []
    for img, labels in tqdm(loader):
        img = img.to(model.device)
        pred = model(img, if_convert=True)
        pred_L.append(pred.cpu().numpy())
        cell_type_indices.append(labels[0].cpu().numpy())
        if args.cell_type:
            cell_type_L.append(np.array([args.cell_type], dtype=str))
        else:
            cell_type_L.append(np.array(labels[2], dtype=str))
        img_name.append(np.array(labels[1], dtype=str))
    pred_L = np.concatenate(pred_L, axis=0)
    cell_type_L = np.concatenate(cell_type_L, axis=0)
    cell_type_indices = np.concatenate(cell_type_indices, axis=0)
    img_name = np.concatenate(img_name, axis=0)
    # save the predictions
    np.save(os.path.join(args.output_dir, f"{args.name}_predictions.npy"), pred_L)
    np.save(os.path.join(args.output_dir, f"{args.name}_cell_types.npy"), cell_type_L)
    np.save(os.path.join(args.output_dir, f"{args.name}_cell_type_indices.npy"), cell_type_indices)
    np.save(os.path.join(args.output_dir, f"{args.name}_img_names.npy"), img_name)
    print("Inference finished. Predictions saved to ", args.output_dir)

if __name__ == "__main__":
    main()



