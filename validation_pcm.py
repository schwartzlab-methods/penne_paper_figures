'''
Perform inference of the model on PCM data
'''

import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from model import GeneExpPredVisiumHD
from dataset import LiveCellDataset, U373Dataset, TrizinaCaco2Dataset, ShaneMCF10ADataset, ShaneSeqDataset, ShaneSeqConfluencyDataset
from transformers import AutoModel
import argparse
from _feature_extractors import init_spaghetti, pre_processing_phikon
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
    parser.add_argument('--spaghetti_model', type=str, default=None, help='Path to the Spaghetti model. If none, will try to load from model state dict')
    parser.add_argument('--no_spaghetti', action="store_true", help='Whether to not use the Spaghetti model')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument("--u373_dataset", action="store_true", help="use the u373 dataset instead of the livecell data")
    parser.add_argument("--caco2_dataset", action="store_true", help="use the cao2 dataset instead of the livecell data")
    parser.add_argument("--shane_mcf10a_dataset", action="store_true", help="use the MCF10A dataset from Shane instead of the livecell data")
    parser.add_argument("--shane_seq_dataset", action="store_true", help="use the Sequencing dataset from Shane instead of the livecell data")
    parser.add_argument("--shane_confluency_dataset", action="store_true", help="use the Confluency dataset from Shane instead of the livecell data")
    parser.add_argument('--name', type=str, default="gene_predictor", help='Name of the model for logging')
    parser.add_argument('--cell_type', type=str, default=None, help='Cell type of the image. Supply this if all cells are the same type.')
    parser.add_argument('--if_scramble', action="store_true", help='Whether to scramble the input images for testing as a baseline.')
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
    elif args.shane_seq_dataset:
        dataset = ShaneSeqDataset(args.img_dir[0])
    elif args.shane_confluency_dataset:
        dataset = ShaneSeqConfluencyDataset(args.img_dir[0])
    else:
        dataset = LiveCellDataset(args.img_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    # create feature extractor
    extractor = AutoModel.from_pretrained("owkin/phikon-v2").eval()
    image_processor = pre_processing_phikon()
    feature_extractor = (image_processor, extractor)
    if args.spaghetti_model:
        converter = init_spaghetti(args.spaghetti_model)
    else:
        converter = None
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
                                converter = converter, feature_extractor = feature_extractor,
                                bio_feature_size = 960, domain_feature_size = 64)
    model.freeze()
    model.eval()
    # inference
    pred_L = []
    cell_type_L = []
    cell_type_indices = [] # useful later for cell type classification (if needed)
    img_name = []
    features_L = []
    img_exp_type_L = []
    exp_cell_L = []
    for img, labels in tqdm(loader):
        img = img.to(model.device)
        if len(img.shape) > 4:
            img = img.squeeze(0)
        pred = model(img, if_convert=not args.no_spaghetti, scramble=args.if_scramble) #shape: (n, num_genes)
        pred = pred.mean(dim=0, keepdim=True) #shape: (1, num_genes)
        # features
        features = model.compute_feature(img, if_convert=not args.no_spaghetti, scramble=args.if_scramble) #shape: (n, feature_dim)
        features = features.mean(dim=0, keepdim=True) #shape: (1, feature_dim)
        pred_L.append(pred.cpu().numpy())
        features_L.append(features.cpu().numpy())
        cell_type_indices.append(labels[0].cpu().numpy())
        if args.cell_type:
            cell_type_L.append(np.array([args.cell_type], dtype=str))
        else:
            cell_type_L.append(np.array(labels[2], dtype=str))
        if len(labels) > 3:
            img_exp_type_L.append(np.array(labels[3], dtype=str))
            exp_cell_L.append(np.array(labels[4], dtype=str))
        img_name.append(np.array(labels[1], dtype=str))
    pred_L = np.concatenate(pred_L, axis=0)
    cell_type_L = np.concatenate(cell_type_L, axis=0)
    cell_type_indices = np.concatenate(cell_type_indices, axis=0)
    img_name = np.concatenate(img_name, axis=0)
    features_L = np.concatenate(features_L, axis=0)
    if len(img_exp_type_L) > 0:
        img_exp_type_L = np.concatenate(img_exp_type_L, axis=0)
        exp_cell_L = np.concatenate(exp_cell_L, axis=0)
        np.save(os.path.join(args.output_dir, f"{args.name}_experiment_types.npy"), img_exp_type_L)
        np.save(os.path.join(args.output_dir, f"{args.name}_exp_cell_types.npy"), exp_cell_L)
    # save the predictions
    np.save(os.path.join(args.output_dir, f"{args.name}_predictions.npy"), pred_L)
    np.save(os.path.join(args.output_dir, f"{args.name}_cell_types.npy"), cell_type_L)
    np.save(os.path.join(args.output_dir, f"{args.name}_cell_type_indices.npy"), cell_type_indices)
    np.save(os.path.join(args.output_dir, f"{args.name}_img_names.npy"), img_name)
    np.save(os.path.join(args.output_dir, f"{args.name}_features.npy"), features_L)
    print("Inference finished. Predictions saved to ", args.output_dir)

if __name__ == "__main__":
    main()



