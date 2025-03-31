import os
import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
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

def find_checkpoint(dir: str):
    '''
    Find the latest checkpoint in the directory
    args:
        dir: str, the directory to search for the checkpoint for Pytorch Lightning
    return:
        str or None. The str of the path to the latest checkpoint that ends with .ckpt
        If no checkpoint is found, return None
    '''
    files = []
    for path, _, file in os.walk(dir):
        for f in file:
            if f.endswith(".ckpt"):
                files.append(os.path.join(path, f))
    if len(files) == 0:
        return None
    else:
        return max(files, key=os.path.getctime)
    
def train(train_loader, val_loader, 
          num_genes, converter, feature_extractor,
          batch_size=1, domain_weight=5.0,
          lr = 0.0001, save_dir = None, epochs=100, name="gene_predictor"):
    '''
    Train the gene expression prediction model using PyTorch Lightning.
    args:
        train_loader: the PyTorch Dataloader for the training dataset
        val_loader: the PyTorch Dataloader for the validation dataset
        num_genes: int, the number of genes in the dataset to predict
        converter: the converter module to convert the image to the right format
        feature_extractor: the feature extractor module to extract the features from the image
        batch_size: int, the batch size for the model, default 1
        domain_weight: float, the weight for the domain adaptation loss, default 5.0
        lr: float, the learning rate for the model, default 0.0001
        save_dir: str, the directory to save the model checkpoints and logs. Default current directory
        epochs: int, the number of epochs to train the model, default 100
        name: str, the name of the model for the logger, default "gene_predictor"
    '''
    ngpus_per_node = torch.cuda.device_count()
    num_nodes = int(os.environ.get("SLURM_NNODES"))
    if save_dir is None:
        final_save_dir = os.getcwd()
    else:
        final_save_dir = save_dir
    # create model
    lit_model = GeneExpPredVisiumHD(num_genes, 
                                    converter, feature_extractor,
                                    domain_weight = domain_weight, lr=lr)
    # train model
    logger = CSVLogger(final_save_dir, name=name)
    trainer = pl.Trainer(max_epochs=epochs, devices=ngpus_per_node, num_nodes=num_nodes,
                            use_distributed_sampler=True, enable_progress_bar=True,
                            strategy="ddp",
                            default_root_dir=final_save_dir, logger=logger)
    print("Trainer initialized with ", ngpus_per_node, "GPU(s) per node on ", num_nodes, "node(s)")
    print("Training Starting...")
    ckpt = find_checkpoint(final_save_dir)
    if ckpt:
        print("Checkpoint found. Resuming from ", ckpt)
    else:
        print("Starting from epoch 0")
    trainer.fit(lit_model, train_loader, val_loader, None, ckpt)
    print("Training ended.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--visiumhd_dir', type=str, help='Directory containing the VisiumHD patches')
    parser.add_argument('--livecell_dir', type=str, help='Directory containing the LIVECell patches')
    parser.add_argument('--mtx_dir', type=str, help='Directory containing the mtx files')
    parser.add_argument('--spaghetti_model', type=str, help='Path to the Spaghetti model')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--domain_weight', type=float, default=5.0, help='Domain weight for training')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--name', type=str, default="gene_predictor", help='Name of the model for logging')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # create dataset
    dataset = VisiumHD_Livecell_Dataset(args.visiumhd_dir, args.mtx_dir, args.livecell_dir)
    # split dataset into train and val
    train_dataset, val_dataset = random_split(dataset, [0.8, 0.2])
    # create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    # create feature extractor
    feature_extractor = AutoModel.from_pretrained("/fs01/home/richarddong/.cache/huggingface/hub/phikon-v2")
    image_processor = AutoImageProcessor.from_pretrained("owkin/phikon-v2")
    # prepare spaghetti model
    spaghetti = init_spaghetti(args.spaghetti_model)
    # start training
    train(train_loader, val_loader, 
          num_genes=dataset.num_genes,
          converter=lambda x: spaghetti_convertion(spaghetti, device, x),
          feature_extractor=lambda x: owkin_features(feature_extractor, device, image_processor, x), 
          batch_size=args.batch_size, domain_weight=args.domain_weight,
          lr=args.lr, save_dir=args.output_dir, epochs=args.epochs, name=args.name)


if __name__ == "__main__":
    main()