import os
import torch
from torch.utils.data import random_split, DataLoader, ConcatDataset
import pytorch_lightning as pl
import torchvision.transforms.v2 as v2
import torch.nn.functional as F
from pytorch_lightning.loggers import CSVLogger
from model import GeneExpPredVisiumHD
from dataset import VisiumHD_Livecell_Dataset
from transformers import AutoImageProcessor, AutoModel
from _feature_extractors import init_spaghetti, pre_processing_phikon
import argparse
from tqdm import tqdm
import pandas as pd

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
        
def read_tsv(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip().split('\t') for line in f]
    df = pd.DataFrame(lines).T #number of genes x number of cells
    df.columns = [cell.lower() for cell in df.iloc[0].tolist()]  # set the first row as header
    df = df[2:]  # remove the first row
    return df

def train(train_loader, val_loader, 
          num_genes, converter, feature_extractor,
          num_cell_types, 
          end_to_end=False,
          up_marker_genes=None,
          gene_names=None,
          pcm_cell_to_idx=None,
          celltypes=None,
          cell_type_loss_weight=0.0,
          marker_gene_loss_weight=0,
          marker_across_cell=False,
          domain_weight=5.0, coral_loss_weight=0.1,
          lr = 0.0001, save_dir = None, epochs=100, name="gene_predictor"):
    '''
    Train the gene expression prediction model using PyTorch Lightning.
    args:
        train_loader: the PyTorch Dataloader for the training dataset
        val_loader: the PyTorch Dataloader for the validation dataset
        num_genes: int, the number of genes in the dataset to predict
        converter: the converter module to convert the image to the right format
        feature_extractor: the feature extractor module to extract the features from the image
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
    if up_marker_genes:
        print("Attempting to start Stage Two training")
        celltype_of_interest = [x.lower() for x in celltypes]
        pcm_cell_to_idx_lower = {k.lower(): v for k, v in pcm_cell_to_idx.items()}
        signature = read_tsv(up_marker_genes)
        # convert the signature from gene names to gene symbols
        if gene_names.endswith(".tsv.gz"):
            template = pd.read_csv(gene_names, sep='\t', header=None)
            feature_names = template.iloc[:, 0].tolist()
            # print(feature_names)
            name_to_symbol = {row[1]: row[0] for _, row in template.iterrows()}  # map from gene name to gene symbol
            # convert the signature to gene symbols by mapping the dataframe
            signature = signature.map(lambda x: name_to_symbol.get(x, x))  # map gene names to symbols
        else:
            with open(gene_names, 'r') as f:
                feature_names = [line.strip() for line in f.readlines() if "Unnamed" not in line]
                assert len(feature_names) == num_genes, "Number of genes in dataset does not match number of feature names"
        # select only the cell type of interest
        enriched_gene_sets_name = {
            pcm_cell_to_idx_lower[celltype]: signature[celltype].dropna().tolist() for celltype in celltype_of_interest if celltype in signature.columns
        }
        # convert to a list of indices
        enriched_gene_sets = {
            celltype: [1 if gene in enriched_genes else 0 for gene in feature_names]
            for celltype, enriched_genes in enriched_gene_sets_name.items()
        }
        print("Gene sets processing completed. Number of marker genes for a celltype is as follows: ")
        for celltype, genes in enriched_gene_sets.items():
            print(f"{celltype}: {sum(genes)}")
        print("The cell type name to index is:")
        print(pcm_cell_to_idx_lower)
    else:
        print("No marker genes provided. Starting stage ONE training...")
        enriched_gene_sets = None
        feature_names = None

    # create model
    print("Preparing training model ...")
    lit_model = GeneExpPredVisiumHD(num_genes, 
                                    converter, feature_extractor,
                                    end_to_end=end_to_end,
                                    num_cell_types=num_cell_types,
                                    up_marker_genes=enriched_gene_sets,
                                    domain_weight = domain_weight, 
                                    second_order_weight=coral_loss_weight,
                                    marker_gene_weight=marker_gene_loss_weight,
                                    cell_type_weight=cell_type_loss_weight,
                                    lr=lr, do_gmlp=True,
                                    across_cell=marker_across_cell)
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
        checkpoint = torch.load(ckpt, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        lit_model.load_state_dict(checkpoint["state_dict"], strict=False)  # Fuzzy loading due to different training stage
    else:
        print("Starting from epoch 0")       
    trainer.fit(lit_model, train_loader, val_loader, None)
    print("Training ended.")

def main():
    # seeds for reproducibility
    torch.manual_seed(42)
    pl.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision('high')
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--visiumhd_dir', type=str, nargs="+", help='Directory containing the VisiumHD patches')
    parser.add_argument('--livecell_dir', type=str, nargs="+", help='Directory containing the LIVECell patches')
    parser.add_argument('--mtx_dir', type=str, nargs="+", help='Directory containing the mtx files')
    parser.add_argument('--spaghetti_model', type=str, help='Path to the Spaghetti model')
    parser.add_argument('--end_to_end', action='store_true', help='If set, train the model end to end including SPAGHETTI')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--domain_weight', type=float, default=5.0, help='Domain weight for training')
    parser.add_argument('--coral_loss_weight', type=float, default=0.1, help='Weight for the CORAL loss')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--cell_type_loss_weight', type=float, default=0, help='Weight for the cell type loss')
    parser.add_argument('--marker_gene_loss_weight', type=float, default=0, help='Weight for the marker gene loss')
    parser.add_argument('--gene_names', type=str, default=None, help="Path to the tsv storing the list of gene names")
    parser.add_argument('--up_marker_genes', type=str, default=None, 
                            help='Path to up marker genes for PCM cell type. If supplied, stage two training will be performed.')
    parser.add_argument('--marker_across_cell', action='store_true',
                            help='If set, the marker gene loss will be calculated across all cells. If not set, the marker gene loss will be calculated within each cell type.')
    parser.add_argument('--name', type=str, default="gene_predictor", help='Name of the model for logging')
    args = parser.parse_args()
    print("Starting the training script with the following arguments:")
    print(args)

    # create dataset
    print("Preparing datasets ...")
    dataset_L = []
    for i in tqdm(range(len(args.visiumhd_dir))):
        dataset_L.append(VisiumHD_Livecell_Dataset(args.visiumhd_dir[i], args.mtx_dir[i], args.livecell_dir))
    dataset = ConcatDataset(dataset_L)
    # split dataset into train and val
    train_dataset, val_dataset = random_split(dataset, [0.8, 0.2])
    # create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    print("Data loader created")
    # create feature extractor
    extractor = AutoModel.from_pretrained("owkin/phikon-v2").eval()
    # image_processor = AutoImageProcessor.from_pretrained("owkin/phikon-v2")
    image_processor = pre_processing_phikon()
    feature_extractor = (image_processor, extractor)
    # prepare spaghetti model
    spaghetti = init_spaghetti(args.spaghetti_model)
    converter = spaghetti
    # start training
    train(train_loader, val_loader, 
          num_genes=dataset.datasets[0].num_genes,
          num_cell_types=dataset.datasets[0].num_pcm_classes,
          end_to_end=args.end_to_end,
          up_marker_genes=args.up_marker_genes,
          gene_names=args.gene_names,
          pcm_cell_to_idx=dataset.datasets[0].livecell_class_to_idx,
          celltypes=dataset.datasets[0].livecell_class_to_idx.keys(),
          cell_type_loss_weight=args.cell_type_loss_weight,
          marker_gene_loss_weight=args.marker_gene_loss_weight,
          marker_across_cell=args.marker_across_cell,
          converter=converter,
          feature_extractor=feature_extractor, 
          domain_weight=args.domain_weight, coral_loss_weight=args.coral_loss_weight,
          lr=args.lr, save_dir=args.output_dir, epochs=args.epochs, name=args.name)


if __name__ == "__main__":
    main()
