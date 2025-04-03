from torch.utils.data import Dataset
import os
import numpy as np
import tifffile as tiff
from PIL import Image
import pandas as pd
import anndata as ad
import scanpy as sc
import torchvision.transforms.v2 as v2
import torch

class PanNukeDataset(Dataset):
    '''
    Generate the PanNuke Dataset class for testing purposes
    The PanNuke dataset is stored in three numpy files:
    - masks/fold1/images.npy: size (N, 256, 256, 3)
    - images/fold1/masks.npy: size (N, 256, 256, 6), where each channel is the mask for one class
    - images/fold1/types.npy: size (N,), where each element is the tissue type of the image
    '''
    def __init__(self, root_dir):
        super(PanNukeDataset, self).__init__()
        self.images = np.load(os.path.join(root_dir, 'images/fold1/images.npy'))
        self.masks = np.load(os.path.join(root_dir, 'masks/fold1/masks.npy'))
        self.type = np.load(os.path.join(root_dir, 'images/fold1/types.npy'))
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        return self.images[idx], self.masks[idx], self.type[idx]

class VisiumDataset(Dataset):
    '''
    Generate the Visium dataset class
    The Visium dataset consists of:
    - tissue_img.btf: tiff file containing the whole tissue image
    - filtered_feature_bc_matrix/barcodes.tsv.gz: spot barcodes for the features
    - filtered_feature_bc_matrix/features.tsv.gz: features (genes) for the dataset
    - filtered_feature_bc_matrix/matrix.mtx.gz: the feature matrix of features x barcodes
    - spatial/tissue_positions.csv: the spatial positions of the spots
    '''
    def __init__(self, root_dir, img_dir):
        super(VisiumDataset, self).__init__()
        self.spatial = pd.read_csv(os.path.join(root_dir, 'spatial/tissue_positions.csv'), sep=",", header = None)
        self.img = Image.fromarray(tiff.imread(img_dir))
        self.data = sc.read_10x_mtx(os.path.join(root_dir, 'filtered_feature_bc_matrix'))
    
    def __len__(self):
        return len(self.data.n_obs)
    
    def __getitem__(self, idx):
        spot = self.data.obs_names[idx]
        spot_data = self.data.X[idx]
        # return as a list
        spot_exp = spot_data.toarray().flatten().tolist()
        # get the spatial position of the spot
        spot_pos = self.spatial[self.spatial["barcode"] == spot]
        x, y = spot_pos["pxl_row_in_fullres"].values[0], spot_pos["pxl_col_in_fullres"].values[0]
        # get the region of the image, 256x256
        region = self.img.crop((x-128, y-128, x+128, y+128))
        return region, spot_exp
    
class VisiumHDDataset(Dataset):
    ''''
    Generate the VisiumHD dataset class
    The VisiumHD dataset consists of:
    - tissue_img/: a directory of images of the tissue
    - mtx/: a directory of truncated matrix files, one for each tissue image, in anndata .h5ad format, which contains:
        - the expression matrix in cells x features (in anndata.X)
        - the spatial coordinates of the spots (in anndata.obs["cell_position_xmin/ymin/xmax/ymax"])
        - the gene names (in anndata.var_names)
    '''
    def __init__(self, tissue_dir, mtx_dir):
        super(VisiumHDDataset, self).__init__()
        self.tissue_dir = tissue_dir
        self.mtx_dir = mtx_dir
        self.imgs = os.listdir(tissue_dir)
        self.mtxs = os.listdir(mtx_dir)

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        name = self.imgs[idx].split(".")[0]
        image_path = os.path.join(self.tissue_dir, f"{name}.png")
        mtx_path = os.path.join(self.mtx_dir, f"{name}.h5ad")
        image = Image.open(image_path).convert('RGB')
        mtx = ad.read_h5ad(mtx_path)
        # put in tensor
        transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32),
            v2.Resize((224, 224)),
        ])
        image = transforms(image)
        # get the region of the image
        # region = image.crop((x1, y1, x2, y2))
        return image, mtx
    
class VisiumHD_Livecell_Dataset(Dataset):
    ''''
    Generate the VisiumHD and LIVECell dataset class
    The VisiumHD dataset consists of:
    - tissue_img/: a directory of images of the tissue
    - mtx/: a directory of truncated matrix files, one for each tissue image, in anndata .h5ad format, which contains:
        - the expression matrix in cells x features (in anndata.X)
        - the spatial coordinates of the spots (in anndata.obs["cell_position_xmin/ymin/xmax/ymax"])
        - the gene names (in anndata.var_names)
    '''
    def __init__(self, tissue_dir, mtx_dir, livecell_dir):
        super(VisiumHD_Livecell_Dataset, self).__init__()
        self.tissue_dir = tissue_dir
        self.mtx_dir = mtx_dir
        self.imgs = np.array(os.listdir(tissue_dir))
        self.mtxs = np.array(os.listdir(mtx_dir))
        self.livecell_path = np.array(self._find_all_files(livecell_dir))
        # transformations
        self.he_transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32),
            v2.Resize((224, 224)),
        ])
        self.pcm_transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32),
            v2.RandomCrop((224,224)),
            v2.Resize((224, 224)),
        ])
        # get the number of genes from the mtx file
        mtx = np.load(os.path.join(mtx_dir, self.mtxs[0]))
        self.num_genes = mtx.shape[1]

    def __len__(self):
        return self.imgs.size
    
    def __getitem__(self, idx):
        name = self.imgs[idx].split(".")[0]
        image_path = os.path.join(self.tissue_dir, f"{name}.png")
        mtx_path = os.path.join(self.mtx_dir, f"{name}.npy")
        image = Image.open(image_path).convert('RGB')
        mtx = np.load(mtx_path)
        # put in tensor
        image = self.he_transforms(image) / 255 # scale to [0,1]
        mtx_tensor = torch.tensor(mtx).float().view(-1)
        # select a random image from the livecell dataset
        livecell_path = np.random.choice(self.livecell_path)
        livecell_img = Image.open(livecell_path).convert('RGB')
        livecell_img = self.pcm_transforms(livecell_img) / 255 # scale to [0,1]
        return image, mtx_tensor, livecell_img
    
    @staticmethod
    def _find_all_files(path):
        all_files = []
        for root, _, files in os.walk(path):
            for file in files:
                all_files.append(os.path.join(root, file))
        return all_files