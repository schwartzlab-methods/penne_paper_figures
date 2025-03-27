from torch.utils.data import Dataset
import os
import numpy as np
import tifffile as tiff
from PIL import Image
import pandas as pd
import anndata as ad
import scanpy as sc

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
        self.spatial = pd.read_csv(os.path.join(root_dir, 'spatial/tissue_positions.csv'))
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

    def _get_region(df, img_dir):
        '''
        return the image
        '''
        # select the rows with colium "in_tissue" == 1 and greater than 0
        df = df[df["in_tissue"] == 1]
        df = df[df["pxl_row_in_fullres"] > 0]
        df = df[df["pxl_col_in_fullres"] > 0]
        # get the region of the image by looking at the max and min of the x and y
        x_min = int(df["pxl_row_in_fullres"].min())
        x_max = int(df["pxl_row_in_fullres"].max())
        y_min = int(df["pxl_col_in_fullres"].min())
        y_max = int(df["pxl_col_in_fullres"].max())
        # get the image
        img = tiff.imread(img_dir) #shape (x, y, 3)
        img = img[x_min:x_max, y_min:y_max, :]
        img_PIL = Image.fromarray(img)
        return img_PIL