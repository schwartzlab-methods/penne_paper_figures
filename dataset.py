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
    
#todo: implement the mtx files accordingly; the mtx files are in anndata format, including the coordinates in the obsm
#todo: also update coordinate the accordingly, to match the patch instead of the whole image
#todo: structure the data to be a list of single-cell features in npy + a list of expression values (normalized) in npy
class VisiumHDDataset(Dataset):
    ''''
    Generate the VisiumHD dataset class
    The VisiumHD dataset consists of:
    - tissue_img/: a directory of images of the tissue
    - mtx/: a directory of truncated matrix files, one for each tissue image, in anndata .h5ad format
    - cells_csv/: the segmentations of the cells, with coordinates x1, y1, x2, y2, 
                where (x1, y1) is the top left corner and (x2, y2) is the bottom right corner
    '''
    def __init__(self, tissue_dir, mtx_dir, cells_dir):
        super(VisiumHDDataset, self).__init__()
        self.imgs = tissue_dir
        self.mtxs = mtx_dir
        self.cells = cells_dir

    def __len__(self):
        return len(self.cells)
    
    def __getitem__(self, idx):
        image = Image.open(self.imgs[idx]).convert('RGB')
        mtx = ad.read_h5ad(self.mtxs[idx])
        cells = pd.read_csv(self.cells[idx])
        # track returns
        exp_L = []
        coordinate_L = []
        for each_cell in cells.iterrows():
        # get coordinates of the cell
            x1, y1, x2, y2 = each_cell["x1"], each_cell["y1"], each_cell["x2"], each_cell["y2"]
            # get the barcode
            barcodes = mtx.obsm['spatial'][(mtx.obsm['spatial']['pxl_row_in_fullres'] >= x1) 
                                        & (mtx.obsm['spatial']['pxl_row_in_fullres'] <= x2) 
                                        & (mtx.obsm['spatial']['pxl_col_in_fullres'] >= y1) 
                                        & (mtx.obsm['spatial']['pxl_col_in_fullres'] <= y2)]
            barcodes_L = barcodes['barcode'].values
            # get only the barcodes
            new_mtx = mtx[self.anndata_mtx.obs.index.isin(barcodes_L)]
            # combine all gene expression values
            new_mtx = new_mtx.X.sum(axis=0)
            # return as a list
            spot_exp = new_mtx.toarray().flatten().tolist()
            # append to the list
            exp_L.append(spot_exp)
            # append the coordinates
            coordinate_L.append((x1, y1, x2, y2))
        # get the region of the image
        # region = image.crop((x1, y1, x2, y2))
        return image, exp_L, coordinate_L