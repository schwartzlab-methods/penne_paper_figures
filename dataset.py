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
import torch.nn.functional as F
from cellpose import models
import torchvision.transforms.v2.functional as TF

#! Microscopy only datasets
class LiveCellDataset(Dataset):
    '''
    Construct the LiveCell dataset class
    Returns the image as well as the cell type
    Attributes:
        paths (list[str]): List of paths to the image directories.
        images (list[str]): List of paths to the image files.
        classes (list[str]): List of class labels for each image.
        class_to_idx (dict[str, int]): Mapping from class labels to class indices.
        targets (list[int]): List of class indices for each image.
        class_count_dict (dict[str, int]): Mapping from class labels to the number of images in each class.
    '''
    def __init__(self, paths: list[str]):
        '''Initialize the LiveCell dataset for validation

        Args:
            paths (list[str]): List of paths to the image directories.
        '''
        super(LiveCellDataset, self).__init__()
        self.paths = paths
        self.images: list[str] = []
        self.classes: list[str] = []
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32),
            # v2.RandomCrop((256,256)),
            # v2.Resize((256, 256)),
        ])
        self._write_attributes() # this will write class_to_idx and targets
    
    @staticmethod
    def extract_full_patches(img: torch.Tensor, patch_size=256):
        # img: (C, H, W)
        C, H, W = img.shape
        img_batched = img.unsqueeze(0)  # (1, C, H, W)

        # Only patches that fully fit will be returned
        patches = F.unfold(
            img_batched, 
            kernel_size=patch_size, 
            stride=patch_size
        )

        # Each column is a flattened patch
        patches = patches.transpose(1, 2)  # (1, num_patches, C*ps*ps)
        patches = patches.reshape(-1, C, patch_size, patch_size)
        return patches

    def _write_attributes(self):
        '''Write the attributes for the dataset.
        '''
        for path in self.paths:
            all_cls = [x for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))]
            for cls in all_cls:
                imgs = [os.path.join(root, img) for root, _, imgs in os.walk(os.path.join(path, cls)) for img in imgs]
                self.images.extend(imgs)
                self.classes.extend([cls]*len(imgs))
        # get the class to idx mapping
        self.class_to_idx = {cls: i for i, cls in enumerate(np.unique(self.classes).tolist())}
        self.targets = [self.class_to_idx[x] for x in self.classes] # targets are the class indices
        assert len(self.images) == len(self.targets) == len(self.classes)
        self.class_count_dict = {k: self.classes.count(k) for k in np.unique(self.classes)}

    def __getitem__(self, idx):
        img = Image.open(self.images[idx])
        img = np.array(img, dtype=np.uint16)
        img = self.transform(img)
        if img.max().item() > 1:
            img = img / 255 # rescale to [0,1]
        img = torch.clamp(img, max=1, min=0) #ensure no float overflow
        imgs = self.extract_full_patches(img)  # (num_patches, C, 256, 256)
        cls_idx = self.targets[idx]
        cls = self.classes[idx]
        # return the image as x and the class int label, image path, and img cls as y
        return imgs, (cls_idx, self.images[idx], cls)

    def __len__(self):
        return len(self.images)

class U373Dataset(Dataset):
    '''
    Validation using the U373 dataset by Deiber et al. (2005)
    '''
    def __init__(self, path: str):
        super(U373Dataset, self).__init__()
        self.images: list[str] = [os.path.join(path, img) for img in os.listdir(path)]
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32),
            v2.RandomCrop((256,256)),
            v2.Resize((256, 256)),
        ])

    def __getitem__(self, idx):
        img = Image.open(self.images[idx])
        img = np.array(img, dtype=np.uint16)
        img = self.transform(img)
        img = img / 255 # rescale to [0,1]
        img = torch.clamp(img, max=1, min=0) #ensure no float overflow
        return img, (0, self.images[idx], "U373")
    
    def __len__(self):
        return len(self.images)
    
class TrizinaCaco2Dataset(Dataset):
    '''
    Validation using the Cao-2 dataset by Trizina et al. (2023)
    '''
    def __init__(self, path):
        super(TrizinaCaco2Dataset, self).__init__()
        self.path = [os.path.join(path, x) for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))]
        self.images = []
        self.treatment = [] # treatment is ctrl or Cam (treated with Camptothecin)
        for each in self.path:
            pcm_image_path = [os.path.join(each, x) for x in os.listdir(each) if x.endswith("0.jpg")]
            self.images.extend(pcm_image_path)
            self.treatment.extend([os.path.basename(os.path.normpath(each)).split("_")[2]]*len(pcm_image_path))
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32),
            # v2.RandomCrop((256,256)),
            v2.Resize((256, 256)),
        ])
    
    def __getitem__(self, idx):
        img = Image.open(self.images[idx])
        img = np.array(img, dtype=np.uint16)
        img = self.transform(img)
        img = img / 255
        img = torch.clamp(img, max=1, min=0) #ensure no float overflow
        treatment = self.treatment[idx]
        treatment_idx = 0 if treatment.lower() == "ctrl" else 1
        return img, (treatment_idx, self.images[idx], 
                    "camptothecin" if treatment_idx == 1 else "Ctrl")
    
    def __len__(self):
        return len(self.images)

class ShaneMCF10ADataset(Dataset):
    '''
    Validation using Shane's MCF10A dataset (internal)
    '''
    def __init__(self, path):
        super(ShaneMCF10ADataset, self).__init__()
        self.path = [os.path.join(path, x) for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))]
        self.images = []
        self.treatment = [] # treatment is NIR, 2Gy, or 5Gy
        for each in self.path:
            pcm_image_path = [os.path.join(each, x) for x in os.listdir(each) if (x.split(".")[0].split("_")[-1][0:3] == "00d") and (x.split(".")[0].split("_")[1] == "A1")] # we get only 0 day
            self.images.extend(pcm_image_path)
            self.treatment.extend([os.path.basename(os.path.normpath(each)).split("_")[0]]*len(pcm_image_path))
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32),
            v2.RandomCrop((256,256)),
            v2.Resize((256, 256)),
        ])
    
    def __getitem__(self, idx):
        img = Image.open(self.images[idx])
        img = np.array(img, dtype=np.uint16)
        img = self.transform(img)
        img = img / 255
        img = torch.clamp(img, max=1, min=0) #ensure no float overflow
        treatment = self.treatment[idx]
        treatment_idx = (0 if treatment == "NIR" 
                        else 1 if treatment == "2Gy"
                        else 2)
        return img, (treatment_idx, self.images[idx], treatment)
    
    def __len__(self):
        return len(self.images)

#! H&E Related datasets
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
    
#! ST + Microscopy datasets
class VisiumHD_Livecell_Dataset(Dataset):
    ''''
    Generate the VisiumHD and LIVECell dataset class
    The VisiumHD dataset consists of:
    - tissue_img/: a directory of images of the tissue
    - mtx/: a directory of truncated matrix files, one for each tissue image, in numpy format
        Each contains a long vector of gene expression counts for each spot in the shape of
        (1, num_genes).

    The LIVECell dataset main directory consists of multiple directories,
    each directory is the name of the cell type, and contains images of that cell type.

    Attributes:
        tissue_dir (str): Path to the tissue image directory.
        mtx_dir (str): Path to the matrix directory.
        livecell_dir (str): Path to the LIVECell image directory.
        imgs (np.ndarray): Array of image file names.
        mtxs (np.ndarray): Array of matrix file names.
        livecell_path (list[str]): List of paths to the LIVECell images.
        livecell_classes (list[str]): List of cell type labels for each LIVECell image.
        livecell_class_to_idx (dict[str, int]): Mapping from cell type labels to class indices.
        livecell_class_count_dict (dict[str, int]): Mapping from cell type labels to the number of images in each class.
        livecell_targets (list[int]): List of class indices for each LIVECell image.
        he_transforms (torchvision.transforms.Compose): Transformations for HE images.
        pcm_transforms (torchvision.transforms.Compose): Transformations for PCM images.
        num_genes (int): Number of genes in the dataset.
        num_pcm_classes (int): Number of PCM classes (ie: cell types) in the dataset.
    '''
    def __init__(self, tissue_dir: str, mtx_dir: str, livecell_dir: str, use_mtx: bool = True):
        '''Initialize the VisiumHD and LIVECell dataset.

        Args:
            tissue_dir (str): Path to the tissue image directory.
            mtx_dir (str): Path to the matrix directory.
            livecell_dir (str): Path to the LIVECell image main directory.
            use_mtx (bool): Whether to use the matrix files. If False, a zero tensor will be returned for the matrix.
        '''
        super(VisiumHD_Livecell_Dataset, self).__init__()
        self.tissue_dir = tissue_dir
        self.mtxs = np.array(os.listdir(mtx_dir))
        if use_mtx:
            self.mtx_dir = mtx_dir
        else:
            self.mtx_dir = None
        self.imgs = np.array(os.listdir(tissue_dir))
        self.livecell_path = []
        self.livecell_classes = [] #classes are string labels of the cell types
        self._write_attributes(livecell_dir)
        # transformations
        self.he_transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32),
            v2.Resize((256, 256)),
        ])
        self.pcm_transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32),
            v2.RandomCrop((256,256)),
            v2.Resize((256, 256)),
        ])
        # get the number of genes from the mtx file
        mtx = np.load(os.path.join(mtx_dir, self.mtxs[0]))
        self.num_genes = mtx.shape[1]
        # get the total number of pcm classes
        self.num_pcm_classes = len(self.livecell_class_count_dict)

    def __len__(self):
        return self.imgs.size
    
    def __getitem__(self, idx: int):
        '''Get the item at the specified index.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, int]: 
                The HE image, exp matrix tensor, LIVECell image, HE image path, and LIVECell class index.
        '''
        name = self.imgs[idx].split(".")[0]
        he_image_path = os.path.join(self.tissue_dir, f"{name}.png")
        if self.mtx_dir:
            mtx_path = os.path.join(self.mtx_dir, f"{name}.npy")
            mtx = np.load(mtx_path)
            mtx_tensor = torch.tensor(mtx).float().view(-1)
        else:
            mtx_tensor = torch.zeros((self.num_genes,), dtype=torch.float32) # if no mtx_dir is provided, return a zero tensor
        image = Image.open(he_image_path).convert('RGB')
        # put in tensor
        image = self.he_transforms(image)
        if image.max() > 1:
            image = image / 255 # rescale to [0,1]
        # select a random image from the livecell dataset
        livecell_path = self.livecell_path[idx % len(self.livecell_path)]
        livecell_img = Image.open(livecell_path).convert('RGB')
        livecell_img = self.pcm_transforms(livecell_img)
        if livecell_img.max() > 1:
            livecell_img = livecell_img / 255 # scale to [0,1]
        return image, mtx_tensor, livecell_img, he_image_path, self.livecell_class_to_idx[self.livecell_classes[idx % len(self.livecell_classes)]]
    
    @staticmethod
    def _find_all_files(path):
        '''Find all files in a directory.

        Args:
            path (str): Path to the directory.

        Returns:
            list[str]: List of file paths.
        '''
        all_files = []
        for root, _, files in os.walk(path):
            for file in files:
                all_files.append(os.path.join(root, file))
        return all_files
    
    def _write_attributes(self, livecell_dir):
        '''Write attributes for the LIVECell dataset.

        Args:
            livecell_dir (list[str]): List of paths to the LIVECell directories.
        '''
        for path in livecell_dir:
            all_cls = [x for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))]
            for cls in all_cls:
                imgs = [os.path.join(root, img) for root, _, imgs in os.walk(os.path.join(path, cls)) for img in imgs]
                self.livecell_path.extend(imgs)
                self.livecell_classes.extend([cls]*len(imgs))
        # get the class to idx mapping
        self.livecell_class_to_idx = {cls: i for i, cls in enumerate(np.unique(self.livecell_classes).tolist())}
        self.livecell_targets = [self.livecell_class_to_idx[x] for x in self.livecell_classes] # targets are the class indices
        assert len(self.livecell_path) == len(self.livecell_targets) == len(self.livecell_classes)
        self.livecell_class_count_dict = {k: self.livecell_classes.count(k) for k in np.unique(self.livecell_classes)}

#! validation datasets
#* Shane's sequencing for correlation
class ShaneSeqDataset(Dataset):
    '''Dataset for Shane's sequencing data.
    This is used to see the correlation between images and the ground truth gene expression.
    '''
    def __init__(self, path: str):
        super(ShaneSeqDataset, self).__init__()
        experiments = os.listdir(path)
        self.imgs = []
        self.experiments = []
        for exp in experiments:
            exp_dir = os.path.join(path, exp, "Phase")
            if os.path.isdir(exp_dir):
                # grab only the 02d images, since only those have sequencing data
                self.experiments.extend([f"{exp}_{img}" for img in os.listdir(exp_dir) if "02d" in img])
                self.imgs.extend([os.path.join(exp_dir, img) for img in os.listdir(exp_dir) if "02d" in img])

        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32),
        ])
    
    @staticmethod
    def extract_full_patches(img: torch.Tensor, patch_size=256):
        # img: (C, H, W)
        C, H, W = img.shape
        img_batched = img.unsqueeze(0)  # (1, C, H, W)

        # Only patches that fully fit will be returned
        patches = F.unfold(
            img_batched, 
            kernel_size=patch_size, 
            stride=patch_size
        )

        # Each column is a flattened patch
        patches = patches.transpose(1, 2)  # (1, num_patches, C*ps*ps)
        patches = patches.reshape(-1, C, patch_size, patch_size)
        return patches

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        exp = self.experiments[idx]
        cell_type = ("MCF10A" if "GFP" in exp else 
                     "HCT116" if "HCT" in exp else
                     "Mixed")
        treatment = ("NIR" if "NIR" in exp else "IR")
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        if img.max().item() > 1:
            img = img / 255.0
        # chop into patches
        imgs = self.extract_full_patches(img)  # (num_patches, 3, 256, 256)
        return imgs, (torch.tensor([0 if cell_type == "MCF10A" else 1 if cell_type == "HCT116" else 2]), 
                      img_path, cell_type, treatment, exp)

#* Shane's sequencing for cell type identification
class ShaneSeqCellTypeDataset(Dataset):
    '''Dataset for Shane's sequencing data for cell type identification.
    In this dataset, there are phase images and green chanel GFP images
    The more GFP, the more likely the cell is MCF10A. Otherwise, the cell is HCT116.
    The dataset will return the image and the green channel value (as a percentage of the total image size)

    This dataset class can also be used to estimate the amount of mitotic cells in the image if the GFP is a mitotic marker.
    '''
    def __init__(self, path, load_mitotic: bool = False):
        super(ShaneSeqCellTypeDataset, self).__init__()
        self.path = path
        self.imgs = []
        self.image_names = []
        self.gfp_imgs = []
        self.load_mitotic = load_mitotic
        if load_mitotic:
            self._load_data_coculture_mitotic()
        else:
            self._load_data_coculture()
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32),
        ])
    
    @staticmethod
    def extract_full_patches(img: torch.Tensor, patch_size=256):
        # img: (C, H, W)
        C, H, W = img.shape
        img_batched = img.unsqueeze(0)  # (1, C, H, W)

        # Only patches that fully fit will be returned
        patches = F.unfold(
            img_batched, 
            kernel_size=patch_size, 
            stride=patch_size
        )

        # Each column is a flattened patch
        patches = patches.transpose(1, 2)  # (1, num_patches, C*ps*ps)
        patches = patches.reshape(-1, C, patch_size, patch_size)
        return patches

    def _load_data_coculture(self):
        # Load images and GFP values from the dataset
        # this case, path is a str to the coculture directory
        for img in os.listdir(os.path.join(self.path, "Phase")):
            # if "02d" in img: # only load 02d images
            self.imgs.append(os.path.join(self.path, "Phase", img))
            # compute GFP values
            self.gfp_imgs.append(os.path.join(self.path, "GFP", img))
            self.image_names.append(img)
    
    def _load_data_coculture_mitotic(self):
        # self.path is a tuple of (phase_path, gfp_path)
        img_path = self.path[0]
        gfp_path = self.path[1]
        for img in os.listdir(img_path):
            self.imgs.append(os.path.join(img_path, img))
            self.gfp_imgs.append(os.path.join(gfp_path, img))
            self.image_names.append(img)

    @staticmethod
    def rgb_to_hsv(rgb: torch.Tensor) -> torch.Tensor:
        """
        rgb: (B,3,H,W) in [0,1]
        returns hsv: (B,3,H,W) with H∈[0,1], S∈[0,1], V∈[0,1]
        """
        r, g, b = rgb[:,0], rgb[:,1], rgb[:,2]  # (B,H,W)

        maxc, _ = rgb.max(dim=1)
        minc, _ = rgb.min(dim=1)
        v = maxc
        deltac = maxc - minc

        # Saturation
        s = torch.zeros_like(maxc)
        mask = maxc > 1e-6
        s[mask] = deltac[mask] / maxc[mask]

        # Hue
        h = torch.zeros_like(maxc)
        mask_r = (maxc == r) & (deltac > 0)
        mask_g = (maxc == g) & (deltac > 0)
        mask_b = (maxc == b) & (deltac > 0)

        h[mask_r] = ((g - b)[mask_r] / deltac[mask_r]) % 6
        h[mask_g] = ((b - r)[mask_g] / deltac[mask_g]) + 2
        h[mask_b] = ((r - g)[mask_b] / deltac[mask_b]) + 4

        h = h / 6.0  # normalize to [0,1]
        return torch.stack([h, s, v], dim=1)
    
    @staticmethod
    def rgb_to_lab(rgb):
        """
        Approximate RGB->Lab using D65 white point.
        rgb: (B,3,H,W) in [0,1], assumed sRGB.
        returns (B,3,H,W): L∈[0,100], a*, b*
        """
        # sRGB to linear
        mask = rgb > 0.04045
        rgb_lin = torch.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)

        # RGB to XYZ
        mat = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                            [0.2126729, 0.7151522, 0.0721750],
                            [0.0193339, 0.1191920, 0.9503041]], device=rgb.device)
        xyz = torch.einsum('ij,bjhw->bihw', mat, rgb_lin)

        # Normalize by D65
        xyz_ref = torch.tensor([0.95047, 1.0, 1.08883], device=rgb.device).view(1,3,1,1)
        xyz = xyz / xyz_ref

        # f(t) helper
        eps, kappa = 216/24389, 24389/27
        def f(t):
            return torch.where(t > eps, t**(1/3), (kappa * t + 16)/116)

        fX, fY, fZ = f(xyz[:,0]), f(xyz[:,1]), f(xyz[:,2])
        L = 116*fY - 16
        a = 500*(fX - fY)
        b = 200*(fY - fZ)
        return torch.stack([L,a,b], dim=1)

    @staticmethod
    def otsu_threshold(values, nbins=256):
        """Compute Otsu threshold for 1D values in [0,1]."""
        values = torch.tensor(values) # ensure tensor
        if values.numel() == 0:
            return 1.0
        hist = torch.histc(values, bins=nbins, min=0.0, max=1.0)
        prob = hist / hist.sum()
        omega = torch.cumsum(prob, 0)
        mu = torch.cumsum(prob * torch.arange(1,nbins+1, device=values.device), 0)
        mu_t = mu[-1]
        sigma_b = (mu_t*omega - mu)**2 / (omega*(1-omega) + 1e-9)
        idx = torch.argmax(sigma_b)
        return ((idx.float()+0.5)/nbins).item()

    def segment_cells_from_phase(self, phase_imgs, sigma=3):
        """
        phase_imgs: (B,3,H,W) in [0,1]
        Returns binary mask (B,H,W) of cells
        """
        from torchvision.transforms.functional import gaussian_blur

        # Smooth to reduce noise
        smooth = gaussian_blur(phase_imgs, kernel_size=11)

        # Invert (cells usually darker in phase)
        inv = 1.0 - smooth
        # inv = smooth

        masks = []
        for i in range(phase_imgs.shape[0]):
            thr = self.otsu_threshold(inv[i,0].flatten())
            mask = (inv[i,0] >= thr)
            # morphological cleanup
            mask = torch.nn.functional.max_pool2d(
                mask[None,None].float(), 3, stride=1, padding=1).bool()[0,0]
            masks.append(mask)
        return torch.stack(masks, dim=0)  # (B,H,W)

    def segment_cells_cellpose(self, imgs):
        '''
        Use CellPose to segment cells from phase or GFP images.
        imgs: (B,3,H,W) or (B,H,W) or (B,1,H,W) in [0,1]
        returns: (B,H,W) with integer labels of masks of cells
        '''
        model = models.CellposeModel(model_type='cyto3', gpu=torch.cuda.is_available())
        B = imgs.shape[0]
        all_masks = []
        for i in range(B):
            img = imgs[i]
            img_np = img.permute(1,2,0).cpu().numpy()  # (H,W,3) or (H,W) or (H,W,1)
            img_gray = img_np.mean(axis=2) if img_np.ndim == 3 else img_np.squeeze()  # (H,W)
            try:
                masks, _, _ = model.eval(img_gray, diameter=None, channels=[0,0],
                                        flow_threshold=0.4, cellprob_threshold=0, min_size=10) # (H,W) with integer labels
            except IndexError:
                # sometimes cellpose throws an index error, in that case return empty mask
                print("CellPose index error, returning empty mask")
                masks = np.zeros_like(img_gray, dtype=np.int32)
            all_masks.append(torch.tensor(masks, device=imgs.device).long())
        return torch.stack(all_masks, dim=0) # (B,H,W) with integer labels

    def _extract_gfp_value_threshold(self, batch_rgb: torch.Tensor, phase_imgs: torch.Tensor):
        """
        batch_rgb: (B,3,H,W) in [0,1]
        returns: (B,) tensor of % bright green area
        """
        from skimage.filters import threshold_otsu
        batch_size = batch_rgb.shape[0]
        gfp_vals = []
        # cell_masks = self.segment_cells_from_phase(phase_imgs)
        # thr = 55 / 255.0  # fixed threshold for GFP, select only very bright areas
        # mean = batch_rgb[:,1].mean()  # mean green channel per patch globally
        # std = batch_rgb[:,1].std()    # std dev of green channel per patch globally
        # thr = max(mean - 1 * std, 1.0)

        for i in range(batch_size):
            gfp_img = batch_rgb[i].cpu().numpy() # (3,H,W)
            # get only the green channel
            gfp_img = gfp_img[1,:,:]  # (H,W)
            bg_mean = np.mean(gfp_img)
            bg_std = np.std(gfp_img)
            thr = bg_mean + 1.5 * bg_std # dynamic threshold based on background

            # compute simple threshold
            bright_mask = gfp_img >= thr
            # # estimate % area of bright green cells
            # cell_mask = cell_masks[i].cpu().numpy()
            # cell_pixels = cell_mask.sum()
            # bright_pixels = (bright_mask & cell_mask).sum()
            # percent = bright_pixels / (cell_pixels + 1e-9)

            percent = bright_mask.sum() / (gfp_img.size + 1e-9)
            gfp_vals.append(percent)
        return torch.tensor(gfp_vals, device=batch_rgb.device) #shape (B,)

    def _extract_gfp_value_filter(self, batch_rgb: torch.Tensor, phase_imgs: torch.Tensor):
        from scipy.ndimage import gaussian_filter
        from skimage.restoration import rolling_ball
        from skimage import exposure
        from skimage.filters import apply_hysteresis_threshold

        batch_size = batch_rgb.shape[0]
        gfp_vals = []
        cell_masks = self.segment_cells_from_phase(phase_imgs)
        for i in range(batch_size):
            gfp_img = batch_rgb[i].cpu().numpy() # (3,H,W)
            # get only the green channel
            gfp_img = gfp_img[1,:,:]  # (H,W)
            # #* compute illumination-corrected GFP
            # illum = gaussian_filter(gfp_img, sigma=50)
            # gfp_corr = gfp_img / (illum + 1e-6)
            # gfp_corr = np.clip(gfp_corr, 0, None)
            # # Detect bright mitotic spots
            # thr = self.otsu_threshold(gfp_corr.flatten())
            # mitotic_mask = gfp_corr > (thr * 1.4)   # 1.4× Otsu = high-specificity threshold
            #* uses a rolling ball algorithm for background subtraction
            background = rolling_ball(gfp_img, radius=50)
            img_corr = gfp_img - background
            img_corr = np.clip(img_corr, 0, None)
            img_norm = exposure.rescale_intensity(img_corr, in_range="image", out_range=(0, 1))
            th_high = np.percentile(img_norm, 99.2) # high-specificity threshold
            th_low = th_high * 0.5
            mitotic_mask = apply_hysteresis_threshold(img_norm, th_low, th_high)
            # estimate % area of mitotic cells
            cell_mask = cell_masks[i].cpu().numpy()
            cell_pixels = cell_mask.sum()
            mitotic_pixels = (mitotic_mask & cell_mask).sum()
            percent = mitotic_pixels / (cell_pixels + 1e-9)
            gfp_vals.append(percent)
        return torch.tensor(gfp_vals, device=batch_rgb.device) #shape (B,)

    def _extract_gfp_value(self, batch_rgb: torch.Tensor, phase_imgs: torch.Tensor, 
                           sat_thresh=0.5, a_thresh=-6) -> float:
        """
        batch_rgb: (B,3,H,W) in [0,1]
        returns: (B,) tensor of % bright green area
        """
        hsv = self.rgb_to_hsv(batch_rgb)
        lab = self.rgb_to_lab(batch_rgb)

        H, S, V = hsv[:,0], hsv[:,1], hsv[:,2]
        L, a, b = lab[:,0], lab[:,1], lab[:,2]

        # Hue in green range (approx 70°–170° → 0.2–0.47 in [0,1])
        green_band = (H > 70/360) & (H < 100/360)
        sat_ok = S > sat_thresh
        a_green = a < a_thresh

        green_mask = green_band & sat_ok & a_green

        cell_mask = self.segment_cells_from_phase(phase_imgs)

        B,_,Hh,Ww = batch_rgb.shape
        results = []
        for i in range(B):
            vals = V[i][green_mask[i]]
            thr = self.otsu_threshold(vals.flatten())
            bright_mask = green_mask[i] & (V[i] >= thr) & cell_mask[i]
            cell_pixels = cell_mask[i].sum().item()
            bright_pixels = bright_mask.sum().item()
            percent = bright_pixels / (cell_pixels+1e-9)
            results.append(percent)
        return torch.tensor(results, device=batch_rgb.device) #shape (B,)
    
    @staticmethod
    def morphological_top_hat(gfp_tensor, kernel_size=65):
        """
        Applies Morphological Top-Hat transform to a batch of images.
        
        Args:
            gfp_tensor (torch.Tensor): Input tensor of shape (B, C, H, W).
                                    Values should typically be normalized (0-1).
            kernel_size (int): Size of the structural element (must be odd). 
                            Larger kernel = removes larger background blobs.
                            
        Returns:
            torch.Tensor: The Top-Hat transformed tensor (B, C, H, W).
        """
        
        # Ensure kernel_size is odd for symmetric padding
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        pad = kernel_size // 2

        # 1. EROSION (Min Pooling)
        # PyTorch doesn't have min_pool, so we do: -MaxPool(-x)
        eroded = -F.max_pool2d(-gfp_tensor, kernel_size=kernel_size, stride=1, padding=pad)

        # 2. DILATION (Max Pooling)
        # Dilation is simply Max Pooling with stride 1
        # We apply this to the eroded image to get the "Opening"
        opening = F.max_pool2d(eroded, kernel_size=kernel_size, stride=1, padding=pad)

        # 3. TOP-HAT
        # Original - Opening
        top_hat = gfp_tensor - opening
        
        # Clamp to ensure no negative values (optional but recommended for image data)
        return torch.clamp(top_hat, min=0.0) #shape (B, C, H, W)

    def calculate_gfp_structure_score(self, image_tensor, min_contrast=0.05):
        """
        Calculates GFP score based on structural contrast, ignoring bright noise.
        
        Args:
            image_tensor (Tensor): Input images (B, C, H, W). Scaled 0.0-1.0.
            min_contrast (float): Minimum standard deviation required to consider 
                                the image as containing "real" objects.
        """
        batch_size = image_tensor.shape[0]
        scores = []
        
        # Handle Grayscale vs RGB
        c_idx = 1 if image_tensor.shape[1] == 3 else 0
        
        for i in range(batch_size):
            img = image_tensor[i, c_idx:c_idx+1, :, :] # Keep (1, H, W) for pooling
            
            # --- STEP 1: BLUR (Remove high-freq noise) ---
            # We average small 5x5 blocks. Real cells survive this; noise disappears.
            blurred = F.avg_pool2d(img, kernel_size=5, stride=1, padding=2)
            
            # --- STEP 2: THE VARIANCE GATE (Check for Structure) ---
            # Calculate how much the pixel values "spread out"
            std_dev = torch.std(blurred)
            
            # If the image is flat (just grey noise or flat green), std_dev will be tiny.
            if std_dev < min_contrast:
                scores.append(0.0)
                continue
                
            # --- STEP 3: RELATIVE THRESHOLDING ---
            # If we passed the gate, we know there are objects.
            # We find them by looking at the histogram of the BLURRED image.
            
            # Find the range of the image (ignoring single hot pixels)
            p5 = torch.quantile(blurred, 0.05)
            p95 = torch.quantile(blurred, 0.95)
            
            # Define "Green" as anything significantly brighter than the bottom 5%
            # We use a dynamic threshold: Bottom + 20% of the range
            threshold = p5 + 0.2 * (p95 - p5)
            
            mask = (blurred > threshold).float()
            
            # Calculate Percentage
            green_pixels = torch.sum(mask)
            # total_pixels = mask.numel()
            total_pixels = img.numel()
            score = (green_pixels / total_pixels) * 100
            scores.append(score.item())

        return torch.tensor(scores) #shape (B,)

    def _top_hat_gfp(self, batch_rgb: torch.Tensor, phase_imgs: torch.Tensor, 
                    z_score_cutoff=3.0, min_absolute_threshold=0.05) -> float:
        """
        batch_rgb: (B,3,H,W) in [0,1]
        returns: (B,) tensor of % bright green area
        """
        B,_,Hh,Ww = batch_rgb.shape
        cell_mask = self.segment_cells_from_phase(phase_imgs)
        # green_masks = self.morphological_top_hat(batch_rgb[:,1:2,:,:], kernel_size=25)
        green_masks = batch_rgb[:,1:2,:,:]
        results = []
        for i in range(B):
            mean_val = torch.mean(green_masks[i,0])
            std_val = torch.std(green_masks[i,0])
            adaptive_thresh = mean_val + (z_score_cutoff * std_val)
            threshold = max(adaptive_thresh, min_absolute_threshold)
            bright_mask = (green_masks[i,0] >= threshold) & cell_mask[i]
            # vals = green_masks[i,0][cell_mask[i]]
            # thr = self.otsu_threshold(vals.flatten())
            # bright_mask = (green_masks[i,0] >= thr) & cell_mask[i]
            cell_pixels = cell_mask[i].sum().item()
            bright_pixels = bright_mask.sum().item()
            percent = bright_pixels / (cell_pixels+1e-9)
            results.append(percent)
        return torch.tensor(results, device=batch_rgb.device), cell_mask #shape ((B,), (B, H,W))

    def _extract_gfp_value_per_patch(self, batch_rgb: torch.Tensor, phase_imgs: torch.Tensor, 
                           sat_thresh=0.3, a_thresh=-6):
        """
        batch_rgb: (B,3,H,W) in [0,1]
        returns: (B,) tensor of % bright green area
        """
        B,_,Hh,Ww = batch_rgb.shape
        cell_mask = self.segment_cells_from_phase(phase_imgs)
        results = []
        for i in range(B):
            current_patch = batch_rgb[i].unsqueeze(0)  # (1,3,H,W)
            hsv = self.rgb_to_hsv(current_patch)
            lab = self.rgb_to_lab(current_patch)

            H, S, V = hsv[:,0], hsv[:,1], hsv[:,2]
            L, a, b = lab[:,0], lab[:,1], lab[:,2]

            # Hue in green range (approx 70°–170° → 0.2–0.47 in [0,1])
            green_band = (H > 70/360) & (H < 140/360)
            sat_ok = S > sat_thresh
            a_green = a < a_thresh

            green_mask = green_band & sat_ok & a_green
            vals = V[0][green_mask[0]]
            thr = self.otsu_threshold(vals.flatten())
            bright_mask = green_mask[0] & (V[0] >= thr) & cell_mask[i]
            cell_pixels = cell_mask[i].sum().item()
            bright_pixels = bright_mask.sum().item()
            percent = bright_pixels / (cell_pixels+1e-9)
            results.append(percent)
        return torch.tensor(results, device=batch_rgb.device) #shape (B,)
    
    def _extract_gfp_value_per_patch_cellpose(self, batch_rgb: torch.Tensor, phase_imgs: torch.Tensor):
        '''
        Use CellPose to segment cells from phase and GFP images.
        Then compute a percentage of the number of green cells over total cells.
        batch_rgb: (B,3,H,W) in [0,1]
        phase_imgs: (B,3,H,W) in [0,1]
        returns: (B,) tensor of % green cells
        '''
        B,_,Hh,Ww = batch_rgb.shape
        phase_seg = self.segment_cells_cellpose(phase_imgs)  # (B,H,W) integer labels
        gfp_seg = self.segment_cells_cellpose(batch_rgb)    # (B,H,W) integer labels
        results = []
        for i in range(B):
            phase_labels = torch.unique(phase_seg[i])
            gfp_labels = torch.unique(gfp_seg[i])
            num_phase_cells = phase_labels.shape[0] - (1 if 0 in phase_labels else 0)  # exclude background
            num_gfp_cells = gfp_labels.shape[0] - (1 if 0 in gfp_labels else 0)        # exclude background
            if num_phase_cells < 2 or num_gfp_cells < 2:
                percent = 0.0
            else:
                percent = num_gfp_cells / (num_phase_cells + 1e-9)
            results.append(percent)
        return torch.tensor(results, device=batch_rgb.device) #shape (B,)

    
    def calculate_gfp_score_gatekeeper(self, image_tensor, pcm_tensor, 
                                        noise_floor=0.15, saturation_check=False):
        """
        Robust GFP scoring using a Gatekeeper approach.
        
        Args:
            image_tensor (torch.Tensor): Input images (B, C, H, W). 
                                        Values MUST be 0.0 to 1.0.
            noise_floor (float): The absolute minimum intensity (0.0-1.0) for a pixel 
                                to be considered 'real' GFP. 0.15 (15%) is a good start.
                                
        Returns:
            scores (Tensor): Percentage of greenness (0-100)
        """
        batch_size = image_tensor.shape[0]
        # cell_mask = self.segment_cells_from_phase(pcm_tensor)
        # cell_mask = self.segment_cells_cellpose(pcm_tensor) > 0  # binary mask of cells
        scores = []
        
        # Ensure we are looking at the Green channel (usually channel 1 in RGB)
        # If input is already grayscale (B, 1, H, W), use channel 0
        c_idx = 1 if image_tensor.shape[1] == 3 else 0
        
        for i in range(batch_size):
            img = image_tensor[i, c_idx, :, :] # Extract Green Channel
            
            # --- STEP 1: THE GATEKEEPER ---
            # We calculate the 99th percentile brightness (robust max)
            # This tells us: "How bright is the brightest stuff in this image?"
            max_val = torch.quantile(img, 0.99)
            
            # If the brightest parts of the image are dimmer than our floor,
            # the whole image is just noise.
            if max_val < noise_floor:
                scores.append(0.0)
                continue
                
            # --- STEP 2: SIGNAL COUNTING ---
            # Since the image passed the gate, we count the pixels.
            # We use the same noise_floor as the threshold.
            
            # mask = (img > noise_floor) & cell_mask[i]  # Only consider pixels above noise floor within cells
            mask = (img > noise_floor)  # Only consider pixels above noise floor
            
            # total_pixels = cell_mask[i].sum()
            total_pixels = torch.numel(img)
            green_pixels = torch.sum(mask)
            
            # Calculate Percentage
            score = green_pixels / total_pixels
            scores.append(score.item())

        return torch.tensor(scores)  #shape (B,)

    def _mean_gfp_intensity(self, batch_rgb, phase_imgs):
        g = batch_rgb[:,1]   # assume GFP = channel 1
        # cell_mask = self.segment_cells_from_phase(phase_imgs)
        # return (g*cell_mask).sum(dim=[1,2]) / (cell_mask.sum(dim=[1,2])+1e-9) #shape (B,)
        thr = 40 / 255.0
        g = g * (g >= thr) # threshold to remove background
        return g.mean(dim=[1,2])  # shape (B,)

    def calculate_gfp_texture_score(self, image_tensor, window_size=9, texture_threshold=0.01):
        """
        Calculates GFP score based on TEXTURE (Local Variance), which is robust 
        to changing background brightness.
        
        Args:
            image_tensor: Input (B, C, H, W). Scaled 0.0-1.0.
            window_size: Size of the neighborhood to check for texture (approx cell size).
                        Must be an odd number (e.g., 9, 15).
            texture_threshold: Sensitivity. 
                            0.01 picks up faint cells. 
                            0.05 requires very sharp contrast.
        """
        batch_size = image_tensor.shape[0]
        scores = []
        
        # Ensure Green Channel
        c_idx = 1 if image_tensor.shape[1] == 3 else 0
        
        # Padding for the averaging window
        pad = window_size // 2
        
        for i in range(batch_size):
            img = image_tensor[i, c_idx:c_idx+1, :, :] # Keep (1, H, W)
            
            # --- THE MATH: Local Variance ---
            # Var(X) = E[X^2] - (E[X])^2
            
            # 1. Average of the squared image
            img_sq = img ** 2
            mean_sq = F.avg_pool2d(img_sq, kernel_size=window_size, stride=1, padding=pad)
            
            # 2. Square of the average image
            sq_mean = F.avg_pool2d(img, kernel_size=window_size, stride=1, padding=pad) ** 2
            
            # 3. Variance = Term 1 - Term 2
            # (Clamp to 0 to avoid tiny negative floating point errors)
            local_var = torch.clamp(mean_sq - sq_mean, min=0.0)
            
            # 4. Convert to Standard Deviation (more intuitive scale)
            local_std = torch.sqrt(local_var)
            
            # --- THE DECISION ---
            # If a region has high texture (std > threshold), it's a cell.
            # This ignores "Smooth Bright Green" AND "Smooth Black".
            mask = (local_std > texture_threshold).float()
            
            # Optional: Clean up speckles (Erosion)
            # mask = F.max_pool2d(-mask, 3, stride=1, padding=1) * -1 
            
            # Score
            green_pixels = torch.sum(mask)
            # total_pixels = mask.numel()
            total_pixels = img.numel()
            score = (green_pixels / total_pixels) * 100
            scores.append(score.item())

        return torch.tensor(scores)

    def calculate_gfp_bandpass_score(self, image_tensor, min_signal=0.05):
        """
        Calculates GFP score using Band-Pass filtering (Difference of Gaussians).
        This isolates "cell-sized" objects from both NOISE (too small) and 
        BACKGROUND HAZE (too big).
        
        Args:
            image_tensor: Input (B, C, H, W). Scaled 0.0-1.0.
            min_signal: Sensitivity. How much brighter than the local background 
                        must a cell be? (0.05 = 5% brighter).
        """
        batch_size = image_tensor.shape[0]
        scores = []
        
        # Ensure Green Channel
        c_idx = 1 if image_tensor.shape[1] == 3 else 0
        
        for i in range(batch_size):
            img = image_tensor[i, c_idx:c_idx+1, :, :] 
            
            # --- STEP 1: Gaussian Blurs ---
            # Sigma 2: Removes the "grain" (High frequency noise)
            blur_small = TF.gaussian_blur(img, kernel_size=9, sigma=2.0)
            
            # Sigma 20: Estimates the "background haze" (Low frequency trends)
            # We use a large kernel to average out the illumination
            blur_large = TF.gaussian_blur(img, kernel_size=61, sigma=20.0)
            
            # --- STEP 2: Band-Pass (Subtraction) ---
            # "Signal" = Structure that exists in small blur but not large blur
            # This deletes the flat glowing background effectively
            signal = blur_small - blur_large
            
            # --- STEP 3: Thresholding ---
            # We only count pixels that are locally brighter than their surroundings
            mask = (signal > min_signal).float()
            
            # Score
            green_pixels = torch.sum(mask)
            # total_pixels = mask.numel()
            total_pixels = img.numel()
            score = (green_pixels / total_pixels) * 100
            scores.append(score.item())

        return torch.tensor(scores)

    def calculate_gfp_bandpass_score_per_cell(self, image_tensor, phase_tensor, min_signal=0.05, integrated=False, saturation_point=0.15):
        """
        Calculates GFP score using Band-Pass filtering (Difference of Gaussians),
        normalized per cell area.
        
        Args:
            image_tensor: Input (B, C, H, W). Scaled 0.0-1.0.
            phase_tensor: Input (B, C, H, W). Scaled 0.0-1.0.
            min_signal: Sensitivity. How much brighter than the local background 
                        must a cell be? (0.05 = 5% brighter).
        """
        batch_size = image_tensor.shape[0]
        scores = []

        cell_masks = self.segment_cells_cellpose(phase_tensor) > 0  # binary mask of cells
        
        # Ensure Green Channel
        c_idx = 1 if image_tensor.shape[1] == 3 else 0
        
        for i in range(batch_size):
            # # save the phase tensor for debugging
            # tfm = v2.ToPILImage()
            # phase_img = tfm(phase_tensor[i].cpu())
            # phase_img.save(f"/home/zf2dong/scratch/temp/tem_phase_tensor/debug_phase_{i}.png")

            img = image_tensor[i, c_idx:c_idx+1, :, :] # shape： (1, H, W)
            cell_mask = cell_masks[i]
            
            # --- STEP 1: Gaussian Blurs ---
            blur_small = TF.gaussian_blur(img, kernel_size=9, sigma=2.0)
            blur_large = TF.gaussian_blur(img, kernel_size=61, sigma=20.0)
            
            # --- STEP 2: Band-Pass (Subtraction) ---
            signal = blur_small - blur_large
            
            # --- STEP 3: Thresholding ---
            mask = (signal > min_signal).float() * cell_mask.float()
                       
            # Score
            total_pixels = torch.sum(cell_mask)
            if integrated: # use integrated density, normalize by saturation point
                norm_signal = torch.clamp(signal / saturation_point, min=0.0, max=1.0)
                valid_signal = norm_signal * mask
                green_value = torch.sum(valid_signal)
                score = (green_value / (total_pixels) * 100).item()
            else:
                green_value = torch.sum(mask)
                # catch very small or very large cell areas, return negative score to indicate unreliable measuremen
                img_size = img.shape[1] * img.shape[2]
                if total_pixels < 0.2*img_size or total_pixels > 0.8*img_size:
                    score = -1.0
                else:
                    score = (green_value / (total_pixels) * 100).item()
            scores.append(score)

        return torch.tensor(scores)  #shape (B,)

    def _quantify_greenness_with_integrated_density(self, gfp_img: torch.Tensor):
        """
        Quantifies green fluorescence from an RGB image.
        
        Args:
            gfp_img (torch.Tensor): Shape (B, 3, H, W). 
                                   Assumes channel 0=R, 1=G, 2=B.
        
        Returns:
            torch.Tensor: Shape (B,). Integrated density of green fluorescence per image.
        """
        from skimage.filters import threshold_otsu
        # 1. Extract Green Channel
        # Shape is (3, H, W), so Green is at index 1
        green_channel = gfp_img[:, 1, :, :] # Shape (B, H, W)
        green_channel = green_channel.cpu().numpy()  # Convert to NumPy for processing

        # 3. Background Subtraction
        # We estimate background intensity by looking at the dimmest pixels.
        # A safe bet is the 5th percentile (to ignore dead pixels/absolute black).
        background_level = np.percentile(green_channel, 5)
        
        # Subtract background, clipping negative values to 0
        green_corrected = np.maximum(green_channel - background_level, 0)

        # get a global ostsu threshold across the batch
        # thresh_val = threshold_otsu(green_corrected.flatten())

        # # get a threshold using mean + 3*std
        # mean_intensity = np.mean(green_corrected)
        # std_intensity = np.std(green_corrected)
        # thresh_val = mean_intensity + 3 * std_intensity

        green_L = []
        for i in range(gfp_img.shape[0]):
            current_green = green_corrected[i]
            # get a threshold using mean and std
            mean_intensity = np.mean(current_green)
            std_intensity = np.std(current_green)
            thresh_val = mean_intensity + 2 * std_intensity

            binary_mask = current_green > thresh_val
            green = np.sum(current_green[binary_mask]) # shape: scalar
         
            # Mean Intensity: Average brightness of the spots only
            # green = np.mean(current_green[binary_mask]) if np.sum(binary_mask) > 0 else 0.0
            
        #     # Spot Area: How many pixels are glowing?
        #     spot_area_pixels = np.sum(binary_mask)

            # return {
            #     "integrated_density": integrated_density, # Best for "Total Amount"
            #     "mean_intensity": mean_intensity,         # Best for "Concentration"
            #     "background_level": background_level,
            #     "mask": binary_mask
            # }
            green_L.append(green)
        return torch.tensor(green_L, device=gfp_img.device)  # shape (B,)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img_name = self.image_names[idx]
        gfp_path = self.gfp_imgs[idx]
        img = self.transform(Image.open(img_path).convert("RGB"))
        gfp_img = self.transform(Image.open(gfp_path).convert("RGB"))
        if img.max().item() > 1:
            img = img / 255.0
        if gfp_img.max().item() > 1:
            gfp_img = gfp_img / 255.0
        # only get the central part the image to avoid edge effects
        _, H, W = img.shape
        h_start = int(0.1 * H)
        h_end = int(0.9 * H)
        w_start = int(0.1 * W)
        w_end = int(0.9 * W)
        img = img[:, h_start:h_end, w_start:w_end]
        gfp_img = gfp_img[:, h_start:h_end, w_start:w_end]
        # make them all 256x256 patches
        img_patches = self.extract_full_patches(img)
        gfp_patches = self.extract_full_patches(gfp_img)
        # gfp_bright = self.morphological_top_hat(gfp_img.unsqueeze(0))[0]
        # gfp_patches = self.extract_full_patches(gfp_bright)
        #* compute the gfp value for each patch
        # gfp_values = self._extract_gfp_value(gfp_patches, img_patches).view(-1, 1) #* normalize by cell area through a rough segmentation; compute per big image
        # gfp_values = self._extract_gfp_value_per_patch(gfp_patches, img_patches).view(-1, 1) #* same as above but compute per patch
        # gfp_values, cell_masks = self._top_hat_gfp(gfp_patches, img_patches) #* top-hat morphological transform to enhance bright spots; normalize by cell area
        # gfp_values, cell_masks = self.calculate_gfp_score_gatekeeper(gfp_patches, img_patches) #* gatekeeper method for robust GFP scoring
        # gfp_values = self.calculate_gfp_score_gatekeeper(gfp_patches, img_patches, noise_floor=0.5)  #* gatekeeper method for robust GFP scoring
        # gfp_values = self._extract_gfp_value_per_patch_cellpose(gfp_patches, img_patches)  #* cellpose segmentation for robust cell masking
        # gfp_values = self.calculate_gfp_structure_score(gfp_patches, min_contrast=0.1)  #* structure-based GFP scoring
        # gfp_values = self.calculate_gfp_texture_score(gfp_patches, window_size=9, texture_threshold=0.02)  #* texture-based GFP scoring
        # gfp_values = self.calculate_gfp_bandpass_score(gfp_patches, min_signal=0.1)  #* band-pass filtering GFP scoring, normaliuzed by total area
        # gfp_values = self._mean_gfp_intensity(gfp_patches, img_patches).view(-1, 1)
        # gfp_values = self._extract_gfp_value_filter(gfp_patches, img_patches).view(-1, 1)
        # gfp_values = self._extract_gfp_value_threshold(gfp_patches, img_patches).view(-1, 1)
        if self.load_mitotic:
            gfp_values = self.calculate_gfp_bandpass_score_per_cell(gfp_patches, img_patches, min_signal=0.01, integrated=True)  #* band-pass filtering GFP scoring, normalized by cell area, integrated density
            # gfp_values = self.calculate_gfp_bandpass_score_per_cell(gfp_patches, img_patches, min_signal=0.07, integrated=False)  #* band-pass filtering GFP scoring, normalized by cell area, no integrated density
        else:
            gfp_values = self.calculate_gfp_bandpass_score_per_cell(gfp_patches, img_patches, min_signal=0.07, integrated=False)  #* band-pass filtering GFP scoring, normalized by cell area
        gfp_values = gfp_values.view(-1, 1)
        return img_patches, (gfp_values, img_name, gfp_patches)#, cell_masks)
    
#* Shane's seuqnecing for confluency
class ShaneSeqConfluencyDataset(Dataset):
    '''
    Dataset for extracting confluency features from Shane's sequencing data.
    In this dataset, only none IR images are used.
    Images are in 3 classes:
    - day 0: low confluency
    - day 1: medium confluency
    - day 2: high confluency
    '''
    def __init__(self, path: str):
        super(ShaneSeqConfluencyDataset, self).__init__()
        experiments = os.listdir(path)
        self.imgs = []
        self.experiments = []
        self.cell_types = []
        for exp in experiments:
            exp_dir = os.path.join(path, exp, "Phase")
            if os.path.isdir(exp_dir) and ("NIR" in exp): # only use NIR images
                # grab only the 00d, 01d, and 02d images
                self.experiments.extend([f"{'day0' if '00h' in img else 'day1' if '23h' in img else 'day2'}" for img in os.listdir(exp_dir)])
                self.cell_types.extend([ "MCF10A" if "GFP" in exp else
                                         "HCT116" if "HCT" in exp else
                                         "Mixed"]*len(os.listdir(exp_dir)))
                self.imgs.extend([os.path.join(exp_dir, img) for img in os.listdir(exp_dir)])

        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32),
        ])
    
    @staticmethod
    def extract_full_patches(img: torch.Tensor, patch_size=256):
        # img: (C, H, W)
        C, H, W = img.shape
        img_batched = img.unsqueeze(0)  # (1, C, H, W)

        # Only patches that fully fit will be returned
        patches = F.unfold(
            img_batched, 
            kernel_size=patch_size, 
            stride=patch_size
        )

        # Each column is a flattened patch
        patches = patches.transpose(1, 2)  # (1, num_patches, C*ps*ps)
        patches = patches.reshape(-1, C, patch_size, patch_size)
        return patches

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        exp_name = self.experiments[idx]
        img = self.transform(Image.open(img_path).convert("RGB"))
        if img.max().item() > 1:
            img = img / 255.0
        imgs = self.extract_full_patches(img)  # (num_patches, 3, 256, 256)
        return imgs, (torch.tensor([0 if exp_name == "day0" else 1 if exp_name == "day1" else 2]), 
                      img_path, self.cell_types[idx], exp_name, f"{self.cell_types[idx]}_{exp_name}")

#* mitosis dataset
class MitoticDataset(Dataset):
    '''
    Dataset for external mitosis images from the Asmar et al. (2024) dataset.
    The mitotic percentage is calculated using mitosis_inferenced images, where
    mitotic_per = (number of 2 + 3) / (number of 1 + 2 + 3)
    '''
    def __init__(self, path: str):
        super(MitoticDataset, self).__init__()
        self.imgs = [os.path.join(path, "phase", img) for img in os.listdir(os.path.join(path, "phase")) if img.endswith(".tif")]
        self.mitosis_labels = [os.path.join(path, "mitosis_inferenced", img) for img in os.listdir(os.path.join(path, "mitosis_inferenced")) if img.endswith(".tif")]
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32),
        ])
    
    def __len__(self):
        return len(self.imgs)
    
    @staticmethod
    def extract_full_patches(img: torch.Tensor, patch_size=256):
        # img: (C, H, W)
        C, H, W = img.shape
        img_batched = img.unsqueeze(0)  # (1, C, H, W)

        # Only patches that fully fit will be returned
        patches = F.unfold(
            img_batched, 
            kernel_size=patch_size, 
            stride=patch_size
        )

        # Each column is a flattened patch
        patches = patches.transpose(1, 2)  # (1, num_patches, C*ps*ps)
        patches = patches.reshape(-1, C, patch_size, patch_size)
        return patches
    
    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        mitosis_path = self.mitosis_labels[idx]
        img = self.transform(Image.open(img_path).convert("RGB"))
        mit = Image.open(mitosis_path)
        mitosis_label = torch.tensor(np.array(mit)).unsqueeze(0).to(torch.float32) # (1, H, W)
        if img.max().item() > 1:
            img = img / 255.0
        imgs = self.extract_full_patches(img)  # (num_patches, 3, 256, 256)
        mitosis_labels = self.extract_full_patches(mitosis_label) # (num_patches, 1, 256, 256)
        mitosis_percentages = [0 if (torch.isclose(ml, torch.tensor(2.0))).sum().item() == 0
                                else (torch.isclose(ml, torch.tensor(2.0))).sum().item() / ((torch.isclose(ml, torch.tensor(1.0))).sum().item() 
                                + (torch.isclose(ml, torch.tensor(2.0))).sum().item() + (torch.isclose(ml, torch.tensor(3.0))).sum().item()) 
                                for ml in mitosis_labels] # (num_patches,)
        return imgs, (mitosis_percentages, img_path)
