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
            v2.RandomCrop((256,256)),
            v2.Resize((256, 256)),
        ])
        self._write_attributes() # this will write class_to_idx and targets
    
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
        img = img / 255 # rescale to [0,1]
        img = torch.clamp(img, max=1, min=0) #ensure no float overflow
        cls_idx = self.targets[idx]
        cls = self.classes[idx]
        # retrun the image as x and the class int label, image path, and img cls as y
        return img, (cls_idx, self.images[idx], cls)
    
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
    def __init__(self, tissue_dir: str, mtx_dir: str, livecell_dir: str):
        '''Initialize the VisiumHD and LIVECell dataset.

        Args:
            tissue_dir (str): Path to the tissue image directory.
            mtx_dir (str): Path to the matrix directory.
            livecell_dir (str): Path to the LIVECell image main directory.
        '''
        super(VisiumHD_Livecell_Dataset, self).__init__()
        self.tissue_dir = tissue_dir
        self.mtx_dir = mtx_dir
        self.imgs = np.array(os.listdir(tissue_dir))
        self.mtxs = np.array(os.listdir(mtx_dir))
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
        mtx_path = os.path.join(self.mtx_dir, f"{name}.npy")
        image = Image.open(he_image_path).convert('RGB')
        mtx = np.load(mtx_path)
        # put in tensor
        image = self.he_transforms(image) / 255 # scale to [0,1]
        mtx_tensor = torch.tensor(mtx).float().view(-1)
        # select a random image from the livecell dataset
        livecell_path = self.livecell_path[idx % len(self.livecell_path)]
        livecell_img = Image.open(livecell_path).convert('RGB')
        livecell_img = self.pcm_transforms(livecell_img) / 255 # scale to [0,1]
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
                # grab only the 02d images
                self.experiments.extend([f"{exp}_{img[9]}" for img in os.listdir(exp_dir) if "02d" in img ])
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
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        if img.max().item() > 1:
            img = img / 255.0
        # chop into patches
        imgs = self.extract_full_patches(img)  # (num_patches, 3, 256, 256)
        return imgs, (torch.tensor([0]), img_path, exp)

#* Shane's seuqnecing for cell type identification
class ShaneSeqCellTypeDataset(Dataset):
    '''Dataset for Shane's sequencing data for cell type identification.
    In this dataset, there are phase images and green chanel GFP images
    The more GFP, the more likely the cell is MCF10A. Otherwise, the cell is HCT116.
    The dataset will return the image and the green channel value (as a percentage of the total image size)
    '''
    def __init__(self, path: str):
        super(ShaneSeqCellTypeDataset, self).__init__()
        self.path = path
        self.imgs = []
        self.image_names = []
        self.gfp_imgs = []
        self._load_data()
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

    def _load_data(self):
        # Load images and GFP values from the dataset
        for img in os.listdir(os.path.join(self.path, "Phase")):
            self.imgs.append(os.path.join(self.path, "Phase", img))
            # compute GFP values
            self.gfp_imgs.append(os.path.join(self.path, "GFP", img))
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

    def otsu_threshold(values, nbins=256):
        """Compute Otsu threshold for 1D values in [0,1]."""
        if values.numel() == 0:
            return 1.0
        hist = torch.histc(values, bins=nbins, min=0.0, max=1.0)
        prob = hist / hist.sum()
        omega = torch.cumsum(prob, 0)
        mu = torch.cumsum(prob * torch.arange(1,nbins+1, device=values.device), 0)
        mu_t = mu[-1]
        sigma_b = (mu_t*omega - mu)**2 / (omega*(1-omega) + 1e-9)
        idx = torch.argmax(sigma_b)
        return (idx.float()+0.5)/nbins
    
    def _extract_gfp_value(self, batch_rgb: torch.Tensor, sat_thresh=0.3, a_thresh=-6) -> float:
        """
        batch_rgb: (B,3,H,W) in [0,1]
        returns: (B,) tensor of % bright green area
        """
        hsv = self.rgb_to_hsv(batch_rgb)
        lab = self.rgb_to_lab(batch_rgb)

        H, S, V = hsv[:,0], hsv[:,1], hsv[:,2]
        L, a, b = lab[:,0], lab[:,1], lab[:,2]

        # Hue in green range (approx 70°–170° → 0.2–0.47 in [0,1])
        green_band = (H > 70/360) & (H < 170/360)
        sat_ok = S > sat_thresh
        a_green = a < a_thresh

        green_mask = green_band & sat_ok & a_green

        B,_,Hh,Ww = batch_rgb.shape
        results = []
        for i in range(B):
            vals = V[i][green_mask[i]]
            thr = self.otsu_threshold(vals.flatten())
            bright_mask = green_mask[i] & (V[i] >= thr)
            percent = bright_mask.float().mean()
            results.append(percent)
        return torch.tensor(results, device=batch_rgb.device) #shape (B,)

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
        # make them all 256x256 patches
        img_patches = self.extract_full_patches(img)
        gfp_patches = self.extract_full_patches(gfp_img)
        # compute the gfp value for each patch
        gfp_values = self._extract_gfp_value(gfp_patches).view(-1, 1)
        return img_patches, (gfp_values, img_name)