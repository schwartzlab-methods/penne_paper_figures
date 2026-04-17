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
                score = (green_value / (total_pixels + 1e-8) * 100).item()
            else:
                green_value = torch.sum(mask)
                # catch very small or very large cell areas, return negative score to indicate unreliable measurement
                img_size = img.shape[1] * img.shape[2]
                if total_pixels < 0.2*img_size or total_pixels > 0.8*img_size:
                    score = -1.0
                else:
                    score = (green_value / (total_pixels) * 100).item()
            scores.append(score)

        return torch.tensor(scores)  #shape (B,)

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
        #* compute the gfp value for each patch
        if self.load_mitotic:
            gfp_values = self.calculate_gfp_bandpass_score_per_cell(gfp_patches, img_patches, min_signal=0.01, integrated=True)  #* band-pass filtering GFP scoring, normalized by cell area, integrated density
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
