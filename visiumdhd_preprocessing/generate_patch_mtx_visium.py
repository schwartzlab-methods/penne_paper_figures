'''
Process Visium regular data
Chop the original image into patches according to the location of each barcode
Generate a truncated matrix file in the mtx directory for each patch
This matrix only the gene expression values for the barcode present in the image
'''
import os
import numpy as np
import pandas as pd
import argparse
import scanpy as sc
import anndata as ad
import concurrent.futures
import tifffile
from PIL import Image
from tqdm import tqdm

def process(img: np.array, position_matrix: pd.DataFrame, 
            cell_matrix: ad.AnnData, name: str, each_barcode: str,
            out: dir) -> None:
    '''Process a single image patch and its corresponding spatial and expression data.

    Args:
        img (np.array): The image patch to process.
        position_matrix (pd.DataFrame): The position matrix containing barcode locations.
        cell_matrix (ad.AnnData): The cell matrix containing gene expression data.
        name (str): The name of the output files.
        each_barcode (str): The barcode corresponding to the current patch.
        out (dir): The output directory.
    '''
    try:
        img_width = img.shape[1]
        img_height = img.shape[0]
        x = position_matrix.loc[position_matrix.barcode == each_barcode, 'pxl_col_in_fullres'].values[0]
        y = position_matrix.loc[position_matrix.barcode == each_barcode, 'pxl_row_in_fullres'].values[0]
        mtx = cell_matrix[cell_matrix.obs_names == each_barcode].X
        # check if save is needed
        if x < 112 or y < 112 or x + 112 > img_width or y + 112 > img_height:
            return None  # Skip if the patch is out of bounds
        if mtx.shape[0] == 0:
            return None  # Skip if there are no barcode found
        if mtx.sum() == 0:
            return None  # Skip if there are no expression values found
        # process image
        cropped_image = img[y - 112:y + 112, x - 112:x + 112]
        Image.fromarray(cropped_image).save(os.path.join(out, "tissue_img", f"{name}_{each_barcode}.png"))
        # process exp
        np.save(os.path.join(out, "mtx", f"{name}_{each_barcode}.npy"), mtx)
    except Exception as e:
        print(e)

def find_common_genes(cell_matrices: list, out: str) -> list:
    '''Find common genes across all cell matrices.

    Args:
        cell_matrices (list): A list of AnnData objects containing cell matrices.
        out (str): The output directory.

    Returns:
        list: A list of common gene names.
    '''
    genes_list = []
    for cell_matrix in cell_matrices:
        genes_list.append(set(cell_matrix.var_names))
    common_genes = set.intersection(*genes_list)
    common_genes_L = sorted(list(common_genes))
    with open(os.path.join(out, "common_genes.txt"), "w") as f:
        for gene in sorted(common_genes_L):
            f.write(f"{gene}\n")
    return common_genes_L

def main(all_dir: str, output: str, common_genes: str) -> None:
    sample_dir = [os.path.join(all_dir, each) for each in os.listdir(all_dir)]
    mtx_save = os.path.join(output, "mtx")
    img_save = os.path.join(output, "tissue_img")
    if not os.path.exists(mtx_save):
        os.makedirs(mtx_save)
    if not os.path.exists(img_save):
        os.makedirs(img_save)
    pos_mtx_list = []
    cell_mtx_list = []
    img_list = []
    output_name = os.path.dirname(output)
    for each in tqdm(sample_dir):
        if each.endswith(output_name):
            continue
        # find the image
        for file in os.listdir(each):
            if file.endswith(".tif") or file.endswith(".tiff") or file.endswith(".btf"):
                img_list.append(os.path.join(each, file))
        try:
            position_matrix = pd.read_csv(os.path.join(each, "spatial",
                                                   "tissue_positions_list.csv"), sep=",", header=None)
        except FileNotFoundError:
            position_matrix = pd.read_csv(os.path.join(each, "spatial",
                                                   "tissue_positions.csv"), sep=",", header=None)
        position_matrix.columns = ['barcode', 'in_tissue', 'array_row', 'array_col', 'pxl_row_in_fullres', 'pxl_col_in_fullres']
        cell_matrix = sc.read_10x_mtx(os.path.join(each, "filtered_feature_bc_matrix"))
        # normalization
        sc.pp.normalize_total(cell_matrix, target_sum=1e6)
        sc.pp.log1p(cell_matrix)
        pos_mtx_list.append(position_matrix)
        cell_mtx_list.append(cell_matrix)
    if common_genes:
        with open(common_genes, "r") as f:
            common_genes_list = [line.strip() for line in f.readlines()]
    else:
        common_genes_list = find_common_genes(cell_mtx_list, output)
    print("Common genes number: ", len(common_genes_list))
    # filter the cell matrices to only contain common genes
    for i in range(len(cell_mtx_list)):
        cell_mtx_list[i] = cell_mtx_list[i][:,cell_mtx_list[i].var["gene_ids"].isin(common_genes_list)].copy()
        # sort according to the common genes
        cell_mtx_list[i].var_names = cell_mtx_list[i].var_names.astype(str)
        cell_mtx_list[i] = cell_mtx_list[i][:,cell_mtx_list[i].var.sort_values("gene_ids").index].copy()
    # process by saving the cropped image and mtx according to the coordinates
    print("Files loaded and matrix normalized. Start processing")
    total_tasks = sum(len(pos_mtx.barcode) for pos_mtx in pos_mtx_list)
    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        futures = []
        for i in tqdm(range(len(img_list))):
            img = tifffile.imread(img_list[i])
            name = os.path.basename(os.path.dirname(img_list[i]))
            for each_barcode in pos_mtx_list[i].barcode:
                futures.append(executor.submit(process, img, pos_mtx_list[i], cell_mtx_list[i], name, each_barcode, output))
        for _ in tqdm(concurrent.futures.as_completed(futures), total=total_tasks, desc="Processing patches"):
            pass
    print("All images have been processed")

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dir', type=str, help='Main directory containing the Visium image')
    argparser.add_argument('--common_genes', type=str, default=None, help='Path to the common genes file')
    argparser.add_argument('--output_dir', type=str, help='Output directory')
    args = argparser.parse_args()
    main(args.dir, args.output_dir, args.common_genes)