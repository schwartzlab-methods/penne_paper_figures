'''
Process VisiumHD data
For each image in the tissue_img directory, generate a truncated matrix file in the mtx directory
This matrix only contains the barcodes in the image and combines the gene expression values
'''
import os
import numpy as np
import pandas as pd
import argparse
import scanpy as sc
import anndata as ad
import concurrent.futures

def find_barcodes(x1, y1, x2, y2, position_matrix) -> list:
    barcodes = position_matrix[(position_matrix['pxl_row_in_fullres'] >= x1) & (position_matrix['pxl_row_in_fullres'] <= x2) 
                       & (position_matrix['pxl_col_in_fullres'] >= y1) & (position_matrix['pxl_col_in_fullres'] <= y2)]
    barcodes_L = barcodes['barcode'].values
    return barcodes_L

def combine_barcodes(anndata_mtx, barcodes):
    '''
    Return a new Anndata objects with only the barcodes in the list
    '''
    new_mtx = anndata_mtx[anndata_mtx.obs.index.isin(barcodes)]
    # combine all gene expression values
    new_mtx = new_mtx.X.sum(axis=0)
    # get the number of transcripts
    num_transcripts = new_mtx.sum()
    new_mtx = ad.AnnData(np.array(new_mtx))
    new_mtx.var_names = anndata_mtx.var_names
    # normalization
    # sc.pp.normalize_total(new_mtx, target_sum=1e6)
    # sc.pp.log1p(new_mtx)
    return new_mtx, num_transcripts

def process_image(dir, each, x_off, y_off, position_matrix, cell_matrix, name):
    # generate new anndata object for each image
    barcodes = find_barcodes(x_off, y_off, x_off+224, y_off+224, position_matrix)
    new_mtx, num_trans = combine_barcodes(cell_matrix, barcodes)
    print(f"Found {len(barcodes)} barcodes in {each}. Total number of transcripts: {num_trans}")
    new_mtx.write(os.path.join(dir, f"{name}.h5ad"))

def main(dir, output, paraquet, cellranger):
    if not os.path.exists(output):
        os.makedirs(output)
    position_matrix = pd.read_parquet(paraquet)
    cell_matrix = sc.read_10x_mtx(cellranger)
    print("Files loaded. Start processing")
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        for each in os.listdir(dir):
            x_off, y_off = [int(i) for i in each.split("_")[:2]]
            executor.submit(process_image, output, each, x_off, y_off, position_matrix, cell_matrix, 
                            each.split(".")[0])
        print("All images have been processed")

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dir', type=str, help='Directory containing the VisiumHD patches')
    argparser.add_argument('--paraquet', type=str, help='Paraquet file containing the regions of the image')
    argparser.add_argument('--cellranger', type=str, help='Cellranger output file directory')
    argparser.add_argument('--output_dir', type=str, help='Output directory')
    args = argparser.parse_args()
    main(args.dir, args.output_dir, args.paraquet, args.cellranger)